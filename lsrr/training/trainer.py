import time
import math
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from lsrr.utils.logging import ExperimentTracker
from lsrr.losses.composite import CompositeLoss
from lsrr.utils.flops import count_parameters

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: Optional[float] = None,
    min_lr_ratio: Optional[float] = None
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a 1-cycle half-cosine learning rate schedule without restarts (Mamba-2 recipe).
    - 0–10% steps (0 to num_warmup_steps): Linear warmup from 0 to eta_max (base_lr)
    - 10–100% steps (num_warmup_steps to num_training_steps - 1):
      Cosine decay following:
        LR(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))
      where at the final step (last iteration of last epoch), LR(T) = eta_min = 1e-5 exactly.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    if min_lr is not None:
        min_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
    elif min_lr_ratio is not None:
        min_ratio = float(min_lr_ratio)
    else:
        min_ratio = 1e-5 / float(base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # t / T where at final step (num_training_steps - 1), progress = 1.0 -> LR = eta_min
        total_decay_steps = max(1, num_training_steps - 1 - num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / float(total_decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_batch_accuracy(
    preds: torch.Tensor,
    target_ids: torch.Tensor,
    target_lens: Optional[torch.Tensor] = None
) -> Tuple[int, int, int, int]:
    """Computes exact match correct count, total samples, correct token count, total tokens.
    Returns:
        (exact_match_correct, total_samples, token_correct, total_tokens)
    """
    B = preds.size(0)
    exact_match_correct = 0
    token_correct = 0
    total_tokens = 0

    preds_cpu = preds.detach().cpu()
    targets_cpu = target_ids.detach().cpu()

    for b in range(B):
        if target_lens is not None:
            t_len = int(target_lens[b].item())
        else:
            t_len = targets_cpu.size(1)

        if t_len == 0:
            continue

        p_seq = preds_cpu[b, :t_len]
        t_seq = targets_cpu[b, :t_len]

        min_len = min(p_seq.size(0), t_seq.size(0))
        if min_len > 0:
            matching_tokens = (p_seq[:min_len] == t_seq[:min_len]).sum().item()
            token_correct += matching_tokens
            total_tokens += t_len

            if min_len == t_len and matching_tokens == t_len:
                exact_match_correct += 1
        else:
            total_tokens += t_len

    return exact_match_correct, B, token_correct, max(1, total_tokens)

class Trainer:
    """Trainer supporting AMP, gradient clipping, per-epoch checkpointing, accuracy logging, SGDR cosine LR schedule, and resuming."""
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: CompositeLoss,
        cfg: Dict[str, Any],
        tracker: ExperimentTracker,
        device: torch.device,
        resume_checkpoint: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        dataset_mod: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.cfg = cfg
        self.tracker = tracker
        self.device = device
        self.tokenizer = tokenizer
        self.dataset_mod = dataset_mod

        train_cfg = cfg.get("train", {})
        self.epochs = train_cfg.get("epochs", 5)
        self.lr = float(train_cfg.get("lr", 3e-4))
        self.min_lr = float(train_cfg.get("min_lr", 1e-5))
        self.warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-2))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.use_amp = bool(train_cfg.get("amp", False) and device.type == "cuda")
        # Generation budget for validation accuracy, independent of the gold answer length.
        self.eval_max_new_tokens = int(cfg.get("eval", {}).get("max_new_tokens", 32))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Mamba-2 1-cycle half-cosine decay without restarts:
        # 0–10% steps: linear warmup from 0 to lr (eta_max)
        # 10–100% steps: half-cosine decay reaching exactly min_lr (1e-5) at the final step
        self.total_steps = max(1, self.epochs * len(self.train_loader))
        self.num_warmup_steps = max(1, int(self.total_steps * self.warmup_ratio))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
            min_lr=self.min_lr
        )

        param_stats = count_parameters(self.model)
        print(f"[Trainer] Model parameters: {param_stats}")
        print(f"[Trainer] LR Schedule: Warmup {self.num_warmup_steps}/{self.total_steps} steps (0–{self.warmup_ratio*100:.0f}%), 1-cycle cosine decay {self.lr:.2e} -> {self.min_lr:.2e} at final step.")

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history: List[Dict[str, Any]] = []

        # Restore from resume_checkpoint if provided
        if resume_checkpoint is not None:
            if isinstance(resume_checkpoint, dict) and "model_state_dict" in resume_checkpoint:
                self.model.load_state_dict(resume_checkpoint["model_state_dict"])
                if "optimizer_state_dict" in resume_checkpoint:
                    self.optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
                if self.use_amp and resume_checkpoint.get("scaler_state_dict") is not None:
                    self.scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
                self.start_epoch = resume_checkpoint.get("epoch", 0) + 1
                self.best_val_loss = resume_checkpoint.get("best_val_loss", float("inf"))
                self.best_val_acc = resume_checkpoint.get("best_val_acc", 0.0)
                self.history = list(resume_checkpoint.get("history", []))

                prev_epochs = resume_checkpoint.get("epochs", self.epochs)
                if prev_epochs == self.epochs and "scheduler_state_dict" in resume_checkpoint and resume_checkpoint["scheduler_state_dict"] is not None:
                    self.scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
                else:
                    import warnings
                    curr_step = self.start_epoch * len(self.train_loader)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        for _ in range(curr_step):
                            self.scheduler.step()

                print(f"[Trainer] Resumed from epoch {self.start_epoch + 1}/{self.epochs} (Best Val Acc: {self.best_val_acc*100:.2f}%)")
            else:
                self.model.load_state_dict(resume_checkpoint)
                print("[Trainer] Loaded model state_dict weights.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        step_count = 0
        total_exact = 0
        total_samples = 0
        total_token_correct = 0
        total_tokens = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch in pbar:
            H = batch["H"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            target_lens = batch.get("target_lens")
            if target_lens is not None and isinstance(target_lens, torch.Tensor):
                target_lens = target_lens.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(H, target_ids=target_ids, is_eval=False)
                loss_dict = self.loss_fn(outputs, {"target_ids": target_ids})
                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            step_count += 1

            curr_lr = self.optimizer.param_groups[0]["lr"]
            # Compute accuracy
            logits = outputs.get("logits")
            if logits is not None:
                preds = logits.argmax(dim=-1)
                e_corr, b_tot, t_corr, t_tot = compute_batch_accuracy(preds, target_ids, target_lens)
                total_exact += e_corr
                total_samples += b_tot
                total_token_correct += t_corr
                total_tokens += t_tot
                curr_acc = total_exact / max(1, total_samples)
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{curr_acc:.4f}", "lr": f"{curr_lr:.2e}"})
            else:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{curr_lr:.2e}"})

            # Log diagnostics
            if outputs.get("diagnostics"):
                diags_summary = [d.delta_state for d in outputs["diagnostics"]]
                self.tracker.log_diagnostics({
                    "epoch": epoch,
                    "step": step_count,
                    "loss": loss_val,
                    "deltas": diags_summary
                })

        avg_loss = total_loss / max(1, step_count)
        train_acc = total_exact / max(1, total_samples) if total_samples > 0 else 0.0
        train_tok_acc = total_token_correct / max(1, total_tokens) if total_tokens > 0 else 0.0

        return {
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "train_token_acc": train_tok_acc
        }

    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        step_count = 0
        total_cycles = 0
        sample_count = 0
        total_exact = 0
        total_samples = 0
        total_token_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                H = batch["H"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                target_lens = batch.get("target_lens")
                if target_lens is not None and isinstance(target_lens, torch.Tensor):
                    target_lens = target_lens.to(self.device)

                # 1. Validation loss via shifted teacher-forcing (uncheated cross-entropy)
                outputs = self.model(H, target_ids=target_ids, is_eval=True)
                loss_dict = self.loss_fn(outputs, {"target_ids": target_ids})
                total_loss += loss_dict["total_loss"].item()
                step_count += 1

                if outputs.get("stopping_cycles") is not None:
                    total_cycles += outputs["stopping_cycles"].sum().item()
                    sample_count += outputs["stopping_cycles"].numel()

                # 2. Validation Accuracy: Real Autoregressive Generation without target_ids!
                # Fixed budget: deriving it from target_ids.size(1) would leak the gold
                # answer length into generation.
                gen_tokens, _, _ = self.model.generate_answer(
                    H, max_new_tokens=self.eval_max_new_tokens
                )

                metas = batch.get("meta")
                if self.tokenizer is not None and self.dataset_mod is not None:
                    for b in range(H.size(0)):
                        pred_str = self.tokenizer.decode(gen_tokens[b], skip_special_tokens=True).strip()
                        t_len = target_lens[b].item() if target_lens is not None else target_ids.size(1)
                        target_str = self.tokenizer.decode(target_ids[b, :t_len], skip_special_tokens=True).strip()
                        meta_sample = metas[b] if metas and b < len(metas) else {}

                        is_corr = self.dataset_mod.evaluate_answer(pred_str, target_str, meta_sample)
                        if is_corr:
                            total_exact += 1
                        total_samples += 1

                    # Also track token-level accuracy for completeness
                    _, _, t_corr, t_tot = compute_batch_accuracy(gen_tokens, target_ids, target_lens)
                    total_token_correct += t_corr
                    total_tokens += t_tot
                else:
                    e_corr, b_tot, t_corr, t_tot = compute_batch_accuracy(gen_tokens, target_ids, target_lens)
                    total_exact += e_corr
                    total_samples += b_tot
                    total_token_correct += t_corr
                    total_tokens += t_tot

        avg_loss = total_loss / max(1, step_count)
        avg_cycles = (total_cycles / max(1, sample_count)) if sample_count > 0 else 0.0
        val_acc = total_exact / max(1, total_samples) if total_samples > 0 else 0.0
        val_tok_acc = total_token_correct / max(1, total_tokens) if total_tokens > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "val_acc": val_acc,
            "val_token_acc": val_tok_acc,
            "avg_stopping_cycles": avg_cycles
        }

    def fit(self) -> Dict[str, Any]:
        start_time = time.time()

        if self.start_epoch >= self.epochs:
            print(f"[Trainer] Already completed {self.start_epoch} epochs (Target: {self.epochs}). Nothing to train.")
            return {
                "best_val_loss": self.best_val_loss,
                "best_val_acc": self.best_val_acc,
                "total_train_time_sec": 0.0,
                "final_epoch_metrics": self.history[-1] if self.history else {}
            }

        for epoch in range(self.start_epoch, self.epochs):
            ep_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            ep_time = time.time() - ep_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            combined = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": current_lr,
                "epoch_time_sec": round(ep_time, 2)
            }
            self.history.append(combined)

            # Log epoch metrics to history.json and history.csv
            self.tracker.log_epoch(combined)

            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            val_acc = val_metrics.get("val_acc", train_metrics.get("train_acc", 0.0))

            is_best_loss = val_loss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = val_loss
                # Save best_model.pt (pure model state_dict for eval.py compatibility)
                self.tracker.save_checkpoint(self.model.state_dict(), "best_model.pt")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

            # Full checkpoint for resuming (saved every epoch)
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True) if hasattr(self.cfg, "_content") or isinstance(self.cfg, DictConfig) else self.cfg
            checkpoint_data = {
                "epoch": epoch,
                "epochs": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "best_val_loss": self.best_val_loss,
                "best_val_acc": self.best_val_acc,
                "history": self.history,
                "cfg": cfg_dict
            }
            self.tracker.save_checkpoint(checkpoint_data, "checkpoint_last.pt")
            self.tracker.save_checkpoint(checkpoint_data, f"checkpoint_epoch_{epoch+1}.pt")

            t_loss = train_metrics.get("train_loss", 0.0)
            t_acc = train_metrics.get("train_acc", 0.0)
            v_loss = val_metrics.get("val_loss", 0.0)
            v_acc = val_metrics.get("val_acc", 0.0)
            print(
                f"[Epoch {epoch+1:02d}/{self.epochs:02d}] "
                f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | "
                f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}% | "
                f"LR: {current_lr:.2e} | Best Val Acc: {self.best_val_acc*100:.2f}% ({ep_time:.1f}s)"
            )

        total_time = time.time() - start_time
        final_summary = {
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "total_train_time_sec": total_time,
            "final_epoch_metrics": self.history[-1] if self.history else {}
        }
        self.tracker.log_metrics_and_finish(final_summary)
        return final_summary
