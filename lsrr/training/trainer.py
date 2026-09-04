import time
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lsrr.utils.logging import ExperimentTracker
from lsrr.losses.composite import CompositeLoss
from lsrr.utils.flops import count_parameters

class Trainer:
    """Trainer supporting AMP, gradient clipping, checkpointing, and diagnostic logging."""
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: CompositeLoss,
        cfg: Dict[str, Any],
        tracker: ExperimentTracker,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.cfg = cfg
        self.tracker = tracker
        self.device = device

        train_cfg = cfg.get("train", {})
        self.epochs = train_cfg.get("epochs", 5)
        self.lr = float(train_cfg.get("lr", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-2))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.use_amp = bool(train_cfg.get("amp", False) and device.type == "cuda")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        param_stats = count_parameters(self.model)
        print(f"[Trainer] Model parameters: {param_stats}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        step_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch in pbar:
            H = batch["H"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

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

            loss_val = loss.item()
            total_loss += loss_val
            step_count += 1

            # Log diagnostics
            if outputs.get("diagnostics"):
                diags_summary = [d.delta_state for d in outputs["diagnostics"]]
                self.tracker.log_diagnostics({
                    "epoch": epoch,
                    "step": step_count,
                    "loss": loss_val,
                    "deltas": diags_summary
                })

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        return {"train_loss": total_loss / max(1, step_count)}

    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        step_count = 0
        total_cycles = 0
        sample_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                H = batch["H"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                outputs = self.model(H, target_ids=target_ids, is_eval=True)
                loss_dict = self.loss_fn(outputs, {"target_ids": target_ids})
                total_loss += loss_dict["total_loss"].item()
                step_count += 1

                if outputs.get("stopping_cycles") is not None:
                    total_cycles += outputs["stopping_cycles"].sum().item()
                    sample_count += outputs["stopping_cycles"].numel()

        avg_loss = total_loss / max(1, step_count)
        avg_cycles = (total_cycles / max(1, sample_count)) if sample_count > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "avg_stopping_cycles": avg_cycles
        }

    def fit(self) -> Dict[str, Any]:
        best_val_loss = float("inf")
        history = []

        start_time = time.time()
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            combined = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            history.append(combined)

            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.tracker.save_checkpoint(self.model.state_dict(), "best_model.pt")

            print(f"[Epoch {epoch+1}] {combined}")

        total_time = time.time() - start_time
        final_summary = {
            "best_val_loss": best_val_loss,
            "total_train_time_sec": total_time,
            "final_epoch_metrics": history[-1] if history else {}
        }
        self.tracker.log_metrics_and_finish(final_summary)
        return final_summary
