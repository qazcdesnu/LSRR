"""학습 루프 (제안서 §5).

학습 대상은 **레이어 어댑터 + SSM 엔진 + 풀링/융합 헤드**뿐이다. 백본은 완전
동결하며 LoRA도 쓰지 않는다.

학습 스텝의 형태 (runtime/README):
    1. backbone.encode(question)          무그래디언트, 1회 (I2)
    2. memory.pipeline(bundle)     → R0
    3. recurrence.run_train(R0, hooks) → R*, 사이클별 판독
    4. readout.readout(R*, h_ctx)  → logits    훅이 쓴 것과 동일 인스턴스 (I3)
    5. objectives(trace, batch)    → loss
    6. backward / clip / step       백본에는 그래디언트 없음 (I1, I4)

개작: Legacy_LSRR/lsrr/training/trainer.py (LR 스케줄·클리핑·체크포인트 계승)
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lsrr.core.invariants import assert_no_grad
from lsrr.core.types import ReasoningTrace
from lsrr.recurrence.hooks import DiagnosticsRecorder, ReadoutHook
from lsrr.runtime.checkpoint import save_checkpoint
from lsrr.telemetry.traces import summarize_trace


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """1-cycle half-cosine (Mamba-2 레시피).

    이식: Legacy_LSRR/lsrr/training/trainer.py:get_cosine_schedule_with_warmup
    """
    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        total = max(1, num_training_steps - 1 - num_warmup_steps)
        progress = min(max((step - num_warmup_steps) / total, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """최소 학습 루프.

    Args:
        model: 조립된 `LSRRModel`.
        objective: 합성 손실.
        tracker: 실험 기록기.
        cfg: 해석 완료 설정 (`train:` 절을 읽는다).
    """

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        tracker: Any,
        cfg: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.tracker = tracker
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        train_cfg = cfg.get("train", {}) if hasattr(cfg, "get") else {}
        self.lr = float(train_cfg.get("lr", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-2))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.epochs = int(train_cfg.get("epochs", 1))
        self.log_every = int(train_cfg.get("log_every", 20))
        self.min_lr = float(train_cfg.get("min_lr", 1e-5))
        self.warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "학습 가능한 파라미터가 없다. 어댑터·엔진·융합 헤드가 조립되었는지 "
                "확인하라 (제안서 §5)."
            )
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.global_step = 0

        # 깊은 감독이 켜져 있으면 사이클별 판독이 필요하다
        self.readout_per_cycle = bool(
            _get(cfg, "recurrence.hooks.readout_per_cycle", False)
        )
        self.deep_supervision = bool(
            _get(cfg, "objective.deep_supervision.enabled", False)
        )

    # ------------------------------------------------------------ 스텝

    def forward_batch(self, batch: dict[str, Any]) -> ReasoningTrace:
        """학습 순전파 1회. 판독 경로는 전 사이클 동일 인스턴스다 (I3)."""
        model = self.model
        model.encode_counter.reset()

        context = model.encode(batch["input_ids"], batch.get("attention_mask"))
        R0 = model.build_memory(context)

        recorder = DiagnosticsRecorder()
        hooks: list[Any] = [recorder]
        readout_hook: Optional[ReadoutHook] = None
        if self.readout_per_cycle or self.deep_supervision:
            readout_hook = ReadoutHook(
                readout=model.readout,
                context=context,
                answer_ids=batch.get("target_ids"),
            )
            hooks.append(readout_hook)

        trace = model.refine(R0, hooks=hooks, is_eval=False)
        trace.R0 = R0
        trace.h_ctx = context.h_ctx

        result = model.read(trace.R_star, context, answer_ids=batch.get("target_ids"))
        trace.logits = result.logits
        trace.h_fusion = result.h_fusion
        trace.alpha = result.alpha
        if readout_hook is not None:
            trace.per_cycle_readout = readout_hook.results
            trace.supervised_cycles = readout_hook.supervised_cycles()
        trace.meta["encode_count"] = model.encode_counter.count
        return trace

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = _to_device(batch, self.device)
        trace = self.forward_batch(batch)
        losses = self.objective(trace, batch)

        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.grad_clip
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {k: v for k, v in losses.items() if k != "loss"}
        metrics["grad_norm"] = float(grad_norm)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return {"metrics": metrics, "trace": trace}

    # ------------------------------------------------------------ 루프

    def fit(
        self,
        loader: DataLoader,
        eval_fn: Optional[Callable[[int], dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        total_steps = max(1, len(loader) * self.epochs)
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
            min_lr=self.min_lr,
        )

        history: list[dict[str, Any]] = []
        started = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            for batch in loader:
                out = self.training_step(batch)
                self.global_step += 1

                if self.global_step % self.log_every == 0 or self.global_step == 1:
                    record = {
                        "epoch": epoch,
                        **{k: _scalar(v) for k, v in out["metrics"].items()},
                    }
                    self.tracker.log_metrics(self.global_step, **record)
                    self.tracker.log_diagnostics(
                        self.global_step, summarize_trace(out["trace"])
                    )
                    history.append(record)

            save_checkpoint(
                self.model,
                self.tracker.checkpoint_dir / f"epoch_{epoch}.pt",
                optimizer=self.optimizer,
                step=self.global_step,
                meta={"epoch": epoch},
            )
            if eval_fn is not None:
                self.tracker.log_metrics(self.global_step, **eval_fn(epoch))

        # I1·I4: 학습이 끝난 뒤 백본이 그대로인지 확인한다
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            encoder.verify_frozen()
            assert_no_grad(encoder.model, what="backbone")

        return {
            "steps": self.global_step,
            "seconds": time.time() - started,
            "history": history,
        }


# ---------------------------------------------------------------- 헬퍼


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


def _scalar(value: Any) -> Any:
    return float(value) if torch.is_tensor(value) and value.numel() == 1 else value


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    from lsrr.config.schema import get_path

    return get_path(cfg, path, default)


__all__ = ("Trainer", "cosine_schedule_with_warmup")
