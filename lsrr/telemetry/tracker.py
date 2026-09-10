"""실험 기록 — 런 디렉터리·설정 스냅샷·지표 (telemetry/README).

이 패키지는 **쓰기만** 한다. 지표를 계산하지 않고(metrics), 분석하지 않는다(analysis).
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from lsrr.config.snapshot import config_hash, make_run_id, run_metadata, save_snapshot


class ExperimentTracker:
    """런 디렉터리를 만들고 그 안의 모든 파일을 소유한다.

    구조:
        runs/<run_id>/
          config.yaml        해석 완료 설정 (런 신원)
          meta.json          백본 해시, 학습 파라미터 수·비율, 활성 안정화 단, …
          metrics.jsonl      스텝별 지표
          diagnostics.jsonl  사이클별 트레이스
          checkpoints/
          eval/
    """

    def __init__(
        self,
        cfg: DictConfig,
        exp_name: str = "exp",
        seed: int = 0,
        root: str | Path = "runs",
        run_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id or make_run_id(cfg, exp_name, seed)
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        (self.dir / "eval").mkdir(exist_ok=True)

        save_snapshot(cfg, self.dir / "config.yaml")
        self.meta = run_metadata(cfg, exp_name, seed, extra=self._environment())
        self._flush_meta()

        self._metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._diagnostics = (self.dir / "diagnostics.jsonl").open("a", encoding="utf-8")

    # ------------------------------------------------------------ 메타

    @staticmethod
    def _environment() -> dict[str, Any]:
        import torch

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }

    def update_meta(self, **fields: Any) -> None:
        """표에 병기해야 하는 값을 채운다 (백본 해시, 파라미터 비율, 배치 크기 등)."""
        self.meta.update(fields)
        self._flush_meta()

    def _flush_meta(self) -> None:
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------ 기록

    def log_metrics(self, step: int, **values: Any) -> None:
        record = {"step": step, **{k: _plain(v) for k, v in values.items()}}
        self._metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._metrics.flush()

    def log_diagnostics(self, step: int, record: dict[str, Any]) -> None:
        payload = {"step": step, **{k: _plain(v) for k, v in record.items()}}
        self._diagnostics.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._diagnostics.flush()

    def write_eval(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.dir / "eval" / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_plain),
            encoding="utf-8",
        )
        return path

    @property
    def checkpoint_dir(self) -> Path:
        return self.dir / "checkpoints"

    def close(self) -> None:
        self._metrics.close()
        self._diagnostics.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _plain(value: Any) -> Any:
    """텐서·넘파이 스칼라를 JSON 직렬화 가능한 값으로."""
    if hasattr(value, "item") and getattr(value, "numel", lambda: 1)() == 1:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = ("ExperimentTracker",)
