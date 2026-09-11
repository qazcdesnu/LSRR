"""ProsQA — 홉 수 통제 다중 홉 추론, 메커니즘 분석의 주 무대 (제안서 §6.1).

v2 의 핵심 가설이 걸린 과제다. §2.2 는 단일 벡터 압축의 병목 근거로 "다중 홉
추론에서 파생되는 다수의 중간 브리지 엔티티, 조건 분기, 중간 연산 결과" 를 들고,
§7.1-1 의 Ablation A 가설은 **"3-hop 이상의 다중 홉에서 급격한 성능 저하"** 를
예측한다. 곱셈은 계산 깊이 과제라 이 가설을 검증할 수 없다 — 브리지 엔티티가
생기지 않기 때문이다.

각 샘플이 **홉 수 라벨**을 갖는다는 점이 결정적이다. §7.2 의 "홉이 많을수록
오래 생각한다"(홉 수 ↔ 수렴 사이클 M 의 단조성)를 같은 런에서 검증할 수 있다.

출처: facebookresearch/coconut 의 prosqa_{train,valid,test}.json.
스키마는 `{question, answer, steps, edges, idx_to_symbol, root, target, neg_target}`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from lsrr.core.errors import ConfigError
from lsrr.core.interfaces import BaseDataModule
from lsrr.core.registry import DATA_REGISTRY
from lsrr.core.types import DataSample
from lsrr.data.schema import normalize_split
from lsrr.metrics.accuracy import match_free_form

#: split → 파일명 후보. coconut 배포본과 일반 명명을 모두 받는다.
_FILES: dict[str, tuple[str, ...]] = {
    "train": ("prosqa_train.json", "train.json"),
    "val": ("prosqa_valid.json", "prosqa_val.json", "valid.json", "val.json"),
    "test": ("prosqa_test.json", "test.json"),
}


@DATA_REGISTRY.register("prosqa")
class ProsQADataset(BaseDataModule):
    """ProsQA 로더.

    Args:
        data_dir: `prosqa_*.json` 이 있는 디렉터리.
        sizes: split 별 상한. None 이면 전량. 스모크 런에서 쓴다.
        min_hops / max_hops: 홉 수로 걸러낸다. Ablation A 의 "3-hop 이상" 가설을
            층화해 볼 때, 또는 짧은 홉만으로 예비 확인할 때 쓴다.
    """

    def __init__(
        self,
        data_dir: str = "data/prosqa",
        sizes: Optional[dict[str, int]] = None,
        min_hops: Optional[int] = None,
        max_hops: Optional[int] = None,
        **_: Any,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sizes = dict(sizes or {})
        self.min_hops = min_hops
        self.max_hops = max_hops
        self._cache: dict[str, list[DataSample]] = {}

    # ------------------------------------------------------------ 적재

    def _path(self, split: str) -> Path:
        for name in _FILES[split]:
            p = self.data_dir / name
            if p.exists():
                return p
        raise ConfigError(
            f"ProsQA split '{split}' 파일을 {self.data_dir} 에서 찾지 못했다. "
            f"후보: {list(_FILES[split])}. 데이터는 저장소에 포함되지 않으므로 "
            f"facebookresearch/coconut 의 data/ 에서 받아 둔다."
        )

    def _load(self, split: str) -> list[DataSample]:
        raw = json.loads(self._path(split).read_text(encoding="utf-8"))
        samples: list[DataSample] = []
        for i, item in enumerate(raw):
            steps = list(item.get("steps", []))
            hops = len(steps)
            if self.min_hops is not None and hops < self.min_hops:
                continue
            if self.max_hops is not None and hops > self.max_hops:
                continue
            samples.append(
                DataSample(
                    question=str(item["question"]).strip(),
                    answer=str(item["answer"]).strip(),
                    cot_steps=steps,
                    meta={
                        # 홉 수는 §7.2 메커니즘 분석의 층화 축이다.
                        "hops": hops,
                        "sample_id": i,
                        "target": item.get("target"),
                        "neg_target": item.get("neg_target"),
                    },
                )
            )
        return samples

    def get_split(self, split: str) -> list[DataSample]:
        split = normalize_split(split)
        if split not in self._cache:
            samples = self._load(split)
            limit = self.sizes.get(split)
            if limit is not None:
                samples = samples[: int(limit)]
            if not samples:
                raise ConfigError(
                    f"ProsQA split '{split}' 이 비었다. 홉 필터"
                    f"(min_hops={self.min_hops}, max_hops={self.max_hops})가 "
                    f"너무 좁지 않은지 확인하라."
                )
            self._cache[split] = samples
        return self._cache[split]

    # ------------------------------------------------------------ 채점

    def score(self, prediction: str, target: str, meta: dict[str, Any]) -> bool:
        """최종 개체 일치 — 부분 문자열 포함은 받지 않는다.

        판정 자체는 `metrics.match_free_form` 이 한다. 평가 경로는
        `metrics.evaluate.scorer_for("prosqa")` 로 같은 함수를 부르므로, 여기서
        따로 구현하면 두 정의가 갈려 같은 런이 두 정확도를 갖게 된다.

        정답은 `"Sally is a sterpus."` 형태의 한 문장이고 과제는 두 종점 노드 중
        옳은 것을 고르는 것이므로, 최종 개체어가 맞으면 정답으로 본다.
        """
        return match_free_form(prediction, target)


__all__ = ("ProsQADataset",)
