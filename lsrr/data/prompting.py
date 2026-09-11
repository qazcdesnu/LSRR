"""프롬프트 규약 — 누출 방지의 최전선 (I6).

문맥 인코딩에 들어가는 것은 **질문뿐**이다. 이 계약이 깨지면 전 실험이 무효다.

좌측 패딩(ADR-011): 질문은 좌측, 정답은 우측 패딩한다. 질문의 마지막 실토큰이
항상 인덱스 -1에 놓여야 h⁽ᴸ⁾ 추출과 KV 이어붙이기가 모두 성립한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from lsrr.config.schema import get_path
from lsrr.core.errors import LeakageError
from lsrr.core.invariants import IGNORE_INDEX
from lsrr.core.types import DataSample


@dataclass(frozen=True)
class PromptSpec:
    """질문·정답 렌더링 규약.

    Attributes:
        question_template: `{question}` 자리표시자를 갖는 템플릿.
        answer_template: `{answer}` 자리표시자를 갖는 템플릿.
        max_question_tokens: 질문 절단 길이.
        max_answer_tokens: 정답 절단 길이.
        append_eos: 정답 끝에 EOS를 붙일지. 생성 종료 학습에 필요하다.
        question_truncation_side: 한도를 넘는 질문을 어느 쪽에서 자를지.
            기본은 `"left"` — **질의는 언제나 꼬리에 있다**. ProsQA 질문은 전제
            수십 줄 뒤에 `"Is Sally a hilpus or sterpus?"`가 붙고 GSM8K도 서술
            뒤에 물음이 온다. 오른쪽에서 자르면 물음 자체가 사라져 과제가 풀 수
            없는 것이 되고, 좌측 패딩 규약(ADR-011)이 보장하려던 "마지막 실토큰
            = 질문의 끝"도 함께 깨진다.
    """

    question_template: str = "{question}"
    answer_template: str = " {answer}"
    max_question_tokens: int = 256
    max_answer_tokens: int = 32
    append_eos: bool = True
    question_truncation_side: str = "left"

    def render_question(self, sample: DataSample) -> str:
        return self.question_template.format(question=sample.question)

    def render_answer(self, sample: DataSample) -> str:
        return self.answer_template.format(answer=sample.answer)


class PromptEncoder:
    """DataSample → 토큰 텐서.

    질문과 정답을 **따로** 토크나이즈한다. 한 문자열로 이어 붙인 뒤 자르면
    경계가 토크나이저 병합에 따라 흔들려 누출이 생길 수 있다.
    """

    def __init__(self, tokenizer: Any, spec: Optional[PromptSpec] = None) -> None:
        self.tokenizer = tokenizer
        self.spec = spec or PromptSpec()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------- 질문

    def encode_questions(
        self, samples: Sequence[DataSample], device: Optional[torch.device] = None
    ) -> dict[str, torch.Tensor]:
        """좌측 패딩된 질문 배치 → {input_ids, attention_mask}."""
        texts = [self.spec.render_question(s) for s in samples]
        # truncation_side는 `__call__` 인자가 아니라 토크나이저 속성이다.
        saved = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = self.spec.question_truncation_side
        try:
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.spec.max_question_tokens,
                padding_side="left",
            )
        finally:
            self.tokenizer.truncation_side = saved
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        return {k: v.to(device) for k, v in out.items()} if device else out

    # -------------------------------------------------------------- 정답

    def encode_answers(
        self, samples: Sequence[DataSample], device: Optional[torch.device] = None
    ) -> dict[str, torch.Tensor]:
        """우측 패딩된 정답 배치 → {target_ids, labels, answer_lens}.

        `target_ids`는 디코더 입력(패딩 = pad_token_id), `labels`는 손실 타깃
        (패딩 = IGNORE_INDEX)이다. 둘을 혼동하면 패딩을 학습한다.
        """
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        limit = self.spec.max_answer_tokens

        seqs: list[list[int]] = []
        for s in samples:
            ids = self.tokenizer(
                self.spec.render_answer(s), add_special_tokens=False
            )["input_ids"][: limit - 1 if self.spec.append_eos else limit]
            if self.spec.append_eos and eos_id is not None:
                ids = ids + [eos_id]
            seqs.append(ids or [eos_id if eos_id is not None else pad_id])

        width = max(len(s) for s in seqs)
        target = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        labels = torch.full((len(seqs), width), IGNORE_INDEX, dtype=torch.long)
        lens = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            target[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            labels[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            lens[i] = len(s)

        out = {"target_ids": target, "labels": labels, "answer_lens": lens}
        return {k: v.to(device) for k, v in out.items()} if device else out

    # -------------------------------------------------------------- 전체

    def encode_batch(
        self,
        samples: Sequence[DataSample],
        device: Optional[torch.device] = None,
        check_leakage: bool = True,
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        batch.update(self.encode_questions(samples, device))
        batch.update(self.encode_answers(samples, device))
        batch["samples"] = list(samples)
        if check_leakage:
            assert_no_answer_text_leakage(samples, self.spec)
        return batch

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """생성 토큰 → 문자열. EOS에서 자른다."""
        eos = self.tokenizer.eos_token_id
        texts: list[str] = []
        for row in token_ids.tolist():
            if eos is not None and eos in row:
                row = row[: row.index(eos)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True))
        return texts


def prompt_spec_from_cfg(cfg: Any) -> PromptSpec:
    """설정의 `prompt:` 절 → PromptSpec.

    train/eval/진단 스크립트가 저마다 PromptSpec을 조립하면 키 하나가 한 곳에만
    반영되는 사고가 난다 (max_question_tokens가 그랬다). 조립은 여기 한 곳이다.
    """
    return PromptSpec(
        max_question_tokens=int(get_path(cfg, "prompt.max_question_tokens", 256)),
        max_answer_tokens=int(get_path(cfg, "prompt.max_answer_tokens", 32)),
        append_eos=bool(get_path(cfg, "prompt.append_eos", True)),
        question_truncation_side=str(
            get_path(cfg, "prompt.question_truncation_side", "left")
        ),
    )


def assert_no_answer_text_leakage(
    samples: Sequence[DataSample], spec: PromptSpec
) -> None:
    """렌더링된 질문에 정답 문자열이 들어 있지 않은지 확인한다 (I6).

    토큰 수준 검사(`core.invariants.assert_no_answer_leakage`)보다 앞선 1차
    방어선이다. 템플릿 실수는 대개 여기서 잡힌다.
    """
    for i, s in enumerate(samples):
        answer = s.answer.strip()
        if not answer:
            continue
        if answer in spec.render_question(s):
            raise LeakageError(
                f"샘플 {i}의 렌더링된 질문에 정답 '{answer}'이 들어 있다. "
                f"문맥 인코딩에 들어가는 것은 질문뿐이다."
            )


__all__ = (
    "PromptSpec",
    "PromptEncoder",
    "prompt_spec_from_cfg",
    "assert_no_answer_text_leakage",
)
