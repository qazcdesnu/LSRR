"""백본 연속 디코딩 (제안서 §4.4, ADR-001).

    "h_fusion을 질문 토큰 시퀀스의 다음 위치 입력 벡터로 1회 주입하고(Coconut의
     연속 사고 주입과 동일한 통로), 이후 답 토큰들을 동결 백본이 자기회귀
     생성한다. 질문 구간의 KV 캐시가 그대로 유효하므로 추가 비용은
     [1 위치 + 답 길이]의 incremental forward에 그치며, 역전파는 h_fusion
     위치를 통해서만 엔진으로 흐른다."

학습 가능한 디코더가 아니다 — 생성 능력은 끝까지 백본 소유다.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.errors import AssemblyError, LSRRError
from lsrr.core.interfaces import BaseAnswerHead
from lsrr.core.invariants import assert_injection_space


def _cache_length(cache: Any) -> int:
    length = getattr(cache, "get_seq_length", None)
    if length is None:
        raise LSRRError(f"KV 캐시에서 길이를 읽을 수 없다: {type(cache).__name__}")
    return int(length())


def rewind_cache(cache: Any, base_len: int) -> None:
    """캐시를 질문 길이로 되돌린다.

    HF 캐시는 순전파에서 **제자리 변경**된다. 깊은 감독은 같은 질문 KV로 여러
    사이클을 판독하므로, 되감지 않으면 두 번째 판독이 첫 번째 판독의 토큰이
    이미 붙은 캐시를 보게 된다 — 조용히 오염된 채로 학습이 진행된다.
    """
    extra = _cache_length(cache) - base_len
    if extra > 0:
        cache.crop(-extra)  # 음수 인자: 뒤에서 그만큼 제거 (5.18+ 규약)
    elif extra < 0:
        raise LSRRError(
            f"KV 캐시가 질문 길이보다 짧다 (len={_cache_length(cache)}, base={base_len})."
        )


class BackboneContinuation(BaseAnswerHead):
    """동결 백본의 연속 디코딩 통로.

    `nn.Module`이 아니다 — 학습 파라미터를 갖지 않으며, 모델 트리에 등록되어
    체크포인트에 백본 가중치가 섞이는 일을 피한다 (I1).
    """

    def __init__(
        self,
        model: nn.Module,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.model = model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.d_in = int(model.config.hidden_size)

    # ------------------------------------------------------------ 내부

    @staticmethod
    def _as_trajectory(injected: torch.Tensor) -> torch.Tensor:
        """`[B, d]` → `[B, 1, d]`, `[B, M, d]` → 그대로.

        v1 단일 벡터를 M=1 궤적의 특수 경우로 흡수한다 (ADR-015) — 두 경로를
        따로 두면 Ablation A 의 조건들이 서로 다른 코드를 타게 된다.
        """
        return injected.unsqueeze(1) if injected.dim() == 2 else injected

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def _step_inputs(
        self,
        inputs_embeds: torch.Tensor,
        question_mask: torch.Tensor,
        q_len: torch.Tensor,
        offset: int,
    ) -> dict[str, torch.Tensor]:
        """주입/생성 스텝의 마스크와 position_ids를 만든다.

        좌측 패딩이므로 각 샘플의 실제 위치는 `q_len + offset`부터 이어진다
        (ADR-011). position_ids를 넘기지 않으면 HF가 패딩을 포함한 절대
        인덱스를 쓰므로 짧은 샘플의 위치가 어긋난다.
        """
        B, T_new, _ = inputs_embeds.shape
        device = inputs_embeds.device
        total_new = offset + T_new
        new_mask = torch.ones(B, total_new, dtype=question_mask.dtype, device=device)
        attn = torch.cat([question_mask, new_mask], dim=1)
        pos = q_len.unsqueeze(1) + offset + torch.arange(T_new, device=device).unsqueeze(0)
        return {"attention_mask": attn, "position_ids": pos.to(torch.long)}

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_cache: Any,
        question_mask: torch.Tensor,
        q_len: torch.Tensor,
        offset: int,
    ) -> Any:
        extra = self._step_inputs(inputs_embeds, question_mask, q_len, offset)
        return self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=kv_cache,
            use_cache=True,
            **extra,
        )

    # ------------------------------------------------------------ 공개

    def teacher_forced(
        self,
        injected: torch.Tensor,
        kv_cache: Any,
        answer_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        q_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """teacher-forcing 로짓.

        입력 구성: `[h⁽¹⁾ … h⁽ᴹ⁾] + embed(answer[:-1])`. 마지막 잠재 토큰 위치의
        로짓이 `answer_ids[:, 0]`을 예측하므로, **앞쪽 M−1 개 위치를 잘라내면**
        라벨과 그대로 정렬된다 (별도의 shift 불필요).

        Args:
            injected: `[B, d_in]` 또는 `[B, M, d_in]`. 백본 입력 임베딩 공간이어야
                한다 (I8). M 개 궤적은 v2.1 §4.4.
            kv_cache: 질문 구간 KV. 호출 후 질문 길이로 되감긴다.
            answer_ids: [B, T_a]
            attention_mask: [B, T_q] 질문 마스크
            q_len: [B] 질문 실길이. None이면 마스크에서 계산한다.

        Returns:
            logits [B, T_a, V]
        """
        assert_injection_space(injected, self.d_in)
        if attention_mask is None:
            raise AssemblyError("연속 디코딩에는 질문 attention_mask가 필요하다.")
        if q_len is None:
            q_len = attention_mask.sum(dim=1)

        base_len = _cache_length(kv_cache)
        traj = self._as_trajectory(injected)
        M = traj.shape[1]
        inject = traj.to(dtype=self._embed(answer_ids[:, :1]).dtype)
        if answer_ids.shape[1] > 1:
            inputs = torch.cat([inject, self._embed(answer_ids[:, :-1])], dim=1)
        else:
            inputs = inject

        out = self._forward(inputs, kv_cache, attention_mask, q_len, offset=0)
        rewind_cache(kv_cache, base_len)
        # 앞선 M−1 개 잠재 위치는 답 토큰을 예측하지 않는다 — 잘라내야 라벨과
        # 정렬된다. M=1 이면 무연산이라 v1 경로와 동일하다.
        return out.logits[:, M - 1 :] if M > 1 else out.logits

    @torch.no_grad()
    def generate(
        self,
        injected: torch.Tensor,
        kv_cache: Any,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        q_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """탐욕적 자기회귀 생성 → [B, T_gen].

        비용은 `[M 위치 + 생성 길이]`의 incremental forward뿐이다 — 질문 KV를
        재사용하므로 질문을 다시 읽지 않는다 (제안서 §4.4).

        **M 개 잠재 토큰은 한 번에 넣는다.** 서로 의존하지 않으므로 한 스텝으로
        족하고, 이 덕분에 궤적 길이가 늘어도 순차 스텝 수는 그대로다.
        """
        assert_injection_space(injected, self.d_in)
        if attention_mask is None:
            raise AssemblyError("연속 디코딩에는 질문 attention_mask가 필요하다.")
        if q_len is None:
            q_len = attention_mask.sum(dim=1)

        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        pad = self.pad_token_id if self.pad_token_id is not None else (eos or 0)

        traj = self._as_trajectory(injected)
        B, M = traj.shape[0], traj.shape[1]
        device = traj.device
        base_len = _cache_length(kv_cache)

        step_input = traj
        tokens: list[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        offset = 0

        for _ in range(max_new_tokens):
            out = self._forward(
                step_input, kv_cache, attention_mask, q_len, offset=offset
            )
            offset += step_input.shape[1]
            nxt = out.logits[:, -1].argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            tokens.append(nxt)
            if eos is not None:
                finished = finished | (nxt == eos)
                if bool(finished.all()):
                    break
            step_input = self._embed(nxt).unsqueeze(1)

        rewind_cache(kv_cache, base_len)
        return torch.stack(tokens, dim=1) if tokens else torch.zeros(B, 0, dtype=torch.long)


__all__ = ("BackboneContinuation", "rewind_cache")
