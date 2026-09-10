"""동결 백본 세션 — 질문을 단 1회 인코딩한다 (I2).

한 번의 순전파로 사고 메모리의 재료(레이어별 상태)와 연속 디코딩의 재료(질문
KV 캐시)를 **동시에** 얻는다. 이것이 ADR-002의 핵심이다 — KV를 디스크에
캐시하는 것은 비현실적이고(GPT-2 기준 GSM8k-Aug 학습셋 전체 약 2.8TB), 백본을
어차피 로드해야 하므로 H 사전 캐시로 아끼려던 비용도 크지 않다.

개작: Legacy_LSRR/lsrr/backbones/extractor.py
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from lsrr.backbone.continuation import BackboneContinuation
from lsrr.backbone.extractor import extract, stack_hidden_states
from lsrr.backbone.freeze import backbone_fingerprint, freeze_backbone
from lsrr.core.errors import AssemblyError, LSRRError
from lsrr.core.interfaces import BaseContextEncoder
from lsrr.core.registry import BACKBONE_REGISTRY
from lsrr.core.types import ContextBundle

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@BACKBONE_REGISTRY.register("hf_frozen_causal")
@BACKBONE_REGISTRY.register("gpt2")
class HFFrozenCausalBackbone(BaseContextEncoder):
    """HuggingFace causal LM을 동결 문맥 인코더로 쓴다.

    `nn.Module`이 아니다 — 모델 트리에 등록되지 않으므로 학습 파라미터와
    체크포인트에서 구조적으로 배제된다 (I1).
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        dtype: str = "float32",
        device: Optional[str] = None,
        include_embedding: bool = False,
        position_rule: str = "last_token",
        keep_hidden_stack: bool = True,
        **_: Any,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.include_embedding = include_embedding
        self.position_rule = position_rule
        self.keep_hidden_stack = keep_hidden_stack
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch_dtype = _DTYPES.get(dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 연속 디코딩은 좌측 패딩을 요구한다 (ADR-011).
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=torch_dtype
        ).to(self.device)
        freeze_backbone(self.model)

        self._num_layers = int(self.model.config.num_hidden_layers) + (
            1 if include_embedding else 0
        )
        self._hidden_dim = int(self.model.config.hidden_size)
        self._fingerprint = backbone_fingerprint(self.model)

        self.answer_head = BackboneContinuation(
            self.model,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ------------------------------------------------------------ 속성

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def weight_hash(self) -> str:
        return self._fingerprint

    def num_parameters(self) -> int:
        return int(self.model.num_parameters())

    def verify_frozen(self) -> None:
        """학습 후 호출해 가중치가 변하지 않았음을 확인한다 (I1)."""
        from lsrr.core.invariants import assert_frozen, assert_weights_unchanged

        assert_frozen(self.model, what=self.model_name)
        assert_weights_unchanged(
            self._fingerprint, backbone_fingerprint(self.model), what=self.model_name
        )

    # ------------------------------------------------------------ 인코딩

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ContextBundle:
        """질문 1회 순전파 → ContextBundle.

        Args:
            input_ids: [B, T] 좌측 패딩된 질문 토큰.
            attention_mask: [B, T]

        Returns:
            H_last / hidden_stack / h_ctx / kv_cache를 담은 ContextBundle.
            H_pool은 여기서 채우지 않는다 — 풀러는 학습 대상이므로 모델의
            자식으로 등록되어 모델이 호출한다 (ADR-012).
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)

        if not bool(attention_mask[:, -1].all()):
            raise LSRRError(
                "질문 배치가 우측 패딩되어 있다. 연속 디코딩은 KV 정렬을 위해 "
                "좌측 패딩을 요구한다 (ADR-011)."
            )

        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True,
            )

        stack = stack_hidden_states(out.hidden_states, self.include_embedding)
        H_last = extract(stack, attention_mask, self.position_rule)
        # h_ctx: 어댑터 통과 전 백본 원본 h⁽ᴸ⁾ — 융합 잔차 앵커 (ADR-003)
        h_ctx = out.hidden_states[-1][:, -1, :]

        return ContextBundle(
            H_last=H_last,
            h_ctx=h_ctx,
            H_pool=None,
            hidden_stack=stack if self.keep_hidden_stack else None,
            kv_cache=out.past_key_values,
            attention_mask=attention_mask,
            q_len=attention_mask.sum(dim=1),
            meta={
                "backbone_id": self.model_name,
                "weight_hash": self._fingerprint,
                "include_embedding": self.include_embedding,
                "position_rule": self.position_rule,
                "kv_len": int(out.past_key_values.get_seq_length()),
            },
        )

    def __repr__(self) -> str:
        return (
            f"HFFrozenCausalBackbone({self.model_name!r}, L={self._num_layers}, "
            f"d={self._hidden_dim}, device={self.device})"
        )


__all__ = ("HFFrozenCausalBackbone",)
