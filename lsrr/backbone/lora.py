"""LoRA 수신 정렬 — 디코딩 전용 (ADR-014, 제안서 v2.1 §5.0).

LoRA 를 쓰는 이유는 **전적으로 수신 측**에 있다: 토큰 임베딩만 읽도록 학습된
백본이 `[토큰 ⊕ 연속 잠재]` 시퀀스를 읽어야 하는 불일치다. 그래서 어댑터는
디코딩 국면에서만 활성이고, `H` 추출 인코딩은 전 학습 과정에서 항상 순수 base
가중치로 수행한다 (I9).

**이 파일이 존재하는 이유는 그 한정이 조용히 깨지기 때문이다.** PEFT 는 대상
모듈을 **제자리에서** 교체하므로, 인코딩과 디코딩이 같은 `nn.Module` 인스턴스를
공유하는 현 구조에서는 어댑터가 인코딩 패스에도 자동 적용된다. 그리고 그
위반은 소리 없이 일어난다 — 학습은 정상적으로 돌고, `H` 캐시는 유효하다고
믿긴 채 오염되며, 결과만 설명 불가가 된다. 그래서 경고가 아니라 불변식이다.

    인코딩 ①  어댑터 **비활성** (`adapters_disabled`)  input_ids     → H, base KV
    디코딩 ②  어댑터 **활성**                          inputs_embeds → 로짓
               + 캐시된 base KV

②의 정합성은 프레임워크가 이미 보장한다. HF causal LM 은 `inputs_embeds` 와
`past_key_values` 동시 사용을 지원하고, 어텐션이 "캐시된 K/V + 새로 계산된 K/V"
를 섞는 것은 증분 디코딩의 표준 동작이다 — **새 위치의 투영만 LoRA 를 통과하고
질문 KV 는 base 로 남는다.**
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import torch
import torch.nn as nn

from lsrr.core.errors import AssemblyError

#: GPT-2 는 `Conv1D`, LLaMA 계열은 `nn.Linear` 다. PEFT 가 `Conv1D` 를 감지해
#: `fan_in_fan_out` 을 자동 보정하므로 경고가 떠도 정상이다 (ADR-014).
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("c_attn", "c_proj")

LORA_MARKER = "lora_"
"""LoRA 파라미터 이름에 반드시 포함되는 조각. I1 의 base/델타 구분 기준이다."""


def is_lora_param(name: str) -> bool:
    return LORA_MARKER in name


def attach_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    **_: Any,
) -> Any:
    """base 모델에 LoRA 어댑터를 제자리 주입하고 `PeftModel` 을 돌려준다.

    반환된 래퍼는 `adapters_disabled` 에 필요하다 — `disable_adapter()` 는
    `PeftModel` 의 메서드다. base 모델 인스턴스 자체도 이 호출 뒤에는 LoRA 층을
    품고 있으므로, 디코딩 경로는 base 인스턴스를 그대로 불러도 어댑터를 탄다.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:  # pragma: no cover - 의존성 누락은 조립 오류다
        raise AssemblyError(
            "LoRA 를 요청했으나 peft 를 불러올 수 없다. `uv sync` 로 의존성을 "
            "맞추라 (pyproject 의 peft>=0.20.0)."
        ) from e

    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    # base 는 여전히 동결이어야 한다. get_peft_model 이 이미 그렇게 두지만,
    # '했다' 와 '되었다' 는 다르므로 여기서 확인한다.
    for name, param in peft_model.named_parameters():
        param.requires_grad_(is_lora_param(name))
    return peft_model


@contextmanager
def adapters_disabled(peft_model: Optional[Any]) -> Iterator[None]:
    """어댑터를 끈 채로 실행한다 (I9).

    `None` 이면 무연산이다 — Phase A 에는 어댑터가 아예 없으므로, 호출부가
    LoRA 유무로 분기하지 않아도 되게 한다.
    """
    if peft_model is None or not hasattr(peft_model, "disable_adapter"):
        yield
        return
    with peft_model.disable_adapter():
        yield


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if is_lora_param(n)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA 델타만 뽑는다. base 를 같이 저장하면 '무엇이 학습되었는가' 가 흐려진다."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if is_lora_param(k)
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = [k for k in state if k not in model.state_dict()]
    if missing:
        raise AssemblyError(
            f"LoRA 상태의 키 {len(missing)}개가 현재 모델에 없다: {missing[:3]}. "
            f"r·alpha·target_modules 가 저장 시점과 다른지 확인하라."
        )
    model.load_state_dict(state, strict=False)


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


__all__ = (
    "DEFAULT_TARGET_MODULES",
    "LORA_MARKER",
    "attach_lora",
    "adapters_disabled",
    "is_lora_param",
    "lora_parameters",
    "lora_state_dict",
    "load_lora_state_dict",
    "count_lora_parameters",
)
