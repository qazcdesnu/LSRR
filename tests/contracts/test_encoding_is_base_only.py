"""I9 — 인코딩 패스는 어댑터가 비활성인 상태에서 수행된다 (ADR-014).

LoRA 를 쓰는 논거는 **전적으로 수신 측**에 있다: 토큰 임베딩만 읽도록 학습된
백본이 `[토큰 ⊕ 연속 잠재]` 시퀀스를 읽어야 하는 불일치다. 그러므로 송신
측(인코딩)까지 건드리는 것은 논거를 넘어선 적용이고, `H` 와 질문 KV 의 재사용
경제도 함께 잃는다.

**이 계약이 필요한 이유는 위반이 조용하기 때문이다.** PEFT 는 대상 모듈을
제자리에서 교체하므로, 인코딩과 디코딩이 같은 인스턴스를 공유하는 구조에서는
어댑터가 인코딩에도 자동 적용된다. 그래도 학습은 정상적으로 돌고 손실은
내려간다 — `H` 캐시는 유효하다고 믿긴 채 오염되고, 결과만 설명 불가가 된다.
"""

from __future__ import annotations

import pytest
import torch

from lsrr.backbone.lora import adapters_disabled, count_lora_parameters
from lsrr.core.errors import EncodingNotBaseOnly
from lsrr.core.invariants import assert_encoding_is_base_only

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def bare_backbone():
    """**전용** 백본 인스턴스.

    공유 `gpt2_backbone` 픽스처에 어댑터를 붙이면 PEFT 가 모듈을 제자리에서
    교체하므로 같은 세션의 다른 테스트가 전부 오염된다 — 이 파일이 막으려는
    바로 그 사고를 테스트가 저지르는 꼴이다.
    """
    from lsrr.backbone import HFFrozenCausalBackbone

    return HFFrozenCausalBackbone("sshleifer/tiny-gpt2", device="cpu")


@pytest.fixture(scope="module")
def lora_backbone():
    """어댑터를 장착하고 `lora_B` 를 비영으로 만든다.

    초기값 `lora_B = 0` 이면 델타가 0 이라 오염이 **일어나지 않는다.** 그 상태로
    통과하는 테스트는 가드가 아니라 초기화를 검증한 것이다.
    """
    from lsrr.backbone import HFFrozenCausalBackbone

    backbone = HFFrozenCausalBackbone("sshleifer/tiny-gpt2", device="cpu")
    backbone.attach_lora(r=4, alpha=8)
    with torch.no_grad():
        for name, p in backbone.model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.5)
    return backbone


def _ids(backbone):
    return torch.tensor([[15496, 11, 314, 716, 257]], device=backbone.device)


def test_adapters_actually_contaminate_encoding_without_the_guard(lora_backbone):
    """가드가 막고 있는 것이 실재하는지 먼저 보인다."""
    ids = _ids(lora_backbone)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        active = lora_backbone.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-1]
    with torch.no_grad(), adapters_disabled(lora_backbone.peft_model):
        base = lora_backbone.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-1]
    assert not torch.equal(active, base)


def test_encode_is_bit_identical_to_base(lora_backbone):
    """비교는 비트 단위다 — 허용 오차를 두면 '조금 오염됐지만 통과' 가 생긴다."""
    ids = _ids(lora_backbone)
    guarded = lora_backbone.encode(ids).H_last
    with adapters_disabled(lora_backbone.peft_model):
        base = lora_backbone.encode(ids).H_last
    assert torch.equal(guarded, base)
    assert_encoding_is_base_only(lora_backbone, ids)


def test_invariant_catches_a_leaking_encoder(lora_backbone, monkeypatch):
    """가드를 떼면 불변식이 잡아야 한다 — 통과가 우연이 아님을 보인다."""
    import lsrr.backbone.session as session

    from contextlib import contextmanager

    @contextmanager
    def _no_guard(_peft):
        yield

    monkeypatch.setattr(session, "adapters_disabled", _no_guard)
    with pytest.raises(EncodingNotBaseOnly, match="I9"):
        assert_encoding_is_base_only(lora_backbone, _ids(lora_backbone))


def test_base_weights_survive_lora_attachment(bare_backbone):
    """I1 재정의: base 지문은 어댑터 장착 전후로 같다."""
    before = bare_backbone.weight_hash()
    bare_backbone.attach_lora(r=4, alpha=8)
    assert bare_backbone.weight_hash() == before
    bare_backbone.verify_frozen()


def test_only_lora_params_are_trainable(lora_backbone):
    trainable = [n for n, p in lora_backbone.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert count_lora_parameters(lora_backbone.model) > 0


def test_guard_is_a_noop_without_adapters(gpt2_backbone):
    """Phase A 는 어댑터가 없다 — 호출부가 LoRA 유무로 분기하지 않아야 한다."""
    assert not gpt2_backbone.has_lora
    assert_encoding_is_base_only(gpt2_backbone, _ids(gpt2_backbone))  # 자명하게 통과
    with adapters_disabled(None):
        pass
