"""레지스트리 계약."""

from __future__ import annotations

import pytest

from lsrr.core.errors import RegistryError
from lsrr.core.registry import ALL_REGISTRIES, Registry, registry_snapshot


def test_all_slots_have_registry():
    """ARCHITECTURE.md §3 슬롯 맵의 모든 슬롯에 레지스트리가 있다."""
    expected = {
        "backbone", "pooler", "composer", "scope", "adapter", "engine",
        "schedule", "termination", "stability", "fusion", "readout",
        "objective", "data", "analysis", "gate",
    }
    assert set(ALL_REGISTRIES) == expected


def test_register_and_build():
    reg = Registry("test")

    @reg.register("thing")
    class Thing:
        def __init__(self, a: int = 1, b: int = 2):
            self.a, self.b = a, b

    obj = reg.build({"type": "thing", "a": 10}, b=20)
    assert (obj.a, obj.b) == (10, 20)


def test_build_filters_unknown_kwargs():
    """생성자가 **kwargs를 받지 않으면 모르는 주입값은 걸러진다."""
    reg = Registry("test")

    @reg.register("narrow")
    class Narrow:
        def __init__(self, a: int = 1):
            self.a = a

    obj = reg.build({"type": "narrow"}, a=5, d_model=768, num_layers=12)
    assert obj.a == 5


def test_build_passes_all_kwargs_when_var_keyword():
    reg = Registry("test")

    @reg.register("wide")
    class Wide:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    obj = reg.build({"type": "wide"}, d_model=768)
    assert obj.kwargs["d_model"] == 768


def test_missing_type_raises():
    reg = Registry("test")
    with pytest.raises(RegistryError, match="'type' 키가 없다"):
        reg.build({"a": 1})


def test_unknown_key_lists_available():
    reg = Registry("test")
    reg.register("known")(lambda **kw: None)
    with pytest.raises(RegistryError, match="known"):
        reg.get("unknown")


def test_alias_resolves_to_canonical():
    """별칭은 정본 키로 정규화된다 — 런 기록·표 표기의 일관성."""
    reg = Registry("test")

    @reg.register("canonical")
    @reg.register("alias")
    class Impl:
        pass

    # 아래쪽 데코레이터가 먼저 적용되므로 'alias'가 첫 등록 = 정본
    assert reg.canonical_key("canonical") == reg.canonical_key("alias")


def test_string_shorthand_becomes_type():
    reg = Registry("test")

    @reg.register("thing")
    class Thing:
        def __init__(self):
            pass

    assert isinstance(reg.build("thing"), Thing)


def test_snapshot_lists_registered_keys():
    snap = registry_snapshot()
    assert set(snap) == set(ALL_REGISTRIES)
    assert all(isinstance(v, list) for v in snap.values())
