"""ProsQA 로더 — 홉 라벨, 채점 엄격성, 정답 누출 (I6).

이 과제를 v2 의 주 무대로 쓰는 이유는 **홉 수가 샘플마다 라벨로 붙는다**는
점이다. §7.1-1 의 "3-hop 이상에서 급격한 저하" 가설과 §7.2 의 "홉이 많을수록
오래 생각한다"를 같은 런에서 볼 수 있다. 그 라벨이 조용히 사라지면 두 분석이
모두 불가능해지므로 여기서 고정한다.

데이터 파일은 저장소에 없다(.gitignore). 없으면 건너뛴다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lsrr.core.errors import ConfigError
from lsrr.data.datasets.prosqa import ProsQADataset
from lsrr.data.prompting import PromptSpec, assert_no_answer_text_leakage

DATA_DIR = Path("data/prosqa")
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "prosqa_test.json").exists(),
    reason="data/prosqa 가 없다 — coconut 배포본을 받아 두어야 한다",
)


@pytest.fixture(scope="module")
def ds() -> ProsQADataset:
    return ProsQADataset(data_dir=str(DATA_DIR))


def test_every_sample_carries_a_hop_label(ds):
    """홉 라벨이 §7.2 메커니즘 분석의 층화 축이다."""
    for split in ("train", "val", "test"):
        samples = ds.get_split(split)
        assert samples
        assert all(s.meta["hops"] >= 1 for s in samples)
        assert all(s.meta["hops"] == len(s.cot_steps) for s in samples)


def test_the_hypothesis_range_is_actually_covered(ds):
    """3-hop 이상이 표본에 충분히 있어야 §7.1-1 가설을 볼 수 있다."""
    hops = [s.meta["hops"] for s in ds.get_split("test")]
    assert min(hops) >= 3 and max(hops) >= 5


def test_valid_alias_reaches_the_val_split(ds):
    """split 이름이 달라 빈 split 을 조용히 읽는 사고를 막는다."""
    assert ds.get_split("valid") is ds.get_split("val")


def test_answers_do_not_appear_verbatim_in_questions(ds):
    """I6 의 1차 방어선. 정답 문장이 질문에 그대로 있으면 복사로 풀린다."""
    spec = PromptSpec(max_question_tokens=640, max_answer_tokens=12)
    for split in ("train", "val", "test"):
        assert_no_answer_text_leakage(ds.get_split(split), spec)


def test_hop_filter_narrows_the_sample(ds):
    narrow = ProsQADataset(data_dir=str(DATA_DIR), min_hops=5)
    assert {s.meta["hops"] for s in narrow.get_split("test")} <= {5, 6}


def test_an_empty_hop_filter_fails_loudly():
    """빈 split 으로 학습이 시작되면 손실이 없는데도 런이 성공으로 끝난다."""
    ds = ProsQADataset(data_dir=str(DATA_DIR), min_hops=99)
    with pytest.raises(ConfigError, match="비었다"):
        ds.get_split("test")


def test_missing_data_dir_says_where_to_get_it():
    ds = ProsQADataset(data_dir="data/does-not-exist")
    with pytest.raises(ConfigError, match="coconut"):
        ds.get_split("test")


# ---------------------------------------------------------------- 채점


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ("Sally is a sterpus.", True),
        ("sally is a sterpus", True),      # 정규화 후 일치
        (" Sally is a sterpus. ", True),
        ("sterpus", True),                  # 최종 개체어만 내놓아도 정답
        ("Sally is a grimpus.", False),     # 오답 분기
        ("", False),
        ("is a", False),                    # 부분 문자열 포함을 받으면 이게 정답이 된다
        ("Sally is a sterpus and a grimpus.", False),
    ],
)
def test_scoring_is_exact_on_the_final_entity(prediction, expected):
    ds = ProsQADataset(data_dir=str(DATA_DIR))
    assert ds.score(prediction, "Sally is a sterpus.", {}) is expected
