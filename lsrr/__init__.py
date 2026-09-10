"""LSRR — Layer-State Recurrent Reasoner.

동결 트랜스포머를 문맥 인코더로 단 1회 실행해 레이어별 은닉 상태 전개를 얻고,
분리된 경량 양방향 SSM 사고 엔진이 정제 사이클 축에서 이를 반복 정제하며,
사고 메모리의 수렴을 기준으로 자율 종료한다.

패키지 루트에는 조립 코드만 둔다 (lsrr/README.md).
"""

__version__ = "0.1.0"

__all__ = ("__version__",)
