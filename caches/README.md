# `caches/` — 은닉 상태 캐시

버전 관리 대상이 아니다. `scripts/extract_h.py`가 여기에 쓴다.

## 구조
```
caches/<backbone_id>/<cache_key>/
  manifest.json
  shard_0000.safetensors
  ...
```
`cache_key = sha256(backbone_id, dataset_id, split, tokenizer_hash, position_rule, precision, include_embedding)[:16]`

## 이 캐시는 보조 경로다 (ADR-002)
기본 학습 경로는 매 스텝 동결 백본을 1회 순전파해 `H`와 **질문 KV 캐시**를 동시에 얻는다. KV 캐시는 디스크에 담기에 너무 크다(GPT-2 기준 샘플당 약 7.4MB, GSM8k-Aug 학습셋 전체로는 약 2.8TB).

따라서 이 캐시에는 `H`만 있으며, **캐시 경로로는 연속 디코딩을 할 수 없다.** 캐시로 학습을 시도하면 명시적 오류를 던진다. 용도는 엔진 전용 프로파일링과 어댑터 단독 실험에 한정된다.
