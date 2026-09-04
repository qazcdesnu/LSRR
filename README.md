# LSRR: Layer-State Recurrent Reasoner

Research codebase for **Layer-State Recurrent Reasoner (LSRR)**: Decoupling Transformer Context and SSM Reasoning Engine with Convergence-based Termination.

## Architecture Overview

```
Frozen LLM Backbone (GPT-2, Llama-3.2, Mistral-7B)
       │  ( 단 1회 전파, 질문 마지막 토큰 수직 1열 추출 )
       ▼
   H ∈ ℝ^{L × d} ( 디스크 샤딩 safetensors 캐시 )
       │
       ▼
  LayerAdapter ( per_layer_affine + RMSNorm + Layer Pos Emb )
       │
       ▼
  R^{(0)} ∈ ℝ^{L × d_model}
       │
       ▼
  RefinementEngine ( HydraQS, Mamba-Up, Mamba-Down, Bidir-Add, AttnBlock )
   ↺ M 사이클 반복 정제 ( Truncated BPTT, Δ < ε 수렴 시 조기 종료 )
       │
       ▼
  R* = R^{(M)}
       │
       ▼
  FusionHead ( Attention Pooling across layers: α_l, h_fusion )
       │
       ▼
  AnswerDecoder ( Trained Light Decoder / Frozen Backbone Prefix )
```

---

## Quickstart & Environment

Set up environment using `uv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

---

## Running Experiments

### 1. Extract Hidden States Cache
Extract layer-wise hidden states $H \in \mathbb{R}^{L \times d}$ using frozen GPT-2 on synthetic multiplication:
```bash
python scripts/extract_h.py backbone=gpt2 data=mult_4x4 split=train
python scripts/extract_h.py backbone=gpt2 data=mult_4x4 split=val
```

### 2. End-to-End Training
Run training with YAML config and CLI overrides:
```bash
python scripts/train.py exp=ablation_C_direction engine.type=mamba_up
```

With custom hyperparameters:
```bash
python scripts/train.py exp=ablation_C_direction engine.type=hydra_qs train.lr=5e-4 train.epochs=10
```

### 3. Grid Sweeps
Run grid sweeps over all engines defined in YAML `sweep:` section:
```bash
python scripts/sweep.py exp=ablation_C_direction
```

### 4. Evaluation and Pareto Sweep
Evaluate a checkpoint under different termination rules ($\Delta < \varepsilon$ thresholds):
```bash
python scripts/eval.py run_dir=runs/<run_id>
```

### 5. Generate Paper Tables
Merge local runs index (`runs/runs_index.csv`) with published baselines (`configs/published_numbers.yaml`):
```bash
python scripts/make_tables.py
```

---

## Running Tests

Run the complete test verification suite:
```bash
pytest -v tests/
```

- `test_freeze.py`: Validates that frozen backbone weights and hash are 100% invariant after training.
- `test_engine_equiv.py`: Validates numerical equivalence between HydraQS forward branch and Mamba-Up.
- `test_cache.py`: Validates cache key uniqueness and tensor reconstruction fidelity ($< 10^{-5}$).
- `test_termination.py`: Validates early stopping on converging sequences and $M_{\max}$ fallback on oscillating sequences.
- `test_reproducibility.py`: Validates deterministic multi-run reproducibility bit-for-bit.
- `test_param_matching.py`: Validates parameter budget matching between AttentionBlock and HydraQS ($< 5\%$).
