# 연구계획서: 계층 상태 순환 추론
(Layer-State Recurrent Reasoning)
**Decoupling Transformer Context and SSM Reasoning Engine with Convergence-based Termination**

## 1. 한 줄 주장
기존의 잠재 추론(Latent Reasoning) 모델들은 사고 스텝마다 무거운 트랜스포머 본체를 반복 호출하는 구조적 비효율을 안고 있다. 본 연구는 트랜스포머를 '문맥 인코더'로 단 한 번만 실행하여 계층적 은닉 상태(Layer-wise Hidden States)를 추출하고, 이를 경량화된 양방향 SSM(State Space Model) '사고 엔진'이 반복 정제(Recursive Refinement)하는 새로운 아키텍처를 제안한다. 이를 통해 연산 병목을 제거하고, 은닉 상태의 수렴(Convergence)을 기준으로 자율적인 종료(termination)를 수행하여 과사고(Overthinking)를 억제한다.

## 2. 문제 설정 및 기존 연구의 한계
최근 ICoT, Coconut, CODI 등의 연구는 사고 과정을 잠재 공간으로 옮겨 언어적 정보 병목을 해소하려 했으나, 아키텍처 관점에서 치명적인 비효율성을 갖는다.

* **트랜스포머 반복 호출의 늪 (Sequential Rollout Bottleneck):** $k$번의 잠재 사고 스텝을 밟기 위해 수백억 파라미터의 트랜스포머를 $k$번 통과해야 하므로 추론 지연시간(Latency)과 FLOPs가 급증한다.
* **계층적 표현의 낭비:** 기존 연구는 마지막 레이어($L$)의 출력만을 잠재 사고의 입력으로 사용한다. 그러나 하위 레이어의 어휘적(Lexical) 정보부터 상위 레이어의 의미론적(Semantic) 정보까지, 트랜스포머가 쌓아 올린 층(Layer)의 전개 자체가 이미 풍부한 '사고 메모리(Reasoning Memory)'임에도 이를 활용하지 못한다.
* **작위적인 스텝 수 및 종료(termination) 기준:** 고정된 잠재 토큰 길이를 강제하거나, 전환 헤드를 별도로 학습해야 하는 부담이 존재한다.

> **본 연구의 철학:** 트랜스포머는 '새로운 사고 상태를 생성'하는 데 쓰기엔 너무 무겁다. "Transformer는 단 1회 실행되는 문맥 인코더(Context Encoder)로, SSM은 이미 존재하는 계층적 표현을 반복 정제하는 사고 엔진(Reasoning Engine)으로 분리한다."

## 3. 핵심 아키텍처 설계

### 3.1. 계층적 사고 메모리 (Layer-wise Reasoning Memory)
질문의 마지막 토큰 위치를 $k$라 할 때, 트랜스포머를 단 1회 순전파하여 모든 레이어의 은닉 상태를 추출한다. (단, 단일 토큰의 정보 병목을 막기 위해 $k$번째 토큰뿐만 아니라 질문 전체의 어텐션 풀링 결과를 결합하여 글로벌 문맥을 보완한다.)

$$ H = [h^{(1)}, h^{(2)}, \dots, h^{(L)}] \in \mathbb{R}^{L \times d} $$

이 $H$ 행렬이 단순한 특징(Feature)의 모음이 아니라, SSM이 읽고 쓸 수 있는 최초의 '초기 사고 메모리' $R^{(0)} = H$가 된다.

### 3.2. 양방향 SSM 사고 엔진 (Bi-directional SSM Reasoning Block)
초기 메모리 $R^{(0)}$를 경량화된 Mamba 계열의 양방향 SSM 블록 $S_\phi$에 투입하여 $M$번 반복(Iteration) 정제한다.

$$ R^{(m+1)} = S_\phi(R^{(m)}) $$

* **동시 처리와 순차 반복:** $L$개의 레이어 상태는 SSM의 병렬 스캔(Parallel Scan)을 통해 깊이 방향으로 동시에(Simultaneously) 처리된다. 오직 정제 사이클($m$)만이 순차적(Sequential)이다.
* **양방향 교환:** 추상화 레벨이 다른 하위 층과 상위 층 간의 논리적 인과관계를 모델링하기 위해 양방향(Bi-directional) 스캔을 적용한다.

### 3.3. 수렴 기반 자율 종료 (Convergence-based Termination)
본 모델은 사고를 멈추기 위해 외부의 학습된 전환 헤드에 의존하지 않는다. 매 반복($m$)마다 메모리의 변화량을 측정하여, 논리적 추론이 더 이상 의미 있는 변화를 만들어내지 못할 때 사고를 자율 종료(termination)한다.

$$ \Delta^{(m)} = \frac{1}{L} \sum_{l} \| r_l^{(m+1)} - r_l^{(m)} \|_2 $$

* **종료(termination) 조건:** $\Delta^{(m)} < \epsilon$

이는 Deep Equilibrium Model의 고정점(Fixed-point) 정제와 유사한 해석을 가지며, 문제 난이도에 따라 동적으로 연산량이 결정되는 가장 우아한 적응적 계산(Adaptive Computation) 방식이다.

### 3.4. 통합 및 잔차 결합 (Representation Fusion)
수렴하여 종료(termination)된 최종 메모리 $R^* = R^{(M)}$에서 어텐션 풀링을 통해 하나의 사고 표현을 통합한다.

$$ \alpha_l = \text{softmax}(w^T r_l^*) $$
$$ h_{SSM} = \sum_{l} \alpha_l r_l^* $$

이후, 원래 트랜스포머가 가지고 있던 문맥 표현($h^{(L)}$)에 정제된 사고 결과($h_{SSM}$)를 잔차(Residual) 형태로 결합하여 디코딩한다.

$$ h_{fusion} = h^{(L)} + W_r \cdot h_{SSM} $$

## 4. 학습 목표 (Training Objectives)
전체 아키텍처는 다음 두 가지 손실 함수의 결합으로 최적화된다.

* **정답 생성 손실 (Answer Loss):** 최종 결합된 $h_{fusion}$을 기반으로 정답 토큰을 예측하는 $L_{NLL}$
* **추론 증류 손실 (Reasoning Distillation, 선택적):** 교사 모델(Teacher CoT)이 산출한 최종 은닉 상태 궤적과 $h_{SSM}$ 사이의 거리를 좁히는 지식 증류를 통해, SSM이 올바른 논리적 정제 방향을 학습하도록 유도한다.

$$ L = L_{NLL} + \lambda \cdot L_{KD} $$

## 5. 핵심 절제 실험 (Ablation Studies)
"트랜스포머 층 전체를 사용해야 하며, SSM이 사고 엔진으로서 타당하다"는 가설을 증명하기 위해 다음 세 가지 핵심 비교를 수행한다.

| Ablation 조건 | 비교 대상 | 실험 목적 및 증명 대상 |
| :--- | :--- | :--- |
| **A. 메모리 범위** | $h^{(L)}$ (단일 레이어) vs 전체 레이어 $H$ | 깊이 방향의 계층적 은닉 상태가 '사고 메모리'로서 압도적인 효용이 있음을 증명 (ICoT형 구조와의 차별화) |
| **B. 종료(termination) 규칙** | 고정 반복 횟수 vs 수렴 종료(termination) ($\epsilon$) | 수렴 기반 종료(termination)가 쉬운 문제의 과사고를 막고 동적 적응을 완벽히 수행함을 실증 |
| **C. 사고 엔진 구조** | 단순 MLP/Transformer Block vs 양방향 SSM | Mamba 구조 특유의 선형 복잡도와 상태 압축 능력이 반복 정제에 최적화되어 있음을 속도/성능으로 입증 |

## 6. 예상되는 장점 및 기여
* **아키텍처 혁신 (Decoupling):** 추론(Reasoning) 연산을 무거운 거대 언어 모델 본체에서 독립된 SSM 블록으로 완벽하게 분리해 낸 최초의 구조이다.
* **연산 효율의 극대화:** Coconut이 $O(N)$회 트랜스포머를 호출할 때, 제안 모델은 단 1회의 트랜스포머 호출만으로 동일하거나 더 깊은 사고 스텝을 전개하므로 실측 추론 속도(Latency)를 극적으로 단축한다.
* **매니폴드 붕괴 위험 원천 차단:** 기존 모델처럼 생성된 잠재 토큰을 다시 입력층(Input Embedding)으로 밀어 넣지 않고 닫힌 공간에서 상태를 업데이트하므로, 매니폴드 불일치(Manifold Mismatch)로 인한 훈련 붕괴가 발생하지 않는다.

---

# 💡 제안하는 모델 이름 (Model Name Recommendations)

위 연구계획서의 핵심 철학인 **트랜스포머 분리, 계층적 사고 메모리, 양방향 SSM, 수렴 기반 자율 종료(termination)** 특성을 반영하여 다음과 같은 모델 이름들을 제안합니다.

### 1. 직관적이고 기능적인 이름 (학술 논문용)
* **LSRR (Layer-State Recurrent Reasoner):** 현재 부제목으로 쓰인 약어를 그대로 살려, 학술적으로 가장 명확하고 직관적인 인상을 줍니다.
* **CoRe-SSM (Context-Reasoning Decoupled SSM):** 트랜스포머를 문맥(Context)으로, SSM을 추론(Reasoning)으로 완전히 분리(Decoupling)했다는 핵심 기여를 강조합니다.
* **L-SSM (Layer-wise SSM for Reasoning):** 기존 마지막 층만 쓰던 방식과 달리, '트랜스포머의 전체 레이어(Layer-wise)'를 사고 메모리로 활용한다는 차별점을 직관적으로 보여줍니다.

### 2. 약어를 활용한 인상적인 이름 (브랜딩용)
* **LORE (Layer-Oriented Reasoning Engine):** '설화, 지식'이라는 뜻의 영단어 lore와 형태가 같아 기억하기 쉬우며, 계층 중심적(Layer-Oriented) 사고 엔진이라는 의미를 훌륭하게 담고 있습니다.
* **DART (Decoupled Architecture for Recursive Thinking):** '반복적 사고를 위한 분리형 아키텍처'라는 뜻으로, 빠르고 날카롭게 정답을 찾아내는 '다트(Dart)'의 민첩한 이미지를 줍니다.
* **AURA (Adaptive Unrolling Reasoning Architecture):** 고정된 스텝이 아니라 수렴 조건에 따라 유연하게 적응(Adaptive)하며 사고를 전개(Unrolling)한다는 점을 강조합니다.

### 3. '수렴/종료(termination)' 및 '상태(State)' 강조형
* **REST (Recursive State Transformer-decoupled Engine):** 휴식(Rest)이라는 단어처럼, 무거운 트랜스포머 본체의 반복 연산 부담을 '쉬게' 해주는 경량화된 엔진임을 재치 있게 표현합니다.
* **Converge-R (Convergence-based Reasoner):** 외부 헤드 없이 자율적으로 판단하는 동적 계산의 핵심인 '수렴(Convergence)' 방식을 가장 전면에 내세운 이름입니다.