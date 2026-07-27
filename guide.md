# PhysSet-DOA: Geometry-Aware Physics-Guided Set Prediction for Multi-Source 3D DOA Estimation

---

## 1. 연구 목표

본 연구는 마이크의 개수와 배치가 달라지는 환경에서 복수 음원의 3차원 도래방향(Direction of Arrival, DOA)을 추정하는 것을 목표로 한다.

기존 딥러닝 기반 DOA 추정 모델은 특정 마이크 배열에 종속되거나, 고정된 방향 그리드에 대해 분류를 수행하는 경우가 많다. 이러한 방식은 학습 시 사용하지 않은 마이크 배열에 대한 일반화가 어렵고, 방향 해상도가 미리 설정된 그리드 간격에 제한되는 문제가 있다.

이를 해결하기 위해 PhysSet-DOA는 다음 요소를 결합한다.

1. 마이크 배열을 순서 없는 기하학적 집합으로 처리하는 geometry-aware encoder
2. 다중 음원을 순서 없는 방향 집합으로 출력하는 set prediction decoder
3. 각 방향을 연속적인 3차원 단위벡터로 표현하는 grid-free DOA estimation
4. 예측된 방향으로부터 마이크 간 위상 및 공간 공분산을 복원하는 physics-guided decoder
5. 마이크 배열 회전, 채널 제거 및 위치 오차에 대한 consistency learning

---

## 2. 문제 정의

마이크의 개수를 $M$, 시간 프레임의 수를 $T$, 주파수 bin의 수를 $F$라고 정의한다.

다채널 입력 신호의 STFT는 다음과 같이 표현한다.

$$
\mathbf X
=
\left\{
X_m(f,t)
\right\}_{m=1}^{M}
\in
\mathbb C^{M\times F\times T}
$$

각 마이크의 3차원 좌표는 다음과 같다.

$$
\mathbf r_m
=
[x_m,y_m,z_m]^\top
\in
\mathbb R^3
$$

모델은 오디오 신호와 마이크 위치를 입력받는다.

$$
f_\theta:
(\mathbf X,\mathbf R)
\rightarrow
\hat{\mathcal S}
$$

여기서

$$
\mathbf R
=
\{\mathbf r_m\}_{m=1}^{M}
$$

이며, 출력 $\hat{\mathcal S}$는 순서가 없는 복수 음원의 방향 집합이다.

$$
\hat{\mathcal S}
=
\left\{
\hat s_k
\right\}_{k=1}^{K_{\max}}
$$

각 source slot $\hat s_k$는 다음 값을 포함한다.

$$
\hat s_k
=
\left(
\hat p_k,
\hat{\mathbf u}_k,
\hat\kappa_k
\right)
$$

- $\hat p_k$: $k$번째 source slot이 활성 음원일 확률
- $\hat{\mathbf u}_k$: 예측된 3차원 DOA 단위벡터
- $\hat\kappa_k$: 방향 추정의 확신도를 나타내는 농도 파라미터
- $K_{\max}$: 모델이 동시에 추정할 수 있는 최대 음원 수

실제 음원 수 $K$는 고정하지 않으며 다음을 만족한다.

$$
0\le K\le K_{\max}
$$

---

## 3. 3차원 DOA 표현

방위각과 고도를 직접 회귀하면 방위각의 $0^\circ$와 $360^\circ$ 경계에서 불연속성이 발생할 수 있다. 또한 고도각의 극점 부근에서는 동일한 방향이 서로 다른 각도 조합으로 표현되는 문제가 발생한다.

따라서 본 연구에서는 DOA를 Cartesian 단위벡터로 표현한다.

방위각을 $\phi$, 고도각을 $\theta$라고 할 때 방향벡터는 다음과 같다.

$$
\mathbf u
=
\begin{bmatrix}
\cos\theta\cos\phi\\
\cos\theta\sin\phi\\
\sin\theta
\end{bmatrix}
$$

모델은 정규화 이전의 벡터 $\mathbf v_k\in\mathbb R^3$를 출력하고 다음과 같이 단위벡터로 변환한다.

$$
\hat{\mathbf u}_k
=
\frac{\mathbf v_k}
{\|\mathbf v_k\|_2+\epsilon}
$$

최종 방위각과 고도각은 다음과 같이 계산한다.

$$
\hat\phi_k
=
\operatorname{atan2}
(\hat u_{k,y},\hat u_{k,x})
$$

$$
\hat\theta_k
=
\operatorname{atan2}
\left(
\hat u_{k,z},
\sqrt{
\hat u_{k,x}^2+\hat u_{k,y}^2
}
\right)
$$

---

## 4. 전체 모델 구조

PhysSet-DOA는 다음 다섯 개 모듈로 구성한다.

```text
Multi-channel waveform
        │
        ▼
STFT and pairwise spatial feature extraction
        │
        ▼
Geometry-aware pair encoder
        │
        ▼
Microphone set / graph encoder
        │
        ▼
Multi-source query decoder
        │
        ├── Activity probability
        ├── Continuous 3D DOA vector
        ├── Directional uncertainty
        └── Time-frequency source responsibility
        │
        ▼
Physics-guided spatial covariance decoder
```

---

## 5. 신호 전처리 및 공간 특징 추출

### 5.1 STFT

각 마이크 신호 $x_m[n]$에 대해 STFT를 수행한다.

$$
X_m(f,t)
=
\sum_n
x_m[n]
w[n-tH]
e^{-j2\pi fn/N}
$$

여기서 $N$은 FFT 크기, $H$는 hop size, $w[\cdot]$는 window function이다.

권장 초기 설정은 다음과 같다.

| 항목 | 설정 예시 |
|---|---:|
| Sampling rate | 16 kHz 또는 24 kHz |
| FFT size | 512 또는 1024 |
| Window length | 32 ms |
| Hop length | 8–16 ms |
| Maximum source count | 3 또는 4 |

---

### 5.2 마이크 쌍별 위상 특징

각 마이크 쌍 $(i,j)$에 대해 cross-spectrum을 계산한다.

$$
G_{ij}(f,t)
=
X_i(f,t)X_j^*(f,t)
$$

GCC-PHAT과 유사하게 크기 정보를 정규화한 complex phase spectrum을 사용한다.

$$
C_{ij}(f,t)
=
\frac{
G_{ij}(f,t)
}{
|G_{ij}(f,t)|+\epsilon
}
$$

실수부와 허수부를 각각 특징으로 사용한다.

$$
\mathbf c_{ij}(f,t)
=
\left[
\operatorname{Re}C_{ij}(f,t),
\operatorname{Im}C_{ij}(f,t)
\right]
$$

추가적으로 다음 특징을 함께 사용할 수 있다.

$$
\mathbf a_{ij}(f,t)
=
\left[
\log|X_i|,
\log|X_j|,
\operatorname{coh}_{ij},
\operatorname{Re}C_{ij},
\operatorname{Im}C_{ij}
\right]
$$

여기서 $\operatorname{coh}_{ij}$는 두 채널 사이의 magnitude-squared coherence 또는 학습 가능한 유사도 특징이다.

---

## 6. 마이크 배열 기하학 표현

### 6.1 좌표 중심화

배열의 절대 위치에 대한 의존성을 제거하기 위해 마이크 좌표를 배열 중심 기준으로 변환한다.

$$
\mathbf r_c
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf r_m
$$

$$
\tilde{\mathbf r}_m
=
\mathbf r_m-\mathbf r_c
$$

이 처리를 통해 배열 전체가 공간상에서 평행 이동해도 동일한 상대 기하학을 갖게 된다.

---

### 6.2 마이크 쌍 기하학 특징

마이크 쌍 $(i,j)$의 baseline vector를 다음과 같이 정의한다.

$$
\mathbf d_{ij}
=
\tilde{\mathbf r}_i-\tilde{\mathbf r}_j
$$

baseline 길이는 다음과 같다.

$$
l_{ij}
=
\|\mathbf d_{ij}\|_2
$$

단위 baseline 방향은 다음과 같다.

$$
\bar{\mathbf d}_{ij}
=
\frac{\mathbf d_{ij}}
{l_{ij}+\epsilon}
$$

최대 물리적 시간지연은 다음과 같이 계산할 수 있다.

$$
\tau_{ij}^{\max}
=
\frac{l_{ij}}{c}
$$

여기서 $c$는 음속이다.

최종 geometry feature는 다음과 같이 구성한다.

$$
\mathbf g_{ij}
=
\left[
\mathbf d_{ij},
\bar{\mathbf d}_{ij},
l_{ij},
\tau_{ij}^{\max}
\right]
$$

배열의 실제 크기는 DOA에 따른 위상차에 직접적인 영향을 주므로 좌표를 단순히 단위 크기로 정규화하지 않는다. 대신 중심화된 실제 좌표와 배열 aperture 정보를 함께 입력한다.

$$
D_{\mathrm{array}}
=
\max_{i,j}
\|\mathbf r_i-\mathbf r_j\|_2
$$

---

## 7. Geometry-Aware Pair Encoder

각 마이크 쌍에 대해 음향 특징과 기하학 특징을 결합한다.

$$
\mathbf z_{ij}(f,t)
=
\left[
\mathbf a_{ij}(f,t),
\mathbf g_{ij},
\frac{f}{F}
\right]
$$

모든 마이크 쌍에는 동일한 파라미터를 가진 shared encoder를 적용한다.

$$
\mathbf h_{ij}(f,t)
=
E_{\mathrm{pair}}
\left(
\mathbf z_{ij}(f,t)
\right)
$$

shared encoder를 사용하면 마이크 개수와 마이크 쌍의 수가 달라져도 동일한 모델을 적용할 수 있다.

Pair encoder는 다음 구조로 구현할 수 있다.

```text
Pair feature
    │
Linear projection
    │
Frequency convolution or frequency attention
    │
Temporal convolution / conformer block
    │
Geometry-conditioned feature modulation
    │
Pair embedding
```

기하학 특징을 단순 concatenate하는 것 외에도 FiLM 방식으로 음향 특징을 조절할 수 있다.

$$
\boldsymbol\gamma_{ij},
\boldsymbol\beta_{ij}
=
\operatorname{MLP}_{\mathrm{geo}}(\mathbf g_{ij})
$$

$$
\tilde{\mathbf h}_{ij}
=
\boldsymbol\gamma_{ij}
\odot
\mathbf h_{ij}
+
\boldsymbol\beta_{ij}
$$

이 구조는 동일한 위상 패턴이라도 마이크 간 거리와 방향에 따라 다르게 해석하도록 한다.

---

## 8. 마이크 집합 및 그래프 인코더

마이크는 node, 마이크 쌍은 edge인 완전 그래프로 표현한다.

$$
\mathcal G
=
(\mathcal V,\mathcal E)
$$

$$
\mathcal V
=
\{1,\ldots,M\}
$$

$$
\mathcal E
=
\{(i,j)\mid i\neq j\}
$$

각 edge는 pair embedding $\mathbf h_{ij}$를 가진다.

마이크 순서 변화에 영향을 받지 않도록 permutation-invariant attention aggregation을 수행한다.

$$
\mathbf h_i
=
\sum_{j\neq i}
\alpha_{ij}
\mathbf W_v\mathbf h_{ij}
$$

$$
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{
(\mathbf W_q\mathbf q_i)^\top
(\mathbf W_k\mathbf h_{ij})
}{
\sqrt d
}
\right)
$$

이후 모든 마이크 node를 다시 집합 형태로 통합한다.

$$
\mathbf H
=
E_{\mathrm{set}}
\left(
\{\mathbf h_i\}_{i=1}^{M}
\right)
$$

실제 구현에서는 다음 중 하나를 사용할 수 있다.

- Set Transformer
- Graph Attention Network
- Perceiver-style latent encoder
- SE(3)-equivariant graph neural network

초기 구현에서는 Set Transformer 또는 Graph Attention Network를 사용하는 것이 현실적이며, 후속 실험에서 SE(3)-equivariant 구조를 적용한다.

---

## 9. Multi-Source Set Prediction Decoder

### 9.1 Source query

최대 음원 수 $K_{\max}$만큼 학습 가능한 source query를 정의한다.

$$
\mathbf Q
=
\{\mathbf q_k\}_{k=1}^{K_{\max}}
$$

각 query는 encoder의 전체 시공간 표현 $\mathbf H$에 cross-attention을 수행한다.

$$
\mathbf z_k
=
D_{\mathrm{query}}
(\mathbf q_k,\mathbf H)
$$

각 query는 하나의 잠재적인 음원을 나타내며, 실제 음원 수가 $K_{\max}$보다 적은 경우 나머지 query는 비활성 상태를 출력한다.

---

### 9.2 Activity head

각 source query의 활성 확률을 예측한다.

$$
\hat p_k
=
\sigma
\left(
\operatorname{MLP}_{\mathrm{act}}
(\mathbf z_k)
\right)
$$

추론 단계에서는 다음 조건을 만족하면 활성 음원으로 판단한다.

$$
\hat p_k>\delta
$$

여기서 $\delta$는 activity threshold이다.

---

### 9.3 Direction head

방향 예측 head는 3차원 벡터를 출력한다.

$$
\mathbf v_k
=
\operatorname{MLP}_{\mathrm{dir}}
(\mathbf z_k)
$$

$$
\hat{\mathbf u}_k
=
\frac{\mathbf v_k}
{\|\mathbf v_k\|_2+\epsilon}
$$

이 방식은 고정 방향 grid를 사용하지 않기 때문에 이론적으로 임의의 방향 해상도를 가질 수 있다.

---

### 9.4 Direction uncertainty head

각 방향에 대한 확신도 $\kappa_k$를 예측한다.

$$
\hat\kappa_k
=
\operatorname{softplus}
\left(
\operatorname{MLP}_{\mathrm{unc}}
(\mathbf z_k)
\right)
+
\kappa_{\min}
$$

$\kappa_k$가 클수록 예측 방향 주변에 확률이 집중되고, 작을수록 방향 불확실성이 크다는 것을 의미한다.

방향 분포는 von Mises–Fisher 분포로 모델링한다.

$$
p(\mathbf u\mid\boldsymbol\mu,\kappa)
=
C_3(\kappa)
\exp
\left(
\kappa\boldsymbol\mu^\top\mathbf u
\right)
$$

여기서

$$
C_3(\kappa)
=
\frac{\kappa}
{4\pi\sinh\kappa}
$$

이다.

---

## 10. Time-Frequency Source Responsibility

다중 음원 환경에서는 하나의 시간-주파수 bin에 특정 음원의 에너지가 더 강하게 나타나는 경우가 많다. 이를 활용하기 위해 각 source query가 TF bin별 책임도 또는 soft mask를 예측하도록 한다.

$$
q_k(f,t)
=
\operatorname{softmax}_k
\left(
\mathbf z_k^\top
\mathbf h(f,t)
\right)
$$

배경 또는 diffuse noise를 표현하기 위해 background slot $q_0(f,t)$를 추가할 수 있다.

$$
\sum_{k=0}^{K_{\max}}
q_k(f,t)
=
1
$$

TF responsibility는 반드시 완전한 음원분리 mask일 필요는 없으며, 각 방향을 지지하는 공간적 관측을 구분하는 latent assignment로 사용한다.

---

## 11. Physics-Guided Spatial Decoder

### 11.1 Plane-wave steering vector

원거리 음원과 자유음장 근사를 적용하면 방향 $\mathbf u_k$에서 입사하는 평면파의 steering vector는 다음과 같다.

$$
\mathbf a_f(\mathbf u_k)
=
\begin{bmatrix}
e^{-j\omega_f
\tilde{\mathbf r}_1^\top\mathbf u_k/c}\\
\vdots\\
e^{-j\omega_f
\tilde{\mathbf r}_M^\top\mathbf u_k/c}
\end{bmatrix}
$$

여기서

$$
\omega_f=2\pi f
$$

이다.

마이크 쌍 $(i,j)$에서 예측되는 이론적 위상차는 다음과 같다.

$$
\Delta\hat\phi_{ij,k}(f)
=
-\frac{\omega_f}{c}
\mathbf d_{ij}^\top
\hat{\mathbf u}_k
$$

---

### 11.2 관측 공간 공분산

단일 STFT frame의 outer product는 변동성이 크므로 인접 프레임을 이용해 local spatial covariance matrix를 계산한다.

$$
\mathbf R_{\mathrm{obs}}(f,t)
=
\frac{
\sum_{\tau\in\mathcal W_t}
w_\tau
\mathbf x(f,\tau)
\mathbf x^H(f,\tau)
}{
\sum_{\tau\in\mathcal W_t}w_\tau+\epsilon
}
$$

여기서

$$
\mathbf x(f,t)
=
[X_1(f,t),\ldots,X_M(f,t)]^\top
$$

이다.

---

### 11.3 예측 공간 공분산

각 음원 방향과 TF responsibility를 이용해 예측 공분산을 구성한다.

$$
\hat{\mathbf R}(f,t)
=
\sum_{k=1}^{K_{\max}}
\gamma_k(f,t)
\mathbf a_f(\hat{\mathbf u}_k)
\mathbf a_f^H(\hat{\mathbf u}_k)
+
\gamma_0(f,t)\mathbf\Gamma_{\mathrm{diff}}(f)
+
\sigma^2(f,t)\mathbf I
$$

여기서

$$
\gamma_k(f,t)
=
\hat p_k
q_k(f,t)
\rho(f,t)
$$

이며, $\rho(f,t)$는 관측 신호 에너지 또는 별도의 network head가 예측한 source power이다.

$\mathbf\Gamma_{\mathrm{diff}}$는 diffuse noise 또는 잔향 성분을 근사하는 공간 공분산이고, $\sigma^2\mathbf I$는 비상관 센서 잡음을 나타낸다.

---

### 11.4 정규화된 공분산 비교

음원 크기보다 공간적 위상 및 상관 구조에 집중하기 위해 공분산 행렬을 trace로 정규화한다.

$$
\bar{\mathbf R}_{\mathrm{obs}}
=
\frac{
\mathbf R_{\mathrm{obs}}
}{
\operatorname{tr}
(\mathbf R_{\mathrm{obs}})
+\epsilon
}
$$

$$
\bar{\mathbf R}_{\mathrm{pred}}
=
\frac{
\hat{\mathbf R}
}{
\operatorname{tr}
(\hat{\mathbf R})
+\epsilon
}
$$

physics reconstruction loss는 다음과 같다.

$$
\mathcal L_{\mathrm{phys}}
=
\frac{1}{FT}
\sum_{f,t}
w(f,t)
\left\|
\bar{\mathbf R}_{\mathrm{obs}}(f,t)
-
\bar{\mathbf R}_{\mathrm{pred}}(f,t)
\right\|_F^2
$$

$w(f,t)$는 direct-path dominance, coherence 또는 신호 에너지를 기반으로 정할 수 있다.

잔향이 강하거나 신뢰도가 낮은 TF bin에서는 물리 손실의 비중을 줄인다.

---

## 12. Permutation-Invariant Set Matching

다중 음원의 정답 순서는 정의되지 않는다. 예를 들어 정답 방향이 $\{\mathbf u_1,\mathbf u_2\}$인 경우 모델의 첫 번째 query가 $\mathbf u_2$를 출력하고 두 번째 query가 $\mathbf u_1$을 출력해도 동일한 결과로 평가해야 한다.

이를 위해 Hungarian matching을 적용한다.

예측 source $k$와 정답 source $j$ 사이의 matching cost를 다음과 같이 정의한다.

$$
C_{k,j}
=
\lambda_{\mathrm{ang}}
d_{\mathrm{ang}}
(\hat{\mathbf u}_k,\mathbf u_j)
+
\lambda_{\mathrm{act}}
\operatorname{BCE}
(\hat p_k,1)
+
\lambda_{\mathrm{unc}}
\mathcal L_{\mathrm{vMF}}(k,j)
$$

각도 오차는 다음과 같다.

$$
d_{\mathrm{ang}}
(\hat{\mathbf u},\mathbf u)
=
\arccos
\left(
\operatorname{clip}
(\hat{\mathbf u}^\top\mathbf u,-1,1)
\right)
$$

최적 matching은 다음과 같이 구한다.

$$
\hat\pi
=
\arg\min_{\pi}
\sum_{j=1}^{K}
C_{\pi(j),j}
$$

matching되지 않은 source query는 inactive target으로 학습한다.

---

## 13. 손실함수

전체 손실함수는 다음과 같이 정의한다.

$$
\mathcal L_{\mathrm{total}}
=
\lambda_{\mathrm{set}}
\mathcal L_{\mathrm{set}}
+
\lambda_{\mathrm{phys}}
\mathcal L_{\mathrm{phys}}
+
\lambda_{\mathrm{rot}}
\mathcal L_{\mathrm{rot}}
+
\lambda_{\mathrm{sub}}
\mathcal L_{\mathrm{sub}}
+
\lambda_{\mathrm{count}}
\mathcal L_{\mathrm{count}}
+
\lambda_{\mathrm{cal}}
\mathcal L_{\mathrm{cal}}
$$

---

### 13.1 Set prediction loss

$$
\mathcal L_{\mathrm{set}}
=
\mathcal L_{\mathrm{ang}}
+
\beta_{\mathrm{act}}
\mathcal L_{\mathrm{act}}
+
\beta_{\mathrm{vMF}}
\mathcal L_{\mathrm{vMF}}
$$

Angular loss는 다음과 같다.

$$
\mathcal L_{\mathrm{ang}}
=
\frac{1}{K}
\sum_{j=1}^{K}
d_{\mathrm{ang}}
\left(
\hat{\mathbf u}_{\hat\pi(j)},
\mathbf u_j
\right)
$$

Activity loss는 활성 query와 비활성 query 모두에 대해 binary cross-entropy를 계산한다.

$$
\mathcal L_{\mathrm{act}}
=
-\sum_{k=1}^{K_{\max}}
\left[
y_k\log\hat p_k
+
(1-y_k)\log(1-\hat p_k)
\right]
$$

---

### 13.2 vMF uncertainty loss

von Mises–Fisher negative log-likelihood는 다음과 같다.

$$
\mathcal L_{\mathrm{vMF}}
=
-\log C_3(\hat\kappa_k)
-
\hat\kappa_k
\hat{\mathbf u}_k^\top
\mathbf u_j
$$

모델이 틀린 방향에 대해 지나치게 높은 확신도를 출력하면 큰 손실을 받는다.

---

### 13.3 Rotation equivariance loss

좌표계 회전행렬 $\mathbf Q\in SO(3)$를 생성한다.

마이크 좌표와 정답 방향에 동일한 회전을 적용한다.

$$
\mathbf r'_m
=
\mathbf Q\mathbf r_m
$$

$$
\mathbf u'_j
=
\mathbf Q\mathbf u_j
$$

오디오 신호는 동일하게 유지하면서 좌표 표현만 회전시킨다. 모델은 다음 조건을 만족해야 한다.

$$
f_\theta(\mathbf X,\mathbf Q\mathbf R)
\approx
\mathbf Qf_\theta(\mathbf X,\mathbf R)
$$

rotation consistency loss는 다음과 같다.

$$
\mathcal L_{\mathrm{rot}}
=
d_{\mathrm{set}}
\left(
f_\theta(\mathbf X,\mathbf Q\mathbf R),
\mathbf Qf_\theta(\mathbf X,\mathbf R)
\right)
$$

이 손실은 모델이 특정 전역 좌표축에 과적합되는 것을 방지한다.

---

### 13.4 Microphone subset consistency loss

전체 마이크 집합에서 일부 채널을 무작위로 제거한다.

$$
\mathcal M'
\subset
\{1,\ldots,M\}
$$

전체 배열 출력과 부분 배열 출력이 가능한 범위 내에서 일관되도록 한다.

$$
\mathcal L_{\mathrm{sub}}
=
d_{\mathrm{set}}
\left(
f_\theta(\mathbf X,\mathbf R),
f_\theta
(\mathbf X_{\mathcal M'},\mathbf R_{\mathcal M'})
\right)
$$

다만 제거된 배열이 3차원 방향을 물리적으로 구분할 수 없는 경우에는 해당 loss의 가중치를 줄여야 한다.

---

### 13.5 Source count loss

모델의 예상 음원 수는 다음과 같이 계산한다.

$$
\hat K
=
\sum_{k=1}^{K_{\max}}
\hat p_k
$$

음원 수 손실은 다음처럼 정의할 수 있다.

$$
\mathcal L_{\mathrm{count}}
=
|\hat K-K|
$$

또는 별도의 count classification head를 두고 $0,\ldots,K_{\max}$ 범주에 대한 cross-entropy를 사용할 수 있다.

---

## 14. 3차원 배열 관측 가능성

모든 배열이 동일한 수준의 3차원 정보를 제공하는 것은 아니다.

마이크 baseline으로 geometry matrix를 구성한다.

$$
\mathbf G
=
\sum_{i<j}
w_{ij}
\mathbf d_{ij}
\mathbf d_{ij}^\top
$$

3차원 관측 가능성 지표를 다음과 같이 정의한다.

$$
o_{\mathrm{3D}}
=
\frac{
\lambda_{\min}(\mathbf G)
}{
\operatorname{tr}(\mathbf G)+\epsilon
}
$$

모든 마이크가 하나의 평면에 위치하면 최소 고유값이 0에 가까워지므로 $o_{\mathrm{3D}}$가 작아진다.

이 경우 평면 위쪽과 아래쪽 방향이 유사한 TDOA를 생성할 수 있어 고도각 부호가 모호해질 수 있다.

따라서 $o_{\mathrm{3D}}$를 geometry token에 추가한다.

$$
\mathbf g_{\mathrm{global}}
=
[
D_{\mathrm{array}},
o_{\mathrm{3D}},
\operatorname{cond}(\mathbf G),
M
]
$$

이를 uncertainty head에 입력하여 관측 가능성이 낮은 배열에서 $\kappa$가 과도하게 커지는 것을 억제한다.

---

## 15. 학습 전략

### Stage 1. 단일 음원 기초 학습

- 단일 음원
- 높은 SNR
- 낮은 잔향
- 비공면 마이크 배열
- angular loss 중심 학습

목적은 마이크 위치, 위상차 및 방향 사이의 기본 물리 관계를 학습하는 것이다.

---

### Stage 2. 배열 일반화 학습

- 마이크 개수 무작위화
- 마이크 순서 무작위 permutation
- linear, circular, planar, spherical, random 배열
- 배열 aperture 무작위화
- 마이크 좌표 오차 추가
- channel dropout 적용
- rotation consistency 적용

마이크 위치 오차는 다음과 같이 생성한다.

$$
\mathbf r_m'
=
\mathbf r_m+\boldsymbol\epsilon_m
$$

$$
\boldsymbol\epsilon_m
\sim
\mathcal N
(\mathbf 0,\sigma_r^2\mathbf I)
$$

---

### Stage 3. 다중 음원 학습

- 동시 음원 수 1–4개
- 음원 간 최소 각도 점진적 감소
- 부분 중첩 및 완전 중첩
- 동일 화자 및 서로 다른 음향 이벤트
- Hungarian matching 적용
- source responsibility head 활성화

초기에는 음원 간 각도를 크게 설정하고 학습 후반에 가까운 음원 조건을 증가시킨다.

---

### Stage 4. Physics-guided 학습

학습 초기부터 physics loss를 크게 적용하면 잔향과 noise 때문에 최적화가 불안정해질 수 있다.

따라서 다음과 같은 weight scheduling을 사용한다.

$$
\lambda_{\mathrm{phys}}(e)
=
\lambda_{\max}
\min
\left(
1,
\frac{e}{E_{\mathrm{warmup}}}
\right)
$$

여기서 $e$는 epoch이고, $E_{\mathrm{warmup}}$은 physics loss warm-up epoch이다.

---

### Stage 5. Sim-to-Real 적응

- 실제 RIR 사용
- microphone gain mismatch
- microphone phase mismatch
- diffuse noise
- 센서 고장
- 실제 녹음 데이터 fine-tuning
- label이 부족한 경우 physics loss와 consistency loss 활용

---

## 16. 추론 과정

추론 단계는 다음과 같다.

```text
Input:
    Multichannel audio X
    Microphone coordinates R

1. Perform STFT.
2. Construct all microphone-pair spatial features.
3. Encode pairwise acoustic and geometric information.
4. Aggregate variable-size microphone representations.
5. Decode K_max source queries.
6. Estimate activity probability, 3D direction and uncertainty.
7. Remove slots whose activity probability is below threshold.
8. Optionally merge duplicated directions using angular NMS.
9. Convert Cartesian unit vectors to azimuth and elevation.
```

중복 방향을 제거하기 위해 angular non-maximum suppression을 사용할 수 있다.

두 예측 방향 사이의 각도가 임계값 이하이면 낮은 activity score를 가진 예측을 제거한다.

$$
d_{\mathrm{ang}}
(\hat{\mathbf u}_i,\hat{\mathbf u}_j)
<
\delta_{\mathrm{merge}}
$$

---

## 17. 학습 알고리즘

```python
for audio, mic_positions, gt_directions in dataloader:

    # 1. STFT
    X = stft(audio)

    # 2. Pairwise acoustic features
    pair_features = extract_pair_features(X)

    # 3. Relative microphone geometry
    geometry_features = build_geometry_features(mic_positions)

    # 4. Geometry-aware pair encoding
    pair_embeddings = pair_encoder(
        pair_features,
        geometry_features
    )

    # 5. Variable-size microphone set encoding
    scene_embedding = set_graph_encoder(
        pair_embeddings,
        mic_positions
    )

    # 6. Multi-source query decoding
    activity, doa_vector, kappa, tf_responsibility = query_decoder(
        scene_embedding
    )

    # 7. Hungarian matching
    matching = hungarian_match(
        activity,
        doa_vector,
        gt_directions
    )

    # 8. Supervised set loss
    loss_set = compute_set_loss(
        activity,
        doa_vector,
        kappa,
        gt_directions,
        matching
    )

    # 9. Physics-guided covariance reconstruction
    steering_vectors = build_steering_vectors(
        doa_vector,
        mic_positions
    )

    predicted_covariance = reconstruct_covariance(
        steering_vectors,
        tf_responsibility,
        activity
    )

    observed_covariance = estimate_local_covariance(X)

    loss_phys = covariance_loss(
        predicted_covariance,
        observed_covariance
    )

    # 10. Rotation and microphone subset consistency
    loss_rot = rotation_consistency(...)
    loss_sub = microphone_subset_consistency(...)

    # 11. Total loss
    loss = (
        lambda_set * loss_set
        + lambda_phys * loss_phys
        + lambda_rot * loss_rot
        + lambda_sub * loss_sub
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 18. 예상 연구 기여점

본 연구의 기여점은 다음과 같이 정리할 수 있다.

### Contribution 1. 배열 비종속 다중 음원 3D DOA 추정

고정된 마이크 개수나 특정 배열 형태를 가정하지 않고, 마이크 좌표와 pairwise spatial feature를 이용하여 다양한 배열에 하나의 모델을 적용한다.

### Contribution 2. Grid-free set prediction

방향 grid 분류 대신 다중 음원의 DOA를 연속적인 3차원 단위벡터 집합으로 직접 추정한다. 이를 통해 방향 해상도가 사전 정의된 grid 간격에 제한되지 않는다.

### Contribution 3. 다중 음원 물리 디코더

예측된 복수 음원 방향과 TF responsibility로부터 공간 공분산을 재구성하고 실제 관측 공분산과 비교함으로써, 방향 예측이 물리적으로 관측된 위상 관계를 설명하도록 학습한다.

### Contribution 4. 배열 변환 일관성

좌표계 회전, 마이크 permutation, channel dropout 및 microphone subset consistency를 통해 학습하지 않은 배열에 대한 일반화를 강화한다.

### Contribution 5. Geometry-aware uncertainty

마이크 배열의 3차원 관측 가능성을 분석하고, 평면 배열이나 작은 aperture에서 발생하는 물리적 모호성을 uncertainty prediction에 반영한다.

---

## 19. 핵심 Ablation Study

| 실험 | 제거하거나 변경할 요소 | 확인 목적 |
|---|---|---|
| A1 | Geometry feature 제거 | 마이크 좌표 입력의 효과 |
| A2 | Physics loss 제거 | 물리 일관성 학습 효과 |
| A3 | Set decoder를 grid classifier로 교체 | 연속 방향 집합 출력 효과 |
| A4 | Rotation consistency 제거 | 좌표계 변화 일반화 효과 |
| A5 | Subset consistency 제거 | 채널 손실 강건성 |
| A6 | vMF uncertainty 제거 | 신뢰도 calibration 효과 |
| A7 | TF responsibility 제거 | 근접 다중 음원 분리 효과 |
| A8 | 실제 좌표 대신 배열 ID 사용 | geometry generalization 검증 |
| A9 | 공분산 decoder 대신 pairwise phase loss 사용 | 물리 decoder 설계 비교 |
| A10 | 평면 배열과 비공면 배열 비교 | 3D 관측 가능성 검증 |

---

## 20. 연구 범위와 한계

본 방법은 기본적으로 원거리 plane-wave 조건을 가정한다.

따라서 음원이 배열에 매우 가까운 경우에는 거리 정보와 spherical-wave steering vector를 추가해야 한다.

$$
a_{m,f}(\mathbf s)
=
\frac{1}{\|\mathbf s-\mathbf r_m\|}
\exp
\left(
-j\frac{\omega_f}{c}
\|\mathbf s-\mathbf r_m\|
\right)
$$

또한 강한 잔향 환경에서는 단순한 direct-path steering vector만으로 실제 공간 공분산을 완전히 설명할 수 없다.

이를 완화하기 위해 다음 요소가 필요하다.

- diffuse covariance component
- learned residual covariance
- coherence-based TF weighting
- 실제 RIR 기반 fine-tuning
- physics loss warm-up

마지막으로 공면 배열에서는 순수한 TDOA 및 위상 정보만으로 상하 대칭 방향을 완전히 구분하지 못할 수 있다. 따라서 모든 배열에서 동일한 수준의 3차원 DOA 성능을 보장한다는 주장은 피해야 한다.

---

## 21. 최종 방법론 요약

PhysSet-DOA는 다채널 음향 신호와 가변적인 마이크 좌표를 입력받아, 마이크 쌍별 위상 특징과 상대 기하학 정보를 학습한다.

마이크 배열은 순서 없는 집합 또는 그래프로 처리되며, 복수의 source query가 활성 확률, 연속적인 3차원 방향벡터 및 방향 불확실성을 출력한다.

예측된 방향은 steering vector로 변환되고, TF별 source responsibility와 결합되어 다중 음원 공간 공분산을 재구성한다. 재구성된 공분산이 실제 관측 공분산과 일치하도록 학습함으로써, 모델은 정답 각도뿐 아니라 실제 마이크 간 위상 및 공간 상관관계를 설명하는 방향을 추정하도록 제약된다.

전체 구조는 다음과 같이 요약할 수 있다.

$$
\boxed{
\text{Pairwise Acoustic Features}
+
\text{Geometry-Aware}
}