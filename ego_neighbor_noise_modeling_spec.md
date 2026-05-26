# Ego-centric Neighbor Observation Noise Modeling 구현 명세서

## 0. 목적

이 문서는 ego vehicle이 neighbor agent를 관측할 때 발생하는 observation noise를 여러 방식으로 모델링하고, 각 모델의 성능을 비교하기 위한 구현 가이드이다.

구현 대상 noise model은 크게 다음과 같다.

1. Distance-dependent Gaussian noise
2. Range/Bearing noise model
3. AI 모델 활용
   - v1: current ego-centric feature 기반 MLP
   - v2: richer ego-centric feature 기반 heteroscedastic MLP
   - v3: history sequence 기반 GRU/LSTM/Transformer
4. 추가 실험
   - observation confidence / covariance prediction
   - residual correction model

최종 목표는 동일한 dataset과 동일한 evaluation protocol 하에서 각 noise modeling 방식이 얼마나 잘 noise를 설명하거나, 관측값을 보정하거나, downstream tracking/state estimation 성능을 개선하는지 비교하는 것이다.

---

## 1. 사용 가능한 원본 정보

각 timestep `t`에서 사용 가능한 정보는 다음과 같다고 가정한다.

### 1.1 Neighbor 관측 상태

```python
x_obs
y_obs
vx_obs
vy_obs
dx_obs
dy_obs
```

의미:

- `x_obs`, `y_obs`: 관측된 neighbor global position
- `vx_obs`, `vy_obs`: 관측된 neighbor global velocity
- `dx_obs`, `dy_obs`: ego 기준 neighbor relative position  
  일반적으로 다음과 같이 정의된다.

```python
dx_obs = x_obs - ego_x
dy_obs = y_obs - ego_y
```

단, 데이터 생성 파이프라인에서 이미 정의된 값이 있다면 그 정의를 따른다.

### 1.2 Ego 상태

```python
ego_x
ego_y
ego_vx
ego_vy
```

### 1.3 시간 정보

```python
step
time_sec
```

### 1.4 History

과거 관측 history를 사용할 수 있다.

예:

```python
history[t - H + 1 : t + 1]
```

각 history timestep에는 현재 timestep과 동일하거나 일부 feature가 포함될 수 있다.

---

## 2. Ground-truth 및 label 정의

noise model을 학습 또는 평가하려면 noise-free true state가 필요하다.

가능하다면 다음 값들이 dataset에 있어야 한다.

```python
x_true
y_true
vx_true
vy_true
dx_true
dy_true
```

여기서:

```python
dx_true = x_true - ego_x
dy_true = y_true - ego_y
```

로 계산할 수 있다.

---

## 3. 기본 noise label 정의

### 3.1 Global coordinate noise

```python
noise_x = x_obs - x_true
noise_y = y_obs - y_true
noise_vx = vx_obs - vx_true
noise_vy = vy_obs - vy_true
```

label vector:

```python
noise_global = [noise_x, noise_y, noise_vx, noise_vy]
```

### 3.2 Ego-centric relative coordinate noise

ego 입장에서 neighbor 관측 noise를 모델링하는 것이 목적이므로, 이 label을 기본 추천값으로 사용한다.

```python
noise_dx = dx_obs - dx_true
noise_dy = dy_obs - dy_true
noise_vx = vx_obs - vx_true
noise_vy = vy_obs - vy_true
```

label vector:

```python
noise_rel = [noise_dx, noise_dy, noise_vx, noise_vy]
```

### 3.3 Range/Bearing coordinate noise

range/bearing 기반 센서 모델을 만들기 위해 다음 변환을 사용한다.

```python
r_obs = sqrt(dx_obs**2 + dy_obs**2)
theta_obs = atan2(dy_obs, dx_obs)

r_true = sqrt(dx_true**2 + dy_true**2)
theta_true = atan2(dy_true, dx_true)
```

noise:

```python
noise_r = r_obs - r_true
noise_theta = wrap_angle(theta_obs - theta_true)
```

label vector:

```python
noise_polar = [noise_r, noise_theta]
```

velocity까지 polar coordinate로 확장하려면 radial/tangential velocity를 정의한다.

```python
distance = sqrt(dx**2 + dy**2) + eps

unit_r_x = dx / distance
unit_r_y = dy / distance

unit_t_x = -unit_r_y
unit_t_y = unit_r_x

dvx = vx_neighbor - ego_vx
dvy = vy_neighbor - ego_vy

v_r = dvx * unit_r_x + dvy * unit_r_y
v_t = dvx * unit_t_x + dvy * unit_t_y
```

그러면 다음 label도 가능하다.

```python
noise_vr = vr_obs - vr_true
noise_vt = vt_obs - vt_true
```

label vector:

```python
noise_polar_full = [noise_r, noise_theta, noise_vr, noise_vt]
```

---

## 4. 공통 feature engineering

여러 모델에서 공통적으로 사용할 ego-centric feature를 정의한다.

### 4.1 기본 relative feature

```python
dvx_obs = vx_obs - ego_vx
dvy_obs = vy_obs - ego_vy

distance_obs = sqrt(dx_obs**2 + dy_obs**2)
bearing_obs = atan2(dy_obs, dx_obs)

ego_speed = sqrt(ego_vx**2 + ego_vy**2)
neighbor_speed_obs = sqrt(vx_obs**2 + vy_obs**2)
relative_speed_obs = sqrt(dvx_obs**2 + dvy_obs**2)
```

### 4.2 Closing rate

neighbor가 ego에 접근하는지 멀어지는지를 나타내는 feature이다.

```python
eps = 1e-6
closing_rate = (dx_obs * dvx_obs + dy_obs * dvy_obs) / (distance_obs + eps)
```

해석:

- `closing_rate < 0`: neighbor가 ego에 가까워지는 방향
- `closing_rate > 0`: neighbor가 ego에서 멀어지는 방향

단, 정의에 따라 부호 해석이 달라질 수 있으므로 실험 코드에서 일관되게 사용한다.

### 4.3 Heading-aligned feature, 선택 사항

ego heading이 별도로 없고 velocity 방향을 ego heading proxy로 사용한다면 다음을 만들 수 있다.

```python
ego_heading = atan2(ego_vy, ego_vx)
relative_bearing = wrap_angle(bearing_obs - ego_heading)
```

ego가 정지 상태에 가까울 때는 heading이 불안정할 수 있으므로 아래 조건을 둔다.

```python
if ego_speed < min_speed:
    relative_bearing = bearing_obs
```

### 4.4 Time feature

단순히 `step`, `time_sec`를 넣을 수도 있지만, scale이 크면 학습이 불안정할 수 있다.

권장:

```python
time_sec_normalized = time_sec / max_episode_time
step_normalized = step / max_episode_step
```

또는 주기성이 있는 환경이면 sinusoidal encoding을 사용할 수 있다.

```python
sin_time = sin(2 * pi * time_sec / period)
cos_time = cos(2 * pi * time_sec / period)
```

---

## 5. Experiment 1: Distance-dependent Gaussian noise

### 5.1 모델 개념

neighbor가 멀리 있을수록 관측 noise가 커진다고 가정한다.

기본 형태:

```python
epsilon ~ Normal(0, sigma(distance)^2)
```

position에 대해:

```python
sigma_dx(distance) = sigma_dx_0 + alpha_dx * distance
sigma_dy(distance) = sigma_dy_0 + alpha_dy * distance
```

velocity에 대해:

```python
sigma_vx(distance) = sigma_vx_0 + alpha_vx * distance
sigma_vy(distance) = sigma_vy_0 + alpha_vy * distance
```

### 5.2 구현 방식

두 가지 구현을 모두 지원하면 좋다.

#### Option A: rule-based fixed parameter

config에 parameter를 수동으로 지정한다.

```yaml
distance_gaussian:
  sigma_dx_0: 0.05
  sigma_dy_0: 0.05
  sigma_vx_0: 0.02
  sigma_vy_0: 0.02
  alpha_dx: 0.01
  alpha_dy: 0.01
  alpha_vx: 0.005
  alpha_vy: 0.005
  min_sigma: 1.0e-4
  max_sigma: 10.0
```

#### Option B: fit parameter from data

train set에서 실제 noise label을 사용해 다음 regression을 fit한다.

```python
abs_noise_i ≈ sigma_i_0 + alpha_i * distance
```

좀 더 정확히는 variance를 fit한다.

```python
noise_i**2 ≈ sigma_i(distance)**2
```

추천 parameterization:

```python
sigma_i(distance) = softplus(a_i + b_i * distance) + min_sigma
```

이때 `a_i`, `b_i`를 최적화한다.

### 5.3 입력

```python
distance_obs
```

또는 true distance를 쓸 수 있다면 calibration 단계에서는 `distance_true`도 사용 가능하다.  
하지만 실제 inference 상황을 고려하면 `distance_obs` 사용을 기본으로 한다.

### 5.4 출력

이 모델은 noise mean을 0으로 가정한다.

```python
mu = [0, 0, 0, 0]
sigma = [
    sigma_dx(distance),
    sigma_dy(distance),
    sigma_vx(distance),
    sigma_vy(distance),
]
```

### 5.5 Sampling

noise를 simulation에 주입할 때:

```python
noise_dx ~ Normal(0, sigma_dx(distance)**2)
noise_dy ~ Normal(0, sigma_dy(distance)**2)
noise_vx ~ Normal(0, sigma_vx(distance)**2)
noise_vy ~ Normal(0, sigma_vy(distance)**2)
```

### 5.6 Evaluation

이 모델은 다음 metric으로 평가한다.

- Negative Log Likelihood, NLL
- predicted sigma calibration
- noise magnitude vs distance plot
- corrected state RMSE는 기본적으로 적용하지 않음. mean이 0이므로 보정 효과는 없음.

### 5.7 추천 class 이름

```python
class DistanceDependentGaussianNoiseModel:
    def fit(self, train_dataset): ...
    def predict_distribution(self, batch): ...
    def sample(self, batch): ...
    def nll(self, batch): ...
```

---

## 6. Experiment 2: Range/Bearing noise model

### 6.1 모델 개념

ego-centric 관측은 Cartesian `dx, dy`보다 polar coordinate인 `range, bearing`으로 해석하는 것이 자연스럽다.

```python
r = sqrt(dx**2 + dy**2)
theta = atan2(dy, dx)
```

noise는 다음처럼 둔다.

```python
r_obs = r_true + noise_r
theta_obs = theta_true + noise_theta
```

```python
noise_r ~ Normal(0, sigma_r(r)**2)
noise_theta ~ Normal(0, sigma_theta(r)**2)
```

### 6.2 Distance-dependent polar sigma

```python
sigma_r(r) = sigma_r_0 + alpha_r * r
sigma_theta(r) = sigma_theta_0 + alpha_theta * r
```

혹은 안정적인 parameterization:

```python
sigma_r = softplus(a_r + b_r * r) + min_sigma
sigma_theta = softplus(a_theta + b_theta * r) + min_sigma
```

### 6.3 입력

```python
dx_obs
dy_obs
distance_obs
bearing_obs
```

### 6.4 출력

```python
mu_polar = [0, 0]
sigma_polar = [sigma_r, sigma_theta]
```

velocity까지 포함하면:

```python
mu_polar_full = [0, 0, 0, 0]
sigma_polar_full = [sigma_r, sigma_theta, sigma_vr, sigma_vt]
```

### 6.5 Cartesian 변환

polar noise를 sample한 뒤 noisy relative position을 생성한다.

```python
r_noisy = r_true + noise_r
theta_noisy = theta_true + noise_theta

dx_noisy = r_noisy * cos(theta_noisy)
dy_noisy = r_noisy * sin(theta_noisy)
```

관측값을 평가할 때는 observed polar error를 직접 계산한다.

```python
noise_r = r_obs - r_true
noise_theta = wrap_angle(theta_obs - theta_true)
```

### 6.6 주의: angle wrapping

bearing noise는 반드시 angle wrapping이 필요하다.

```python
def wrap_angle(angle):
    return (angle + pi) % (2 * pi) - pi
```

MSE나 NLL 계산 전에 항상 `wrap_angle`을 적용한다.

### 6.7 Evaluation

- polar noise NLL
- Cartesian으로 변환한 후 `dx, dy` RMSE
- bearing error distribution
- distance bucket별 `r`, `theta` error
- outlier rate

### 6.8 추천 class 이름

```python
class RangeBearingNoiseModel:
    def fit(self, train_dataset): ...
    def predict_distribution(self, batch): ...
    def sample_polar_noise(self, batch): ...
    def sample_cartesian_noise(self, batch): ...
    def nll(self, batch): ...
```

---

## 7. Experiment 3: AI model v1 - ego-centric MLP baseline

### 7.1 목적

현재 timestep의 간단한 ego-centric feature만으로 noise mean을 직접 예측한다.

### 7.2 입력 feature

최소 feature set:

```python
features_v1 = [
    dx_obs,
    dy_obs,
    dvx_obs,
    dvy_obs,
    distance_obs,
    bearing_obs,
]
```

### 7.3 Label

기본 label:

```python
target = [
    noise_dx,
    noise_dy,
    noise_vx,
    noise_vy,
]
```

### 7.4 모델 구조

```python
input_dim = 6
hidden_dim = 64 or 128
output_dim = 4
```

예시:

```python
MLP(
    Linear(input_dim, hidden_dim),
    ReLU,
    Linear(hidden_dim, hidden_dim),
    ReLU,
    Linear(hidden_dim, output_dim)
)
```

### 7.5 출력

```python
noise_hat = [noise_dx_hat, noise_dy_hat, noise_vx_hat, noise_vy_hat]
```

### 7.6 Loss

```python
loss = MSE(noise_hat, target_noise)
```

또는 dimension별 scale 차이가 크다면 weighted MSE를 사용한다.

```python
loss = mean(((noise_hat - target_noise) / target_std) ** 2)
```

### 7.7 Inference

관측 보정값:

```python
dx_corrected = dx_obs - noise_dx_hat
dy_corrected = dy_obs - noise_dy_hat
vx_corrected = vx_obs - noise_vx_hat
vy_corrected = vy_obs - noise_vy_hat
```

global position으로 되돌리려면:

```python
x_corrected = ego_x + dx_corrected
y_corrected = ego_y + dy_corrected
```

### 7.8 추천 class 이름

```python
class EgoCentricMLPNoiseRegressorV1(nn.Module):
    def forward(self, features): ...
```

---

## 8. Experiment 4: AI model v2 - richer ego-centric heteroscedastic MLP

### 8.1 목적

v1은 noise의 평균만 예측한다.  
v2에서는 noise의 평균뿐 아니라 상황별 uncertainty, 즉 variance도 예측한다.

즉 다음 분포를 예측한다.

```python
noise | features ~ Normal(mu(features), sigma(features)^2)
```

### 8.2 입력 feature

추천 feature set:

```python
features_v2 = [
    dx_obs,
    dy_obs,
    dvx_obs,
    dvy_obs,

    distance_obs,
    bearing_obs,
    relative_speed_obs,
    closing_rate,

    ego_vx,
    ego_vy,
    ego_speed,

    vx_obs,
    vy_obs,
    neighbor_speed_obs,

    step_normalized,
    time_sec_normalized,
]
```

선택적으로 추가:

```python
relative_bearing
sin_bearing
cos_bearing
sin_relative_bearing
cos_relative_bearing
```

angle은 raw `bearing` 하나만 넣기보다 다음처럼 sin/cos로 넣는 것이 안정적일 수 있다.

```python
sin_bearing = sin(bearing_obs)
cos_bearing = cos(bearing_obs)
```

이 경우 `bearing_obs` 대신 `sin_bearing`, `cos_bearing`를 넣는다.

### 8.3 Label

```python
target = [
    noise_dx,
    noise_dy,
    noise_vx,
    noise_vy,
]
```

### 8.4 모델 구조

모델 출력은 `mu`와 `log_sigma` 또는 `log_var`이다.

```python
output_dim = 8
```

출력 split:

```python
mu = output[:, :4]
log_sigma = output[:, 4:]
sigma = softplus(log_sigma) + min_sigma
```

혼동을 줄이기 위해 실제 구현에서는 network가 `raw_scale`을 출력하고, 이를 softplus로 변환하는 것을 권장한다.

```python
raw_scale = output[:, 4:]
sigma = F.softplus(raw_scale) + min_sigma
```

### 8.5 Gaussian NLL loss

dimension independent diagonal Gaussian을 가정한다.

```python
def gaussian_nll(target, mu, sigma):
    var = sigma ** 2
    return 0.5 * (((target - mu) ** 2) / var + torch.log(var)).mean()
```

constant term `0.5 * log(2*pi)`는 모델 비교에서는 생략 가능하지만, 정확한 NLL 값을 원하면 포함한다.

```python
def gaussian_nll_full(target, mu, sigma):
    var = sigma ** 2
    return 0.5 * (((target - mu) ** 2) / var + torch.log(var) + math.log(2 * math.pi)).mean()
```

### 8.6 Optional corrected state loss

noise distribution 학습과 함께 실제 corrected state도 잘 맞추도록 보조 loss를 추가할 수 있다.

```python
corrected = obs_state - mu
loss_state = MSE(corrected, true_state)
loss = loss_nll + lambda_state * loss_state
```

여기서:

```python
obs_state = [dx_obs, dy_obs, vx_obs, vy_obs]
true_state = [dx_true, dy_true, vx_true, vy_true]
```

### 8.7 장점

- distance가 먼 경우 더 큰 sigma를 예측 가능
- 측방/후방 등 관측 geometry에 따른 anisotropic uncertainty 가능
- noise mean bias가 있는 경우 보정 가능
- downstream filter에서 uncertainty로 사용 가능

### 8.8 추천 class 이름

```python
class EgoCentricHeteroscedasticMLPNoiseModelV2(nn.Module):
    def forward(self, features):
        # returns mu, sigma
        ...
```

---

## 9. Experiment 5: AI model v3 - history sequence model

### 9.1 목적

관측 noise가 시간적으로 correlated되어 있거나, sensor lag, jitter, tracking instability가 있는 경우 history를 사용하는 sequence model이 유리하다.

### 9.2 입력 sequence

sequence length를 `H`라고 할 때:

```python
sequence_features.shape = [batch_size, H, feature_dim]
```

각 timestep feature는 v2 feature의 일부 또는 전체를 사용한다.

추천 기본 timestep feature:

```python
features_t = [
    dx_obs,
    dy_obs,
    dvx_obs,
    dvy_obs,

    distance_obs,
    sin_bearing_obs,
    cos_bearing_obs,

    relative_speed_obs,
    closing_rate,

    ego_speed,
    neighbor_speed_obs,

    delta_time,
]
```

`bearing`은 sequence model에서도 sin/cos encoding을 권장한다.

### 9.3 History length 후보

시뮬레이션 step size에 따라 다르지만 다음을 실험한다.

```yaml
history_lengths: [1, 5, 10, 20]
```

예:

- dt = 0.1 sec
- H = 10이면 1초 history

### 9.4 모델 후보

#### Option A: GRU

가장 추천하는 baseline.

```python
GRU(input_dim, hidden_dim, batch_first=True)
last_hidden -> MLP -> mu, sigma
```

#### Option B: LSTM

GRU와 비슷하지만 parameter가 더 많다.

```python
LSTM(input_dim, hidden_dim, batch_first=True)
last_hidden -> MLP -> mu, sigma
```

#### Option C: Transformer Encoder

데이터가 충분히 많고 history가 길다면 사용한다.

```python
input projection
positional encoding
TransformerEncoder
pooling or last token
MLP -> mu, sigma
```

처음 구현은 GRU를 우선 추천한다.

### 9.5 Output

v2와 동일하게 heteroscedastic output을 추천한다.

```python
mu = [mu_dx, mu_dy, mu_vx, mu_vy]
sigma = [sigma_dx, sigma_dy, sigma_vx, sigma_vy]
```

### 9.6 Loss

```python
loss = GaussianNLL(target_noise, mu, sigma)
```

선택적으로 corrected state loss를 추가한다.

```python
loss = loss_nll + lambda_state * loss_corrected_state
```

### 9.7 Missing history 처리

episode 초반에는 충분한 history가 없을 수 있다.

가능한 처리:

1. padding with first observation
2. zero padding + mask
3. 해당 timestep은 training에서 제외

추천:

- GRU/LSTM: first observation repeat padding
- Transformer: padding mask 사용

### 9.8 추천 class 이름

```python
class EgoCentricHistoryNoiseModelV3(nn.Module):
    def __init__(self, encoder_type="gru", ...):
        ...
    def forward(self, sequence_features):
        # returns mu, sigma
        ...
```

---

## 10. 추가 실험 A: Observation confidence / covariance prediction

### 10.1 목적

noise 자체를 직접 예측하지 않고, 현재 관측값을 얼마나 신뢰할 수 있는지 예측한다.

이 방식은 Kalman filter, particle filter, tracking module과 결합하기 좋다.

### 10.2 출력

가장 단순한 형태는 diagonal covariance이다.

```python
R_diag = [
    sigma_dx**2,
    sigma_dy**2,
    sigma_vx**2,
    sigma_vy**2,
]
```

모델은 다음을 출력한다.

```python
raw_scale = model(features)
sigma = softplus(raw_scale) + min_sigma
R = diag(sigma**2)
```

### 10.3 Full covariance

더 표현력이 필요한 경우 full covariance를 예측한다.

4D noise에 대해 covariance matrix는 `4 x 4`이다.

양의 정부호를 보장하기 위해 Cholesky factor `L`을 예측한다.

```python
R = L @ L.T + min_jitter * I
```

구현 방식:

```python
# model outputs 10 values for lower triangular 4x4 matrix
# diagonal entries are softplus-transformed
```

4D lower triangular entry 개수:

```python
4 * (4 + 1) / 2 = 10
```

### 10.4 Loss

실제 noise residual을 `e`라고 하면:

```python
e = target_noise
```

Gaussian NLL:

```python
NLL = 0.5 * (e.T @ inv(R) @ e + logdet(R) + D * log(2*pi))
```

diagonal covariance인 경우:

```python
NLL = 0.5 * sum(e_i**2 / sigma_i**2 + log(sigma_i**2) + log(2*pi))
```

### 10.5 Confidence score로 변환

모델이 예측한 uncertainty를 confidence로 바꿀 수 있다.

예:

```python
confidence = 1 / (1 + mean_sigma)
```

또는

```python
confidence = exp(-mean_log_sigma)
```

주의: confidence는 downstream에서 쓰기 쉽게 만든 scalar이며, 학습 자체는 covariance/NLL로 하는 것을 추천한다.

### 10.6 Evaluation

- NLL
- calibration curve
- predicted sigma vs actual absolute error
- distance bucket별 sigma calibration
- high-confidence sample에서 error가 정말 작은지
- low-confidence sample에서 error가 큰지

### 10.7 추천 class 이름

```python
class ObservationCovariancePredictor(nn.Module):
    def forward(self, features):
        # returns R or sigma
        ...
```

---

## 11. 추가 실험 B: Residual correction model

### 11.1 목적

관측 noise를 예측해 관측값에서 빼서 true state에 더 가까운 corrected state를 만든다.

```python
corrected_state = observed_state - predicted_noise
```

### 11.2 입력

v1, v2, v3 feature를 그대로 사용할 수 있다.

추천:

```python
features_v2
```

또는 history가 있으면:

```python
sequence_features_v3
```

### 11.3 출력

```python
predicted_noise = [
    noise_dx_hat,
    noise_dy_hat,
    noise_vx_hat,
    noise_vy_hat,
]
```

### 11.4 Corrected state

```python
obs_state = [dx_obs, dy_obs, vx_obs, vy_obs]
corrected_state = obs_state - predicted_noise
```

true state:

```python
true_state = [dx_true, dy_true, vx_true, vy_true]
```

### 11.5 Loss 구성

#### Option A: noise prediction loss only

```python
loss = MSE(predicted_noise, target_noise)
```

#### Option B: corrected state loss only

```python
loss = MSE(corrected_state, true_state)
```

이 경우 target noise를 직접 쓰지 않아도 된다.

#### Option C: combined loss, 추천

```python
loss_noise = MSE(predicted_noise, target_noise)
loss_state = MSE(corrected_state, true_state)

loss = loss_noise + lambda_state * loss_state
```

일반적으로 다음부터 시작한다.

```yaml
lambda_state: 1.0
```

단, noise와 state scale이 같으면 두 loss가 거의 동치일 수 있다.  
velocity와 position scale 차이가 있으면 dimension별 normalization을 사용한다.

### 11.6 Heteroscedastic residual correction

residual correction model도 uncertainty를 같이 출력할 수 있다.

```python
mu_noise, sigma_noise = model(features)
corrected_state = obs_state - mu_noise
loss = GaussianNLL(target_noise, mu_noise, sigma_noise) + lambda_state * MSE(corrected_state, true_state)
```

이 구조가 최종 추천 구조 중 하나이다.

### 11.7 Evaluation

- observed state RMSE vs corrected state RMSE
- noise prediction MSE
- corrected trajectory smoothness
- downstream planning/tracking 성능
- outlier sample에서 correction이 악화되는 비율

특히 중요한 지표:

```python
improvement_ratio = RMSE_corrected / RMSE_observed
```

`improvement_ratio < 1`이면 보정이 효과가 있다.

---

## 12. Dataset 구현 명세

### 12.1 Raw sample format

하나의 row/sample은 다음 정보를 포함한다고 가정한다.

```python
sample = {
    "x_obs": ...,
    "y_obs": ...,
    "vx_obs": ...,
    "vy_obs": ...,
    "dx_obs": ...,
    "dy_obs": ...,

    "ego_x": ...,
    "ego_y": ...,
    "ego_vx": ...,
    "ego_vy": ...,

    "step": ...,
    "time_sec": ...,

    "x_true": ...,
    "y_true": ...,
    "vx_true": ...,
    "vy_true": ...,

    "episode_id": ...,
    "neighbor_id": ...,
}
```

`episode_id`, `neighbor_id`는 history sequence를 구성할 때 필요하다.

### 12.2 Derived fields

dataset class에서 다음 derived field를 생성한다.

```python
dx_true = x_true - ego_x
dy_true = y_true - ego_y

dvx_obs = vx_obs - ego_vx
dvy_obs = vy_obs - ego_vy

distance_obs = sqrt(dx_obs**2 + dy_obs**2)
bearing_obs = atan2(dy_obs, dx_obs)

ego_speed = sqrt(ego_vx**2 + ego_vy**2)
neighbor_speed_obs = sqrt(vx_obs**2 + vy_obs**2)
relative_speed_obs = sqrt(dvx_obs**2 + dvy_obs**2)

closing_rate = (dx_obs * dvx_obs + dy_obs * dvy_obs) / (distance_obs + eps)
```

### 12.3 Target 생성

```python
target_noise_rel = [
    dx_obs - dx_true,
    dy_obs - dy_true,
    vx_obs - vx_true,
    vy_obs - vy_true,
]
```

```python
obs_state_rel = [
    dx_obs,
    dy_obs,
    vx_obs,
    vy_obs,
]
```

```python
true_state_rel = [
    dx_true,
    dy_true,
    vx_true,
    vy_true,
]
```

### 12.4 Train/Val/Test split

중요: 같은 episode나 같은 neighbor trajectory가 train/test에 섞이지 않도록 한다.

추천 split 기준:

```python
split by episode_id
```

예:

```yaml
train: 70%
val: 15%
test: 15%
```

### 12.5 Normalization

AI 모델은 feature normalization이 중요하다.

train set 통계만 사용해 mean/std를 계산한다.

```python
feature_norm = (feature - train_mean) / train_std
target_norm = target / target_std
```

target을 normalize해서 학습했다면 inference에서 inverse transform이 필요하다.

```python
noise_hat = noise_hat_norm * target_std
sigma = sigma_norm * target_std
```

주의:

- validation/test 통계로 normalize하면 data leakage
- angle을 raw로 넣는 경우 discontinuity 문제 있음
- angle은 sin/cos encoding 권장

---

## 13. 공통 평가 지표

모든 모델을 같은 test set에서 평가한다.

### 13.1 Noise prediction metrics

```python
MSE_noise
RMSE_noise
MAE_noise
```

dimension별로도 계산한다.

```python
RMSE_dx
RMSE_dy
RMSE_vx
RMSE_vy
```

### 13.2 Distribution metrics

분포를 예측하는 모델에 대해:

```python
NLL
calibration_error
predicted_sigma_mean
```

distance bucket별로도 계산한다.

```python
distance_bins = [0, 5, 10, 20, 30, 50, inf]
```

각 bucket에서:

```python
actual_rmse
predicted_sigma_mean
nll
```

### 13.3 Corrected state metrics

residual correction 또는 mean prediction 모델에 대해:

```python
observed_rmSE = RMSE(obs_state, true_state)
corrected_RMSE = RMSE(corrected_state, true_state)
improvement_ratio = corrected_RMSE / observed_RMSE
```

dimension별:

```python
RMSE_dx_obs
RMSE_dx_corrected
RMSE_dy_obs
RMSE_dy_corrected
...
```

### 13.4 Outlier metrics

큰 noise sample에 대한 성능을 따로 본다.

예:

```python
outlier = norm(target_noise_position) > percentile_95
```

outlier subset에서:

```python
RMSE
MAE
NLL
correction_worsening_rate
```

### 13.5 Correction worsening rate

보정 후 오히려 나빠진 비율.

```python
error_obs = norm(obs_state - true_state)
error_corrected = norm(corrected_state - true_state)

worsening_rate = mean(error_corrected > error_obs)
```

이 값이 너무 높으면 correction model이 위험할 수 있다.

---

## 14. 추천 프로젝트 구조

```text
noise_modeling/
  configs/
    distance_gaussian.yaml
    range_bearing.yaml
    ai_v1_mlp.yaml
    ai_v2_hetero_mlp.yaml
    ai_v3_history_gru.yaml
    covariance_predictor.yaml
    residual_correction.yaml

  data/
    dataset.py
    feature_builder.py
    normalization.py
    sequence_builder.py

  models/
    distance_gaussian.py
    range_bearing.py
    mlp_v1.py
    hetero_mlp_v2.py
    history_model_v3.py
    covariance_predictor.py
    residual_correction.py

  losses/
    gaussian_nll.py
    covariance_nll.py
    correction_losses.py

  evaluation/
    metrics.py
    calibration.py
    plots.py
    evaluate.py

  train.py
  evaluate.py
  run_experiments.py
  utils/
    geometry.py
    angle.py
    seed.py
```

---

## 15. Utility functions

### 15.1 angle.py

```python
import torch
import math

def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

```python
import numpy as np

def wrap_angle_np(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

### 15.2 geometry.py

```python
def compute_distance(dx, dy, eps=1e-6):
    return torch.sqrt(dx ** 2 + dy ** 2 + eps)

def compute_bearing(dx, dy):
    return torch.atan2(dy, dx)

def compute_closing_rate(dx, dy, dvx, dvy, eps=1e-6):
    distance = compute_distance(dx, dy, eps)
    return (dx * dvx + dy * dvy) / distance
```

---

## 16. Config 예시

### 16.1 ai_v2_hetero_mlp.yaml

```yaml
experiment_name: ai_v2_hetero_mlp

data:
  label_type: relative
  use_history: false
  normalize_features: true
  normalize_targets: true

features:
  use:
    - dx_obs
    - dy_obs
    - dvx_obs
    - dvy_obs
    - distance_obs
    - sin_bearing_obs
    - cos_bearing_obs
    - relative_speed_obs
    - closing_rate
    - ego_vx
    - ego_vy
    - ego_speed
    - vx_obs
    - vy_obs
    - neighbor_speed_obs
    - step_normalized
    - time_sec_normalized

model:
  type: heteroscedastic_mlp
  hidden_dims: [128, 128, 64]
  activation: relu
  dropout: 0.0
  output_dim: 8
  min_sigma: 1.0e-4

training:
  batch_size: 256
  epochs: 100
  lr: 1.0e-3
  weight_decay: 1.0e-5
  early_stopping_patience: 10
  loss: gaussian_nll
  lambda_state: 0.0

evaluation:
  distance_bins: [0, 5, 10, 20, 30, 50]
```

### 16.2 ai_v3_history_gru.yaml

```yaml
experiment_name: ai_v3_history_gru

data:
  label_type: relative
  use_history: true
  history_length: 10
  history_padding: repeat_first
  normalize_features: true
  normalize_targets: true

features:
  use:
    - dx_obs
    - dy_obs
    - dvx_obs
    - dvy_obs
    - distance_obs
    - sin_bearing_obs
    - cos_bearing_obs
    - relative_speed_obs
    - closing_rate
    - ego_speed
    - neighbor_speed_obs
    - delta_time

model:
  type: history_gru
  input_dim: auto
  hidden_dim: 128
  num_layers: 1
  bidirectional: false
  output_type: heteroscedastic
  min_sigma: 1.0e-4

training:
  batch_size: 128
  epochs: 100
  lr: 1.0e-3
  weight_decay: 1.0e-5
  loss: gaussian_nll
  lambda_state: 0.1
```

---

## 17. 실험 비교 테이블

최종적으로 아래 표 형태의 결과를 만들면 된다.

| Model | Input | Output | Loss | Noise RMSE | NLL | Corrected RMSE | Improvement Ratio | Worsening Rate |
|---|---|---|---|---:|---:|---:|---:|---:|
| Distance Gaussian | distance | sigma | NLL | - | - | - | - | - |
| Range/Bearing | range, bearing | polar sigma | NLL | - | - | - | - | - |
| AI v1 MLP | current simple | noise mean | MSE | - | N/A | - | - | - |
| AI v2 Hetero MLP | current rich | mean + sigma | NLL | - | - | - | - | - |
| AI v3 GRU | history rich | mean + sigma | NLL | - | - | - | - | - |
| Covariance Predictor | current/history | covariance R | NLL | N/A | - | N/A | N/A | N/A |
| Residual Correction | current/history | corrected residual | MSE/NLL | - | optional | - | - | - |

---

## 18. 구현 우선순위

### Phase 1: 데이터 및 공통 평가 코드

1. Dataset loader 구현
2. Feature builder 구현
3. Label builder 구현
4. Train/val/test split 구현
5. Normalizer 구현
6. 공통 metrics 구현

### Phase 2: Baseline 구현

1. Distance-dependent Gaussian
2. Range/Bearing noise model
3. Noise statistics visualization

### Phase 3: AI 모델 구현

1. AI v1 MLP
2. AI v2 heteroscedastic MLP
3. AI v3 GRU history model

### Phase 4: 추가 실험

1. Observation covariance predictor
2. Residual correction model
3. Heteroscedastic residual correction model

### Phase 5: 결과 분석

1. 전체 test metric 비교
2. distance bucket별 비교
3. bearing sector별 비교
4. outlier subset 비교
5. qualitative trajectory plot

---

## 19. 중요한 구현 주의사항

### 19.1 Data leakage 방지

- normalization 통계는 train set에서만 계산한다.
- 같은 episode가 train/test에 섞이지 않게 한다.
- history sequence 구성 시 미래 timestep을 사용하지 않는다.

### 19.2 Angle discontinuity

- `theta`나 `bearing`을 label로 쓸 때는 wrap_angle을 반드시 적용한다.
- input angle은 raw angle보다 sin/cos encoding을 권장한다.

### 19.3 Sigma 안정성

heteroscedastic model에서 sigma가 0으로 가면 NLL이 폭발한다.

항상 다음을 적용한다.

```python
sigma = softplus(raw_sigma) + min_sigma
```

추천:

```python
min_sigma = 1e-4
```

필요하면 max clamp도 사용한다.

```python
sigma = torch.clamp(sigma, min=min_sigma, max=max_sigma)
```

### 19.4 Scale 차이

position noise와 velocity noise scale이 다르면 MSE가 특정 dimension에 치우칠 수 있다.

해결:

- target normalization
- weighted loss
- dimension별 metric 확인

### 19.5 Corrected state가 악화될 수 있음

residual correction은 평균적으로 좋아져도 일부 case에서는 더 나빠질 수 있다.

반드시 다음을 기록한다.

```python
worsening_rate
```

---

## 20. 최종 추천 실험 세트

최소한 아래 실험을 모두 실행한다.

```text
E0: No correction / observed state baseline
E1: Distance-dependent Gaussian
E2: Range/Bearing Gaussian
E3: AI v1 ego-centric MLP
E4: AI v2 heteroscedastic MLP
E5: AI v3 history GRU
E6: Observation covariance predictor
E7: Residual correction MLP
E8: Heteroscedastic residual correction GRU
```

가장 기대되는 모델:

```text
AI v3 history GRU + heteroscedastic output + corrected state auxiliary loss
```

실용성과 안정성을 고려한 추천 모델:

```text
AI v2 heteroscedastic MLP
```

가장 해석 가능한 baseline:

```text
Range/Bearing Gaussian noise model
```

---

## 21. Codex에게 요청할 구현 태스크 예시

아래 지시를 local Codex에게 그대로 전달해도 된다.

```text
Implement an ego-centric neighbor observation noise modeling experiment framework.

Please create the project structure described in this markdown.

Implement:
1. Dataset and feature builder for ego-centric features.
2. Label builders for relative Cartesian noise and range/bearing noise.
3. DistanceDependentGaussianNoiseModel.
4. RangeBearingNoiseModel.
5. EgoCentricMLPNoiseRegressorV1.
6. EgoCentricHeteroscedasticMLPNoiseModelV2.
7. EgoCentricHistoryNoiseModelV3 using GRU first.
8. ObservationCovariancePredictor with diagonal covariance first.
9. ResidualCorrectionModel.
10. Shared losses: MSE, Gaussian NLL, corrected state loss.
11. Shared metrics: noise RMSE, NLL, corrected RMSE, improvement ratio, worsening rate, distance bucket metrics.
12. YAML config driven train/evaluate scripts.

Use PyTorch.
Use train-set-only normalization.
Split data by episode_id.
Avoid future leakage in history construction.
Use sin/cos encoding for bearing.
Ensure all models can be evaluated with a common evaluate.py script.
```

---

## 22. Acceptance criteria

구현 완료 기준:

1. 모든 모델이 같은 dataset interface를 사용한다.
2. `train.py --config configs/ai_v2_hetero_mlp.yaml` 형태로 학습 가능하다.
3. `evaluate.py --checkpoint ... --split test` 형태로 평가 가능하다.
4. 각 실험 결과가 하나의 CSV 또는 JSONL로 저장된다.
5. 결과 파일에 최소한 다음 column이 포함된다.

```text
experiment_name
model_type
split
noise_rmse
noise_rmse_dx
noise_rmse_dy
noise_rmse_vx
noise_rmse_vy
nll
observed_rmse
corrected_rmse
improvement_ratio
worsening_rate
```

6. distance bucket별 metric 파일이 별도로 저장된다.
7. v1, v2, v3, distance gaussian, range/bearing model이 모두 같은 test split에서 비교된다.
