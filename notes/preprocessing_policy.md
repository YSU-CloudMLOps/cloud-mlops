# AI4I 2020 데이터셋 탐색적 데이터 분석(EDA) 및 전처리 정책 수립서

본 문서는 [`dataset/ai4i2020.csv`](file:///home/ubuntu/cloud-mlops/dataset/ai4i2020.csv)에 대한 심층 탐색적 데이터 분석(EDA) 결과와 이를 기반으로 도출된 **공식 데이터 전처리 정책(Data Preprocessing Policy)**을 정리한 문서입니다.

분석 결과 생성된 시각화 차트는 [`figures/`](file:///home/ubuntu/cloud-mlops/figures) 디렉토리에 고해상도 이미지로 저장되어 있습니다.

---

## 1. 탐색적 데이터 분석 (EDA) 핵심 결과 요약

### 1.1 타겟 및 고장 모드 분포 ([`figures/01_target_and_failure_distribution.png`](file:///home/ubuntu/cloud-mlops/figures/01_target_and_failure_distribution.png))
- **극심한 클래스 불균형**: 전체 10,000건 중 정상은 9,661건(96.61%), 장비 고장(`Machine failure = 1`)은 339건(3.39%)으로 불균형 비율이 약 **28.5 : 1**에 달합니다.
- **세부 고장 모드 빈도**:
  - `HDF` (방열 고장): 115건 (전체의 1.15%, 고장 중 33.9%)
  - `OSF` (과부하 고장): 98건 (전체의 0.98%, 고장 중 28.9%)
  - `PWF` (전력 이상): 95건 (전체의 0.95%, 고장 중 28.0%)
  - `TWF` (공구 마모): 46건 (전체의 0.46%, 고장 중 13.6%)
  - `RNF` (무작위 고장): 19건 (전체의 0.19%, 고장 중 5.6%)
- **제품 품질 등급(`Type`)별 고장률**:
  - `L` (Low 품질): 235건 / 6,000건 (**3.92%**)
  - `M` (Medium 품질): 83건 / 2,997건 (**2.77%**)
  - `H` (High 품질): 21건 / 1,003건 (**2.09%**)
  - 생산량이 많고 가격이 저렴한 `L` 제품 생산 시 부하가 커 고장률이 가장 높습니다.

### 1.2 센서 수치 분포 비교 ([`figures/02_sensor_features_by_failure.png`](file:///home/ubuntu/cloud-mlops/figures/02_sensor_features_by_failure.png))
- **`Torque [Nm]`**: 고장 그룹의 평균/중앙값(50.3 Nm / 53.6 Nm)이 정상 그룹(39.6 Nm / 40.0 Nm)보다 현저히 높음.
- **`Tool wear [min]`**: 고장 그룹의 중앙값이 약 165분으로 정상(107분) 대비 50% 이상 길며, 특히 200분 이상 구간에서 집중 고장 발생.
- **`Rotational speed [rpm]`**: 정상 그룹은 1,500 rpm 부근에 정규분포를 이루는 반면, 고장 그룹은 1,380 rpm 이하의 저속 영역(HDF 영향)과 2,500 rpm 이상의 초고속 이상치 영역(PWF 영향)으로 양극화(Bimodal)되는 패턴을 보임.
- **`Air / Process temperature [K]`**: 고장 그룹에서 주변 대기 온도와 공정 온도가 평균 1K가량 높게 형성됨.

### 1.3 다중공선성 및 상관관계 분석 ([`figures/03_correlation_matrix.png`](file:///home/ubuntu/cloud-mlops/figures/03_correlation_matrix.png))
- **`Torque` vs `Rotational speed`**: 피어슨 상관계수 **-0.88** (스피어만: -0.92)의 매우 강한 음의 상관관계. 기계 동력학상 $P = \tau \times \omega$ 관계식에 의해 모터 속도가 증가하면 토크가 감소하는 물리적 특성을 그대로 반영.
- **`Air temperature` vs `Process temperature`**: 피어슨 상관계수 **+0.88**로 매우 강한 양의 상관관계. 외부 기온이 높을수록 공정 온도도 비례하여 상승함.
- *MLOps 시사점*: 선형 모델(Logistic Regression, Linear SVM 등) 사용 시 심각한 다중공선성(Multicollinearity) 문제가 발생하므로, 도메인 파생 변수($\Delta T$, 전력 $P$)를 생성하여 이를 해소해야 함.

### 1.4 물리적 고장 발생 메커니즘 검증 ([`figures/04_physical_failure_mechanisms.png`](file:///home/ubuntu/cloud-mlops/figures/04_physical_failure_mechanisms.png))
데이터셋 내 세부 고장은 다음의 정밀한 물리 공식에 의해 100% 결정론적으로 트리거됨을 실증 검증함:
1. **`HDF` (Heat Dissipation Failure)**:
   - 조건: $\Delta T = (\text{Process Temp} - \text{Air Temp}) < 8.6\text{ K}$ **AND** $\text{Rotational Speed} \le 1,380\text{ rpm}$
   - 검증 정확도: **100% 매칭** (115건 전체)
2. **`PWF` (Power Failure)**:
   - 조건: 기계적 전력 $P = \text{Torque} \times \left(\text{Speed} \times \frac{2\pi}{60}\right)$ 에 대해 $P < 3,500\text{ W}$ 또는 $P > 9,000\text{ W}$
   - 검증 정확도: **100% 매칭** (95건 전체)
3. **`OSF` (Overstrain Failure)**:
   - 조건: $\text{Strain} = \text{Tool wear [min]} \times \text{Torque [Nm]} > \text{Threshold}$
   - 기준치: `Type L` > 11,000, `Type M` > 12,000, `Type H` > 13,000
   - 검증 정확도: **100% 매칭** (98건 전체)
4. **`TWF` (Tool Wear Failure)**:
   - 조건: 공구 마모 시간이 200~240분 사이의 마모 임계치에 도달할 때 확률적으로 파손 (평균 마모 시간: 216.4분)
5. **`RNF` (Random Failure)**:
   - 센서 조건과 독립적으로 0.1%의 확률로 발생. 19건 중 18건은 `Machine failure = 0`으로 유지됨.

### 1.5 파생 변수의 분리력 확인 ([`figures/05_engineered_features_distribution.png`](file:///home/ubuntu/cloud-mlops/figures/05_engineered_features_distribution.png))
- $\Delta T$ (온도차), $P$ (전력), $\text{Strain}$ (마모×토크) 3가지 물리 파생 변수를 추가할 경우 정상 데이터와 고장 데이터의 확률 밀도 분포가 뚜렷하게 분리됨을 확인함.

---

## 2. 전처리 정책 수립 (Data Preprocessing Policy)

탐색적 데이터 분석 결과를 토대로 프로덕션 MLOps 파이프라인에서 반드시 준수해야 할 전처리 정책을 7가지 항목으로 확정합니다.

```
                  [ 원시 데이터 (Raw Data) ]
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 1. Data Cleaning: UDI, Product ID 컬럼 제거    │
     │    BOM(\ufeff) 제거 및 표준 스네이크 표기 변환     │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 2. Outlier Policy: 센서 극단치는 고장 신호!     │
     │    (임의의 IQR Drop/Winsorizing 절대 금지)    │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 3. Feature Engineering: 3대 물리 공식 피처 생성 │
     │    - Temp_Diff (ΔT)  = Process Temp - Air Temp │
     │    - Power_W         = Torque * (RPM * 2π/60)  │
     │    - Strain_min_Nm   = Tool wear * Torque      │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 4. Categorical Encoding: Type 컬럼 순서형 변환 │
     │    - Ordinal Encoding: L: 0, M: 1, H: 2       │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 5. Leakage Prevention: 세부 고장 5개 라벨 배제  │
     │    (TWF, HDF, PWF, OSF, RNF는 Feature 사용 금지)│
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 6. Data Splitting: Stratified K-Fold (불균형) │
     │    - Train : Test = 80 : 20 (Stratified)      │
     │    - 평가 지표: Recall, PR-AUC, F1-Score        │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
     ┌───────────────────────────────────────────────┐
     │ 7. Scaling Policy: 모델군별 차등 적용         │
     │    - Tree 계열: Scaling 불필요                │
     │    - 선형/신경망: RobustScaler 적용             │
     └───────────────────────────────────────────────┘
```

---

### 정책 1: 데이터 정제 및 식별자 제거 (Data Cleaning)
- **식별자 제거**:
  - `UDI` (1~10,000 단순 일련번호)와 `Product ID` (개별 시리얼 넘버)는 설비 물리 상태와 무관한 단순 메타데이터이므로 **입력 피처에서 반드시 제거**합니다.
- **BOM 및 컬럼명 표준화**:
  - CSV 로드시 첫 번째 컬럼의 UTF-8 BOM(`\ufeff`)을 제거하고, 영문 소문자 스네이크 표기법(예: `air_temperature_k`, `rotational_speed_rpm`, `tool_wear_min`)으로 정규화합니다.
- **결측치(Missing Values)**:
  - 현재 데이터셋 결측치는 0건입니다.
  - *서빙 파이프라인 방어 정책*: 추론 서빙 시 센서 패킷 유실로 `NaN`이 유입될 경우, 이동 평균(Rolling Mean) 또는 설비별 정상 중앙값(Median)으로 대치하는 결측 방어 계층을 구축합니다.

---

### 정책 2: 이상치 보존 원칙 (Outlier Handling Policy)
> [!CAUTION]
> **IQR 기반의 임의 이상치 제거(Drop) 및 절단(Winsorizing/Clipping)은 절대 금지합니다.**

- **분석적 근거**:
  - `Torque [Nm]` IQR 이상치(69건)의 **89.86%(62건)**가 실제 장비 고장 샘플입니다.
  - 파생 전력(`Power_W`) IQR 이상치(60건)의 **100.0%(60건 전원)**이 실제 장비 고장(`PWF`) 샘플입니다.
  - 단순 통계 기반 이상치 제거 시 모델이 학습해야 할 고장 시그널의 대다수가 영구적으로 유실됩니다.
- **적용 규칙**:
  - 센서 하드웨어 고장으로 인한 물리적 불가능 값(예: 절대온도 < 0K, RPM < 0 등)만 유효 범위 필터링(Data Sanity Check)을 거치고, 그 외의 극단값은 **온전히 보존**하여 학습 데이터로 활용합니다.

---

### 정책 3: 도메인 기반 피처 엔지니어링 필수화 (Feature Engineering)
EDA에서 입증된 3대 물리 공식을 파이프라인의 필수 변환 단계로 포함합니다:

1. **온도차 (`temp_diff_k`)**:
   $$\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$$
   - *목적*: `HDF` 고장 탐지 직결 및 두 온도 센서 간 다중공선성 완화.
2. **기계적 소비 전력 (`power_w`)**:
   $$P = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$$
   - *목적*: `PWF` 고장 탐지 직결 및 토크-속도 간 강한 음의 상관관계($r=-0.88$) 정보 통합.
3. **과부하/변형 지수 (`strain_min_nm`)**:
   $$\text{Strain} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
   - *목적*: `OSF` 고장 임계치 판별.
4. **마모 위험 임계 지시자 (`tool_wear_critical`)**:
   - `tool_wear_min >= 200` 여부의 이진 플래그 (TWF 발생 고위험군 마킹).

---

### 정책 4: 범주형 변수 인코딩 정책 (Categorical Encoding)
- 대상: 제품 등급 `Type` (`L`, `M`, `H`)
- **정책**: **순서형 인코딩 (Ordinal Encoding)** 적용
  - 매핑: `{'L': 0, 'M': 1, 'H': 2}`
  - *이유*: 제품 품질 등급은 마모율 및 허용 과부하 한계(11,000 / 12,000 / 13,000 min·Nm)와 단조(Monotonic) 증가 관계를 가지므로 순서 정보를 보존하는 인코딩이 모델 해석력과 수렴성에 유리합니다.
  - (선형 모델 실험 시에 한하여 One-Hot Encoding 병행 허용)

---

### 정책 5: 데이터 누수(Data Leakage) 원천 차단
> [!IMPORTANT]
> **세부 고장 모드(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) 5개 컬럼은 `Machine failure` 예측 모델의 입력 특성으로 절대 사용할 수 없습니다.**

- **이유**: 실시간 운영 환경에서 설비가 셧다운되기 전에는 세부 고장 원인을 알 수 없으며, 세부 고장 컬럼을 특성으로 포함하면 타겟 정보가 그대로 유출되는 전형적인 Data Leakage가 발생합니다.
- **파이프라인 아키텍처 제언**:
  - **Stage 1 (고장 조기 경보)**: 순수 센서 데이터 및 파생 변수만을 입력으로 받아 장비 고장 여부(`Machine failure`) 이진 분류 수행.
  - **Stage 2 (고장 원인 진단)**: Stage 1에서 고장(`1`)으로 예측된 인스턴스에 한하여 5개 원인 라벨을 예측하는 다중 라벨/분류 진단 모델로 전달.

---

### 정책 6: 데이터 분할 및 평가 전략 (Data Splitting & Validation)
- **분할 기법**: **Stratified K-Fold / Stratified Split** 필수
  - 3.39%의 소수 클래스이므로 무작위 분할(Random Split) 시 특정 Fold에 고장 샘플이 과소 할당되는 문제가 발생합니다.
  - 분할 비율: `Train (80%)` : `Test (20%)`, 층화 5-Fold Cross Validation 채택.
- **평가 지표 가이드라인**:
  - `Accuracy` 사용 금지 (모두 정상으로 예측해도 96.61% 달성).
  - 1순위 지표: **`Recall` (재현율)** — 고장 미탐지로 인한 설비 셧다운 비용 방지.
  - 2순위 지표: **`PR-AUC` (Precision-Recall AUC)** 및 **`F1-Score`** — 불균형 데이터셋에 가장 견고한 종합 평가 지표.
- **불균형 처리 기법**:
  - 알고리즘 수준: 트리 모델의 `scale_pos_weight = 9661 / 339 ≈ 28.5` 또는 `class_weight='balanced'` 적용.
  - 의사결정 임계값(Threshold): 기본 0.5 대신 Recall 95% 이상을 달성하는 최적 Decision Threshold 튜닝.

---

### 정책 7: 특성 스케일링 정책 (Feature Scaling)
- **트리 기반 모델 (LightGBM, XGBoost, CatBoost, Random Forest)**:
  - 변수 분할 기준에 영향이 없으므로 **스케일링 생략** (원래 수치 단위 보존하여 설명 가능성 향상).
  - 파일: [`train.csv`](file:///home/ubuntu/cloud-mlops/dataset/train.csv), [`test.csv`](file:///home/ubuntu/cloud-mlops/dataset/test.csv)
- **거리 기반 / 선형 모델 (Logistic Regression, SVM, KNN, Deep Learning/MLP)**:
  - `StandardScaler`는 센서의 고장 신호 극단치에 의해 평균과 분산이 왜곡되므로, 중앙값과 사분위수(IQR) 기반의 **`RobustScaler`**를 적용합니다.
  - 범주형 특성(`Type`)은 거리 왜곡 방지를 위해 **One-Hot Encoding** (`type_l`, `type_m`, `type_h`)을 적용합니다.
  - Data Leakage 방지를 위해 Train 세트에만 `fit`하고, Test 세트에는 `transform`만 적용합니다.
  - 전용 산출물:
    - **[`dataset/train_scaled.csv`](file:///home/ubuntu/cloud-mlops/dataset/train_scaled.csv)** (8,000행 × 18열)
    - **[`dataset/test_scaled.csv`](file:///home/ubuntu/cloud-mlops/dataset/test_scaled.csv)** (2,000행 × 18열)
    - **[`dataset/robust_scaler.joblib`](file:///home/ubuntu/cloud-mlops/dataset/robust_scaler.joblib)** (학습된 Scaler 객체)
    - 실행 스크립트: [`scripts/preprocess_scaled.py`](file:///home/ubuntu/cloud-mlops/scripts/preprocess_scaled.py)

---

## 3. 요약 및 체크리스트

| 단계 | 전처리 항목 | 권장 정책 | 비권장/금지 사항 |
| :---: | :--- | :--- | :--- |
| **정제** | 식별자 처리 | `UDI`, `Product ID` 제거 | 모델 특성으로 포함 |
| **이상치** | 극단 센서값 | 센서 극단값 전량 보존 (고장 신호) | IQR Trimming / Winsorizing |
| **인코딩** | `Type` 컬럼 | Ordinal Encoding (`L:0, M:1, H:2`) | Target Encoding (과적합 위험) |
| **피처 생성** | 도메인 물리 변수 | $\Delta T$, 전력($P$), 과부하지수($\text{Strain}$) 추가 | 미검증 임의 조합 변수 남발 |
| **누수 방지** | 세부 고장 라벨 | 5대 고장 라벨 입력 특성에서 배제 | 세부 고장 컬럼을 Feature로 사용 |
| **분할** | 데이터 분할 | Stratified 5-Fold CV (층화 추출) | 단순 무작위 분할 (Random Split) |
| **평가** | 모델 검증 지표 | Recall, PR-AUC, F1-Score | 단순 정확도 (Accuracy) |
| **스케일링** | 정규화 방식 | Tree: 생략 / 선형·신경망: RobustScaler | 극단치를 압축하는 Min-Max Scaler |
