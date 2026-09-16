# AI4I 2020 Predictive Maintenance Dataset 설명서

이 문서는 [`ai4i2020.csv`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/cloud-mlops/dataset/ai4i2020.csv) 데이터셋의 구성, 컬럼 정의, 기초 통계 및 MLOps 관점에서의 분석 포인트를 정리한 문서입니다.

---

## 1. 데이터셋 개요 (Overview)

- **데이터셋 명칭**: AI4I 2020 Predictive Maintenance Dataset
- **원작자 및 출처**: Stephan Matzka (Hochschule für Technik und Wirtschaft Berlin, 2020), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **데이터 크기**: 10,000 행(Rows) × 14 열(Columns)
- **데이터 유형**: 합성 데이터(Synthetic Dataset) — 실제 밀링(Milling) 공정 장비의 물리적 파라미터와 공학적 도메인 지식을 기반으로 생성되어 실제 산업 현장의 특성을 정밀하게 모사함.
- **주요 용도**:
  - 상태 기반 설비 예지보전(Predictive Maintenance, PdM)
  - 이진 분류(고장 발생 여부 예측) 및 다중 라벨 분류(세부 고장 원인 분석)
  - 클래스 불균형(Class Imbalance) 처리 및 피처 엔지니어링 실습
  - MLOps 파이프라인(데이터 검증, 모델 학습, 서빙, 드리프트 감지) 실습

---

## 2. 컬럼별 상세 정의 (Feature Dictionary)

총 14개의 컬럼은 **식별자(ID)**, **공정 센서 데이터(Features)**, **고장 라벨(Targets)**로 분류됩니다.

| 컬럼명 | 데이터 타입 | 단위 | 구분 | 설명 |
| :--- | :---: | :---: | :---: | :--- |
| **`UDI`** | Integer | - | 식별자 | 레코드 고유 식별자 (1 ~ 10,000) |
| **`Product ID`** | String | - | 식별자 | 제품 품질 등급(L/M/H)과 일련번호로 구성 (예: `M14860`) |
| **`Type`** | String (Categorical) | - | 입력 특성 | 제품 품질 변형 등급 (`L`, `M`, `H`) |
| **`Air temperature [K]`** | Float | Kelvin (K) | 센서 특성 | 대기 온도 (주변 환경 온도) |
| **`Process temperature [K]`** | Float | Kelvin (K) | 센서 특성 | 공정 온도 (가공 중 발생하는 열이 더해진 온도) |
| **`Rotational speed [rpm]`** | Integer | RPM | 센서 특성 | 회전축 속도 (분당 회전수) |
| **`Torque [Nm]`** | Float | Nm | 센서 특성 | 모터 회전 토크 (돌림힘) |
| **`Tool wear [min]`** | Integer | 분 (min) | 센서 특성 | 공구 누적 마모 시간 |
| **`Machine failure`** | Binary (0 / 1) | - | **주요 타겟** | 장비 고장 발생 여부 (0: 정상, 1: 고장) |
| **`TWF`** | Binary (0 / 1) | - | 세부 고장 | 공구 마모 고장 (Tool Wear Failure) |
| **`HDF`** | Binary (0 / 1) | - | 세부 고장 | 열 방출 실패 고장 (Heat Dissipation Failure) |
| **`PWF`** | Binary (0 / 1) | - | 세부 고장 | 전력 이상 고장 (Power Failure) |
| **`OSF`** | Binary (0 / 1) | - | 세부 고장 | 과부하 고장 (Overstrain Failure) |
| **`RNF`** | Binary (0 / 1) | - | 세부 고장 | 무작위 고장 (Random Failure) |

---

## 3. 세부 속성 및 물리적 특성

### (1) 제품 품질 등급 (`Type`)
가공 공정의 난이도와 장비 부하를 결정하는 제품 등급입니다.
- **`L` (Low)**: 저품질/표준 제품 (데이터의 60.0%, 6,000건)
- **`M` (Medium)**: 중품질 제품 (데이터의 29.97%, 2,997건)
- **`H` (High)**: 고품질 제품 (데이터의 10.03%, 1,003건)
- 공정마다 품질 변형(H/M/L)에 따라 공구 마모가 각각 5분 / 3분 / 2분씩 누적됩니다.

### (2) 센서 수치 통계
| 센서 컬럼 | 최소값 (Min) | 최대값 (Max) | 평균 (Mean) | 표준편차 (Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Air temperature [K]** | 295.30 | 304.50 | 300.00 | 2.00 |
| **Process temperature [K]** | 305.70 | 313.80 | 310.01 | 1.48 |
| **Rotational speed [rpm]** | 1,168.00 | 2,886.00 | 1,538.78 | 179.28 |
| **Torque [Nm]** | 3.80 | 76.60 | 39.99 | 9.97 |
| **Tool wear [min]** | 0.00 | 253.00 | 107.95 | 63.65 |

---

## 4. 고장 메커니즘 및 타겟 분석 (Failure Modes)

### (1) 타겟 클래스 분포 (Machine failure)
- **정상 (`0`)**: 9,661건 (96.61%)
- **고장 (`1`)**: 339건 (3.39%)
> 심각한 클래스 불균형(Class Imbalance)이 존재하므로 평가 시 단순 정확도(Accuracy)보다는 **F1-Score, Recall, PR-AUC, ROC-AUC** 지표를 중점적으로 확인해야 합니다.

### (2) 세부 고장 원인 정의 및 통계
각 고장 모드는 명확한 물리적 규칙에 따라 발생하도록 설계되어 있습니다.

1. **`TWF` (Tool Wear Failure, 공구 마모 고장) - 46건**
   - 공구의 누적 마모 시간(`Tool wear`)이 200분에서 240분 사이에 도달하면 마모 한계로 인해 공구가 파손되거나 교체 실패가 발생합니다.
2. **`HDF` (Heat Dissipation Failure, 방열 실패 고장) - 115건**
   - 공정 온도와 대기 온도의 차이가 8.6 K 미만(`Process temperature - Air temperature < 8.6 K`)이고, 회전 속도가 1,380 RPM 이하(`Rotational speed <= 1380 rpm`)일 때 열 방출이 원활하지 않아 과열로 발생합니다.
3. **`PWF` (Power Failure, 전력 이상 고장) - 95건**
   - 소모 전력 $P = \text{Torque [Nm]} \times \text{Rotational speed [rad/s]}$ 계산 시, 전력이 3,500 W 미만이거나 9,000 W를 초과할 때 발생합니다.
4. **`OSF` (Overstrain Failure, 과부하 고장) - 98건**
   - 공구 마모 시간과 토크의 곱($\text{Tool wear} \times \text{Torque}$)이 제품 등급별 허용 임계치를 초과할 때 발생합니다.
     - `L` 타입: 11,000 min·Nm 초과 시
     - `M` 타입: 12,000 min·Nm 초과 시
     - `H` 타입: 13,000 min·Nm 초과 시
5. **`RNF` (Random Failure, 무작위 고장) - 19건**
   - 공정 매개변수와 무관하게 0.1%의 독립적인 확률로 발생하는 외생적 요인에 의한 고장입니다.

### (3) 데이터 분석 시 주의할 특이사항 (Domain Caveats)
- **`RNF`와 `Machine failure`의 관계**:
  - `RNF`가 발생한 19건 중 18건은 `Machine failure`가 **0**입니다. 오직 다른 고장(`TWF`)이 동반된 1건만 `Machine failure = 1`로 기록되어 있습니다. 즉, 단순 무작위 결함 감지와 전체 기계 셧다운 고장 여부는 구분될 수 있습니다.
- **원인 미상 고장 (9건)**:
  - 5가지 고장 모드(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)가 모두 0임에도 `Machine failure = 1`인 케이스가 9건 존재합니다.
- **복합 고장 (24건)**:
  - 2개 이상의 고장 모드가 동시에 트리거된 샘플이 24건 존재합니다.

---

## 5. MLOps 및 Feature Engineering 핵심 포인트

1. **파생 변수(Derived Features) 생성 추천**:
   - **온도차(Temperature Difference)**: `Process temperature [K] - Air temperature [K]` (HDF 예측 기여)
   - **전력(Mechanical Power [W])**: $\text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$ (PWF 예측 기여)
   - **스트레인 지수(Strain Index)**: `Tool wear [min] * Torque [Nm]` (OSF 예측 기여)
2. **불균형 처리 기법**:
   - SMOTE, ADASYN 등의 오버샘플링 또는 Random Under Sampling
   - 손실 함수 가중치(`scale_pos_weight`, `class_weight='balanced'`) 적용
3. **MLOps 파이프라인 활용 시나리오**:
   - **Data Validation**: Great Expectations 또는 Pydantic을 이용해 센서 범위 및 결측치 여부 검증
   - **Feature Store**: 제품별/시간별 특성 저장 및 배치/실시간 추론 파이프라인 연계
   - **Model Serving & Monitoring**: 경량 트리 모델(LightGBM, XGBoost) 또는 심층 신경망을 패키징하고, 입력 센서 분포의 드리프트(KS-Test, PSI) 모니터링 구축
