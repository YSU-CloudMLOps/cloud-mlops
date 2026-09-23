# AI4I 2020 Predictive Maintenance 데이터셋 개요 및 분석 정리

본 문서는 [`dataset/ai4i2020.csv`](file:///home/ubuntu/cloud-mlops/dataset/ai4i2020.csv) 데이터셋의 기본 구조, 피처 정의, 기초 통계량, 고장 메커니즘 및 MLOps 관점의 고려사항을 정리한 문서입니다.

---

## 1. 데이터셋 기본 개요 (Dataset Overview)

- **데이터셋 명칭**: AI4I 2020 Predictive Maintenance Dataset
- **데이터 출처**: Stephan Matzka (HTW Berlin, 2020), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **데이터 파일**: [`dataset/ai4i2020.csv`](file:///home/ubuntu/cloud-mlops/dataset/ai4i2020.csv) (용량: 약 510 KB)
- **데이터 크기**: 10,000개 행(Rows) × 14개 열(Columns)
- **결측치(Missing Values)**: **0건** (전체 컬럼 결측치 없음)
- **데이터 특성**: 밀링(Milling) 공정의 물리적 수식 및 도메인 규칙을 정밀하게 모사한 합성 데이터셋(Synthetic Dataset)
- **주요 목적**: 설비 예지보전(Predictive Maintenance, PdM)을 위한 장비 고장 이진 분류 및 세부 고장 원인 다중 라벨 분류

---

## 2. 컬럼별 상세 정의 (Feature Dictionary)

데이터는 크게 **식별자(Identifier)**, **공정 센서 특성(Sensor Features)**, **타겟 및 세부 고장 라벨(Targets & Failure Modes)**의 세 범주로 구분됩니다.

| 컬럼명 | 데이터 타입 | 단위 | 역할 | 설명 |
| :--- | :---: | :---: | :---: | :--- |
| **`UDI`** | Integer | - | 식별자 | 1부터 10,000까지의 고유 레코드 식별 번호 |
| **`Product ID`** | String | - | 식별자 | 제품 품질 등급(`L`/`M`/`H`) + 일련번호 (10,000개 고유값) |
| **`Type`** | String (Categorical) | - | 범주형 특성 | 제품 품질 변형 등급 (`L`: Low, `M`: Medium, `H`: High) |
| **`Air temperature [K]`** | Float | K (Kelvin) | 연속형 센서 특성 | 외부 대기 환경 온도 |
| **`Process temperature [K]`** | Float | K (Kelvin) | 연속형 센서 특성 | 실제 가공 공정 중 발생하는 열이 반영된 설비 온도 |
| **`Rotational speed [rpm]`** | Integer | RPM | 연속형 센서 특성 | 주축(Spindle) 모터의 분당 회전수 |
| **`Torque [Nm]`** | Float | N·m | 연속형 센서 특성 | 가공 모터에 가해지는 돌림힘(토크) |
| **`Tool wear [min]`** | Integer | 분 (min) | 연속형 센서 특성 | 공구(Tool)의 누적 사용 및 마모 시간 |
| **`Machine failure`** | Binary (0 / 1) | - | **메인 타겟** | 전체 장비 고장 발생 여부 (0: 정상, 1: 고장) |
| **`TWF`** | Binary (0 / 1) | - | 세부 고장 라벨 | 공구 마모 고장 (Tool Wear Failure) |
| **`HDF`** | Binary (0 / 1) | - | 세부 고장 라벨 | 열 방출 실패 고장 (Heat Dissipation Failure) |
| **`PWF`** | Binary (0 / 1) | - | 세부 고장 라벨 | 전력 이상 고장 (Power Failure) |
| **`OSF`** | Binary (0 / 1) | - | 세부 고장 라벨 | 과부하 고장 (Overstrain Failure) |
| **`RNF`** | Binary (0 / 1) | - | 세부 고장 라벨 | 무작위 고장 (Random Failure) |

---

## 3. 기초 통계 및 분포 분석

### 3.1 제품 등급(`Type`) 분포
- **`L` (Low 품질)**: 6,000건 (60.00%) - 고장 235건 (고장률: 3.92%)
- **`M` (Medium 품질)**: 2,997건 (29.97%) - 고장 83건 (고장률: 2.77%)
- **`H` (High 품질)**: 1,003건 (10.03%) - 고장 21건 (고장률: 2.09%)
> 공정 난이도가 낮고 생산량이 많은 `L` 타입 제품 생산 시 고장 발생 비율이 가장 높게 나타납니다.

### 3.2 연속형 센서 특성 기술통계
| 센서 컬럼 | 최소값 (Min) | 1사분위수 (Q1) | 중앙값 (Median) | 3사분위수 (Q3) | 최대값 (Max) | 평균 (Mean) | 표준편차 (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Air temperature [K]** | 295.30 | 298.30 | 300.10 | 301.50 | 304.50 | 300.00 | 2.00 |
| **Process temperature [K]** | 305.70 | 308.80 | 310.10 | 311.10 | 313.80 | 310.01 | 1.48 |
| **Rotational speed [rpm]** | 1,168.00 | 1,423.00 | 1,503.00 | 1,612.00 | 2,886.00 | 1,538.78 | 179.28 |
| **Torque [Nm]** | 3.80 | 33.20 | 40.10 | 46.80 | 76.60 | 39.99 | 9.97 |
| **Tool wear [min]** | 0.00 | 53.00 | 108.00 | 162.00 | 253.00 | 107.95 | 63.65 |

---

## 4. 고장 메커니즘 및 타겟 분석 (Failure Modes)

### 4.1 메인 타겟 (`Machine failure`)
- **정상 (`0`)**: 9,661건 (96.61%)
- **고장 (`1`)**: 339건 (3.39%)
- **불균형 비율**: 약 28.5 : 1 수준의 심각한 클래스 불균형(Class Imbalance)이 존재합니다.

### 4.2 세부 고장 모드별 조건 및 빈도
| 고장 모드 | 빈도 (건수) | 비율 (%) | 발생 조건 및 메커니즘 |
| :--- | :---: | :---: | :--- |
| **HDF** (열 방출 실패) | 115건 | 1.15% | `Process temperature - Air temperature < 8.6 K` 이면서 `Rotational speed <= 1,380 rpm` 인 경우 방열이 불량하여 과열 발생 |
| **OSF** (과부하 고장) | 98건 | 0.98% | $\text{Tool wear [min]} \times \text{Torque [Nm]}$ 값이 제품 등급별 기준치를 초과할 때 발생 (`L`: 11,000, `M`: 12,000, `H`: 13,000 min·Nm) |
| **PWF** (전력 이상) | 95건 | 0.95% | 가공 전력 $P = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$ 이 3,500 W 미만이거나 9,000 W 초과할 때 발생 |
| **TWF** (공구 마모 고장) | 46건 | 0.46% | 공구 마모 시간(`Tool wear`)이 200~240분 사이의 마모 한계 구간에 도달하여 파손 발생 |
| **RNF** (무작위 고장) | 19건 | 0.19% | 센서 수치와 무관하게 0.1%의 독립 확률로 발생하는 외생적 요인의 고장 |

### 4.3 고장 데이터 분석 시 주의사항 (Caveats)
1. **RNF(무작위 고장)의 타겟 불일치**:
   - RNF가 발생한 19건 중 **18건은 `Machine failure`가 0(정상)**으로 표기되어 있습니다. 오직 `TWF`와 동시 발생한 1건만 `Machine failure = 1`입니다.
2. **원인 미상 고장 (9건)**:
   - 5개 세부 고장 라벨(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)이 모두 `0`임에도 `Machine failure = 1`로 기록된 샘플이 9건 존재합니다.
3. **복합 고장 (24건)**:
   - 2가지 이상의 고장이 동시에 발생한 샘플이 24건 존재합니다. (예: `PWF` + `OSF` 동시 발생 11건, `HDF` + `OSF` 동시 발생 6건 등)

---

## 5. MLOps 관점의 핵심 분석 및 파이프라인 제언

### 5.1 도메인 기반 피처 엔지니어링 (Feature Engineering)
물리적 발생 메커니즘을 반영하는 파생 변수를 생성하면 모델 성능(Recall/F1) 향상에 매우 유리합니다.
- **온도차(Temperature Difference)**:  
  $$\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$$
- **기계적 소비 전력(Mechanical Power)**:  
  $$P = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$$
- **과부하/변형 지수(Strain Index)**:  
  $$\text{Strain} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$

### 5.2 모델링 및 평가 전략
- **평가 지표**: 극심한 불균형(고장률 3.39%)으로 인해 단순 `Accuracy`는 부적합하며, 고장 탐지 누락(FN)을 최소화하기 위한 **`Recall`**, **`F1-Score`**, **`PR-AUC(Precision-Recall AUC)`**를 핵심 지표로 선정해야 합니다.
- **불균형 완화 기법**:
  - 모델 레벨: LightGBM / XGBoost의 `scale_pos_weight` 조정, Focal Loss 적용
  - 데이터 레벨: SMOTE 등의 오버샘플링 또는 고장 데이터 증강

### 5.3 MLOps 파이프라인 적용 포인트
- **Data Validation**: 센서 값 범위(예: 온도 290~320K, RPM 1000~3000) 및 결측치 여부를 Great Expectations, Pydantic 등으로 자동 검증
- **Data Drift 모니터링**: 계절/환경 변화에 따른 대기 온도(`Air temperature`) 변화나 공구 교체 주기 패턴의 드리프트를 PSI / KS-test로 지속 감지
- **서빙 파이프라인**: 1차로 고장 여부(`Machine failure`)를 예측하고, 고장으로 판정된 경우 세부 원인(`TWF`, `HDF` 등)을 분석하여 현장 작업자에게 조치 가이드를 제공하는 2단계 서빙 아키텍처 설계 권장
