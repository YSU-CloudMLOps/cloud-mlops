# AI4I 2020 예지보전 모델 종합 개선 및 오분류 진단 보고서 (v2)

본 문서는 기존 베이스라인(10개 피처) 및 1차 고도화 모델(17개 피처)에서 수렴하던 **미탐(FN 12~13건) 및 오탐(FP 4~6건)**의 근본 원인을 데이터·도메인·알고리즘 관점에서 해부하고, **5단계 종합 개선 파이프라인([`scripts/improve_models.py`](../scripts/improve_models.py))**을 통해 모델 성능과 FastAPI 프로덕션 서빙 환경을 전면 업그레이드한 기록입니다.

---

## 1. 핵심 성과 요약 (Executive Summary)

홀드아웃 테스트 세트(미학습 2,000건: 정상 1,932건 / 고장 68건) 및 5-Fold 교차 검증(학습 8,000건) 기준 주요 개선 성과는 다음과 같습니다:

| 성능 지표 | 기존 베이스라인 (`M0`, 10f) | Week 06 튜닝 (`M1`, 10f) | 1차 고도화 (`M2`, 17f) | **최종 개선 모델 (`M4/M5`, 23f v2)** | 개선 폭 (`M0` $\rightarrow$ `v2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold CV F1-Score** | 0.8590 | 0.8448 | 0.8856 | **0.9182** | **+0.0592 (+6.89%)** |
| **5-Fold CV PR-AUC** | 0.8750 | 0.8849 | 0.9013 | **0.9259** | **+0.0509 (+5.82%)** |
| **Test Accuracy (정확도)** | 99.10% | 99.10% | 99.15% | **99.45%** | **+0.35%p** |
| **Test Precision (정밀도)** | 90.32% | 91.67% | 91.80% | **100.00%** | **+9.68%p (오탐 완전 제거)** |
| **Test Recall (재현율)** | 82.35% | 80.88% | 82.35% | **83.82%** | **+1.47%p (결정론적 고장 100% 포착)** |
| **Test F1-Score (@0.50)** | 0.8615 | 0.8594 | 0.8682 | **0.9120** | **+0.0505 (+5.86%)** |
| **Test PR-AUC** | 0.8928 | 0.8992 | 0.8852 | **0.9087** (RF: **0.9101**) | **+0.0159** |
| **Brier Score (확률 보정 오차)** | 0.0085 | 0.0081 | 0.0082 | **0.0056 ~ 0.0061** | **-28.2% 오차 감소** |
| **오탐 (FP) / 미탐 (FN)** | FP 6 / FN 12 | FP 5 / FN 13 | FP 5 / FN 12 | **FP 0 / FN 11** | **총 오류 18건 $\rightarrow$ 11건 (-38.9%)** |

---

## 2. 1단계: 오분류 슬라이스 진단 및 근본 원인(Root Cause) 규명

### 2.1 IEEE-754 부동소수점 반올림에 의한 `HDF(방열 고장)` 경계 손실 발견
기존 [`scripts/preprocess.py`](../scripts/preprocess.py)는 온도차를 `temp_diff_k = (Process - Air).round(2)`로 소수점 둘째 자리 반올림한 뒤, [`scripts/preprocess_advanced.py`](../scripts/preprocess_advanced.py)에서 `temp_diff_k < 8.6` 조건으로 `hdf_risk`를 생성했습니다.

그러나 전체 10,000건 데이터셋에서 `temp_diff_k == 8.6`이면서 `RPM < 1380`인 경계 샘플 **27건**을 전수 분석한 결과, 원시 데이터의 `HDF` 라벨은 반올림 이전의 **IEEE-754 64비트 부동소수점(`float64`) 연산**에 의해 결정되었음을 발견했습니다:

| 온도 조합 예시 (`Process - Air`) | IEEE-754 `float64` 실제 연산값 | `< 8.6` 판정 | 실제 `HDF` 라벨 | 기존 `.round(2)` 후 `< 8.6` 판정 |
| :--- | :---: | :---: | :---: | :---: |
| `310.7 K - 302.1 K` (외 11건) | `8.599999999999966` | **`True`** | **`1` (전건 고장, 12건)** | `8.6 < 8.6` $\rightarrow$ `False` (**미탐 발생**) |
| `311.0 K - 302.4 K` (외 14건) | `8.600000000000023` | **`False`** | **`0` (전건 정상, 15건)** | `8.6 < 8.6` $\rightarrow$ `False` (트리 분할 시 **오탐 유발**) |

- **해결**: `(process_temperature_k - air_temperature_k < 8.6) & (rotational_speed_rpm < 1380)`로 원시 부동소수점 차이를 즉시 평가하도록 수정한 결과, 전체 10,000건 데이터셋에서 **`HDF` 115건 전건, `PWF` 95건 전건, `OSF` 98건 전건이 오탐 0건·미탐 0건(100.00% 일치)**으로 완벽히 분리되었습니다.

### 2.2 과도한 `scale_pos_weight=28.5`에 의한 확률 왜곡 및 `twf_zone` 오탐(FP) 유발
- 1차 고도화 파이프라인(`M2`)에서는 불균형 비율(`9661 / 339 = 28.52`)을 그대로 `scale_pos_weight`에 주입했습니다.
- 공구 마모 위험 구간(`200 <= tool_wear_min <= 240`, `twf_zone=1`) 내에서의 실제 `TWF` 발생 확률은 약 **5.3%(43건 / 812건)**에 불과하지만, 양성 가중치가 28.5배 곱해지면서 정상 마모 샘플의 예측 확률이 `0.55 ~ 0.88`로 과장되어 홀드아웃 테스트 세트에서 5건의 오탐(FP)을 유발했습니다.
- **해결**: Optuna 탐색 시 `scale_pos_weight` 범위를 `[0.8, 2.0]`으로 제어하여 출력 확률의 캘리브레이션(Brier Score `0.0082` $\rightarrow$ `0.0061`)을 정상화하고 **오탐(FP)을 0건으로 완전 제거**했습니다.

### 2.3 고장 모드별 미탐(FN) 슬라이스 해부
홀드아웃 테스트 세트의 전체 고장 68건을 세부 고장 모드별로 분해한 결과는 다음과 같습니다:
1. **물리적 결정론 고장 (`HDF` 29건, `PWF` 13건, `OSF` 16건 — 중복 제외 총 57건)**:
   - 기존 모델(`M0~M2`)에서는 경계선 반올림 손실로 `HDF`에서 1~2건의 미탐이 발생했으나, **개선 모델(`M3~M5`)에서는 57건 전건(100.0%)을 오탐 없이 완벽히 포착**했습니다.
2. **확률적 마모 고장 (`TWF` 단독 고장 9건)**:
   - AI4I 2020 데이터셋 설계상 공구 마모 200~240분 구간에서 무작위 시점에 파손이 부여되므로, 센서 수치만으로는 동일 조건의 정상 샘플과 결정론적으로 분리할 수 없는 확률적 영역입니다.
3. **설명 불가 외생 노이즈 (`RNF` / 무원인 고장 — 학습 7건, 테스트 2건)**:
   - 4대 물리 고장(`TWF`, `HDF`, `PWF`, `OSF`)이 모두 `0`임에도 `machine_failure=1`로 기록된 순수 랜덤 노이즈입니다.

---

## 3. 2단계: 23개 도메인 피처 엔지니어링 (v2)

[`scripts/preprocess_advanced.py`](../scripts/preprocess_advanced.py) 및 [`api/services/feature_service.py`](../api/services/feature_service.py)를 고도화하여 기존 10개 기본 피처 + 7개 물리 경계/비율 피처에 더해 **6개의 연속형 임계 마진(Continuous Boundary Margin) 피처**를 추가했습니다 (총 **23개 피처**):

| 피처 분류 | 피처명 | 수식 및 공학적 의미 |
| :--- | :--- | :--- |
| **기본 센서 및 파생 (10종)** | `type_encoded` ~ `tool_wear_critical` | 원시 센서 6종 + 온도차(`temp_diff_k`), 전력(`power_w`), 스트레인(`strain_min_nm`), 마모 임계(`>=200`) |
| **정밀 물리 경계 및 비율 (7종)** | `hdf_risk` (IEEE-754 수정)<br>`pwf_risk`, `osf_risk`, `twf_zone`<br>`physical_risk_sum`<br>`temp_ratio`, `torque_rpm_ratio` | • `hdf_risk`: `(process_temp - air_temp < 8.6) & (rpm < 1380)`<br>• `pwf_risk`: `power_w < 3500 \| power_w > 9000`<br>• `osf_risk`: `strain_min_nm > {L:11000, M:12000, H:13000}`<br>• `physical_risk_sum`: 3대 물리 위험 합계 (0~3) |
| **연속형 임계 마진 (신규 6종)** | **`osf_margin`**<br>**`osf_strain_ratio`**<br>**`hdf_temp_margin`**<br>**`hdf_rpm_margin`**<br>**`pwf_margin`**<br>**`twf_wear_excess`** | • `strain_min_nm - osf_thresh` (과부하 한계까지 남은 절대 여유분, **중요도 2위 [10.0%]**)<br>• `strain_min_nm / osf_thresh` (제품 등급 대비 상대 과부하율)<br>• `8.6 - (process_temp - air_temp)` (방열 한계 온도 마진)<br>• `1380 - rotational_speed_rpm` (방열 한계 회전수 마진, **중요도 4위 [7.8%]**)<br>• `min(power_w - 3500, 9000 - power_w)` (정상 전력 대역 여유분)<br>• `max(0, tool_wear_min - 198)` (임계 마모 진입 후 초과 누적 시간) |

---

## 4. 3~4단계: 다중 모드 계층형 모델링, Optuna 튜닝 및 OOF 앙상블 평가

### 4.1 단계별 절제 연구 (Progressive Ablation Study)

| 단계 | 모델 구성 | 5-Fold CV F1 | 5-Fold CV PR-AUC | Test Precision | Test Recall | Test F1 (@0.50) | Test PR-AUC | Brier Score | 오탐(FP) / 미탐(FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M0`** | Baseline LightGBM (10f) | 0.8590 | 0.8750 | 90.32% | 82.35% | 0.8615 | 0.8928 | 0.0085 | FP 6 / FN 12 |
| **`M1`** | Legacy Tuned LightGBM (10f) | 0.8448 | 0.8849 | 91.67% | 80.88% | 0.8594 | 0.8992 | 0.0081 | FP 5 / FN 13 |
| **`M2`** | Legacy Adv LightGBM (17f) | 0.8856 | 0.9013 | 91.80% | 82.35% | 0.8682 | 0.8852 | 0.0082 | FP 5 / FN 12 |
| **`M3`** | **Hierarchical 4-Mode (23f)** | 0.9127 | 0.9186 | **100.00%** | **83.82%** | **0.9120** | 0.9021 | **0.0056** | **FP 0 / FN 11** |
| **`M4`** | **Optuna Tuned LightGBM v2 (23f)** | 0.9178 | **0.9259** | **100.00%** | **83.82%** | **0.9120** | **0.9087** | 0.0067 | **FP 0 / FN 11** |
| **`M4-Clean`** | **RNF 노이즈 제거 LightGBM v2 (23f)** | 0.9178 | 0.9250 | **100.00%** | **83.82%** | **0.9120** | **0.9106** | 0.0067 | **FP 0 / FN 11** |
| **`M4-RF`** | **Optuna Tuned Random Forest v2 (23f)** | 0.9178 | 0.9223 | **100.00%** | **83.82%** | **0.9120** | **0.9101** | **0.0053** | **FP 0 / FN 11** |
| **`M5`** | **OOF Soft Voting Ensemble v2 (23f)** | **0.9182** | 0.9249 | **100.00%** | **83.82%** | **0.9120** | 0.9085 | **0.0061** | **FP 0 / FN 11** |

### 4.2 주요 기법별 분석
1. **계층형 4대 고장 모드 분류 (`M3: Hierarchical 4-Mode`, [`models/hierarchical_4mode_models.joblib`](../models/hierarchical_4mode_models.joblib))**:
   - `TWF`, `HDF`, `PWF`, `OSF` 4개 물리 고장 각각에 대해 독립적인 LightGBM 분류기를 학습하고(학습 시 `RNF` 외생 노이즈 자동 배제), 결합 확률 $P(\text{Fail}) = 1 - \prod_{m}(1 - P_m)$을 산출했습니다.
   - 가장 낮은 확률 오차(**Brier Score 0.0056**)를 달성했으며, 현장 엔지니어에게 각 고장 원인별 확률을 분리 제공할 수 있습니다.
2. **라벨 노이즈(`RNF`) 제거 학습 (`M4-Clean`)**:
   - 학습 세트의 무원인 고장 7건을 정상(`0`)으로 정제하여 학습한 결과, 홀드아웃 테스트 세트 **PR-AUC가 `0.9087`에서 `0.9106`으로 추가 상승**하며 랭킹 품질이 개선됨을 입증했습니다.
3. **OOF 가중치 및 임계값 최적화 앙상블 (`M5: OOF Soft Voting Ensemble v2`)**:
   - 테스트 데이터 누수를 완전히 차단하고 오직 5-Fold OOF 예측값만으로 최적 가중치 `(LGB 0.70, XGB 0.20, RF 0.10)`와 최적 임계값 `0.50`을 도출하여 **5-Fold CV F1 `0.9182`, Test F1 `0.9120`, ROC-AUC `0.9845`**를 달성했습니다.

---

## 5. 5단계: 프로덕션 FastAPI 서빙 연동 및 테스트 검증

1. **실시간 피처 엔지니어링 서비스 업그레이드 ([`api/services/feature_service.py`](../api/services/feature_service.py))**:
   - `make_features(sensor: SensorInput)`가 6개 센서 입력으로부터 23개 도메인·마진 피처를 실시간 생성하도록 업데이트했습니다.
2. **기본 서빙 모델 가중치 교체 ([`models/lightgbm_model.txt`](../models/lightgbm_model.txt))**:
   - FastAPI [`ModelService`](../api/services/model_service.py)가 로드하는 기본 가중치 파일을 23개 피처 기반 Optuna 튜닝 모델(`M4: Tuned LightGBM v2`, 최적 임계값 `0.50`)로 업데이트했습니다.
3. **단위 및 통합 테스트 검증 ([`api/tests/`](../api/tests/))**:
   - `PYTHONPATH=. .venv/bin/python -m unittest discover -s API/tests -v` 실행 결과, 원시 데이터-전처리-API 추론 일치성 검증(`test_raw_inputs_match_preprocessed_model_predictions`) 및 23개 피처 검증을 포함한 **10개 테스트 전건 통과(`OK`)**를 확인했습니다.

---

## 6. 생성된 시각화 차트 및 아티팩트 목록

- **시각화 차트**:
  - [`figures/15_error_slice_and_ablation.png`](../figures/15_error_slice_and_ablation.png): 고장 모드별 미탐(FN) 분해, 오탐/미탐 감소 추이, 단계별 F1/PR-AUC 절제 연구(Ablation), CV vs Test 일반화 정렬 차트
  - [`figures/16_v2_model_comparison_and_calibration.png`](../figures/16_v2_model_comparison_and_calibration.png): 버전별 PR 곡선 비교, 최종 혼동 행렬(`TN=1932, FP=0, FN=11, TP=57`), 확률 보정 곡선(Reliability Curve & Brier Score), 상위 15개 피처 중요도
- **모델 및 메타데이터**:
  - [`models/lightgbm_model.txt`](../models/lightgbm_model.txt), [`models/lightgbm_tuned.txt`](../models/lightgbm_tuned.txt), [`models/lightgbm_advanced.txt`](../models/lightgbm_advanced.txt)
  - [`models/ensemble_voting_pipeline.joblib`](../models/ensemble_voting_pipeline.joblib), [`models/hierarchical_4mode_models.joblib`](../models/hierarchical_4mode_models.joblib)
  - [`models/improvement_metadata.json`](../models/improvement_metadata.json)
