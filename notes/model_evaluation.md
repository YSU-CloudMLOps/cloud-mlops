# AI4I 2020 예지보전 모델 학습 및 성능 평가 보고서

본 문서는 [`dataset/train.csv`](file:///home/ubuntu/cloud-mlops/dataset/train.csv) 및 [`dataset/test.csv`](file:///home/ubuntu/cloud-mlops/dataset/test.csv)를 바탕으로 **LightGBM**, **XGBoost**, **Random Forest** 3개 모델을 학습하고, 가중치 및 구조를 영구 직렬화한 결과와 성능 지표를 종합 분석한 평가 보고서입니다.

---

## 1. 모델 학습 및 가중치 저장 내역 ([`models/`](file:///home/ubuntu/cloud-mlops/models))

모든 모델은 학습 완료 후 내부 분기 구조 및 가중치를 프레임워크 네이티브 포맷과 Python 범용 직렬화(`joblib`) 포맷으로 분리 저장했습니다.

| 모델명 | 직렬화 포맷 | 파일 경로 | 설명 |
| :--- | :---: | :--- | :--- |
| **`LightGBM`** | Native Text | [`models/lightgbm_model.txt`](file:///home/ubuntu/cloud-mlops/models/lightgbm_model.txt) | LightGBM 부스터 트리 구조 및 리프 가중치 원본 텍스트 |
| | Scikit-learn API | [`models/lightgbm_model.joblib`](file:///home/ubuntu/cloud-mlops/models/lightgbm_model.joblib) | 파이썬 환경 즉시 서빙용 `LGBMClassifier` 객체 |
| **`XGBoost`** | Native JSON | [`models/xgboost_model.json`](file:///home/ubuntu/cloud-mlops/models/xgboost_model.json) | XGBoost JSON 규격 트리 구조, 분기 임계치 및 가중치 |
| | Scikit-learn API | [`models/xgboost_model.joblib`](file:///home/ubuntu/cloud-mlops/models/xgboost_model.joblib) | 파이썬 환경 즉시 서빙용 `XGBClassifier` 객체 |
| **`Random Forest`** | Scikit-learn API | [`models/random_forest_model.joblib`](file:///home/ubuntu/cloud-mlops/models/random_forest_model.joblib) | 200개 의사결정나무 앙상블 가중치 보존 객체 |
| **메타데이터** | JSON | [`models/model_metadata.json`](file:///home/ubuntu/cloud-mlops/models/model_metadata.json) | 학습 타임스탬프, 피처 목록, 5-Fold CV 및 Test 세트 전체 지표 |

---

## 2. 모델 성능 지표 종합 비교 (Performance Benchmark)

### 2.1 5-Fold Stratified Cross-Validation (Train 세트 8,000건, Mean ± Std)
> 학습 데이터셋의 일반화 성능과 폴드별 안정성을 검증한 결과입니다.

| 모델 | Recall (재현율) | Precision (정밀도) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`LightGBM`** | 81.16% (±4.18%) | **84.04% (±2.22%)** | **0.8250 (±0.024)** | 0.8745 (±0.027) | 0.9732 (±0.014) |
| **`XGBoost`** | **83.02% (±4.32%)** | 71.59% (±2.50%) | 0.7678 (±0.018) | 0.8663 (±0.030) | 0.9777 (±0.012) |
| **`Random Forest`** | 82.28% (±4.79%) | 79.46% (±7.23%) | 0.8063 (±0.047) | **0.8898 (±0.026)** | **0.9806 (±0.014)** |

---

### 2.2 Holdout Test Set 최종 평가 (미학습 평가 세트 2,000건)
> 고장 68건, 정상 1,932건이 포함된 실제 테스트 데이터에 대한 최종 블라인드 평가 결과입니다.

| 모델 | Recall (고장 탐지율) | Precision (정탐률) | F1-Score | PR-AUC | ROC-AUC | 혼동 행렬 (TP / FP / FN / TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LightGBM`** (추천) | 82.35% | **82.35%** | **0.8235** | **0.8814** | **0.9789** | **TP: 56**, FP: 12, FN: 12, TN: 1,920 |
| **`XGBoost`** | 82.35% | 73.68% | 0.7778 | 0.8777 | 0.9782 | **TP: 56**, FP: 20, FN: 12, TN: 1,912 |
| **`Random Forest`** | **85.29%** | 72.50% | 0.7838 | 0.8614 | 0.9775 | **TP: 58**, FP: 22, FN: 10, TN: 1,910 |

---

## 3. 평가 시각화 차트 분석 ([`figures/`](file:///home/ubuntu/cloud-mlops/figures))

### 3.1 ROC 및 Precision-Recall 곡선 ([`figures/06_roc_curves.png`](file:///home/ubuntu/cloud-mlops/figures/06_roc_curves.png), [`figures/07_pr_curves.png`](file:///home/ubuntu/cloud-mlops/figures/07_pr_curves.png))
- **ROC 곡선**: 3개 모델 모두 ROC-AUC 0.977~0.979 구간의 매우 높은 전반적 판별 능력을 입증했습니다.
- **PR 곡선 (핵심 지표)**: 베이스라인 유병률(3.40%) 대비 25배 이상 높은 **PR-AUC 0.8814**를 기록한 **`LightGBM`**이 높은 재현율(Recall > 0.8) 구간에서도 가장 완만한 정밀도 하락세를 보이며 최적의 성능을 보였습니다.

### 3.2 혼동 행렬 비교 ([`figures/08_confusion_matrices.png`](file:///home/ubuntu/cloud-mlops/figures/08_confusion_matrices.png))
- **`LightGBM`**: 오탐(FP) 12건, 미탐(FN) 12건으로 **가장 균형 잡힌 오분류 최소화**를 달성했습니다.
- **`Random Forest`**: 실제 고장 68건 중 58건을 탐지하여 미탐(FN)을 **10건**으로 가장 낮게 억제했으나, 오탐(FP)이 22건으로 다소 증가했습니다.
- **`XGBoost`**: LightGBM과 동일한 고장 56건을 탐지했으나 오탐(FP)이 20건 발생했습니다.

### 3.3 피처 중요도 분석 ([`figures/09_feature_importances.png`](file:///home/ubuntu/cloud-mlops/figures/09_feature_importances.png))
전처리 단계에서 수립하여 주입한 **도메인 기반 물리 파생 변수**가 모든 모델의 의사결정을 지배하고 있음을 확인했습니다:
1. **`rotational_speed_rpm` & `power_w` (기계 전력)**: 전체 중요도의 **30~50%**를 차지하며 고속/저속 및 PWF(전력 이상) 판별의 핵심 변수로 작용.
2. **`strain_min_nm` (과부하 지수)**: 전체 중요도의 **10~15%**를 점유하며 OSF(과부하 파손)의 주 판별 기준으로 활용됨.
3. **`tool_wear_min` & `temp_diff_k` (온도차)**: TWF 및 HDF 고장 감지에 각각 10% 내외의 안정적인 기여도를 제공.

---

## 4. 실무 배포를 위한 종합 평가 및 모델 선정 가이드

```
┌────────────────────────────────────────────────────────────────────────┐
│                        실무 배포 권장 모델                             │
├────────────────────────────────────────────────────────────────────────┤
│  1순위 (운영 최적): LightGBM                                            │
│   - F1-Score: 0.8235 (최고)                                            │
│   - PR-AUC: 0.8814 (최고)                                              │
│   - 오탐(FP): 12건으로 가장 적음 (불필요한 설비 점검 공수 및 비용 최소화) │
│   - 추론 속도: 마이크로초 단위, 네이티브 txt 파일로 초경량 배포 가능     │
├────────────────────────────────────────────────────────────────────────┤
│  2순위 (안전 최우선): Random Forest                                      │
│   - Recall: 85.29% (58건 탐지, 미탐 10건으로 최소)                      │
│   - 설비 셧다운 대파손 리스크가 극도로 커 미탐(FN)을 극도로 방어해야   │
│     하는 라인에 적합                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 저장된 모델 로드 및 추론 예제 코드

```python
import joblib
import pandas as pd
import lightgbm as lgb

# 1. Scikit-learn Joblib 모델 로드
model = joblib.load('models/lightgbm_model.joblib')

# 2. 신규 센서 데이터 추론 예시
sample = pd.DataFrame([{
    'type_encoded': 0,
    'air_temperature_k': 300.0,
    'process_temperature_k': 310.0,
    'rotational_speed_rpm': 1500,
    'torque_nm': 40.0,
    'tool_wear_min': 100,
    'temp_diff_k': 10.0,
    'power_w': 6283.19,
    'strain_min_nm': 4000.0,
    'tool_wear_critical': 0
}])

failure_prob = model.predict_proba(sample)[0, 1]
print(f"장비 고장 확률: {failure_prob:.2%}")
```
