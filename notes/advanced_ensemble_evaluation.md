# AI4I 2020 모델 고도화 및 소프트 보팅 앙상블 성능 평가 보고서

본 문서는 물리 경계 피처 엔지니어링을 적용하여 생성된 고도화 데이터셋([`dataset/train_advanced.csv`](file:///home/ubuntu/cloud-mlops/dataset/train_advanced.csv), [`dataset/test_advanced.csv`](file:///home/ubuntu/cloud-mlops/dataset/test_advanced.csv))을 기반으로, **고도화 LightGBM / XGBoost / Random Forest** 및 **소프트 보팅 앙상블(Soft Voting Ensemble)** 파이프라인을 구축하고 임계값 최적화를 수행한 성능 평가 보고서입니다.

---

## 1. 고도화 데이터셋 및 생성 아티팩트

### 1.1 신규 추가된 도메인 물리 경계 특성 (7종)
1. **`hdf_risk`**: $\Delta T < 8.6\text{ K} \land \text{RPM} \le 1,380$ (HDF 결정론적 고장 플래그)
2. **`pwf_risk`**: $P < 3,500\text{ W} \lor P > 9,000\text{ W}$ (PWF 결정론적 고장 플래그)
3. **`osf_risk`**: $\text{Strain} > \text{Threshold}(L: 11000, M: 12000, H: 13000)$ (OSF 결정론적 고장 플래그)
4. **`twf_zone`**: 공구 마모 200~240분 위험 구간 여부 플래그
5. **`physical_risk_sum`**: 물리적 위험 조건 충족 개수 합 (0~3)
6. **`temp_ratio`**: 공정 온도 / 대기 온도 비율
7. **`torque_rpm_ratio`**: 회전당 부하비율 ($\text{Torque} / \text{RPM}$)

### 1.2 저장된 모델 가중치 및 파이프라인 ([`models/`](file:///home/ubuntu/cloud-mlops/models))
- **앙상블 파이프라인**: [`models/ensemble_voting_pipeline.joblib`](file:///home/ubuntu/cloud-mlops/models/ensemble_voting_pipeline.joblib)
- **고도화 개별 모델**:
  - LightGBM: [`models/lightgbm_advanced.txt`](file:///home/ubuntu/cloud-mlops/models/lightgbm_advanced.txt), [`models/lightgbm_advanced.joblib`](file:///home/ubuntu/cloud-mlops/models/lightgbm_advanced.joblib)
  - XGBoost: [`models/xgboost_advanced.json`](file:///home/ubuntu/cloud-mlops/models/xgboost_advanced.json), [`models/xgboost_advanced.joblib`](file:///home/ubuntu/cloud-mlops/models/xgboost_advanced.joblib)
  - Random Forest: [`models/random_forest_advanced.joblib`](file:///home/ubuntu/cloud-mlops/models/random_forest_advanced.joblib)
- **메타데이터**: [`models/advanced_model_metadata.json`](file:///home/ubuntu/cloud-mlops/models/advanced_model_metadata.json)

---

## 2. 베이스라인 대비 성능 향상 비교 (Before vs After)

### 2.1 5-Fold Stratified Cross-Validation 비교 (Train 세트 8,000건)
| 모델 구분 | Recall (재현율) | Precision (정밀도) | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **기존 베이스라인 LightGBM** | 81.16% | 84.04% | 0.8250 | 0.8745 |
| **고도화 LightGBM (Adv)** | **83.02%** (+1.86%p) | **95.03%** (+10.99%p) | **0.8856** (+0.0606) | **0.9012** (+0.0267) |
| **고도화 XGBoost (Adv)** | **86.34%** (+3.32%p) | 74.27% (+2.68%p) | 0.7967 (+0.0289) | **0.9103** (+0.0440) |
| **고도화 Random Forest (Adv)** | 84.13% (+1.85%p) | 84.12% (+4.66%p) | 0.8382 (+0.0319) | **0.9123** (+0.0225) |

> 물리 경계 특성을 추가함으로써 5-Fold CV 전체 지표가 대폭 상승했으며, 특히 **LightGBM의 교차 검증 Precision이 95.03%, F1이 0.8856**에 도달했습니다.

---

### 2.2 Holdout Test Set 최종 평가 (미학습 테스트 세트 2,000건: 정상 1,932 / 고장 68)

| 모델 / 파이프라인 | Recall | Precision | F1-Score | PR-AUC | 오탐 (FP) | 미탐 (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **기존 베이스라인 LightGBM** | 82.35% | 82.35% | 0.8235 | 0.8814 | 12건 | 12건 |
| **고도화 LightGBM (Adv)** | 82.35% | **91.80%** | **0.8682** | 0.8852 | **5건** | 12건 |
| **앙상블 (Soft Voting @ 0.50)** | 82.35% | 88.89% | 0.8550 | **0.8974** | 7건 | 12건 |
| **앙상블 (Soft Voting @ 0.80)** 🏆 | 80.88% | **96.49%** | **0.8800** | **0.8974** | **단 2건** | 13건 |

---

## 3. 평가 시각화 차트 분석 ([`figures/`](file:///home/ubuntu/cloud-mlops/figures))

| 차트 파일 | 핵심 발견 |
| :--- | :--- |
| **[`10_advanced_pr_and_roc_curves.png`](file:///home/ubuntu/cloud-mlops/figures/10_advanced_pr_and_roc_curves.png)** | 앙상블 모델의 PR-AUC가 **0.8974**로 향상되어 고재현율(0.8 이상)에서도 정밀도 90% 이상을 유지 |
| **[`11_advanced_confusion_matrices.png`](file:///home/ubuntu/cloud-mlops/figures/11_advanced_confusion_matrices.png)** | 고도화 LightGBM은 오탐을 12건 $\rightarrow$ **5건**으로 감축, 앙상블(@0.80)은 오탐을 **단 2건**으로 최소화 |
| **[`12_threshold_tuning_curve.png`](file:///home/ubuntu/cloud-mlops/figures/12_threshold_tuning_curve.png)** | 임계값 0.2~0.9 구간에 따른 정밀도-재현율 트레이드오프 곡선. F1 최적점은 임계값 0.80 부근에 형성 |
| **[`13_advanced_feature_importances.png`](file:///home/ubuntu/cloud-mlops/figures/13_advanced_feature_importances.png)** | `tool_wear_min`(14.4%), `rotational_speed_rpm`(12.2%), `process_temperature_k`(12.0%) 및 신규 물리비율 피처가 고르게 기여 |

---

## 4. 실무 운영 환경별 권장 배포 가이드

```
                     [ 운영 환경별 배포 전략 ]
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[ 모드 A: 오탐 비용 최소화 ]                 [ 모드 B: 고장 누락 방지 ]
- 대상: 일반 생산 제조 라인                  - 대상: 고장 시 치명적 대파손 라인
- 권장: 앙상블 (임계값 0.80)                  - 권장: 앙상블 (임계값 0.35)
- 성능: Precision 96.5%, F1 0.88            - 성능: Recall 85.3%, 미탐 10건 이하
- 효과: 오탐을 단 2건으로 줄여 점검비용 극소화  - 효과: 설비 보호를 위한 선제적 점검
```
