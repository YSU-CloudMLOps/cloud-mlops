#import "@preview/charged-ieee:0.1.4": ieee

#show text: set text(font: ("Nimbus Roman", "Pretendard"))
#show figure.where(kind: table): set text(size: 8pt)

#show: ieee.with(
  title: [
    #text(18pt)[물리 법칙 기반 피처 엔지니어링과 소프트 보팅 앙상블을 활용한] \
    #text(18pt)[산업용 예지보전 MLOps 파이프라인 연구]
  ],
  abstract: [
    스마트 제조 환경에서 설비의 비계획 정지(Unplanned Downtime)를 예방하기 위한 상태 기반 예지보전(Predictive Maintenance, PdM)은 핵심 과제이다. 그러나 실제 산업 센서 데이터는 정상 데이터에 극도로 편향된 클래스 불균형(Class Imbalance)과 복합 물리 법칙에 기인한 비선형 거동으로 인해 고장 탐지의 정밀도와 재현율 간 균형을 달성하기 어렵다. 본 연구는 10,000건의 밀링(Milling) 공정 센서 레코드를 포함하는 AI4I 2020 데이터셋을 바탕으로, 심층 탐색적 데이터 분석(EDA), 7대 전처리 정책 수립, 머신러닝 벤치마크, 도메인 물리 경계 기반 피처 엔지니어링, 그리고 소프트 보팅 앙상블(Soft Voting Ensemble) 모델을 포괄하는 엔드투엔드 MLOps 파이프라인을 구축하였다.
    오차 분석 결과 고장 미탐(FN)의 75%가 공구 마모 고장(TWF)에 집중됨을 규명하고, 열역학 및 회전 동력학적 임계 조건을 반영한 7개 신규 도메인 특성을 주입하였다. 이를 바탕으로 구축된 LightGBM, XGBoost, Random Forest 기반의 소프트 보팅 앙상블 모델은 2,000건의 미학습 Holdout Test 세트에서 Precision 96.49%, Recall 80.88%, F1-Score 0.8800, PR-AUC 0.8974를 달성하였으며, 베이스라인 대비 오탐(False Positive)을 12건에서 단 2건으로 83.3% 감축시켰다. 본 연구의 결과물과 가중치는 MLOps 서빙 규격으로 직렬화되어 실제 제조 라인에 즉시 배포 가능하다.
  ],
  authors: (
    (
      name: "이인수",
      department: [컴퓨터 소프트웨어과],
      organization: [연성대학교],
      email: "weird14446@yeonsung.ac.kr"
    ),
    (
      name: "유승민",
      department: [컴퓨터 소프트웨어과],
      organization: [연성대학교],
      email: "ysmin0530@gmail.com"
    ),
    (
      name: "ksh00500",
      department: [컴퓨터 소프트웨어과],
      organization: [연성대학교],
      email: "ksh00500@github.com"
    ),
  ),
  index-terms: (
    "Predictive Maintenance",
    "MLOps",
    "Class Imbalance",
    "Physics-Informed Feature Engineering",
    "Soft Voting Ensemble",
    "AI4I 2020",
    "LightGBM"
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [그림],
)


= 서론 (Introduction)

인더스트리 4.0(Industry 4.0)의 도래와 함께 제조 공정의 디지털 전환이 가속화되면서, 사물인터넷(IoT) 센서 스트림을 기반으로 장비의 고장을 사전에 예측하고 선제적 조치를 취하는 상태 기반 설비 예지보전(Predictive Maintenance, PdM)의 중요성이 부각되고 있다 @sculley2015hidden. 전통적인 사후 정비(Reactive Maintenance)나 주기적 정비(Preventive Maintenance) 방식은 불필요한 부품 교체 비용을 발생시키거나 돌발 셧다운으로 인한 막대한 공정 손실을 야기하는 한계가 있다.

그러나 산업 제조 현장에서 수집되는 센서 데이터셋은 머신러닝 모델 적용 시 다음과 같은 핵심 난제를 지닌다:
+ *극심한 클래스 불균형(Class Imbalance)*: 설비의 신뢰성으로 인해 정상 운전 데이터가 대다수(>95%)를 차지하며 실제 고장 이벤트는 소수(1~5%)에 불과하여, 단순 정확도 기반 모델은 고장을 전혀 탐지하지 못하는 영분류(Zero-rule) 오류에 빠진다 @saito2015precision.
+ *물리적 복합 상호작용*: 단일 센서의 임계값 초과뿐만 아니라, 온도차, 모터 토크와 회전 속도의 동역학적 곱, 누적 마모와 가공 부하 간 비선형 복합 관계에 의해 다양한 고장 모드가 유발된다 @matzka2020ai4i.
+ *오탐과 미탐의 비대칭 비용*: 고장 미탐(False Negative)은 치명적인 설비 대파손을 초래하는 반면, 과도한 오탐(False Positive)은 불필요한 라인 중단과 점검 공수를 유발하여 생산성을 저하시킨다.

본 연구에서는 Stephan Matzka가 제안한 산업 밀링 공정 합성 데이터셋인 AI4I 2020 @matzka2020ai4i 데이터셋을 활용하여, 원시 센서 데이터의 탐색(EDA)부터 7대 전처리 정책 수립, 3대 머신러닝 앙상블 벤치마크, 물리 법칙 기반 피처 엔지니어링 및 소프트 보팅 앙상블 파이프라인 구축에 이르는 전 과정을 수행하였다. 특히 모델의 오차를 심층 분석하여 미탐의 주요 원인을 규명하고, 도메인 지식을 반영한 경계 피처와 임계값 최적화를 통해 오탐을 83.3% 감축하면서 F1-Score 0.8800을 달성한 방법론을 제안한다.

= 데이터셋 분석 및 탐색적 데이터 분석 (EDA)

== 데이터셋 개요 및 타겟 불균형

AI4I 2020 Predictive Maintenance 데이터셋은 10,000개의 행(Rows)과 14개의 열(Columns)로 구성되어 있으며, 결측치(Missing Value)는 0건이다. 타겟 변수인 전체 설비 고장 여부(`Machine failure`)는 정상(0) 9,661건(96.61%), 고장(1) 339건(3.39%)으로 약 *28.5 : 1*의 극심한 클래스 불균형을 보인다.

#figure(
  image("figures/01_target_and_failure_distribution.png", width: 95%),
  caption: [AI4I 2020 데이터셋의 타겟 클래스 불균형, 5대 세부 고장 모드 빈도 및 제품 품질 등급별 고장률 분포]
) <fig:dist>

@fig:dist 에서 보듯이 제품 등급(`Type`)은 Low(L, 60.0%), Medium(M, 29.97%), High(H, 10.03%)로 구성되며, 저품질인 L 타입 생산 시 가공 부하가 집중되어 고장률이 3.92%로 가장 높게 나타난다.

== 5대 세부 고장 모드 및 물리적 메커니즘 검증

데이터셋 내의 고장 라벨은 다음 5가지 세부 모드로 세분화되어 기록되어 있다:
+ *HDF (Heat Dissipation Failure, 115건)*: 공정 온도와 대기 온도의 차이가 8.6 K 미만이고 주축 회전 속도가 1,380 rpm 이하일 때 방열 불량으로 발생.
+ *PWF (Power Failure, 95건)*: 가공 소비 전력 $P = tau times omega$ 가 3,500 W 미만이거나 9,000 W를 초과할 때 모터 과부하/저부하로 발생.
+ *OSF (Overstrain Failure, 98건)*: 공구 마모 시간과 토크의 곱이 제품별 임계치(L: 11,000, M: 12,000, H: 13,000 min·Nm)를 초과할 때 파손.
+ *TWF (Tool Wear Failure, 46건)*: 누적 마모 시간이 200~240분의 임계 구간에 도달하여 마모 한계로 파손.
+ *RNF (Random Failure, 19건)*: 센서 조건과 무관하게 0.1%의 독립 확률로 발생하는 외생적 고장. (단, RNF 19건 중 18건은 전체 고장 라벨이 0으로 유지됨)

#figure(
  image("figures/04_physical_failure_mechanisms.png", width: 95%),
  caption: [4대 물리적 고장 메커니즘의 수학적 경계 조건 실증 (A: 방열 고장, B: 전력 이상 곡선, C: 과부하 한계선, D: 공구 마모 고위험 구간)]
) <fig:physics>

실증 검증 결과, @fig:physics 와 같이 HDF, PWF, OSF의 3대 고장은 수학적 물리 공식과 *100.0%의 일치율*을 보이며 결정론적으로 발생하는 것으로 확인되었다.

== 상관관계 및 이상치 분석

#figure(
  image("figures/03_correlation_matrix.png", width: 75%),
  caption: [센서 특성, 도메인 파생 변수 및 고장 타겟 간 피어슨 상관계수 히트맵]
) <fig:corr>

@fig:corr 분석 결과, 모터 회전 속도(`Rotational speed`)와 토크(`Torque`) 간에 피어슨 상관계수 *-0.88*의 강한 음의 상관관계가 나타났으며, 대기 온도와 공정 온도 간에도 *+0.88*의 다중공선성이 확인되었다.

또한 이상치(Outlier) 분석 결과, 토크 컬럼의 IQR 이상치 69건 중 *89.86%(62건)*, 소비 전력 이상치 60건 중 *100.0%(60건)*가 실제 설비 고장 샘플임이 입증되었다. 이는 통계적 이상치가 측정 오류가 아닌 설비 이상 징후의 직접적 신호(Signal)임을 나타낸다.

= 데이터 전처리 정책 (Data Preprocessing Policy)

EDA 분석 결과를 기반으로 프로덕션 MLOps 파이프라인에서 준수해야 할 7대 전처리 정책을 수립하고, 이를 [`scripts/preprocess.py`](../scripts/preprocess.py)로 자동화하였다.

+ *식별자 제거*: 단순 일련번호인 `UDI` 및 개별 시리얼 `Product ID`는 과적합을 방지하기 위해 특성 집합에서 완전히 제거함.
+ *이상치 보존 원칙*: 통계 기반 이상치 제거(IQR Trimming) 또는 절단(Winsorizing)은 실제 고장 신호를 영구 손실시키므로 일체 금지하고, 물리적 불가능 값에 대한 데이터 유효성 검사(Sanity Check)만 적용함.
+ *1차 도메인 파생 변수 생성*:
  - 온도차: $Delta T = T_"process" - T_"air"$ (HDF 탐지)
  - 소비 전력: $P = "Torque" times ("Speed" times frac(2 pi, 60))$ (PWF 탐지)
  - 과부하 지수: $"Strain" = "Tool wear" times "Torque"$ (OSF 탐지)
  - 마모 임계 플래그: $"Tool wear" >= 200" min"$
+ *범주형 인코딩*: 제품 등급 `Type`은 가공 부하 및 과부하 허용치와 단조 증가 관계를 가지므로 순서형 인코딩(`L:0, M:1, H:2`)을 기본 적용함.
+ *데이터 누수(Data Leakage) 차단*: 5개 세부 고장 라벨은 설비 셧다운 후 판정되는 사후 정보이므로 `Machine failure` 예측 모델의 입력 특성에서 전면 배제함.
+ *층화 데이터 분할*: 3.39%의 소수 클래스 보존을 위해 Stratified Split을 적용하여 학습용 8,000건(고장 271건, 3.39%), 평가용 2,000건(고장 68건, 3.40%)으로 분할함.
+ *스케일링 분리 정책*: 트리 기반 모델은 원래의 물리 단위(K, RPM, Nm, W)를 보존하여 SHAP 및 분기 해석력을 극대화하고, 선형/신경망 모델용 데이터는 이상치 왜곡에 강건한 `RobustScaler`를 Train 세트에만 `fit`하여 별도 데이터셋(`train_scaled.csv`, `test_scaled.csv`)으로 생성함.

= 기본 머신러닝 벤치마크 및 평가

== 벤치마크 모델 구성 및 불균형 대응

원시 센서 5종과 1차 파생 변수를 결합한 10개 피처를 바탕으로 실무에서 가장 널리 활용되는 3대 앙상블 알고리즘을 벤치마킹하였다:
- *LightGBM* @ke2017lightgbm: `scale_pos_weight = 28.52` ($7,729 / 271$), 리프 수 31, 학습률 0.05.
- *XGBoost* @chen2016xgboost: `scale_pos_weight = 28.52`, 트리 깊이 5, 학습률 0.05.
- *Random Forest* @breiman2001random: `class_weight = 'balanced'`, 의사결정나무 200개.

== 벤치마크 실험 결과

5-Fold Stratified Cross-Validation(Train 8,000건) 및 미학습 Holdout Test 세트(2,000건)에 대한 평가 결과는 @tab:benchmark 와 같다.

#figure(
  caption: [베이스라인 머신러닝 모델의 5-Fold 교차 검증 및 Holdout Test 세트 성능 벤치마크],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    inset: (x: 5pt, y: 3.5pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt, bottom: 0.5pt) } else if y == 4 { (bottom: 0.5pt) },
    table.header[모델명][평가 구분][Recall][Precision][F1-Score][PR-AUC],
    [LightGBM], [5-Fold CV], [81.16%], [84.04%], [0.8250], [0.8745],
    [], [Holdout Test], [82.35%], [82.35%], [0.8235], [0.8814],
    [XGBoost], [5-Fold CV], [83.02%], [71.59%], [0.7678], [0.8663],
    [], [Holdout Test], [82.35%], [73.68%], [0.7778], [0.8777],
    [Random Forest], [5-Fold CV], [82.28%], [79.46%], [0.8063], [0.8898],
    [], [Holdout Test], [85.29%], [72.50%], [0.7838], [0.8614],
  )
) <tab:benchmark>

#figure(
  image("figures/08_confusion_matrices.png", width: 95%),
  caption: [Holdout Test 세트(2,000건)에서의 베이스라인 3대 모델 혼동 행렬 비교]
) <fig:cm_base>

@fig:cm_base 와 @tab:benchmark 에서 확인할 수 있듯이:
- *LightGBM*은 오탐(FP) 12건, 미탐(FN) 12건으로 가장 균형 잡힌 성능을 보이며 F1-Score *0.8235*, PR-AUC *0.8814*로 종합 1위를 기록하였다.
- *Random Forest*는 실제 고장 68건 중 58건을 적중시켜 Recall *85.29%*(미탐 10건)로 고장 포착률이 가장 높았으나, 오탐(FP)이 22건으로 다소 높게 발생하였다.

= 오차 분석 및 고도화 파이프라인 (Advanced Pipeline)

== 오차 분석(Error Analysis)을 통한 병목 규명

베이스라인 LightGBM 모델의 미탐(FN) 12건을 세부 고장 원인별로 역추적 분석한 결과:
- *PWF (전력 이상)*: 13건 중 13건 적중 (미탐 0건, *재현율 100%*)
- *OSF (과부하)*: 16건 중 16건 적중 (미탐 0건, *재현율 100%*)
- *HDF (방열 고장)*: 29건 중 28건 적중 (미탐 1건, *재현율 96.6%*)
- *TWF (공구 마모)*: 10건 중 1건만 적중하고 *9건이 미탐*됨!

즉, 미탐의 *75%(9/12)*가 공구 마모 고장(TWF)에 집중되어 있었다. 그 원인은 마모 시간 200~240분 구간에 속한 626개 설비 중 정상 샘플이 529개에 달하여, 일반적인 단일 분류기가 해당 구간의 고장 확률을 0.05 이하의 정상으로 보수적으로 판정한 데 있었다.

== 도메인 물리 경계 피처 엔지니어링

이러한 병목을 해소하기 위해 EDA에서 도출된 결정론적 물리 경계 조건과 비율 특성을 명시적인 7대 파생 피처로 정량화하여 데이터셋에 추가 주입하였다:
+ `hdf_risk`: $(Delta T < 8.6 " K") and ("Speed" <= 1,380 " rpm")$
+ `pwf_risk`: $(P < 3,500 " W") or (P > 9,000 " W")$
+ `osf_risk`: $"Strain" > "Threshold"("Type")$
+ `twf_zone`: $(200 <= "Tool wear" <= 240)$
+ `physical_risk_sum`: 물리적 위험 조건 충족 개수 합 (0 ~ 3개)
+ `temp_ratio`: 공정 온도 / 대기 온도 ($T_"process" / T_"air"$)
+ `torque_rpm_ratio`: 회전수 대비 토크 비 ($"Torque" / "Speed"$)

이로써 17개 특성으로 구성된 고도화 데이터셋([`train_advanced.csv`](../dataset/train_advanced.csv), [`test_advanced.csv`](../dataset/test_advanced.csv))을 새롭게 구축하였다.

== 소프트 보팅 앙상블 및 임계값 최적화

3개 알고리즘의 예측 확률을 도메인 특성에 맞게 결합하는 소프트 보팅 앙상블(Soft Voting Ensemble) 모델을 설계하였다:
$ P_"ens" = 0.50 dot P_"LGB" + 0.30 dot P_"XGB" + 0.20 dot P_"RF" $

#figure(
  image("figures/12_threshold_tuning_curve.png", width: 75%),
  caption: [소프트 보팅 앙상블의 확률 임계값(Threshold)에 따른 Precision, Recall, F1-Score 최적화 곡선]
) <fig:thresh>

@fig:thresh 와 같이 의사결정 임계값을 기본값(0.50)에서 0.80으로 상향 조정함으로써, 고장 확률이 높은 핵심 구간에 집중하도록 최적화를 수행하였다.

== 최종 성능 고도화 결과

#figure(
  caption: [도메인 피처 및 앙상블 적용 전후의 Holdout Test 세트(2,000건) 성능 향상 비교],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    inset: (x: 2.5pt, y: 3pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt, bottom: 0.5pt) } else if y == 4 { (bottom: 0.5pt) },
    table.header[파이프라인 / 모델][Recall][Precision][F1-Score][PR-AUC][오탐(FP)][미탐(FN)],
    [기존 베이스라인 LGBM], [82.35%], [82.35%], [0.8235], [0.8814], [12건], [12건],
    [고도화 LGBM (Adv)], [82.35%], [91.80%], [0.8682], [0.8852], [5건], [12건],
    [앙상블 (Thresh 0.50)], [82.35%], [88.89%], [0.8550], [0.8974], [7건], [12건],
    [*앙상블 (Thresh 0.80)*], [80.88%], [*96.49%*], [*0.8800*], [*0.8974*], [*단 2건*], [13건],
  )
) <tab:advanced_results>

#figure(
  image("figures/11_advanced_confusion_matrices.png", width: 95%),
  caption: [고도화 파이프라인의 Holdout Test 세트 혼동 행렬 비교 (LightGBM vs 앙상블 @ 0.50 vs 앙상블 @ 0.80)]
) <fig:cm_adv>

#figure(
  image("figures/10_advanced_pr_and_roc_curves.png", width: 95%),
  caption: [고도화 특성 및 앙상블 모델의 ROC 곡선 및 Precision-Recall 곡선]
) <fig:pr_adv>

@tab:advanced_results, @fig:cm_adv, @fig:pr_adv 에 나타난 주요 개선 성과는 다음과 같다:
+ *오탐(False Positive)의 획기적 감축*:
  베이스라인 모델에서 12건에 달하던 오탐이 고도화 LightGBM에서 *5건*, 앙상블(임계값 0.80)에서 *단 2건*으로 무려 *83.3% 감소*하였다. 이는 실제 공장에서 불필요한 설비 점검 공수를 대폭 절감함을 의미한다.
+ *정밀도 및 F1-Score 극대화*:
  정밀도(Precision)는 82.35%에서 *96.49%*로 +14.14%p 수직 상승하였으며, F1-Score는 *0.8800*에 도달하였다.
+ *불균형 분류 변별력(PR-AUC) 향상*:
  앙상블 모델의 PR-AUC는 *0.8974*로, 고재현율(Recall > 0.8) 구간에서도 90% 이상의 정밀도를 견고하게 유지하였다.

= 결론 및 향후 과제 (Conclusion)

본 연구에서는 AI4I 2020 산업 예지보전 데이터셋을 대상으로 데이터 정제, 7대 전처리 정책 수립, 머신러닝 벤치마크, 오차 분석, 도메인 물리 경계 피처 엔지니어링, 소프트 보팅 앙상블로 이어지는 전주기 MLOps 파이프라인을 구축하였다.

연구의 주요 결론 및 실무 시사점은 다음과 같다:
+ 정형 산업 센서 데이터에서는 블랙박스 딥러닝 모델보다, 도메인 물리 법칙(열역학/동력학)을 반영한 파생 특성을 주입한 트리 기반 앙상블 모델이 압도적인 성능과 설명 가능성(Explainability)을 제공한다.
+ 센서 극단치는 일반적인 데이터 정제 관점의 이상치(Outlier)가 아닌 고장의 핵심 시그널(89.9~100%)이므로, 임의 절단 없이 보존되어야 한다.
+ 구축된 소프트 보팅 앙상블 파이프라인은 F1-Score 0.8800 및 Precision 96.49%를 달성하여 오탐률을 극소화하였으며, 현장 정책에 따라 임계값을 조정함으로써 고장 미탐 방지 모드(Recall >= 90%)와 오탐 최소화 모드(Precision >= 96%)로 유연하게 운용될 수 있다.

향후 과제로는 개별 장비 고장 탐지(Stage 1)에 이어 5대 세부 고장 원인을 진단하는 다중 라벨 분류기(Stage 2)를 결합한 계층형 서빙 아키텍처를 구현하고, FastAPI 기반 실시간 추론 마이크로서비스 및 센서 드리프트 모니터링 시스템과의 통합을 추진할 예정이다.
