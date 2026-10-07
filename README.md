# AI4I 2020 설비 예지보전 Cloud MLOps 프로젝트

본 저장소는 UCI Machine Learning Repository의 **AI4I 2020 Predictive Maintenance 데이터셋**을 기반으로 구축된 **엔드투엔드 설비 예지보전(Predictive Maintenance, PdM) MLOps 파이프라인**입니다.

데이터 탐색(EDA), 도메인 물리 기반 전처리, 모델 학습 및 Optuna 하이퍼파라미터 튜닝, 계층형 아키텍처 기반의 **FastAPI 추론 서버**, 비전문가를 위한 **React 19 + Vite + TypeScript 웹 대시보드**, 그리고 AWS EC2 즉시 배포를 위한 **Docker Compose 멀티 컨테이너 오케스트레이션 및 `.env` 포트 제어 시스템**까지 프로덕션 수준으로 구현되어 있습니다.

---

## 📌 목차
1. [데이터셋 개요 및 분석 (Dataset)](#1-데이터셋-개요-및-분석-dataset)
2. [데이터 전처리 정책 및 피처 엔지니어링 (Preprocessing)](#2-데이터-전처리-정책-및-피처-엔지니어링-preprocessing)
3. [모델 학습 및 하이퍼파라미터 튜닝 방법 (Training & Tuning)](#3-모델-학습-및-하이퍼파라미터-튜닝-방법-training--tuning)
4. [모델 성능 평가 및 벤치마크 (Evaluation)](#4-모델-성능-평가-및-벤치마크-evaluation)
5. [서버 아키텍처 상세 설명 (Server Architecture)](#5-서버-아키텍처-상세-설명-server-architecture)
6. [Docker 컨테이너화 및 배포 환경 (Docker & EC2 Orchestration)](#6-docker-컨테이너화-및-배포-환경-docker--ec2-orchestration)
7. [실행 및 배포 가이드 (Quickstart)](#7-실행-및-배포-가이드-quickstart)
8. [관련 문서 및 리소스 링크](#8-관련-문서-및-리소스-링크)

---

## 1. 데이터셋 개요 및 분석 (Dataset)

- **데이터 출처**: Stephan Matzka (HTW Berlin, 2020), [UCI Machine Learning Repository: AI4I 2020 Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **데이터 파일**: [`dataset/ai4i2020.csv`](dataset/ai4i2020.csv) (10,000행 × 14열, 결측치 **0건**)
- **데이터 특성**: 실제 CNC 밀링 머신(Milling Machine)의 가공 물리학 수식 및 도메인 규칙을 정밀하게 반영한 합성 데이터셋(Synthetic Dataset)
- **클래스 불균형 (Class Imbalance)**:
  - **정상 (`0`)**: 9,661건 (**96.61%**)
  - **고장 (`1`)**: 339건 (**3.39%**)
  - 불균형 비율이 약 **28.5 : 1**에 달해, 무조건 정상으로 예측해도 96.6% 정확도가 도출되는 '정확도 역설'이 존재합니다.

### 1.1 피처 명세 (Feature Dictionary)
| 컬럼명 | 데이터 타입 | 단위 | 역할 | 설명 |
| :--- | :---: | :---: | :---: | :--- |
| **`UDI`** | Integer | - | 메타데이터 | 1부터 10,000까지의 고유 레코드 식별 번호 (학습 시 제외) |
| **`Product ID`** | String | - | 메타데이터 | 품질 등급(`L`/`M`/`H`) + 일련번호 (학습 시 제외) |
| **`Type`** | String | - | 범주형 특성 | 제품 품질 변형 등급 (`L`: 60%, `M`: 30%, `H`: 10%) |
| **`Air temperature [K]`** | Float | K | 연속형 센서 | 외부 대기 환경 온도 (평균: 300.0 K, 표준편차: 2.0 K) |
| **`Process temperature [K]`** | Float | K | 연속형 센서 | 가공 중 마찰열이 반영된 설비 온도 (평균: 310.0 K) |
| **`Rotational speed [rpm]`** | Integer | RPM | 연속형 센서 | 주축(Spindle) 분당 회전수 (1,168 ~ 2,886 rpm) |
| **`Torque [Nm]`** | Float | N·m | 연속형 센서 | 모터에 인가되는 돌림힘/토크 (3.8 ~ 76.6 Nm) |
| **`Tool wear [min]`** | Integer | 분 | 연속형 센서 | 절삭 공구의 누적 사용 시간 (0 ~ 253분) |
| **`Machine failure`** | Binary (0/1) | - | **메인 타겟** | 종합 장비 고장 발생 여부 (0: 정상, 1: 고장) |
| **`TWF`** | Binary (0/1) | - | 세부 고장 라벨 | 공구 마모 고장 (Tool Wear Failure, 46건) |
| **`HDF`** | Binary (0/1) | - | 세부 고장 라벨 | 열 방출 실패 고장 (Heat Dissipation Failure, 115건) |
| **`PWF`** | Binary (0/1) | - | 세부 고장 라벨 | 전력 이상 고장 (Power Failure, 95건) |
| **`OSF`** | Binary (0/1) | - | 세부 고장 라벨 | 과부하 고장 (Overstrain Failure, 98건) |
| **`RNF`** | Binary (0/1) | - | 세부 고장 라벨 | 무작위 고장 (Random Failure, 19건) |

### 1.2 5대 물리적 고장 메커니즘
1. **HDF (방열 고장)**: 온도차 $\Delta T < 8.6\text{ K}$ 이고 회전수 $\le 1,380\text{ rpm}$ 일 때 냉각 실패로 과열 발생.
2. **PWF (전력 고장)**: 소비 전력 $P = \text{Torque} \times (\text{Speed} \times \frac{2\pi}{60})$ 이 $3,500\text{ W}$ 미만 또는 $9,000\text{ W}$ 초과 시 발생.
3. **OSF (과부하 고장)**: 공구 마모와 토크의 곱($\text{Strain} = \text{Tool wear} \times \text{Torque}$)이 품질 등급별 한계(`L`: 11,000, `M`: 12,000, `H`: 13,000 min·Nm) 초과 시 발생.
4. **TWF (공구 마모)**: 누적 마모 시간이 200~240분의 임계 피로 구간에 도달할 때 확률적으로 파손.
5. **RNF (무작위 고장)**: 센서 조건과 독립적으로 0.1% 확률로 발생하는 외생적 요인.

---

## 2. 데이터 전처리 정책 및 피처 엔지니어링 (Preprocessing)

EDA 및 물리 공식 검증을 기반으로 수립된 **7대 공식 전처리 정책**([`notes/preprocessing_policy.md`](notes/preprocessing_policy.md))을 엄격히 준수합니다.

```
       [원시 데이터] ───► [1. 식별자(UDI, Product ID) 제거 & 네이밍 정규화]
                              │
                              ▼
                         [2. 이상치 보존 (고장 신호 보존, IQR Drop 금지)]
                              │
                              ▼
                         [3. 도메인 물리 파생 피처 엔지니어링 (ΔT, Power, Strain)]
                              │
                              ▼
                         [4. 범주형 Type 순서형 인코딩 (L:0, M:1, H:2)]
                              │
                              ▼
                         [5. 데이터 누수 차단 (세부 5대 고장 라벨 Feature 배제)]
                              │
                              ▼
                         [6. Stratified Split (80:20 불균형 층화 분할)]
                              │
                              ▼
                         [7. 모델군별 스케일링 정책 (Tree: 원본 보존 / 선형: RobustScaler)]
```

### 2.1 도메인 기반 물리 파생 피처 생성
- **온도차 (`temp_diff_k`)**:
  $$\Delta T = \text{Process temperature [K]} - \text{Air temperature [K]}$$
  - HDF 고장 판별 및 두 온도 센서 간 다중공선성($r=+0.88$) 완화.
- **기계적 소비 전력 (`power_w`)**:
  $$P = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$$
  - PWF 고장 판별 및 토크-속도 간 강한 음의 상관관계($r=-0.88$) 통합.
- **과부하/변형 지수 (`strain_min_nm`)**:
  $$\text{Strain} = \text{Tool wear [min]} \times \text{Torque [Nm]}$$
  - OSF 고장 발생 임계선 직접 포착.
- **공구 마모 위험 임계 지시자 (`tool_wear_critical`)**:
  - `tool_wear_min >= 200` 여부의 이진 플래그 (TWF 고위험군 마킹).

### 2.2 이상치 보존 원칙 (Outlier Handling)
`Torque`의 통계적 이상치(69건) 중 **89.86%(62건)**가 실제 고장이며, `Power`의 이상치(60건)는 **100% 전건이 실제 고장(`PWF`)**입니다. 따라서 통계 기반의 임의 절단(Winsorizing)이나 제거를 금지하고 센서 극단값을 고장 시그널로 온전히 보존합니다.

### 2.3 데이터 누수(Data Leakage) 원천 차단
현장 설비 운영 시 셧다운 이전에는 세부 고장 원인을 알 수 없으므로, 5개 세부 고장 라벨(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)은 모델 입력 특성에서 **완전히 배제**하고 오직 순수 센서 수치와 파생 물리 피처만 사용합니다.

---

## 3. 모델 학습 및 하이퍼파라미터 튜닝 방법 (Training & Tuning)

### 3.1 모델군 선정
산업 현장의 비선형 센서 상호작용 및 고장 임계값 분기 특성을 효과적으로 학습할 수 있는 앙상블 트리 계열 모델을 주력으로 선정하였습니다:
- **LightGBM**: 빠른 추론 속도, 리프 중심(Leaf-wise) 분할을 통한 복합 고장 패턴 학습 최적화
- **XGBoost**: 정규화 페널티($\gamma$, $\alpha$, $\lambda$)를 통한 오탐 억제력 우수
- **Random Forest**: 배깅(Bagging) 기반 다수결 투표로 높은 일반화 안정성 및 미탐(FN) 최소화

### 3.2 클래스 불균형 제어 및 최적화 전략
- **알고리즘 수준 불균형 보정**: `scale_pos_weight` ($9661 / 339 \approx 28.5$ 범위 내 튜닝) 및 `class_weight='balanced'` 적용
- **베이지안 최적화 (Optuna TPE Sampler)**: [`scripts/tune_hyperparameters.py`](scripts/tune_hyperparameters.py)
  - **교차 검증**: 5-Fold Stratified K-Fold (난수 시드 42 고정)
  - **목적 함수**: Out-Of-Fold **PR-AUC (Precision-Recall Area Under Curve)** 극대화
- **의사결정 임계값($\tau^*$) 최적화**:
  - 테스트 데이터셋 누출(Data Leakage)을 차단하기 위해, **학습 세트 내부의 Out-Of-Fold(OOF) 교차 검증 예측 점수**만을 사용하여 F1-Score를 극대화하는 최적 임계값을 도출했습니다.

### 3.3 모델별 최적 하이퍼파라미터
- **LightGBM**: `n_estimators=250`, `learning_rate=0.0155`, `num_leaves=19`, `max_depth=10`, `min_child_samples=22`, `subsample=0.954`, `colsample_bytree=0.722`, `reg_alpha=0.247`, `reg_lambda=1.245`, `scale_pos_weight=1.086`
- **XGBoost**: `n_estimators=250`, `learning_rate=0.0111`, `max_depth=3`, `min_child_weight=2`, `subsample=0.815`, `colsample_bytree=0.954`, `gamma=1.674`, `reg_alpha=0.024`, `reg_lambda=0.034`, `scale_pos_weight=1.166`
- **Random Forest**: `n_estimators=250`, `max_depth=14`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`, `class_weight='balanced'`

---

## 4. 모델 성능 평가 및 벤치마크 (Evaluation)

### 4.1 정확도(Accuracy) 및 다각적 성능 분석
3.39%의 극심한 불균형 환경에서는 "모두 정상"으로 예측해도 96.6%의 정확도가 나오는 한계가 있으므로, 정확도와 함께 **오탐(FP)**, **미탐(FN)**, **PR-AUC**, **F1-Score**를 종합 검증했습니다.

- **LightGBM (최종 서빙 선정)**: 튜닝 후 Holdout 테스트 정확도 **99.10% (1,982건 정상 분류 / 2,000건)** 달성. 오탐이 6건에서 **4건으로 33% 감소**하여 정밀도 **93.10%**, PR-AUC **0.8978**을 기록했습니다.
- **XGBoost**: 튜닝 후 오탐이 **단 1건**으로 극적으로 억제되며 정밀도 **98.08%**, 정확도 **99.10%**를 기록했습니다.
- **Random Forest**: 튜닝 및 OOF 최적 임계값 적용 시 정확도 **99.00%**, 총 오류 건수가 32건에서 **20건으로 37.5% 대폭 감소**했습니다.

### 4.2 Holdout Test Set 최종 성능 비교 (미학습 2,000건: 정상 1,932 / 고장 68)

| 모델 | 평가 모드 | Accuracy (정확도) | Precision (정밀도) | Recall (재현율) | F1-Score | PR-AUC | 혼동 행렬 (TN / FP / FN / TP) | 총 오류 (FP+FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 튜닝 전 (기준 0.50) | 99.10% | 90.32% | 82.35% | 0.8615 | 0.8928 | 1926 / 6 / 12 / 56 | 18건 |
| | **튜닝 후 (기준 0.50)** | **99.10%** | **93.10%** | 79.41% | 0.8571 | **0.8978** | 1928 / **4** / 14 / 54 | 18건 (**오탐 33% 감소**) |
| | **튜닝 후 (OOF 최적 0.48)** | **99.10%** | 91.67% | 80.88% | **0.8594** | **0.8978** | 1927 / 5 / 13 / 55 | 18건 |
| **XGBoost** | 튜닝 전 (기준 0.50) | 99.05% | 90.16% | 80.88% | 0.8527 | 0.9015 | 1926 / 6 / 13 / 55 | 19건 |
| | **튜닝 후 (기준 0.50)** | **99.10%** | **98.08%** | 75.00% | 0.8500 | 0.8991 | 1931 / **1** / 17 / 51 | 18건 (**오탐 단 1건**) |
| | **튜닝 후 (OOF 최적 0.24)** | **99.00%** | 85.29% | **85.29%** | **0.8529** | 0.8991 | 1922 / 10 / 10 / 58 | 20건 |
| **Random Forest** | 튜닝 전 (기준 0.50) | 98.40% | 72.50% | 85.29% | 0.7838 | 0.8608 | 1910 / 22 / 10 / 58 | 32건 |
| | **튜닝 후 (기준 0.50)** | **98.60%** | **77.03%** | 83.82% | **0.8028** | **0.8709** | 1915 / 17 / 11 / 57 | 28건 |
| | **튜닝 후 (OOF 최적 0.56)** | **99.00%** | **87.50%** | 82.35% | **0.8485** | **0.8709** | 1924 / **8** / 12 / 56 | **20건 (-12건 감소)** |

> 📊 **성능 비교 시각화 차트**: [`figures/14_hyperparameter_tuning_comparison.png`](figures/14_hyperparameter_tuning_comparison.png)  
> (CV F1 비교, PR-AUC 비교, 오류 감소량 차트, PR 곡선 수록)

---

## 5. 서버 아키텍처 상세 설명 (Server Architecture)

본 프로젝트는 프로덕션 서빙 품질을 보장하기 위해 관심사 분리(Separation of Concerns)를 구현한 **계층형 FastAPI 백엔드**와 직관적인 UX를 제공하는 **React 19 웹 프론트엔드**로 구성되어 있습니다.

```text
API/
├── __init__.py              # 패키지 초기화 및 하위 호환성 심볼 노출
├── api.py                   # 메인 진입점, lifespan 컨텍스트 매니저, 전역 CORS 미들웨어 구성
├── config.py                # Pydantic Settings 기반 환경 변수 관리 (MODEL_PATH, THRESHOLD 등)
├── schemas.py               # Pydantic v2 데이터 검증 모델 (SensorInput, Prediction, HealthResponse)
├── routers/                 # RESTful 라우팅 모듈
│   ├── health.py            # GET /health (서비스 상태 및 모델 로딩 검사)
│   └── predict.py           # POST /predict (센서 데이터 실시간 고장 추론)
├── services/                # 비즈니스 로직 계층
│   ├── feature_service.py   # 센서 입력 기반 물리 파생 피처(온도차, 전력, 스트레인) 실시간 계산
│   └── model_service.py     # LightGBM Booster 인메모리 로더 및 스레드 세이프 추론 엔진
├── requirements-api.txt     # 백엔드 프로덕션 의존성 목록
└── tests/                   # 10개 단위/통합 테스트 스위트
```

### 5.1 FastAPI 백엔드 주요 모듈 및 엔드포인트 명세
1. **`POST /predict` (실시간 설비 고장 예측)**:
   - 클라이언트로부터 6대 센서 수치를 전달받아 `Pydantic` 스키마로 유효 범위를 엄격히 검증합니다.
   - `feature_service`가 온도차($\Delta T$), 회전 전력($P$), 누적 스트레인 지수를 즉각 계산하여 10개 입력 피처로 변환합니다.
   - 인메모리에 로드된 `LightGBM` 모델이 고장 확률($0.0 \sim 1.0$)과 이진 분류 라벨($0$: 정상, $1$: 고장)을 반환합니다.
2. **`GET /health` (서비스 및 모델 헬스체크)**:
   - 컨테이너 및 오케스트레이터의 Liveness/Readiness Probe를 위해 서버 상태(`status: ok`), 모델 로드 여부(`model_loaded: true`), 버전 정보를 반환합니다.
3. **`GET /docs` & `GET /openapi.json` (대화형 API 문서)**:
   - Swagger UI 및 OpenAPI 규격을 통해 브라우저에서 직접 파라미터 테스트가 가능합니다.
4. **전역 CORS 지원**:
   - 프론트엔드 및 외부 클라이언트 통신을 위해 `CORSMiddleware`(`allow_origins=["*"]`)가 적용되어 있습니다.

### 5.2 React 19 + Vite + TypeScript 프론트엔드 ([`frontend/`](frontend/))
현장 엔지니어 및 비전문가도 머신러닝 코드를 몰라도 즉각 설비 상태를 모니터링할 수 있도록 설계된 산업용 웹 애플리케이션입니다:
- **1-클릭 고장 시나리오 프리셋 (`Presets.tsx`)**:
  - `정상 가동 설비`: 표준 공정 상태 (고장 확률 0.1% 미만)
  - `열 방출 실패 (HDF)`: 저속 회전 및 냉각 불량 조건 반영
  - `과부하/마모 (OSF/TWF)`: 공구 마모 215분 및 고토크 인가 조건 반영
  - `전력 이상 (PWF)`: 9,000W 초과 고전력 조건 반영
- **도메인 물리 파생 지표 실시간 계산 (`DerivedMetrics.tsx`)**:
  - 폼 수치 변경 시 온도차, 기계 소비 전력, 누적 스트레인을 즉시 계산하여 위험 구간 도달 여부를 시각적 뱃지로 표시합니다.
- **직관적 0~100% 게이지 및 현장 조치 권고사항 (`PredictionResult.tsx`)**:
  - 안전 단계(초록, <25%), 주의 단계(노랑, 25~50%), 위험 단계(빨강, $\ge 50\%$)의 3단계 가동 상태 분류 및 구체적 정비 권고안 제시.
- **추론 이력 테이블 (`HistoryTable.tsx`)**:
  - 세션 동안 수행된 추론 이력을 보존하며, '불러오기' 클릭 시 이전 센서 파라미터를 폼으로 1-클릭 복원합니다.

---

## 6. Docker 컨테이너화 및 배포 환경 (Docker & EC2 Orchestration)

본 프로젝트는 AWS EC2 등 클라우드 인프라에서 환경 설정의 번거로움 없이 **단 한 줄의 명령어로 즉시 프로덕션 서비스를 띄울 수 있도록 Docker 멀티 컨테이너 환경**을 완비하였습니다.

```
                  [사용자 브라우저 / 외부 클라이언트]
                                 │
              ┌──────────────────┴──────────────────┐
              │ (PORT: ${FRONTEND_PORT} - 기본 80)   │ (PORT: ${BACKEND_PORT} - 기본 8000)
              ▼                                     ▼
        [mlops-frontend]                      [mlops-backend]
     (Nginx:alpine 리버스 프록시)            (FastAPI Uvicorn)
        ├─ React 19 정적 번들 서빙            ├─ LightGBM 추론 엔진
        ├─ SPA 라우팅 fallback                ├─ OpenMP 병렬 연산 (libgomp1)
        └─ 리버스 프록시 라우팅                └─ 헬스체크 엔드포인트
            (/predict, /health, /docs)
              └───────── Docker Bridge Network ─────────┘
```

### 6.1 백엔드 컨테이너 ([`Dockerfile`](Dockerfile))
- **베이스 이미지**: `python:3.11-slim` 기반 경량화 이미지 (약 180MB)
- **Linux LightGBM 필수 라이브러리**: OpenMP 병렬 런타임인 `libgomp1` 및 헬스체크용 `curl` 설치
- **보안 강화**: 루트 권한 탈취 방지를 위해 비특권 유저(`appuser`, UID 1000) 생성 및 실행
- **무중단 모델 가중치 반영**: `./models:/app/models:ro` 볼륨 마운트를 통해 컨테이너 재빌드 없이 호스트의 모델 파일 갱신 지원
- **헬스체크 정의**: `CMD curl -f http://localhost:8000/health || exit 1`

### 6.2 프론트엔드 컨테이너 ([`frontend/Dockerfile`](frontend/Dockerfile) & [`frontend/nginx.conf`](frontend/nginx.conf))
- **멀티 스테이지 빌드**:
  - 1단계: `node:20-alpine` 환경에서 의존성 설치 및 `npm run build` 번들링 수행
  - 2단계: 초경량 `nginx:alpine` 이미지로 정적 산출물(`dist`)만 복사하여 최종 컨테이너 크기 최소화 (약 25MB)
- **Nginx 리버스 프록시**:
  - 브라우저가 포트 80 하나만으로 React 웹 UI와 백엔드 API(`/predict`, `/health`, `/docs`)를 동시에 호출할 수 있도록 라우팅
  - 브라우저의 CORS 및 방화벽 포트 개방 제약 문제 원천 해결
  - SPA 새로고침 지원 (`try_files $uri $uri/ /index.html`) 및 정적 파일 Gzip 압축 / 1년 장기 캐싱

### 6.3 `.env` 기반 동적 포트 제어 시스템 ([`.env`](.env))
[`docker-compose.yml`](docker-compose.yml)은 루트 디렉토리의 [`.env`](.env) 파일을 자동으로 감지하여 호스트에 노출할 포트를 바인딩합니다:

```env
# 웹 프론트엔드 외부 서비스 포트 (기본값: 80 -> 웹 브라우저에서 포트 번호 없이 접속)
# 필요에 따라 8080, 3000, 5173 등으로 자유롭게 변경 가능
FRONTEND_PORT=80

# 백엔드 API / Swagger 외부 노출 포트 (기본값: 8000)
BACKEND_PORT=8000

# 추론 모델 가중치 파일 경로 및 고장 판정 기본 임계값
MODEL_PATH=/app/models/lightgbm_model.txt
THRESHOLD=0.5
```

> [!TIP]
> 포트를 변경할 때는 [`.env`](.env) 파일의 수치만 수정한 뒤 `docker compose up -d`를 실행하면 컨테이너 재빌드 없이 수 초 내로 변경된 포트로 재배치됩니다.

### 6.4 AWS EC2 배포 최적화 스크립트 ([`scripts/deploy_ec2.sh`](scripts/deploy_ec2.sh))
- **t2.micro / t3.micro 프리티어 OOM 방지**:
  - AWS 프리티어 인스턴스의 1GB RAM 환경에서는 Docker 빌드 중 메모리 고갈로 서버가 다운되는 현상이 자주 발생합니다.
  - `deploy_ec2.sh`는 시스템 메모리가 1.8GB 미만일 경우 **2GB 스왑 메모리(`/swapfile`)를 전자동으로 구성**하여 OOM 크래시를 원천 방지합니다.
- **배포 자동화**: Docker 데몬 가동 확인, `docker compose up -d --build` 실행, 서비스 헬스체크 폴링, 그리고 EC2 퍼블릭 IP를 자동 조회하여 최종 접속 URL을 출력합니다.

---

## 7. 실행 및 배포 가이드 (Quickstart)

### 7.1 Docker Compose를 통한 원클릭 실행 (추천)
호스트 포트는 [`.env`](.env) 파일에서 자유롭게 변경할 수 있습니다 (기본값: 프론트엔드 80, 백엔드 8000).

```bash
# 1. 저장소 복제 및 브랜치 이동
git clone https://github.com/YSU-CloudMLOps/cloud-mlops.git
cd cloud-mlops
git checkout test

# 2. (선택사항) 포트 변경 필요 시 .env 파일 수정
nano .env

# 3. 도커 컨테이너 빌드 및 백그라운드 구동
docker compose up -d --build
```
- **웹 UI 접속**: `http://localhost` (또는 `http://<호스트IP>:<FRONTEND_PORT>`)
- **Swagger API 문서**: `http://localhost:8000/docs` (또는 Nginx 프록시를 통해 `http://localhost/docs`)
- **헬스체크**: `http://localhost:8000/health`

### 7.2 AWS EC2 인스턴스 원터치 배포
```bash
# EC2 인스턴스 터미널에서 실행 (루트/sudo 권한 권장)
sudo ./scripts/deploy_ec2.sh
```
> 세부 보안 그룹 설정(포트 80, 8000 인바운드 규칙 등)은 [`DOCKER_EC2_GUIDE.md`](DOCKER_EC2_GUIDE.md)를 참조하세요.

### 7.3 도커 운영 및 유지보수 명령어 치트시트
```bash
# 실행 중인 컨테이너 상태 및 헬스체크 확인
docker compose ps

# 실시간 컨테이너 로그 스트리밍
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend

# 서비스 재시작
docker compose restart

# 서비스 완전 중지 및 리소스 정리
docker compose down

# 새 모델 가중치 무중단 반영 (컨테이너 재빌드 불필요)
cp my_new_model.txt models/lightgbm_model.txt
docker compose restart backend
```

### 7.4 로컬 개발 환경 직접 실행

**FastAPI 백엔드:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r API/requirements-api.txt
PYTHONPATH=. python -m uvicorn API.api:app --host 127.0.0.1 --port 8000 --reload
```

**React 프론트엔드:**
```bash
cd frontend
npm install
npm run dev
```

**단위 및 통합 테스트 검증:**
```bash
PYTHONPATH=. python -m unittest discover -s API/tests -v
```

---

## 8. 관련 문서 및 리소스 링크
- **[AWS EC2 배포 가이드 (DOCKER_EC2_GUIDE.md)](DOCKER_EC2_GUIDE.md)**: EC2 보안 그룹 설정, Docker 설치 및 컨테이너 관리 매뉴얼
- **[Week 06 종합 프로젝트 보고서 (week06.md)](week06.md)**: 하이퍼파라미터 튜닝, 평가 지표, 프론트엔드 구축 및 Docker 연동 상세 보고서
- **[전처리 정책 보고서 (notes/preprocessing_policy.md)](notes/preprocessing_policy.md)**: 7대 전처리 원칙 및 EDA 검증 보고서
- **[데이터셋 상세 명세 (notes/dataset_info.md)](notes/dataset_info.md)**: 기초 통계 및 세부 고장 발생 조건 정리
- **[하이퍼파라미터 튜닝 스크립트 (scripts/tune_hyperparameters.py)](scripts/tune_hyperparameters.py)**: Optuna 5-Fold Stratified CV 파이프라인
- **[IEEE 학술 연구 보고서 (charged-ieee/main.pdf)](charged-ieee/main.pdf)**: 설비 예지보전 MLOps 파이프라인 연구 논문
