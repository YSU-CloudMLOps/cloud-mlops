# Week 06 — 하이퍼파라미터 튜닝 및 웹 서비스(React + Vite + TS) 구축 보고서

본 문서는 **Week 06** 실습 및 프로젝트 과제로 진행된 **머신러닝 모델 하이퍼파라미터 튜닝(Optuna 기반)**, **정확도(Accuracy)를 포함한 다각적 성능 평가**, 그리고 일반 사용자를 위한 **React + Vite + TypeScript 기반의 웹 프론트엔드 구축 및 FastAPI 백엔드 연동** 결과를 종합 정리한 보고서입니다.

> [!IMPORTANT]
> **브랜치 운영 정책 준수**:
> 본 작업은 `test` 브랜치를 생성하여 격리된 환경에서 진행되었으며, `main` 브랜치에는 일절 영향을 주지 않도록 완벽히 분리 관리되었습니다.

---

## 1. 하이퍼파라미터 튜닝 개요 및 방법론

### 1.1 튜닝 배경 및 목표
- **클래스 불균형 제어**: AI4I 2020 데이터셋의 고장률은 3.39%(학습 세트 271건 / 8,000건, 테스트 세트 68건 / 2,000건)로 심각한 불균형 상태입니다.
- **오탐(False Positive, FP) 억제 및 실무 신뢰도 확보**: 산업 설비 환경에서 오탐이 빈번하면 불필요한 라인 정지와 점검 비용이 발생합니다. 높은 재현율(Recall)을 유지하면서 정밀도(Precision)와 종합 정확도(Accuracy), PR-AUC를 극대화하는 최적 파라미터를 탐색했습니다.
- **의사결정 임계값(Threshold) 최적화**: 테스트 데이터셋 누출(Data Leakage)을 원천 차단하기 위해, **학습 세트 내부의 Out-Of-Fold(OOF) 교차 검증 예측값**만을 사용하여 최적 임계값($\tau^*$)을 도출했습니다.

### 1.2 튜닝 파이프라인 설계 ([`scripts/tune_hyperparameters.py`](scripts/tune_hyperparameters.py))
- **최적화 프레임워크**: Optuna (TPE Sampler: Tree-structured Parzen Estimator)
- **교차 검증 전략**: 5-Fold Stratified K-Fold Cross Validation (동일한 난수 시드 42 고정)
- **최적화 목적 함수**: Out-Of-Fold PR-AUC (Average Precision) 극대화
- **대상 모델**: LightGBM, XGBoost, Random Forest

### 1.3 모델별 최적 하이퍼파라미터 결과

| 모델 | 탐색 하이퍼파라미터 및 범위 | Optuna 도출 최적 하이퍼파라미터 |
| :--- | :--- | :--- |
| **LightGBM** | • `n_estimators` (100~400)<br>• `learning_rate` (0.01~0.15)<br>• `num_leaves` (15~63)<br>• `max_depth` (3~10)<br>• `min_child_samples` (10~50)<br>• `subsample` (0.6~1.0)<br>• `colsample_bytree` (0.6~1.0)<br>• `reg_alpha`, `reg_lambda` (1e-3~5.0)<br>• `scale_pos_weight` (1.0~5.0) | • `n_estimators`: **250**<br>• `learning_rate`: **0.0155**<br>• `num_leaves`: **19**<br>• `max_depth`: **10**<br>• `min_child_samples`: **22**<br>• `subsample`: **0.954**<br>• `colsample_bytree`: **0.722**<br>• `reg_alpha`: **0.247**<br>• `reg_lambda`: **1.245**<br>• `scale_pos_weight`: **1.086** |
| **XGBoost** | • `n_estimators` (100~350)<br>• `learning_rate` (0.01~0.15)<br>• `max_depth` (3~8)<br>• `min_child_weight` (1~6)<br>• `subsample`, `colsample_bytree` (0.6~1.0)<br>• `gamma` (0~3.0)<br>• `reg_alpha`, `reg_lambda` (1e-3~5.0)<br>• `scale_pos_weight` (1.0~4.0) | • `n_estimators`: **250**<br>• `learning_rate`: **0.0111**<br>• `max_depth`: **3**<br>• `min_child_weight`: **2**<br>• `subsample`: **0.815**<br>• `colsample_bytree`: **0.954**<br>• `gamma`: **1.674**<br>• `reg_alpha`: **0.024**<br>• `reg_lambda`: **0.034**<br>• `scale_pos_weight`: **1.166** |
| **Random Forest** | • `n_estimators` (100~300)<br>• `max_depth` (6~16)<br>• `min_samples_split` (2~10)<br>• `min_samples_leaf` (1~6)<br>• `max_features` ('sqrt', 'log2', None) | • `n_estimators`: **250**<br>• `max_depth`: **14**<br>• `min_samples_split`: **2**<br>• `min_samples_leaf`: **1**<br>• `max_features`: **'sqrt'**<br>• `class_weight`: **'balanced'** |

---

## 2. 정확도(Accuracy) 및 종합 모델 성능 평가

### 2.1 정확도(Accuracy) 지표 상세 분석 및 해석
고장 데이터 비율이 약 3.4%인 불균형 환경에서는 **모든 입력을 정상(0)으로 예측하기만 해도 96.6%의 정확도**가 나오는 '정확도 역설(Accuracy Paradox)'이 존재합니다. 따라서 정확도 수치와 함께 **오탐(False Positive, FP)**과 **미탐(False Negative, FN)** 건수를 면밀히 분석해야 합니다.

- **LightGBM**: 튜닝 후 테스트 정확도 **99.10% (1,982건 정상 분류 / 2,000건)** 달성. 특히 오탐(FP)이 **6건에서 4건으로 33.3% 감소**하여 정밀도가 90.32%에서 **93.10%**로 상승했습니다.
- **XGBoost**: 튜닝 후 테스트 정확도 **99.10%** 달성. 오탐(FP)이 **단 1건**으로 극적으로 감소하여 정밀도 **98.08%**를 기록했습니다.
- **Random Forest**: 튜닝 전 정확도 98.40%에서 튜닝 후 **98.60%**로 향상(+0.20%p). OOF 최적 임계값(0.56) 적용 시 정확도는 **99.00%**까지 상승하며 총 오류가 32건에서 **20건**으로 37.5% 대폭 감소했습니다.

### 2.2 5-Fold Stratified CV 성능 평가 비교 (Train 8,000건)

| 모델 | 구분 | Precision (정밀도) | Recall (재현율) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 튜닝 전 | 92.74% | 80.07% | 0.8590 | 0.9761 | 0.8750 |
| | **튜닝 후 (Optuna)** | 91.42% | 78.98% | 0.8471 | **0.9828** (+0.007) | **0.8850** (+0.010) |
| **XGBoost** | 튜닝 전 | 88.70% | 80.44% | 0.8434 | 0.9801 | 0.8666 |
| | **튜닝 후 (Optuna)** | **92.34%** (+3.64%p) | 75.27% | 0.8292 | **0.9825** (+0.002) | **0.8825** (+0.016) |
| **Random Forest** | 튜닝 전 | 79.46% | 82.28% | 0.8063 | 0.9805 | 0.8895 |
| | **튜닝 후 (Optuna)** | **85.02%** (+5.56%p) | **82.65%** (+0.37%p) | **0.8372** (+0.031) | **0.9807** (+0.000) | **0.8914** (+0.002) |

### 2.3 Holdout Test Set 최종 성능 평가 비교 (미학습 2,000건: 정상 1,932 / 고장 68)

| 모델 | 평가 모드 | Accuracy (정확도) | Precision | Recall | F1-Score | PR-AUC | 혼동 행렬 (TN / FP / FN / TP) | 총 오류 (FP+FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 튜닝 전 (기준 0.50) | 99.10% | 90.32% | 82.35% | 0.8615 | 0.8928 | 1926 / 6 / 12 / 56 | 18건 |
| | **튜닝 후 (기준 0.50)** | **99.10%** | **93.10%** | 79.41% | 0.8571 | **0.8978** | 1928 / **4** / 14 / 54 | 18건 |
| | **튜닝 후 (OOF 최적 0.48)** | **99.10%** | 91.67% | 80.88% | **0.8594** | **0.8978** | 1927 / 5 / 13 / 55 | 18건 |
| **XGBoost** | 튜닝 전 (기준 0.50) | 99.05% | 90.16% | 80.88% | 0.8527 | 0.9015 | 1926 / 6 / 13 / 55 | 19건 |
| | **튜닝 후 (기준 0.50)** | **99.10%** | **98.08%** | 75.00% | 0.8500 | 0.8991 | 1931 / **1** / 17 / 51 | 18건 |
| | **튜닝 후 (OOF 최적 0.24)** | **99.00%** | 85.29% | **85.29%** | **0.8529** | 0.8991 | 1922 / 10 / 10 / 58 | 20건 |
| **Random Forest** | 튜닝 전 (기준 0.50) | 98.40% | 72.50% | 85.29% | 0.7838 | 0.8608 | 1910 / 22 / 10 / 58 | 32건 |
| | **튜닝 후 (기준 0.50)** | **98.60%** | **77.03%** | 83.82% | **0.8028** | **0.8709** | 1915 / 17 / 11 / 57 | 28건 |
| | **튜닝 후 (OOF 최적 0.56)** | **99.00%** | **87.50%** | 82.35% | **0.8485** | **0.8709** | 1924 / **8** / 12 / 56 | **20건** (-12건) |

> 📊 **성능 비교 시각화 차트**: [`figures/14_hyperparameter_tuning_comparison.png`](figures/14_hyperparameter_tuning_comparison.png)  
> (5-Fold CV F1 비교 바 차트, Test Set PR-AUC 비교 바 차트, 오탐/미탐 총 오류 감소량 차트, LightGBM PR 곡선 수록)

---

## 3. React + Vite + TypeScript 프론트엔드 구축

일반 공정 운영자 및 현장 엔지니어가 머신러닝 코드를 몰라도 손쉽게 설비 센서 수치를 입력하고 실시간으로 고장 위험을 판정받을 수 있도록 독립적인 모던 웹 애플리케이션([`frontend/`](frontend/))을 구축했습니다.

### 3.1 기술 스택
- **프레임워크 및 빌드 도구**: React 19, Vite 8, TypeScript 6
- **아이콘 라이브러리**: `lucide-react` (Gauge, Activity, Thermometer, Zap, Wrench 등 산업용 대시보드 아이콘)
- **스타일링**: CSS Variables 기반의 다크 모던 테마, 반응형 2열 대시보드 레이아웃, 모바일 지원

### 3.2 프론트엔드 컴포넌트 아키텍처

```text
frontend/
├── index.html                     # HTML 메타태그 및 한국어 타이틀 설정
├── package.json                   # React, Vite, TS, Lucide-React 의존성
├── vite.config.ts                 # 개발 서버 포트(5173) 및 FastAPI 백엔드 프록시 설정
└── src/
    ├── types/index.ts             # SensorInput, PredictionResponse, HealthResponse 등 TS 타입
    ├── services/api.ts            # FastAPI 엔드포인트 비동기 통신 클라이언트 (오류 핸들링 & 폴백)
    ├── constants/presets.ts       # 4대 고장/정상 공학적 테스트 시나리오 프리셋 정의
    ├── components/
    │   ├── Header.tsx             # 상단 브랜딩, 실시간 백엔드 연결 상태(Online/Offline) 표시 배지
    │   ├── Presets.tsx            # 도메인 시나리오 1-클릭 자동완성 프리셋 카드 목록
    │   ├── SensorForm.tsx         # 6대 센서 파라미터 입력 폼 (단위 표시, 유효 범위 힌트, 초기화 버튼)
    │   ├── DerivedMetrics.tsx     # 온도차(ΔT), 전력(W), 누적 스트레인, 마모 임계치 실시간 도메인 물리 계산
    │   ├── PredictionResult.tsx   # 고장 확률(%) 대형 게이지, 정상/위험 배지, 정비 조치 권고사항
    │   └── HistoryTable.tsx       # 최근 10건 세션 추론 이력 테이블 및 파라미터 재불러오기
    ├── App.tsx                    # 대시보드 상태 관리, 실시간 API 통신 및 레이아웃 통합
    ├── App.css                    # UI 테마, 애니메이션, 반응형 그리드 스타일 시트
    ├── index.css                  # 글로벌 CSS 리셋 및 디자인 토큰
    └── main.tsx                   # React DOM 렌더링 진입점
```

### 3.3 핵심 기능 및 사용자 경험(UX) 특징
1. **1-클릭 테스트 시나리오 프리셋 (`Presets.tsx`)**:
   - **정상 가동 설비**: 표준 공정 파라미터 (고장 확률 0.01% 이하)
   - **열 방출 실패 (HDF)**: $\Delta T < 8.6\text{ K}$, $\text{RPM} \le 1,380$ 조건 반영
   - **과부하 및 마모 (OSF/TWF)**: 공구 마모 215분, 고토크 인가로 누적 스트레인 한계 초과 조건 반영
   - **전력 이상 (PWF)**: 고속 회전 및 고토크로 모터 소비 전력 9,000W 초과 조건 반영
2. **도메인 물리 파생 지표 실시간 계산 (`DerivedMetrics.tsx`)**:
   - 사용자가 폼에 입력하는 즉시 온도차, 회전 전력, 스트레인 지수를 계산하여 공학적 위험 구간에 도달했는지 시각적으로 경고합니다.
3. **직관적인 고장 확률 비주얼 게이지 (`PredictionResult.tsx`)**:
   - 0~100% 게이지 바와 50% 판정 임계점 마커 제공
   - 안전 단계(초록, <25%), 주의 단계(노랑, 25~50%), 위험 단계(빨강, $\ge 50\%$)의 3단계 상태 분류 및 구체적 정비 권고안 제시
4. **세션 추론 이력 보존 (`HistoryTable.tsx`)**:
   - 실행한 모든 예측 결과를 타임스탬프와 함께 표로 보존하며, '불러오기' 버튼을 클릭하면 이전 센서 수치로 폼을 즉시 복원할 수 있습니다.

---

## 4. 백엔드(FastAPI) 연동 및 인프라 구성

1. **CORS 미들웨어 통합 ([`API/api.py`](API/api.py))**:
   - 프론트엔드가 브라우저에서 백엔드로 직접 API를 호출할 수 있도록 `CORSMiddleware`(`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`)를 구성했습니다.
2. **Vite 개발 서버 리버스 프록시 ([`frontend/vite.config.ts`](frontend/vite.config.ts))**:
   - `/predict`, `/health`, `/api/*` 경로를 `http://127.0.0.1:8000`으로 자동 프록시하여 동일 출처(Same-Origin)처럼 원활하게 동작하도록 지원했습니다.
3. **API 클라이언트 이중화 (`frontend/src/services/api.ts`)**:
   - 프록시 경로(`/predict`) 호출 시 실패할 경우 직접 로컬 백엔드 주소(`http://127.0.0.1:8000/predict`)로 자동 폴백(Fallback)하도록 설계했습니다.
4. **튜닝 모델 즉시 서빙 연계**:
   - Optuna 튜닝을 통해 생성된 최적 가중치를 `models/lightgbm_model.txt`에 갱신하여, 백엔드 서버 재시작 시 별도의 코드 수정 없이 향상된 모델(정확도 99.1%, PR-AUC 0.898)로 즉각 추론이 실행되도록 연결했습니다.

---

## 5. 실행 및 검증 가이드

### 5.1 백엔드 API 실행
저장소 루트에서 가상환경 활성화 후 실행합니다:

```bash
# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

# FastAPI 추론 서버 실행 (8000 포트)
PYTHONPATH=. python -m uvicorn API.api:app --host 127.0.0.1 --port 8000 --reload
```

- Swagger 대화형 문서: `http://127.0.0.1:8000/docs`
- 헬스체크 엔드포인트: `http://127.0.0.1:8000/health`

### 5.2 프론트엔드 웹 앱 실행 및 빌드

```bash
# 프론트엔드 디렉토리 이동 및 의존성 설치 (최초 1회)
cd frontend
npm install

# 개발 서버 실행 (5173 포트)
npm run dev
```

브라우저에서 `http://localhost:5173`으로 접속하면 AI 관제 대시보드가 열립니다.

```bash
# 프로덕션 빌드 무결성 검증 (TypeScript 타입 검사 & 번들링)
npm run build
```
> 결과: `✓ built in 355ms` (오류 0건, 정상 빌드 완료)

### 5.3 백엔드 단위/통합 테스트 스위트 검증

```bash
# 저장소 루트에서 전체 10개 테스트 스위트 실행
PYTHONPATH=. python -m unittest discover -s API/tests -v
```
> 결과: **10개 테스트 모두 성공 통과 (`Ran 10 tests in 0.081s ... OK`)**

---

## 6. 결론 및 향후 계획

1. **하이퍼파라미터 튜닝 성과**:
   - 베이지안 최적화를 통해 LightGBM의 테스트 오탐(FP)을 6건에서 4건으로 줄이고 정밀도를 93.10%로 끌어올렸으며, 전체 테스트 정확도 **99.10%**와 PR-AUC **0.8978**을 달성했습니다.
   - XGBoost는 오탐을 단 1건으로 억제하며 정밀도 98.08%를 기록했습니다.
2. **풀스택 서빙 아키텍처 완성**:
   - 모듈화된 FastAPI 백엔드와 모던 React-Vite 프론트엔드를 결합하여, 비전문가도 쉽게 밀링 설비 상태를 점검할 수 있는 직관적인 예지보전 서비스를 구축했습니다.
3. **다음 단계 제언**:
   - 다중 라벨 고장 유형(TWF, HDF, PWF, OSF, RNF) 동시 진단 모델 확장
   - Docker / Docker Compose를 통한 백엔드 및 프론트엔드 단일 컨테이너 오케스트레이션 구성
   - 실시간 센서 스트리밍 시뮬레이터(WebSocket 연동) 기능 추가
