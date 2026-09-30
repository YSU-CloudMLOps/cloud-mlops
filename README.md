# cloud-mlops
실습용 리포지토리

## FastAPI 추론 API

저장소 루트에서 실행합니다 (Python 3.10 이상).

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/docs`에서 직접 테스트할 수 있습니다.
`POST /predict`는 기본 센서 정보 6개를 받습니다.

```json
{
  "type": "L",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 100
}
```

PowerShell 호출 예:

```powershell
$body = @{ type = 'L'; air_temperature_k = 300; process_temperature_k = 310; rotational_speed_rpm = 1500; torque_nm = 40; tool_wear_min = 100 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

응답은 `machine_failure` (0: 정상, 1: 고장), `failure_probability`
(0~1의 모델 예측 점수) 두 필드만 포함합니다.
서버에서 학습 코드와 동일한 파생 피처를 생성하며, 시작할 때
`models/lightgbm_model.txt`를 한 번 로드합니다. 기본 LightGBM 모델을
사용하며, 고장 판정 기준은 서버 내부에서 0.5로 적용합니다.
잘못된 등급, 누락된 필드, 음수 마모 시간 등은 HTTP 422를 반환합니다.
`GET /health`는 서버 상태를 반환합니다. 현재 센서 상태의 고장 분류이며,
미래 고장 시점을 예측하는 API는 아닙니다.

검증 실행:

```bash
python -m pip install httpx
python -m unittest discover -s tests
```

## 프로젝트 구성
- **[데이터셋 안내 (dataset/README.md)](dataset/README.md)**: AI4I 2020 Predictive Maintenance 데이터셋 상세 설명 및 분석 가이드
- **[데이터셋 분석 노트 (notes/dataset_info.md)](notes/dataset_info.md)**: AI4I 2020 데이터셋 기초 통계, 피처 명세 및 MLOps 포인트 정리
- **[전처리 정책 및 EDA 보고서 (notes/preprocessing_policy.md)](notes/preprocessing_policy.md)**: 탐색적 데이터 분석 결과 및 7대 전처리 정책 가이드라인
- **[모델 훈련 및 성능 평가 보고서 (notes/model_evaluation.md)](notes/model_evaluation.md)**: LightGBM, XGBoost, Random Forest 5-Fold CV 및 Test 세트 벤치마크 평가 결과
- **[고도화 앙상블 및 임계값 최적화 보고서 (notes/advanced_ensemble_evaluation.md)](notes/advanced_ensemble_evaluation.md)**: 도메인 물리 경계 피처 적용, 소프트 보팅 앙상블 및 F1 0.88 달성 평가 결과
- **[모델 가중치 및 직렬화 디렉토리 (models/)](models/)**: LightGBM, XGBoost, Random Forest, 소프트 보팅 앙상블 파이프라인 및 메타데이터
- **[시각화 차트 디렉토리 (figures/)](figures/)**: EDA 및 모델 평가 차트 (ROC 곡선, PR 곡선, 혼동 행렬, 임계값 최적화 곡선, 피처 중요도)
- **[IEEE 학술 논문 보고서 (charged-ieee/)](charged-ieee/main.pdf)**: AI4I 2020 예지보전 MLOps 파이프라인 학술 연구 논문 ([Typst 소스: main.typ](charged-ieee/main.typ), [PDF 전문: main.pdf](charged-ieee/main.pdf))


