# AWS EC2 Docker 기반 배포 가이드 (Cloud MLOps)

본 문서는 **AI4I 2020 설비 예지보전 AI 서비스**(FastAPI 백엔드 + React 19 웹 프론트엔드 + Nginx 리버스 프록시)를 AWS EC2 인스턴스에서 **Docker 및 Docker Compose**를 활용하여 즉시 배포하고, `.env` 파일을 통해 포트를 자유롭게 설정할 수 있도록 작성된 실전 가이드입니다.

---

## 1. AWS EC2 인스턴스 준비 및 인바운드 규칙 설정

### 1.1 권장 인스턴스 사양
- **인스턴스 유형**: `t2.micro` 또는 `t3.micro` (AWS 프리티어 완전 호환) / `t3.small` 이상 권장
- **운영체제(OS)**: Amazon Linux 2023 또는 Ubuntu 22.04 / 24.04 LTS
- **스토리지(EBS)**: 20GB 이상 gp3 권장

### 1.2 보안 그룹(Security Group) 인바운드 규칙 설정 (필수!)
AWS 콘솔의 **인스턴스 > 보안 > 보안 그룹**에서 다음 포트가 열려 있는지 반드시 확인합니다:

| 유형 (Type) | 프로토콜 | 포트 범위 (Port) | 소스 (Source) | 설명 |
| :--- | :---: | :---: | :---: | :--- |
| **SSH** | TCP | `22` | 내 IP | EC2 원격 터미널 접속 |
| **HTTP** | TCP | `80` (또는 `.env`의 `FRONTEND_PORT`) | `0.0.0.0/0` | **React 웹 대시보드 브라우저 접속** |
| **사용자 지정 TCP** | TCP | `8000` (또는 `.env`의 `BACKEND_PORT`) | `0.0.0.0/0` | **FastAPI Swagger Docs 및 직접 API 호출** |

> [!NOTE]
> `.env` 파일에서 `FRONTEND_PORT=8080` 등으로 변경할 경우, 보안 그룹에서도 해당 포트(`8080`)를 인바운드 규칙에 추가해주셔야 합니다.

---

## 2. 포트 설정 (.env 파일)

저장소 루트에 위치한 `.env` 파일을 통해 호스트 머신에서 외부에 노출할 포트를 손쉽게 커스터마이징할 수 있습니다:

```env
# ========================================================
# Cloud MLOps Server Environment Configuration
# ========================================================

# [웹 프론트엔드 포트]
# EC2 호스트에서 웹 UI를 제공할 외부 포트 번호 (기본값: 80)
# 웹 브라우저 접속: http://<EC2-IP> (기본 포트 80은 포트 번호 생략 가능)
# 8080, 3000 등 다른 포트로 변경 가능
FRONTEND_PORT=80

# [FastAPI 백엔드 포트]
# EC2 호스트에서 Swagger 문서 및 API를 직접 호출할 외부 포트 번호 (기본값: 8000)
# Swagger 문서 접속: http://<EC2-IP>:8000/docs
BACKEND_PORT=8000

# [머신러닝 추론 모델 설정]
MODEL_PATH=/app/models/lightgbm_model.txt

# 고장 분류 판정 임계값 (0.0 ~ 1.0)
THRESHOLD=0.5
```

---

## 3. 원클릭 배포 (Fast Track)

EC2 인스턴스 터미널에 접속한 후, 아래 명령어를 실행하면 끝납니다:

```bash
# 1. 저장소 복제 및 브랜치 이동
git clone https://github.com/YSU-CloudMLOps/cloud-mlops.git
cd cloud-mlops
git checkout test

# 2. (선택사항) 포트 변경이 필요한 경우 .env 파일 수정
nano .env   # 또는 vim .env

# 3. 원클릭 배포 스크립트 실행 (sudo 권한 실행 권장)
sudo ./scripts/deploy_ec2.sh
```

### `deploy_ec2.sh` 스크립트가 자동으로 수행하는 작업:
1. **t2/t3.micro OOM(메모리 부족) 방지**: 1GB RAM 인스턴스 감지 시 **2GB 스왑 메모리(/swapfile)를 자동 할당**하여 빌드 및 구동 중 강제 종료를 원천 차단합니다.
2. **도커 데몬 상태 점검**: Docker 및 Docker Compose 플러그인 상태를 확인하고 데몬을 활성화합니다.
3. **무중단 빌드 & 백그라운드 구동**: `docker compose up -d --build` 실행.
4. **서비스 헬스체크**: 엔드포인트가 정상 응답할 때까지 자동 대기 후, EC2의 Public IP와 함께 접속 URL을 출력합니다.

---

## 4. 수동 배포 단계 (직접 Docker 명령어를 실행할 경우)

### Step 1. Docker 및 Docker Compose 설치 (최초 1회)

**[Amazon Linux 2023]**
```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# 그룹 적용을 위해 로그아웃 후 재접속하거나 newgrp docker 실행
newgrp docker
```

**[Ubuntu 22.04 / 24.04]**
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2. (t2/t3.micro 필수) 스왑 메모리 2GB 활성화
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

### Step 3. 컨테이너 빌드 및 백그라운드 실행
```bash
# .env 파일 확인
cat .env

# 컨테이너 빌드 및 데몬 구동
docker compose up -d --build
```

---

## 5. 서비스 접속 및 확인

배포가 완료되면 브라우저에서 EC2 퍼블릭 IP를 통해 즉시 접속할 수 있습니다:

- **웹 대시보드 (React UI)**:  
  `http://<EC2-퍼블릭-IP>` (또는 `http://<EC2-퍼블릭-IP>:<FRONTEND_PORT>`)
  - 프리셋 시나리오(정상, HDF, OSF, PWF) 1-클릭 테스트
  - 도메인 물리 지표($\Delta T$, Power, Strain) 실시간 연산
  - 고장 확률(%) 게이지 및 정비 권고사항 확인
- **Swagger 대화형 API 문서**:  
  `http://<EC2-퍼블릭-IP>:<BACKEND_PORT>/docs` (또는 Nginx 프록시를 통해 `http://<EC2-퍼블릭-IP>/docs`)
- **헬스체크**:  
  `http://<EC2-퍼블릭-IP>:<BACKEND_PORT>/health` (또는 `http://<EC2-퍼블릭-IP>/health`)

### 터미널 CLI 추론 테스트:
```bash
curl -X POST http://<EC2-퍼블릭-IP>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "L",
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0
  }'
```
**응답 예시**:
```json
{"machine_failure": 0, "failure_probability": 0.00138}
```

---

## 6. 운영 및 유지보수 명령어 치트시트

```bash
# 1. 실행 중인 컨테이너 상태 및 헬스체크 확인
docker compose ps

# 2. 실시간 로그 확인 (전체 또는 특정 서비스)
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend

# 3. 포트 변경 후 서비스 재적용
nano .env # 원하는 포트로 수정
docker compose up -d

# 4. 서비스 재시작
docker compose restart

# 5. 서비스 완전 중지 및 컨테이너 삭제
docker compose down

# 6. 새 모델 가중치 반영 (컨테이너 재빌드 불필요!)
# ./models 폴더가 볼륨 마운트되어 있으므로 파일 교체 후 재시작만 하면 즉시 적용됩니다.
cp my_new_model.txt models/lightgbm_model.txt
docker compose restart backend
```
