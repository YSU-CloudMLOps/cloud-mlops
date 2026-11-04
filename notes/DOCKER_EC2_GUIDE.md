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

# 7. 헬스체크 상세 진단 (컨테이너가 unhealthy 상태일 때 실패 원인 확인)
docker inspect --format='{{json .State.Health}}' mlops-frontend | jq
```

---

## 7. GitHub Actions + AWS SSM 기반 Zero-SSH 자동 배포 (Continuous Deployment)

본 저장소는 코드 푸시 시 자동으로 테스트와 빌드를 수행하고, **AWS Systems Manager (SSM) Run Command**를 통해 **IP 변경이나 EC2 삭제/재생성에 영향받지 않고** 안전하게 무중단 배포되는 **CI/CD 파이프라인([`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml))**을 갖추고 있습니다.

### 7.1 왜 AWS SSM (Zero-SSH) 아키텍처인가?
기존의 SSH 직접 접속 방식(`EC2_HOST` IP 고정)과 달리 다음과 같은 압도적인 이점을 제공합니다:
1. **IP 변경 완벽 대응 (IP-Independent)**: EC2를 중지 후 재시작하여 퍼블릭 IP가 바뀌더라도, AWS 내부 인스턴스 태그(`Name=mlops-server`) 및 SSM 온라인 상태를 기준으로 타겟을 동적 탐색하므로 배포가 절대 깨지지 않습니다.
2. **인스턴스 삭제/재생성 탄력성 (Resilient to Termination)**:
   - **EC2 삭제 시**: 활성 인스턴스가 없음을 자동 감지하여 파이프라인이 에러(Red)로 뻗지 않고 `⚠️ [GRACEFUL SKIP]` 로그를 남기며 안전하게 종료됩니다.
   - **EC2 재생성 시**: 새 EC2에 IAM 역할만 연결하면 Secrets 수정 없이 다음 푸시부터 새 서버에 `git clone` 및 Docker 기동을 자동으로 수행합니다.
3. **최고 수준의 보안 (No Inbound Port 22 & No PEM Key)**: 보안 그룹에서 **22번(SSH) 인바운드 포트를 완전히 폐쇄**할 수 있으며, `.pem` 비밀키 유출 위험이 원천 제거됩니다.

### 7.2 파이프라인 아키텍처

```
[개발자 git push (main/test)]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. CI (Test & Build Job - GitHub Hosted Runner)        │
│   ├─ Python 3.11 환경 구성 & API 의존성 설치           │
│   ├─ 백엔드 12개 단위/통합 테스트 스위트 검증 (unittest) │
│   ├─ Node.js 20 환경 구성 & React 19 프론트엔드 빌드   │
│   └─ Docker Compose 멀티 컨테이너 병렬 빌드 사전 검증  │
└────────────────────────────────────────────────────────┘
           │ (CI 검증 100% 통과 시)
           ▼
┌────────────────────────────────────────────────────────┐
│ 2. CD (Deploy Job - AWS Systems Manager Run Command)   │
│   ├─ AWS IAM 자격 증명 구성 (OIDC Role 또는 IAM Key)   │
│   ├─ SSM Online & Running 상태인 EC2 동적 탐색         │
│   │   (삭제/중지 시 에러 없이 Graceful Skip 처리)      │
│   ├─ aws ssm send-command로 배포 명령 암호화 하달      │
│   └─ deploy_ec2.sh 무중단 배포 실행 및 실시간 로그 수집│
└────────────────────────────────────────────────────────┘
           │
           ▼
 [AWS EC2 프로덕션 서비스 갱신 완료 (포트 80/8000)]
```

### 7.3 설정 가이드 (AWS 콘솔 + GitHub Secrets 1회 설정)

#### Step 1: EC2 인스턴스에 SSM IAM 역할(Role) 연결
Ubuntu 22.04 / 24.04 및 Amazon Linux 이미지에는 이미 **SSM Agent가 기본 설치**되어 있습니다. 권한만 부여하면 즉시 활성화됩니다.
1. **AWS 콘솔 > IAM > 역할(Roles) > [역할 만들기]** 클릭
2. 신뢰할 수 있는 엔터티: **AWS 서비스**, 사용 사례: **EC2** 선택 후 다음
3. 권한 정책 검색창에 **`AmazonSSMManagedInstanceCore`** 검색 및 체크 후 역할 생성 (예: `EC2-SSM-MLOps-Role`)
4. **AWS 콘솔 > EC2 > 인스턴스**에서 배포 대상 EC2 선택 ➔ **[작업] > [보안] > [IAM 역할 수정]** 클릭 ➔ 방금 만든 `EC2-SSM-MLOps-Role` 연결 후 저장
5. (권장) EC2 인스턴스의 `Name` 태그를 **`mlops-server`**로 지정 (계정에 활성 EC2가 1대뿐이면 태그가 없어도 자동 인식됩니다).

#### Step 2: GitHub Repository Secrets 등록
GitHub 저장소의 **Settings > Secrets and variables > Actions**에서 다음 항목을 등록합니다:

| 시크릿 이름 (Secret Name) | 필수 여부 | 권장 값 / 예시 | 설명 |
| :--- | :---: | :--- | :--- |
| **`AWS_ACCESS_KEY_ID`** | **필수*** | `AKIAIOSFODNN7EXAMPLE` | SSM 및 EC2 조회 권한이 있는 IAM Access Key |
| **`AWS_SECRET_ACCESS_KEY`** | **필수*** | `wJalrXUtnFEMI/K7MDENG/bPxRfiCY...` | IAM Secret Access Key (*OIDC 사용 시 `AWS_ROLE_ARN`으로 대체 가능) |
| **`AWS_REGION`** | 선택 | `ap-northeast-2` | AWS 리전 (기본값: `ap-northeast-2`) |
| **`EC2_TAG_VALUE`** | 선택 | `mlops-server` | 탐색할 EC2의 `Name` 태그 값 (기본값: `mlops-server`) |

> [!TIP]
> **3단계 하이브리드 폴백 지원**:
> - **1순위 (AWS SSM)**: `AWS_ACCESS_KEY_ID` 또는 `AWS_ROLE_ARN` 등록 시 SSM Zero-SSH 모드로 작동합니다.
> - **2순위 (Legacy SSH)**: AWS 키 없이 기존 `EC2_HOST`, `EC2_SSH_KEY`만 등록된 경우 SSH 모드로 하위 호환 작동합니다.
> - **3순위 (Dry-Run)**: 아무 시크릿도 등록되지 않은 초기 상태에서는 CI 검증만 수행한 뒤 설정 가이드를 출력하고 정상 종료(Success)합니다.
