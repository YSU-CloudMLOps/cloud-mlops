#!/usr/bin/env bash
# ==============================================================================
# Cloud MLOps - AWS EC2 Automated Docker Deployment Script
# ==============================================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}   Cloud MLOps - AWS EC2 One-Touch Docker Deployment ${NC}"
echo -e "${BLUE}==================================================================${NC}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Check or setup .env file
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo -e "${YELLOW}[!] .env 파일이 존재하지 않아 .env.example로부터 생성합니다.${NC}"
    cp .env.example .env
  else
    echo -e "${RED}[ERROR] .env 및 .env.example 파일이 존재하지 않습니다.${NC}"
    exit 1
  fi
fi

# Load environment variables
set -a
source .env
set +a

FRONTEND_PORT=${FRONTEND_PORT:-80}
BACKEND_PORT=${BACKEND_PORT:-8000}

echo -e "[*] 설정된 포트 정보 (.env):"
echo -e "    - 웹 프론트엔드 포트 : ${GREEN}${FRONTEND_PORT}${NC}"
echo -e "    - 백엔드 API 포트     : ${GREEN}${BACKEND_PORT}${NC}"

# 2. Check RAM & Configure Swap for t2.micro/t3.micro (prevents OOM crashes)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
TOTAL_RAM_MB=$((TOTAL_RAM_KB / 1024))
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")

if [ "$TOTAL_RAM_MB" -gt 0 ] && [ "$TOTAL_RAM_MB" -lt 1800 ] && [ "$SWAP_TOTAL_KB" -lt 500000 ]; then
  echo -e "${YELLOW}[!] 시스템 RAM이 ${TOTAL_RAM_MB}MB로 감지되었습니다 (t2/t3.micro 환경).${NC}"
  echo -e "${YELLOW}[!] 빌드 및 실행 중 OOM(메모리 부족) 방지를 위해 2GB 스왑 메모리를 구성합니다...${NC}"
  if [ "$EUID" -eq 0 ]; then
    fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null 2>&1
    swapon /swapfile >/dev/null 2>&1
    echo "/swapfile swap swap defaults 0 0" >> /etc/fstab 2>/dev/null || true
    echo -e "${GREEN}[✓] 2GB 스왑 메모리 설정이 완료되었습니다.${NC}"
  else
    echo -e "${YELLOW}[!] 스왑 생성을 위해 root 권한(sudo)이 필요할 수 있습니다 (sudo $0 실행 권장).${NC}"
  fi
fi

# 3. Check Docker installation
if ! command -v docker &>/dev/null; then
  echo -e "${RED}[ERROR] Docker가 설치되어 있지 않습니다.${NC}"
  echo -e "다음 명령어로 Docker를 설치해주세요:"
  echo -e "  - Amazon Linux 2023 : sudo dnf install -y docker && sudo systemctl enable --now docker && sudo usermod -aG docker \$USER"
  echo -e "  - Ubuntu            : sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 && sudo systemctl enable --now docker && sudo usermod -aG docker \$USER"
  exit 1
fi

# Check Docker service status
if ! docker info &>/dev/null; then
  echo -e "${YELLOW}[!] Docker 데몬이 실행 중이지 않습니다. 시작을 시도합니다...${NC}"
  if [ "$EUID" -eq 0 ]; then
    systemctl start docker || service docker start
  else
    sudo systemctl start docker || sudo service docker start || true
  fi
fi

# 4. Check Docker Compose availability
COMPOSE_CMD=""
if docker compose version &>/dev/null; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
  COMPOSE_CMD="docker-compose"
else
  echo -e "${RED}[ERROR] Docker Compose 플러그인 또는 docker-compose 명령어를 찾을 수 없습니다.${NC}"
  exit 1
fi

echo -e "[*] Docker Compose 명령어: ${GREEN}${COMPOSE_CMD}${NC}"

# 5. Build and Launch Containers
echo -e "${BLUE}[*] 도커 컨테이너 빌드 및 백그라운드 구동을 시작합니다...${NC}"
$COMPOSE_CMD down --remove-orphans || true
$COMPOSE_CMD up -d --build

# 6. Wait for service readiness
echo -e "[*] 서비스 상태 검증 중 (최대 30초 대기)..."
READY=0
for i in {1..30}; do
  if curl -s -f "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

# 7. Print Results & Public IP
PUBLIC_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || curl -s --connect-timeout 2 ifconfig.me 2>/dev/null || echo "EC2_PUBLIC_IP")

echo ""
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  ✓ 배포 완료! 모든 서비스가 정상 작동 중입니다. ${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "▶ 웹 프론트엔드 대시보드 : ${BLUE}http://${PUBLIC_IP}:${FRONTEND_PORT}${NC}"
echo -e "▶ Swagger API 문서      : ${BLUE}http://${PUBLIC_IP}:${BACKEND_PORT}/docs${NC}"
echo -e "▶ 헬스체크 엔드포인트   : ${BLUE}http://${PUBLIC_IP}:${BACKEND_PORT}/health${NC}"
echo ""
echo -e "[*] 유용한 명령어:"
echo -e "    - 실시간 로그 확인 : ${YELLOW}${COMPOSE_CMD} logs -f${NC}"
echo -e "    - 컨테이너 상태    : ${YELLOW}${COMPOSE_CMD} ps${NC}"
echo -e "    - 컨테이너 중지    : ${YELLOW}${COMPOSE_CMD} down${NC}"
echo -e "    - 포트 변경        : .env 파일 수정 후 ${YELLOW}${COMPOSE_CMD} up -d${NC}"
echo -e "${GREEN}==================================================================${NC}"
