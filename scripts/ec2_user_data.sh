#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 User Data Bootstrap Script (Self-Healing & Auto-Provisioning)
# ==============================================================================
# Executed automatically by cloud-init on first boot when an EC2 instance is
# launched via Launch Template, Auto Scaling Group (ASG), or CI/CD pipeline.
# Logs are streamed to /var/log/mlops-user-data.log
# ==============================================================================

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

exec > >(tee -a /var/log/mlops-user-data.log | logger -t mlops-user-data -s 2>/dev/console) 2>&1

echo "=================================================================="
echo "🚀 [MLOps Self-Healing Bootstrap] Starting EC2 User Data Setup"
echo "   Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=================================================================="

REPO_URL="https://github.com/YSU-CloudMLOps/cloud-mlops.git"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
DEPLOY_USER="ubuntu"
if ! id -u "$DEPLOY_USER" >/dev/null 2>&1; then
    DEPLOY_USER="$(id -un 1000 2>/dev/null || echo root)"
fi

if [ "$DEPLOY_USER" = "root" ]; then
    DEPLOY_DIR="/opt/cloud-mlops"
else
    DEPLOY_DIR="/home/${DEPLOY_USER}/cloud-mlops"
fi

# 1. Ensure AWS SSM Agent is active so CI/CD can reach this instance without SSH
echo "[1/4] Verifying AWS Systems Manager (SSM) Agent..."
if command -v snap >/dev/null 2>&1 && snap list amazon-ssm-agent >/dev/null 2>&1; then
    snap start amazon-ssm-agent || true
elif systemctl list-unit-files | grep -q amazon-ssm-agent; then
    systemctl enable --now amazon-ssm-agent || true
fi

# 2. Install Git & Curl if missing
echo "[2/4] Installing base packages (git, curl, ca-certificates)..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y -qq
    apt-get install -y -qq git curl ca-certificates
elif command -v yum >/dev/null 2>&1; then
    yum install -y -q git curl ca-certificates
fi

# 3. Clone or update repository
echo "[3/4] Fetching repository ($REPO_URL, branch: $TARGET_BRANCH) to $DEPLOY_DIR..."
if [ ! -d "$DEPLOY_DIR/.git" ]; then
    git clone --branch "$TARGET_BRANCH" "$REPO_URL" "$DEPLOY_DIR" || git clone "$REPO_URL" "$DEPLOY_DIR"
else
    cd "$DEPLOY_DIR"
    git fetch --all --prune
    git checkout "$TARGET_BRANCH" || true
    git reset --hard "origin/$TARGET_BRANCH" || true
fi

cd "$DEPLOY_DIR"
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi

# 4. Run zero-downtime Docker deployment script (includes 2GB Swap + Docker install)
echo "[4/4] Running scripts/deploy_ec2.sh..."
chmod +x scripts/deploy_ec2.sh
./scripts/deploy_ec2.sh

if [ "$DEPLOY_USER" != "root" ]; then
    chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "$DEPLOY_DIR" || true
fi

echo "=================================================================="
echo "✅ [MLOps Self-Healing Bootstrap] Instance Provisioned & Healthy!"
echo "=================================================================="
