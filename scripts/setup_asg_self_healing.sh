#!/usr/bin/env bash
# ==============================================================================
# AWS Auto Scaling Group (ASG) + Launch Template Self-Healing Setup Script
# ==============================================================================
# Provisions an idempotent Self-Healing EC2 infrastructure (Option 4):
#   1. IAM Role & Instance Profile with AmazonSSMManagedInstanceCore
#   2. Security Group (HTTP 80 & FastAPI 8000, Zero-SSH required)
#   3. EC2 Launch Template with Base64-encoded scripts/ec2_user_data.sh
#   4. Auto Scaling Group (Min=1, Desired=1, Max=1) with Tag Name=mlops-server
#
# If an EC2 instance is accidentally terminated or fails health checks, AWS ASG
# automatically launches a replacement instance and boots the Docker stack.
# ==============================================================================

set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-northeast-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"
ROLE_NAME="${ROLE_NAME:-EC2-SSM-MLOps-Role}"
PROFILE_NAME="${PROFILE_NAME:-EC2-SSM-MLOps-Profile}"
SG_NAME="${SG_NAME:-mlops-self-healing-sg}"
LT_NAME="${LT_NAME:-mlops-launch-template}"
ASG_NAME="${ASG_NAME:-mlops-self-healing-asg}"
TAG_VALUE="${EC2_TAG_VALUE:-mlops-server}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_FILE="${SCRIPT_DIR}/ec2_user_data.sh"

echo "=================================================================="
echo "🛡️  [Option 4] Configuring Self-Healing AWS Auto Scaling Group"
echo "   Region        : ${AWS_REGION}"
echo "   Instance Type : ${INSTANCE_TYPE}"
echo "   Launch Tmpl   : ${LT_NAME}"
echo "   ASG Name      : ${ASG_NAME}"
echo "   Target Tag    : Name=${TAG_VALUE}"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. IAM Role & Instance Profile for AWS SSM (Zero-SSH)
# ------------------------------------------------------------------------------
echo "[1/5] Ensuring IAM Role (${ROLE_NAME}) & Instance Profile (${PROFILE_NAME})..."

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    echo "   -> Creating IAM Role: $ROLE_NAME"
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
          }]
        }' >/dev/null

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
fi

if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    echo "   -> Creating Instance Profile: $PROFILE_NAME"
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME" || true
    echo "   -> Waiting 10s for IAM Instance Profile propagation..."
    sleep 10
fi

# ------------------------------------------------------------------------------
# 2. Default VPC, Subnets & Security Group (Ports 80 & 8000)
# ------------------------------------------------------------------------------
echo "[2/5] Discovering Default VPC, Subnets, and Security Group (${SG_NAME})..."

VPC_ID=$(aws ec2 describe-vpcs \
    --region "$AWS_REGION" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text)

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    VPC_ID=$(aws ec2 describe-vpcs \
        --region "$AWS_REGION" \
        --query "Vpcs[0].VpcId" --output text)
fi

SUBNET_IDS=$(aws ec2 describe-subnets \
    --region "$AWS_REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[].SubnetId" --output text | tr '\t' ',')

SG_ID=$(aws ec2 describe-security-groups \
    --region "$AWS_REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "   -> Creating Security Group: $SG_NAME in VPC $VPC_ID"
    SG_ID=$(aws ec2 create-security-group \
        --region "$AWS_REGION" \
        --group-name "$SG_NAME" \
        --description "MLOps Predictive Maintenance Web (80) and API (8000) - Zero SSH" \
        --vpc-id "$VPC_ID" \
        --query "GroupId" --output text)

    aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
    aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
fi
echo "   -> VPC: $VPC_ID | SG: $SG_ID"

# ------------------------------------------------------------------------------
# 3. Resolve Latest Official Ubuntu 24.04 LTS AMI ID
# ------------------------------------------------------------------------------
echo "[3/5] Resolving latest Ubuntu 24.04 LTS AMI in ${AWS_REGION}..."

AMI_ID=$(aws ssm get-parameter \
    --region "$AWS_REGION" \
    --name "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id" \
    --query "Parameter.Value" --output text 2>/dev/null || echo "")

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd*/ubuntu-*-24.04-amd64-server-*" "Name=state,Values=available" \
        --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" --output text)
fi
echo "   -> Selected AMI ID: $AMI_ID"

# ------------------------------------------------------------------------------
# 4. Create or Update EC2 Launch Template with User Data
# ------------------------------------------------------------------------------
echo "[4/5] Configuring EC2 Launch Template (${LT_NAME})..."

if [ "$(uname)" = "Darwin" ]; then
    USER_DATA_B64=$(base64 -i "$USER_DATA_FILE" | tr -d '\n')
else
    USER_DATA_B64=$(base64 -w 0 "$USER_DATA_FILE")
fi

LT_DATA=$(cat <<EOF
{
  "ImageId": "${AMI_ID}",
  "InstanceType": "${INSTANCE_TYPE}",
  "IamInstanceProfile": {
    "Name": "${PROFILE_NAME}"
  },
  "UserData": "${USER_DATA_B64}",
  "NetworkInterfaces": [{
    "DeviceIndex": 0,
    "AssociatePublicIpAddress": true,
    "Groups": ["${SG_ID}"],
    "DeleteOnTermination": true
  }],
  "BlockDeviceMappings": [{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 20,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }],
  "TagSpecifications": [{
    "ResourceType": "instance",
    "Tags": [
      {"Key": "Name", "Value": "${TAG_VALUE}"},
      {"Key": "ManagedBy", "Value": "GitHubActions-ASG-SelfHealing"}
    ]
  }]
}
EOF
)

if aws ec2 describe-launch-templates --region "$AWS_REGION" --launch-template-names "$LT_NAME" >/dev/null 2>&1; then
    echo "   -> Updating existing Launch Template ($LT_NAME) with new version..."
    NEW_VER=$(aws ec2 create-launch-template-version \
        --region "$AWS_REGION" \
        --launch-template-name "$LT_NAME" \
        --launch-template-data "$LT_DATA" \
        --query "LaunchTemplateVersion.VersionNumber" --output text)
    aws ec2 modify-launch-template \
        --region "$AWS_REGION" \
        --launch-template-name "$LT_NAME" \
        --default-version "$NEW_VER" >/dev/null
else
    echo "   -> Creating new Launch Template ($LT_NAME)..."
    aws ec2 create-launch-template \
        --region "$AWS_REGION" \
        --launch-template-name "$LT_NAME" \
        --launch-template-data "$LT_DATA" >/dev/null
fi

# ------------------------------------------------------------------------------
# 5. Create or Update Auto Scaling Group (Self-Healing Min=1, Desired=1, Max=1)
# ------------------------------------------------------------------------------
echo "[5/5] Ensuring Auto Scaling Group (${ASG_NAME})..."

EXISTING_ASG=$(aws autoscaling describe-auto-scaling-groups \
    --region "$AWS_REGION" \
    --auto-scaling-group-names "$ASG_NAME" \
    --query "AutoScalingGroups[0].AutoScalingGroupName" --output text 2>/dev/null || echo "None")

if [ "$EXISTING_ASG" != "None" ] && [ -n "$EXISTING_ASG" ]; then
    echo "   -> Updating Auto Scaling Group ($ASG_NAME) to ensure DesiredCapacity=1..."
    aws autoscaling update-auto-scaling-group \
        --region "$AWS_REGION" \
        --auto-scaling-group-name "$ASG_NAME" \
        --launch-template "LaunchTemplateName=${LT_NAME},Version=\$Default" \
        --min-size 1 \
        --max-size 1 \
        --desired-capacity 1 \
        --vpc-zone-identifier "$SUBNET_IDS"
else
    echo "   -> Creating new Auto Scaling Group ($ASG_NAME)..."
    aws autoscaling create-auto-scaling-group \
        --region "$AWS_REGION" \
        --auto-scaling-group-name "$ASG_NAME" \
        --launch-template "LaunchTemplateName=${LT_NAME},Version=\$Default" \
        --min-size 1 \
        --max-size 1 \
        --desired-capacity 1 \
        --health-check-type EC2 \
        --health-check-grace-period 300 \
        --vpc-zone-identifier "$SUBNET_IDS" \
        --tags "ResourceId=${ASG_NAME},ResourceType=auto-scaling-group,Key=Name,Value=${TAG_VALUE},PropagateAtLaunch=true"
fi

echo "=================================================================="
echo "✅ [Option 4] Self-Healing ASG & Launch Template Configured!"
echo "   If any EC2 instance is deleted/terminated, AWS ASG will"
echo "   automatically spawn a replacement and deploy Docker containers."
echo "=================================================================="
