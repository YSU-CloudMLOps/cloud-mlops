import type { MaintenanceAction, RiskTier } from '../types';

// Mirrors API/services/cloudwatch_service.py::recommend_maintenance_action
export const FAILURE_THRESHOLD = 0.5;
export const WARNING_THRESHOLD = 0.25;

export function classifyRisk(probability: number): RiskTier {
  if (probability >= FAILURE_THRESHOLD) return 'CRITICAL';
  if (probability >= WARNING_THRESHOLD) return 'WARNING';
  return 'SAFE';
}

export function recommendAction(probability: number): MaintenanceAction {
  const tier = classifyRisk(probability);
  if (tier === 'CRITICAL') return 'IMMEDIATE_SHUTDOWN_MAINTENANCE';
  if (tier === 'WARNING') return 'PREVENTIVE_INSPECTION';
  return 'CONTINUE_NORMAL_OPERATION';
}

export const ACTION_LABEL: Record<MaintenanceAction, string> = {
  CONTINUE_NORMAL_OPERATION: '연속 가동',
  PREVENTIVE_INSPECTION: '예방 점검 후 가동',
  IMMEDIATE_SHUTDOWN_MAINTENANCE: '즉시 정지·정비',
};

export const TIER_LABEL: Record<RiskTier, string> = {
  SAFE: '정상',
  WARNING: '주의',
  CRITICAL: '위험',
};
