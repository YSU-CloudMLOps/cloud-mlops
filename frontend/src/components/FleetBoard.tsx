import React from 'react';
import { RefreshCw, Factory, AlertTriangle, CheckCircle2, Eye } from 'lucide-react';
import type { EquipmentSnapshot, PredictionResponse, RiskTier } from '../types';
import { ACTION_LABEL, classifyRisk, recommendAction, TIER_LABEL } from '../utils/risk';

export interface FleetStatus {
  equipment: EquipmentSnapshot;
  result: PredictionResponse | null;
  error: string | null;
}

interface FleetBoardProps {
  fleet: FleetStatus[];
  loading: boolean;
  updatedAt: string | null;
  selectedId: string | null;
  onRefresh: () => void;
  onSelect: (id: string) => void;
}

const TIER_ICON: Record<RiskTier, React.ReactNode> = {
  SAFE: <CheckCircle2 size={12} />,
  WARNING: <Eye size={12} />,
  CRITICAL: <AlertTriangle size={12} />,
};

export const FleetBoard: React.FC<FleetBoardProps> = ({
  fleet,
  loading,
  updatedAt,
  selectedId,
  onRefresh,
  onSelect,
}) => {
  const predicted = fleet.filter((f) => f.result);
  const count = (tier: RiskTier) =>
    predicted.filter((f) => classifyRisk(f.result!.failure_probability) === tier).length;
  const avg = predicted.length
    ? (predicted.reduce((s, f) => s + f.result!.failure_probability, 0) / predicted.length) * 100
    : null;

  return (
    <div className="card fleet-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// LINE STATUS BOARD</span>
          <div className="card-title-group">
            <Factory size={18} className="card-icon" />
            <h2 className="card-title">라인 설비 현황 보드</h2>
          </div>
        </div>
        <button type="button" className="btn-outline-pill btn-compact" onClick={onRefresh} disabled={loading}>
          <RefreshCw size={12} className={loading ? 'spin' : ''} />
          <span>{updatedAt ? `${updatedAt} 갱신` : '전체 재진단'}</span>
        </button>
      </div>
      <p className="card-description">
        담당 라인 CNC 설비를 한 화면에서 비교합니다. 위험도 높은 설비가 위로 정렬되며, 카드를 선택하면 아래 투입 시뮬레이터에 해당 설비가 적용됩니다.
      </p>

      <div className="kpi-row">
        <div className="kpi-box">
          <span className="stat-label">// MONITORED</span>
          <span className="stat-number">{fleet.length}<small> 대</small></span>
        </div>
        <div className="kpi-box">
          <span className="stat-label">// CRITICAL</span>
          <span className="stat-number kpi-red">{count('CRITICAL')}<small> 대</small></span>
        </div>
        <div className="kpi-box">
          <span className="stat-label">// WARNING</span>
          <span className="stat-number kpi-yellow">{count('WARNING')}<small> 대</small></span>
        </div>
        <div className="kpi-box">
          <span className="stat-label">// AVG FAILURE PROB</span>
          <span className="stat-number">{avg === null ? '–' : avg.toFixed(1)}<small>%</small></span>
        </div>
      </div>

      <div className="fleet-grid">
        {fleet.map(({ equipment, result, error }) => {
          const prob = result?.failure_probability ?? null;
          const tier = prob === null ? null : classifyRisk(prob);
          const wearPct = Math.min((equipment.sensors.tool_wear_min / 240) * 100, 100);
          return (
            <button
              key={equipment.id}
              type="button"
              className={`fleet-item ${tier ? `tier-${tier.toLowerCase()}` : ''} ${selectedId === equipment.id ? 'selected' : ''}`}
              onClick={() => onSelect(equipment.id)}
            >
              <div className="fleet-item-top">
                <span className="fleet-id">{equipment.id}</span>
                {tier ? (
                  <span className={`risk-pill tier-${tier.toLowerCase()}`}>
                    {TIER_ICON[tier]} {TIER_LABEL[tier]}
                  </span>
                ) : (
                  <span className="risk-pill">{error ? '오류' : '진단 중'}</span>
                )}
              </div>
              <div className="fleet-sub">
                {equipment.line} · {equipment.name} · {equipment.currentJob}
              </div>
              <div className="fleet-prob">
                {prob === null ? '–' : (prob * 100).toFixed(1)}
                <span className="metric-score-percent">%</span>
              </div>
              <div className="wear-track" title={`공구 마모 ${equipment.sensors.tool_wear_min}분 / 한계 240분`}>
                <div className={`wear-fill ${wearPct >= 83 ? 'hot' : ''}`} style={{ width: `${wearPct}%` }}></div>
              </div>
              <div className="fleet-meta">
                <span>공구 마모 {equipment.sensors.tool_wear_min}min</span>
                <span>{prob === null ? '' : ACTION_LABEL[recommendAction(prob)]}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
