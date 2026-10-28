import React, { useState } from 'react';
import { ClipboardCheck, Wrench, Play, AlertTriangle } from 'lucide-react';
import type { EquipmentSnapshot, PredictionResponse, ProductType, SensorInput, SupervisorLogEntry } from '../types';
import { predictFailure } from '../services/api';
import { ACTION_LABEL, classifyRisk, recommendAction, TIER_LABEL } from '../utils/risk';

interface BatchPlannerProps {
  equipment: EquipmentSnapshot;
  onLog: (entry: Omit<SupervisorLogEntry, 'id' | 'timestamp'>) => void;
}

interface Comparison {
  keep: PredictionResponse;
  replaced: PredictionResponse;
}

function decide(c: Comparison): { title: string; detail: string; tone: 'safe' | 'warning' | 'critical' } {
  const keepTier = classifyRisk(c.keep.failure_probability);
  const replacedTier = classifyRisk(c.replaced.failure_probability);
  if (keepTier === 'SAFE') {
    return {
      title: '현재 공구로 연속 가동',
      detail: '공구를 교체하지 않아도 투입 조건이 안전 범위입니다. 불필요한 교체 정지 시간과 공구 비용을 아낄 수 있습니다.',
      tone: 'safe',
    };
  }
  if (replacedTier === 'SAFE') {
    return {
      title: '공구 교체 후 가동',
      detail: '현재 공구로는 위험도가 높지만 교체하면 안전 범위로 내려갑니다. 교체 후 투입을 권장합니다.',
      tone: keepTier === 'CRITICAL' ? 'critical' : 'warning',
    };
  }
  return {
    title: '투입 조건 재조정 또는 정비 요청',
    detail: '공구를 교체해도 위험도가 남습니다. 목표 토크·회전수가 과하거나 방열·전력 문제일 수 있으니 투입 조건을 낮추거나 보전팀에 점검을 요청하세요.',
    tone: 'critical',
  };
}

export const BatchPlanner: React.FC<BatchPlannerProps> = ({ equipment, onLog }) => {
  const [type, setType] = useState<ProductType>(equipment.sensors.type);
  const [rpm, setRpm] = useState<number>(equipment.sensors.rotational_speed_rpm);
  const [torque, setTorque] = useState<number>(equipment.sensors.torque_nm);
  const [wear, setWear] = useState<number>(equipment.sensors.tool_wear_min);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const base: SensorInput = {
      ...equipment.sensors,
      type,
      rotational_speed_rpm: rpm,
      torque_nm: torque,
      tool_wear_min: wear,
    };
    try {
      const [keep, replaced] = await Promise.all([
        predictFailure(base, equipment.id),
        predictFailure({ ...base, tool_wear_min: 0 }, equipment.id),
      ]);
      const result = { keep, replaced };
      setComparison(result);
      onLog({
        kind: '투입 시뮬레이션',
        equipmentId: equipment.id,
        condition: `${type}등급 · ${rpm}rpm · ${torque}Nm · 마모 ${wear}min`,
        probability: keep.failure_probability,
        action: recommendAction(keep.failure_probability),
        note: `${decide(result).title} (교체 시 ${(replaced.failure_probability * 100).toFixed(1)}%)`,
      });
    } catch (err: any) {
      setError(err?.message || '알 수 없는 오류가 발생했습니다.');
      setComparison(null);
    } finally {
      setLoading(false);
    }
  };

  const verdict = comparison ? decide(comparison) : null;

  const renderScenario = (label: string, icon: React.ReactNode, res: PredictionResponse) => {
    const tier = classifyRisk(res.failure_probability);
    return (
      <div className={`scenario-box tier-${tier.toLowerCase()}`}>
        <div className="scenario-head">
          <span className="stat-label">{icon} {label}</span>
          <span className={`risk-pill tier-${tier.toLowerCase()}`}>{TIER_LABEL[tier]}</span>
        </div>
        <div className="fleet-prob">
          {(res.failure_probability * 100).toFixed(1)}
          <span className="metric-score-percent">%</span>
        </div>
        <span className="stat-sub">{ACTION_LABEL[recommendAction(res.failure_probability)]}</span>
      </div>
    );
  };

  return (
    <div className="card planner-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// NEXT BATCH GO / NO-GO</span>
          <div className="card-title-group">
            <ClipboardCheck size={18} className="card-icon" />
            <h2 className="card-title">다음 배치 투입 시뮬레이터</h2>
          </div>
        </div>
        <span className="card-badge-info">{equipment.id} · {equipment.currentJob}</span>
      </div>
      <p className="card-description">
        다음 작업 지시의 제품 등급과 목표 토크·회전수를 넣으면, 현재 공구로 계속 가동할 때와 공구를 교체했을 때의 고장 확률을 나란히 비교합니다.
      </p>

      <form className="planner-form" onSubmit={run}>
        <div className="planner-field">
          <span className="input-label">다음 배치 제품 등급</span>
          <div className="segmented-control">
            {(['L', 'M', 'H'] as ProductType[]).map((t) => (
              <button
                key={t}
                type="button"
                className={`segment-btn ${type === t ? 'active' : ''}`}
                onClick={() => setType(t)}
              >
                <span className="segment-main">{t}</span>
                <span className="segment-sub">{t === 'L' ? '보급형' : t === 'M' ? '표준형' : '고정밀'}</span>
              </button>
            ))}
          </div>
        </div>
        <label className="planner-field">
          <span className="input-label">목표 회전수</span>
          <div className="input-with-unit">
            <input type="number" min={1} step="any" value={rpm} onChange={(e) => setRpm(Number(e.target.value))} required />
            <span className="unit-badge">rpm</span>
          </div>
        </label>
        <label className="planner-field">
          <span className="input-label">목표 토크</span>
          <div className="input-with-unit">
            <input type="number" min={0} step="any" value={torque} onChange={(e) => setTorque(Number(e.target.value))} required />
            <span className="unit-badge">Nm</span>
          </div>
        </label>
        <label className="planner-field">
          <span className="input-label">현재 공구 마모</span>
          <div className="input-with-unit">
            <input type="number" min={0} step="any" value={wear} onChange={(e) => setWear(Number(e.target.value))} required />
            <span className="unit-badge">min</span>
          </div>
        </label>
        <button type="submit" className="btn-primary-pill planner-submit" disabled={loading}>
          {loading ? <span className="btn-content-loading"><span className="btn-spinner"></span>비교 중...</span> : <><Play size={14} /> 투입 시뮬레이션</>}
        </button>
      </form>

      {error && (
        <div className="planner-error">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {comparison && verdict && (
        <>
          <div className="scenario-grid">
            {renderScenario(`현재 공구 유지 (${wear}min)`, <Play size={11} />, comparison.keep)}
            {renderScenario('공구 교체 후 (0min)', <Wrench size={11} />, comparison.replaced)}
          </div>
          <div className={`verdict-box tone-${verdict.tone}`}>
            <span className="eyebrow-mono">// RECOMMENDED DECISION</span>
            <h3 className="verdict-title">{verdict.title}</h3>
            <p className="diagnosis-desc">{verdict.detail}</p>
          </div>
        </>
      )}
    </div>
  );
};
