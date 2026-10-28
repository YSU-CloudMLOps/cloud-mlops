import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ListChecks } from 'lucide-react';
import { FLEET } from '../constants/fleet';
import { predictFailure } from '../services/api';
import { ACTION_LABEL, classifyRisk, recommendAction, TIER_LABEL } from '../utils/risk';
import { BatchPlanner } from './BatchPlanner';
import { SupervisorLog } from './SupervisorLog';
import type { SensorInput, SupervisorLogEntry } from '../types';

const LOG_STORAGE_KEY = 'supervisor-activity-log';

function loadStoredLogs(): SupervisorLogEntry[] {
  try {
    const raw = localStorage.getItem(LOG_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((l) => l?.sensors) : [];
  } catch {
    return [];
  }
}
import { FleetBoard, type FleetStatus } from './FleetBoard';

export const SupervisorView: React.FC = () => {
  const [statuses, setStatuses] = useState<FleetStatus[]>(
    FLEET.map((equipment) => ({ equipment, result: null, error: null })),
  );
  const [loading, setLoading] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string>(FLEET[0].id);
  const [logs, setLogs] = useState<SupervisorLogEntry[]>(loadStoredLogs);
  // 로그에서 불러온 조건. nonce가 바뀌면 시뮬레이터 폼이 새 값으로 다시 만들어진다.
  const [loaded, setLoaded] = useState<{ sensors: SensorInput; nonce: number } | null>(null);

  const addLogs = useCallback((entries: Omit<SupervisorLogEntry, 'id' | 'timestamp'>[]) => {
    const timestamp = new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const stamped = entries.map((e) => ({
      ...e,
      id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      timestamp,
    }));
    setLogs((prev) => [...stamped, ...prev].slice(0, 50));
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    const next = await Promise.all(
      FLEET.map(async (equipment): Promise<FleetStatus> => {
        try {
          const result = await predictFailure(equipment.sensors, equipment.id);
          return { equipment, result, error: null };
        } catch (err: any) {
          return { equipment, result: null, error: err?.message || '추론 실패' };
        }
      }),
    );
    // 위험도 높은 설비를 먼저 보여준다. 진단 실패 설비는 맨 뒤.
    next.sort((a, b) => (b.result?.failure_probability ?? -1) - (a.result?.failure_probability ?? -1));
    setStatuses(next);
    addLogs(
      next
        .filter((n) => n.result)
        .map((n) => {
          const sensors = n.equipment.sensors;
          const prob = n.result!.failure_probability;
          return {
            kind: '전체 진단' as const,
            equipmentId: n.equipment.id,
            sensors,
            condition: `${sensors.type}등급 · ${sensors.rotational_speed_rpm}rpm · ${sensors.torque_nm}Nm · 마모 ${sensors.tool_wear_min}min`,
            probability: prob,
            action: recommendAction(prob),
            note: `${n.equipment.line} · ${n.equipment.currentJob}`,
          };
        }),
    );
    setUpdatedAt(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    setLoading(false);
  }, [addLogs]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    try {
      localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(logs));
    } catch {
      // 저장소를 쓸 수 없는 환경(사생활 보호 모드 등)에서는 메모리에서만 유지한다.
    }
  }, [logs]);

  const handleSelect = (id: string) => {
    setLoaded(null);
    setSelectedId(id);
  };

  const handleLoadLog = (entry: SupervisorLogEntry) => {
    setSelectedId(entry.equipmentId);
    setLoaded({ sensors: entry.sensors, nonce: Date.now() });
    document.getElementById('batch-planner')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const selected = FLEET.find((f) => f.id === selectedId) ?? FLEET[0];
  const handover = useMemo(
    () => statuses.filter((s) => s.result && classifyRisk(s.result.failure_probability) !== 'SAFE'),
    [statuses],
  );
  const failedCount = statuses.filter((s) => s.error).length;

  return (
    <>
      <FleetBoard
        fleet={statuses}
        loading={loading}
        updatedAt={updatedAt}
        selectedId={selectedId}
        onRefresh={refresh}
        onSelect={handleSelect}
      />

      {failedCount > 0 && (
        <div className="planner-error">
          {failedCount}대 설비 진단에 실패했습니다. FastAPI 백엔드가 실행 중인지 확인하세요.
        </div>
      )}

      <div className="workspace-grid">
        {/* key로 설비 변경 시 폼을 해당 설비 센서값으로 초기화 */}
        <BatchPlanner
          key={`${selected.id}-${loaded?.nonce ?? 0}`}
          equipment={selected}
          initial={loaded?.sensors}
          onLog={(e) => addLogs([e])} />

        <div className="card handover-card">
          <div className="card-header">
            <div>
              <span className="eyebrow-mono">// SHIFT HANDOVER</span>
              <div className="card-title-group">
                <ListChecks size={18} className="card-icon" />
                <h2 className="card-title">교대 인계 체크리스트</h2>
              </div>
            </div>
          </div>
          <p className="card-description">
            정상이 아닌 설비만 우선순위 순으로 모았습니다. 다음 조에 인계할 조치 항목입니다.
          </p>
          {handover.length === 0 ? (
            <p className="handover-empty">조치가 필요한 설비가 없습니다. 모든 설비가 정상 범위입니다.</p>
          ) : (
            <ol className="handover-list">
              {handover.map(({ equipment, result }) => {
                const prob = result!.failure_probability;
                const tier = classifyRisk(prob).toLowerCase();
                return (
                  <li key={equipment.id} className={`handover-item tier-${tier}`} onClick={() => handleSelect(equipment.id)} title="클릭하면 투입 시뮬레이터에 이 설비를 적용합니다">
                    <div>
                      <span className="fleet-id">{equipment.id}</span>
                      <div className="fleet-sub">{equipment.line} · {equipment.currentJob}</div>
                    </div>
                    <div className="handover-right">
                      <span className={`risk-pill tier-${tier}`}>{TIER_LABEL[classifyRisk(prob)]} {(prob * 100).toFixed(0)}%</span>
                      <span className="fleet-sub">{ACTION_LABEL[recommendAction(prob)]}</span>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      </div>

      <SupervisorLog logs={logs} onClear={() => setLogs([])} onLoad={handleLoadLog} />
    </>
  );
};
