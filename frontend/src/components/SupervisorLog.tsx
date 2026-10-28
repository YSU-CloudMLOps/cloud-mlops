import React, { useMemo, useState } from 'react';
import { ArrowUpRight, Download, History, Trash2 } from 'lucide-react';
import type { SupervisorLogEntry } from '../types';
import { ACTION_LABEL, classifyRisk, TIER_LABEL } from '../utils/risk';

interface SupervisorLogProps {
  logs: SupervisorLogEntry[];
  onClear: () => void;
  onLoad: (entry: SupervisorLogEntry) => void;
}

type Filter = 'all' | 'fleet' | 'simulation' | 'risky';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: '전체' },
  { key: 'fleet', label: '전체 진단' },
  { key: 'simulation', label: '투입 시뮬레이션' },
  { key: 'risky', label: '위험·주의만' },
];

function downloadCsv(logs: SupervisorLogEntry[]) {
  const header = ['시각', '구분', '설비', '제품등급', '대기온도K', '공정온도K', '회전수rpm', '토크Nm', '공구마모min', '고장확률', '위험도', '권고', '비고'];
  const rows = logs.map((l) => [
    l.timestamp, l.kind, l.equipmentId, l.sensors.type, l.sensors.air_temperature_k, l.sensors.process_temperature_k,
    l.sensors.rotational_speed_rpm, l.sensors.torque_nm, l.sensors.tool_wear_min,
    (l.probability * 100).toFixed(1) + '%', TIER_LABEL[classifyRisk(l.probability)], ACTION_LABEL[l.action], l.note,
  ]);
  const csv = [header, ...rows].map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
  const url = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'shift-activity-log.csv';
  a.click();
  URL.revokeObjectURL(url);
}

export const SupervisorLog: React.FC<SupervisorLogProps> = ({ logs, onClear, onLoad }) => {
  const [filter, setFilter] = useState<Filter>('all');

  const visible = useMemo(
    () =>
      logs.filter((l) => {
        if (filter === 'fleet') return l.kind === '전체 진단';
        if (filter === 'simulation') return l.kind === '투입 시뮬레이션';
        if (filter === 'risky') return classifyRisk(l.probability) !== 'SAFE';
        return true;
      }),
    [logs, filter],
  );

  return (
    <div className="card history-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// SHIFT ACTIVITY LOG</span>
          <div className="card-title-group">
            <History size={18} className="card-icon" />
            <h2 className="card-title">교대 운영 로그 ({visible.length}/{logs.length}건)</h2>
          </div>
        </div>
        <div className="log-actions">
          <button type="button" className="btn-outline-pill btn-compact" onClick={() => downloadCsv(visible)} disabled={visible.length === 0}>
            <Download size={13} /> CSV 내보내기
          </button>
          <button type="button" className="btn-outline-pill btn-compact" onClick={onClear} disabled={logs.length === 0}>
            <Trash2 size={13} /> 로그 비우기
          </button>
        </div>
      </div>
      <p className="card-description">
        라인 전체 진단과 투입 시뮬레이션 결과를 최신순으로 기록합니다. 최근 50건을 이 브라우저에 보관하며, '불러오기'로 당시 조건을 시뮬레이터에 다시 채울 수 있습니다.
      </p>

      <div className="log-filters" role="group" aria-label="로그 필터">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            type="button"
            className={`log-filter ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {visible.length === 0 ? (
        <p className="handover-empty">
          {logs.length === 0 ? '아직 기록이 없습니다.' : '조건에 맞는 기록이 없습니다.'}
        </p>
      ) : (
        <div className="table-wrapper">
          <table className="history-table">
            <thead>
              <tr>
                <th>시각</th>
                <th>구분</th>
                <th>설비</th>
                <th>조건</th>
                <th>위험도</th>
                <th>고장 확률</th>
                <th>권고 / 판단</th>
                <th>작업</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((log) => {
                const tier = classifyRisk(log.probability);
                return (
                  <tr key={log.id}>
                    <td className="cell-time">{log.timestamp}</td>
                    <td>{log.kind}</td>
                    <td className="cell-mono">{log.equipmentId}</td>
                    <td className="cell-mono">{log.condition}</td>
                    <td>
                      <span className={`risk-pill tier-${tier.toLowerCase()}`}>{TIER_LABEL[tier]}</span>
                    </td>
                    <td className="cell-prob">{(log.probability * 100).toFixed(1)}%</td>
                    <td>
                      {ACTION_LABEL[log.action]}
                      <div className="log-note">{log.note}</div>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn-table-action"
                        onClick={() => onLoad(log)}
                        title="이 조건을 투입 시뮬레이터에 다시 불러오기"
                      >
                        <ArrowUpRight size={13} /> 불러오기
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
