import React from 'react';
import { History, Trash2 } from 'lucide-react';
import type { SupervisorLogEntry } from '../types';
import { ACTION_LABEL, classifyRisk, TIER_LABEL } from '../utils/risk';

interface SupervisorLogProps {
  logs: SupervisorLogEntry[];
  onClear: () => void;
}

export const SupervisorLog: React.FC<SupervisorLogProps> = ({ logs, onClear }) => (
  <div className="card history-card">
    <div className="card-header">
      <div>
        <span className="eyebrow-mono">// SHIFT ACTIVITY LOG</span>
        <div className="card-title-group">
          <History size={18} className="card-icon" />
          <h2 className="card-title">교대 운영 로그 ({logs.length}건)</h2>
        </div>
      </div>
      <button type="button" className="btn-outline-pill btn-clear" onClick={onClear} disabled={logs.length === 0}>
        <Trash2 size={13} /> 로그 비우기
      </button>
    </div>
    <p className="card-description">
      라인 전체 진단과 투입 시뮬레이션 결과를 최신순으로 기록합니다. 최근 50건까지 보관하며, 새로고침하면 초기화됩니다.
    </p>

    {logs.length === 0 ? (
      <p className="handover-empty">아직 기록이 없습니다.</p>
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
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => {
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
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    )}
  </div>
);
