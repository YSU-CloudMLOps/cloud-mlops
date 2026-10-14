import React from 'react';
import { History, Trash2, ArrowUpRight } from 'lucide-react';
import type { HistoryItem, SensorInput } from '../types';

interface HistoryTableProps {
  history: HistoryItem[];
  onSelectHistory: (input: SensorInput) => void;
  onClearHistory: () => void;
}

export const HistoryTable: React.FC<HistoryTableProps> = ({
  history,
  onSelectHistory,
  onClearHistory,
}) => {
  if (history.length === 0) return null;

  return (
    <div className="card history-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// AUDIT TRAIL</span>
          <div className="card-title-group">
            <History size={18} className="card-icon" />
            <h2 className="card-title">최근 추론 이력 ({history.length}건)</h2>
          </div>
        </div>
        <button
          type="button"
          className="btn-outline-pill btn-clear"
          onClick={onClearHistory}
        >
          <Trash2 size={13} /> 이력 비우기
        </button>
      </div>

      <div className="table-wrapper">
        <table className="history-table">
          <thead>
            <tr>
              <th>// TIMESTAMP</th>
              <th>// TYPE</th>
              <th>// AIR / PROC TEMP</th>
              <th>// ROTATIONAL SPEED</th>
              <th>// TORQUE</th>
              <th>// TOOL WEAR</th>
              <th>// VERDICT</th>
              <th>// PROBABILITY</th>
              <th>// ACTION</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => {
              const isFail = item.result.machine_failure === 1;
              return (
                <tr key={item.id}>
                  <td className="cell-time">{item.timestamp}</td>
                  <td>
                    <span className="type-badge">{item.input.type}</span>
                  </td>
                  <td className="cell-mono">
                    {item.input.air_temperature_k} / {item.input.process_temperature_k} K
                  </td>
                  <td className="cell-mono">{item.input.rotational_speed_rpm.toLocaleString()} rpm</td>
                  <td className="cell-mono">{item.input.torque_nm} N·m</td>
                  <td className="cell-mono">{item.input.tool_wear_min} min</td>
                  <td>
                    <span className={`status-pill-sm ${isFail ? 'fail' : 'pass'}`}>
                      {isFail ? '고장 (1)' : '정상 (0)'}
                    </span>
                  </td>
                  <td className="cell-prob">
                    {(item.result.failure_probability * 100).toFixed(1)}%
                  </td>
                  <td>
                    <button
                      type="button"
                      className="btn-table-action"
                      onClick={() => onSelectHistory(item.input)}
                      title="이 데이터 폼에 다시 불러오기"
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
    </div>
  );
};
