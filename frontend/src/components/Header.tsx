import React from 'react';
import { Activity, CheckCircle2, AlertCircle, RefreshCw, Cpu } from 'lucide-react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, loadingHealth, onRefreshHealth }) => {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="logo-icon-wrap">
          <Activity className="brand-icon" size={28} />
        </div>
        <div>
          <h1 className="brand-title">AI4I 설비 예지보전 AI 관제 서비스</h1>
          <p className="brand-subtitle">
            스마트 밀링 설비 실시간 센서 고장 진단 및 조기 경보 시스템 (Week 06)
          </p>
        </div>
      </div>

      <div className="header-status-area">
        <div className="model-chip">
          <Cpu size={16} />
          <span>LightGBM Tuned (정확도 99.1% | PR-AUC 0.898)</span>
        </div>

        <div className={`status-badge ${health?.model_loaded ? 'online' : 'offline'}`}>
          {health?.model_loaded ? (
            <>
              <span className="status-dot green"></span>
              <CheckCircle2 size={15} />
              <span>API 정상 작동 (v{health.version})</span>
            </>
          ) : (
            <>
              <span className="status-dot red"></span>
              <AlertCircle size={15} />
              <span>서버 연결 대기 중</span>
            </>
          )}
          <button
            type="button"
            className="icon-btn-refresh"
            onClick={onRefreshHealth}
            disabled={loadingHealth}
            title="서버 상태 새로고침"
          >
            <RefreshCw size={14} className={loadingHealth ? 'spin' : ''} />
          </button>
        </div>
      </div>
    </header>
  );
};
