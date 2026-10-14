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
          <Activity size={22} className="brand-icon" />
        </div>
        <div className="brand-text-group">
          <span className="eyebrow-mono">// TELEMETRY & PREDICTIVE SURVEILLANCE</span>
          <h1 className="brand-title">AI4I 설비 예지보전 AI 관제 시스템</h1>
          <p className="brand-subtitle">
            스마트 CNC 밀링 공작기계 실시간 센서 고장 진단 및 부품 품질 등급(L/M/H)별 조기 경보 인프라
          </p>
        </div>
      </div>

      <div className="header-status-area">
        <div className="model-chip">
          <Cpu size={14} />
          <span>LightGBM Tuned (정확도 99.1% · PR-AUC 0.898)</span>
        </div>

        <div className={`status-badge ${health?.model_loaded ? 'online' : 'offline'}`}>
          {health?.model_loaded ? (
            <>
              <span className="status-dot green"></span>
              <CheckCircle2 size={13} />
              <span>API 온라인 (v{health.version})</span>
            </>
          ) : (
            <>
              <span className="status-dot red"></span>
              <AlertCircle size={13} />
              <span>서버 연결 대기 중</span>
            </>
          )}
          <button
            type="button"
            className="icon-btn-refresh"
            onClick={onRefreshHealth}
            disabled={loadingHealth}
            title="서버 상태 새로고침"
            aria-label="새로고침"
          >
            <RefreshCw size={12} className={loadingHealth ? 'spin' : ''} />
          </button>
        </div>
      </div>
    </header>
  );
};
