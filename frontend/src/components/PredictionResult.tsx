import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, HelpCircle } from 'lucide-react';
import type { PredictionResponse } from '../types';

interface PredictionResultProps {
  result: PredictionResponse | null;
  loading: boolean;
  error: string | null;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  loading,
  error,
}) => {
  if (error) {
    return (
      <div className="card result-card error-state">
        <div className="result-error-box">
          <AlertTriangle size={32} className="error-icon" />
          <h3 className="error-title">추론 오류 발생</h3>
          <p className="error-msg">{error}</p>
          <span className="error-hint">
            FastAPI 백엔드가 실행 중인지 확인하세요:
            <code>python -m uvicorn API.api:app --port 8000</code>
          </span>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="card result-card loading-state">
        <div className="spinner-large"></div>
        <h3 className="loading-title">AI 모델 추론 진행 중...</h3>
        <p className="loading-sub">
          LightGBM 하이퍼파라미터 튜닝 모델이 센서 데이터와 파생 피처를 분석하고 있습니다.
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card result-card empty-state">
        <HelpCircle size={40} className="empty-icon" />
        <h3 className="empty-title">예측 결과 대기 중</h3>
        <p className="empty-sub">
          좌측 폼에 센서 수치를 입력하거나 상단 프리셋을 클릭한 후 <strong>'실시간 고장 예측 실행'</strong> 버튼을 눌러주세요.
        </p>
      </div>
    );
  }

  const isFailure = result.machine_failure === 1;
  const probPercent = (result.failure_probability * 100).toFixed(1);
  const probValue = result.failure_probability;

  // Level classification
  const levelClass = isFailure ? 'critical' : probValue > 0.25 ? 'warning' : 'safe';
  const levelText = isFailure
    ? '설비 고장 위험 (Machine Failure)'
    : probValue > 0.25
    ? '주의 관찰 단계 (Attention Required)'
    : '정상 운용 상태 (Normal Operation)';

  return (
    <div className={`card result-card state-${levelClass}`}>
      <div className="result-header">
        <span className="result-badge-label">추론 판정 결과</span>
        <span className={`result-status-pill ${levelClass}`}>
          {isFailure ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
          {isFailure ? '고장 발생 판정 (1)' : '정상 운용 판정 (0)'}
        </span>
      </div>

      <div className="result-main">
        <div className="result-metric-display">
          <div className="metric-score-label">예측 고장 확률 (Failure Probability)</div>
          <div className="metric-score-value">
            {probPercent}
            <span className="metric-score-percent">%</span>
          </div>
        </div>

        {/* Probability Bar */}
        <div className="prob-meter">
          <div className="prob-meter-labels">
            <span>안전 (0%)</span>
            <span>판정 기준 (50%)</span>
            <span>위험 (100%)</span>
          </div>
          <div className="prob-meter-track">
            <div
              className={`prob-meter-fill ${levelClass}`}
              style={{ width: `${Math.min(Math.max(probValue * 100, 4), 100)}%` }}
            ></div>
            <div className="prob-meter-threshold-marker" style={{ left: '50%' }}>
              <span className="threshold-tooltip">임계값 50%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="result-diagnosis">
        <h4 className="diagnosis-title">진단 상태: {levelText}</h4>
        <p className="diagnosis-desc">
          {isFailure
            ? '기계적 센서 데이터와 도메인 물리 조건 분석 결과, 임계 고장 기준치(50%)를 초과하였습니다. 즉각적인 공구 마모 상태 및 윤활, 발열 점검을 권장합니다.'
            : probValue > 0.25
            ? '현재 정상 범위이나 고장 확률이 다소 상승했습니다. 공구 마모 시간 및 토크 부하를 지속적으로 모니터링하세요.'
            : '모든 센서 지표가 매우 안정적이며, 고장 발생 확률이 극히 낮습니다. 정상 가공 작업을 계속 진행할 수 있습니다.'}
        </p>
      </div>

      <div className="result-footer-specs">
        <div className="spec-item">
          <span className="spec-label">적용 모델</span>
          <span className="spec-val">LightGBM (Week 06 Tuned)</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">테스트 정확도</span>
          <span className="spec-val">99.1% (오탐 4건 / 2,000건)</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">PR-AUC 성능</span>
          <span className="spec-val">0.898</span>
        </div>
      </div>
    </div>
  );
};
