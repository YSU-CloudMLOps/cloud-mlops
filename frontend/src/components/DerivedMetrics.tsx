import React from 'react';
import { Calculator, AlertTriangle, CheckCircle2 } from 'lucide-react';
import type { SensorInput } from '../types';

interface DerivedMetricsProps {
  input: SensorInput;
}

export const DerivedMetrics: React.FC<DerivedMetricsProps> = ({ input }) => {
  const tempDiff = Number((input.process_temperature_k - input.air_temperature_k).toFixed(2));
  const powerWatts = Number((input.torque_nm * input.rotational_speed_rpm * (2 * Math.PI / 60)).toFixed(2));
  const strain = Number((input.tool_wear_min * input.torque_nm).toFixed(2));
  const isCriticalWear = input.tool_wear_min >= 200;

  // Domain thresholds
  const isHdfRisk = tempDiff < 8.6 && input.rotational_speed_rpm <= 1380;
  const isPwfRisk = powerWatts < 3500 || powerWatts > 9000;
  const strainThreshold = input.type === 'L' ? 11000 : input.type === 'M' ? 12000 : 13000;
  const isOsfRisk = strain > strainThreshold;

  return (
    <div className="card metrics-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// PHYSICAL DYNAMICS &amp; DERIVED FEATURES</span>
          <div className="card-title-group">
            <Calculator size={18} className="card-icon" />
            <h2 className="card-title">도메인 물리 파생 지표 (Feature Engineering)</h2>
          </div>
        </div>
      </div>
      <p className="card-description">
        입력된 센서값을 바탕으로 모델 추론에 직접 반영되는 4대 핵심 역학 지표를 실시간 산출합니다.
      </p>

      <div className="metrics-grid">
        {/* Temp Difference */}
        <div className={`metric-box ${isHdfRisk ? 'risk' : 'normal'}`}>
          <div className="metric-header">
            <span className="metric-name">// TEMP DIFF (ΔT)</span>
            {isHdfRisk ? (
              <span className="metric-pill risk">
                <AlertTriangle size={12} /> 방열 위험 (&lt;8.6K)
              </span>
            ) : (
              <span className="metric-pill safe">
                <CheckCircle2 size={12} /> 정상 방열
              </span>
            )}
          </div>
          <div className="metric-value">
            {tempDiff} <span className="metric-unit">K</span>
          </div>
          <div className="metric-calc">공정온도 - 대기온도 (HDF 인자)</div>
        </div>

        {/* Rotational Power */}
        <div className={`metric-box ${isPwfRisk ? 'risk' : 'normal'}`}>
          <div className="metric-header">
            <span className="metric-name">// ROTATIONAL POWER</span>
            {isPwfRisk ? (
              <span className="metric-pill risk">
                <AlertTriangle size={12} /> 전력 이상
              </span>
            ) : (
              <span className="metric-pill safe">
                <CheckCircle2 size={12} /> 정상 전력
              </span>
            )}
          </div>
          <div className="metric-value">
            {powerWatts.toLocaleString()} <span className="metric-unit">W</span>
          </div>
          <div className="metric-calc">기준 3,500 ~ 9,000 W (PWF 인자)</div>
        </div>

        {/* Strain Factor */}
        <div className={`metric-box ${isOsfRisk ? 'risk' : 'normal'}`}>
          <div className="metric-header">
            <span className="metric-name">// ACCUMULATED STRAIN</span>
            {isOsfRisk ? (
              <span className="metric-pill risk">
                <AlertTriangle size={12} /> 한계 초과
              </span>
            ) : (
              <span className="metric-pill safe">
                <CheckCircle2 size={12} /> 스트레인 안전
              </span>
            )}
          </div>
          <div className="metric-value">
            {strain.toLocaleString()} <span className="metric-unit">min·N·m</span>
          </div>
          <div className="metric-calc">임계 {strainThreshold.toLocaleString()} ({input.type}타입)</div>
        </div>

        {/* Tool Wear Status */}
        <div className={`metric-box ${isCriticalWear ? 'risk' : 'normal'}`}>
          <div className="metric-header">
            <span className="metric-name">// TOOL WEAR STATE</span>
            {isCriticalWear ? (
              <span className="metric-pill risk">
                <AlertTriangle size={12} /> 교체 권장 구간
              </span>
            ) : (
              <span className="metric-pill safe">
                <CheckCircle2 size={12} /> 마모 양호
              </span>
            )}
          </div>
          <div className="metric-value">
            {input.tool_wear_min} <span className="metric-unit">min</span>
          </div>
          <div className="metric-calc">임계점: 200 ~ 240 min (TWF 인자)</div>
        </div>
      </div>
    </div>
  );
};
