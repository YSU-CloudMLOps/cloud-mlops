import React from 'react';
import { Gauge, RotateCcw, Play, Thermometer, Zap, Wrench } from 'lucide-react';
import type { ProductType, SensorInput } from '../types';

interface SensorFormProps {
  input: SensorInput;
  onChange: (input: SensorInput) => void;
  onSubmit: (e: React.FormEvent) => void;
  onReset: () => void;
  loading: boolean;
}

export const SensorForm: React.FC<SensorFormProps> = ({
  input,
  onChange,
  onSubmit,
  onReset,
  loading,
}) => {
  const handleTypeChange = (type: ProductType) => {
    onChange({ ...input, type });
  };

  const handleNumberChange = (field: keyof Omit<SensorInput, 'type'>, value: string) => {
    const num = parseFloat(value);
    onChange({
      ...input,
      [field]: isNaN(num) ? 0 : num,
    });
  };

  return (
    <form className="card form-card" onSubmit={onSubmit}>
      <div className="card-header">
        <div className="card-title-group">
          <Gauge size={20} className="card-icon" />
          <h2 className="card-title">설비 센서 파라미터 입력</h2>
        </div>
      </div>
      <p className="card-description">
        현재 가공 공정에서 수집된 6대 핵심 센서 수치를 입력하여 고장 확률을 추론합니다.
      </p>

      {/* Product Quality Type Selector */}
      <div className="form-group">
        <label className="form-label">제품 품질 유형 (Product Type)</label>
        <div className="segmented-control">
          {(['L', 'M', 'H'] as ProductType[]).map((typeOption) => (
            <button
              key={typeOption}
              type="button"
              className={`segment-btn ${input.type === typeOption ? 'active' : ''}`}
              onClick={() => handleTypeChange(typeOption)}
            >
              <strong>{typeOption} 타입</strong>
              <span className="segment-sub">
                {typeOption === 'L' ? '표준/저가 (60%)' : typeOption === 'M' ? '중급형 (30%)' : '고급형 (10%)'}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="form-grid">
        {/* Air Temperature */}
        <div className="input-group">
          <label htmlFor="air_temp" className="input-label">
            <Thermometer size={16} />
            <span>대기 온도 (Air Temp)</span>
          </label>
          <div className="input-with-unit">
            <input
              id="air_temp"
              type="number"
              step="0.1"
              min="250"
              max="350"
              required
              value={input.air_temperature_k || ''}
              onChange={(e) => handleNumberChange('air_temperature_k', e.target.value)}
              placeholder="예: 300.0"
            />
            <span className="unit-badge">K</span>
          </div>
          <span className="input-hint">정상 범위: 약 295.3 ~ 304.5 K</span>
        </div>

        {/* Process Temperature */}
        <div className="input-group">
          <label htmlFor="proc_temp" className="input-label">
            <Thermometer size={16} />
            <span>공정 온도 (Process Temp)</span>
          </label>
          <div className="input-with-unit">
            <input
              id="proc_temp"
              type="number"
              step="0.1"
              min="250"
              max="350"
              required
              value={input.process_temperature_k || ''}
              onChange={(e) => handleNumberChange('process_temperature_k', e.target.value)}
              placeholder="예: 310.0"
            />
            <span className="unit-badge">K</span>
          </div>
          <span className="input-hint">정상 범위: 약 305.7 ~ 313.8 K</span>
        </div>

        {/* Rotational Speed */}
        <div className="input-group">
          <label htmlFor="rot_speed" className="input-label">
            <Zap size={16} />
            <span>회전 속도 (Rotational Speed)</span>
          </label>
          <div className="input-with-unit">
            <input
              id="rot_speed"
              type="number"
              step="1"
              min="500"
              max="4000"
              required
              value={input.rotational_speed_rpm || ''}
              onChange={(e) => handleNumberChange('rotational_speed_rpm', e.target.value)}
              placeholder="예: 1500"
            />
            <span className="unit-badge">rpm</span>
          </div>
          <span className="input-hint">정상 범위: 약 1,168 ~ 2,886 rpm</span>
        </div>

        {/* Torque */}
        <div className="input-group">
          <label htmlFor="torque" className="input-label">
            <Gauge size={16} />
            <span>모터 토크 (Torque)</span>
          </label>
          <div className="input-with-unit">
            <input
              id="torque"
              type="number"
              step="0.1"
              min="0"
              max="150"
              required
              value={input.torque_nm || ''}
              onChange={(e) => handleNumberChange('torque_nm', e.target.value)}
              placeholder="예: 40.0"
            />
            <span className="unit-badge">N·m</span>
          </div>
          <span className="input-hint">정상 범위: 약 3.8 ~ 76.6 N·m</span>
        </div>

        {/* Tool Wear */}
        <div className="input-group full-width">
          <label htmlFor="tool_wear" className="input-label">
            <Wrench size={16} />
            <span>공구 누적 마모 시간 (Tool Wear)</span>
          </label>
          <div className="input-with-unit">
            <input
              id="tool_wear"
              type="number"
              step="1"
              min="0"
              max="350"
              required
              value={input.tool_wear_min ?? ''}
              onChange={(e) => handleNumberChange('tool_wear_min', e.target.value)}
              placeholder="예: 100"
            />
            <span className="unit-badge">분 (min)</span>
          </div>
          <span className="input-hint">주의: 200분 이상 도달 시 공구 파손 위험 (TWF 위험권)</span>
        </div>
      </div>

      <div className="form-actions">
        <button
          type="button"
          className="btn-secondary"
          onClick={onReset}
          disabled={loading}
        >
          <RotateCcw size={16} /> 초기화
        </button>
        <button
          type="submit"
          className="btn-primary"
          disabled={loading}
        >
          {loading ? (
            <span className="btn-content-loading">
              <span className="btn-spinner"></span> 추론 분석 중...
            </span>
          ) : (
            <>
              <Play size={16} /> 실시간 고장 예측 실행
            </>
          )}
        </button>
      </div>
    </form>
  );
};
