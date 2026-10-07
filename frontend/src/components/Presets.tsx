import React from 'react';
import { Bookmark, Sparkles } from 'lucide-react';
import { PRESET_SCENARIOS } from '../constants/presets';
import type { PresetScenario, SensorInput } from '../types';

interface PresetsProps {
  onSelectPreset: (preset: PresetScenario) => void;
  currentInput: SensorInput;
}

export const Presets: React.FC<PresetsProps> = ({ onSelectPreset }) => {
  return (
    <div className="card presets-card">
      <div className="card-header">
        <div className="card-title-group">
          <Bookmark size={20} className="card-icon" />
          <h2 className="card-title">테스트 시나리오 프리셋</h2>
        </div>
        <span className="card-badge-info">
          <Sparkles size={14} /> 클릭 시 자동 입력
        </span>
      </div>
      <p className="card-description">
        공학적 도메인 고장 조건(방열 실패, 과부하, 전력 이상)이 반영된 예제 데이터를 즉시 적용하여 진단해 보세요.
      </p>

      <div className="presets-grid">
        {PRESET_SCENARIOS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className="preset-btn"
            onClick={() => onSelectPreset(preset)}
          >
            <div className="preset-top">
              <span className="preset-title">{preset.title}</span>
              <span className={`preset-tag tag-${preset.badgeColor}`}>
                {preset.badge}
              </span>
            </div>
            <p className="preset-desc">{preset.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
