import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { EquipmentInfo } from './components/EquipmentInfo';
import { Presets } from './components/Presets';
import { SensorForm } from './components/SensorForm';
import { DerivedMetrics } from './components/DerivedMetrics';
import { PredictionResult } from './components/PredictionResult';
import { HistoryTable } from './components/HistoryTable';
import { SupervisorView } from './components/SupervisorView';
import { PRESET_SCENARIOS } from './constants/presets';
import { fetchHealth, predictFailure } from './services/api';
import type { HealthResponse, HistoryItem, PredictionResponse, PresetScenario, SensorInput, UserRole } from './types';
import './App.css';

const DEFAULT_INPUT: SensorInput = PRESET_SCENARIOS[0].data;

export const App: React.FC = () => {
  const [role, setRole] = useState<UserRole>('engineer');
  const [input, setInput] = useState<SensorInput>(DEFAULT_INPUT);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Health check handler
  const checkHealth = useCallback(async () => {
    setLoadingHealth(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    // Poll every 30s
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, [checkHealth]);

  const handleSelectPreset = (preset: PresetScenario) => {
    setInput(preset.data);
    setError(null);
  };

  const handleReset = () => {
    setInput(DEFAULT_INPUT);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const pred = await predictFailure(input);
      setResult(pred);

      const historyItem: HistoryItem = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        timestamp: new Date().toLocaleTimeString('ko-KR', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        input: { ...input },
        result: pred,
      };

      setHistory((prev) => [historyItem, ...prev.slice(0, 9)]);
    } catch (err: any) {
      setError(err?.message || '알 수 없는 오류가 발생했습니다.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectHistory = (selectedInput: SensorInput) => {
    setInput(selectedInput);
    window.scrollTo({ top: 300, behavior: 'smooth' });
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="app-container">
      <Header
        health={health}
        loadingHealth={loadingHealth}
        onRefreshHealth={checkHealth}
        role={role}
        onChangeRole={setRole}
      />

      <main className="main-content">
        {role === 'supervisor' ? (
          <SupervisorView />
        ) : (
        <>
        {/* Target Equipment & Product Scope Section */}
        <section className="section-equipment-info">
          <EquipmentInfo />
        </section>

        {/* Presets Row */}
        <section className="section-presets">
          <Presets
            onSelectPreset={handleSelectPreset}
            currentInput={input}
          />
        </section>

        {/* Core Workspace Grid */}
        <div className="workspace-grid">
          {/* Left Column: Form & Real-time Derived Physics */}
          <div className="workspace-column left-column">
            <SensorForm
              input={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              onReset={handleReset}
              loading={loading}
            />

            <DerivedMetrics input={input} />
          </div>

          {/* Right Column: Prediction Result & MLOps Model Overview */}
          <div className="workspace-column right-column">
            <PredictionResult
              result={result}
              loading={loading}
              error={error}
            />

            {/* Model Architecture Info Card */}
            <div className="card model-info-card">
              <span className="eyebrow-mono">// ARCHITECTURE BENCHMARK</span>
              <h3 className="info-card-title">v2 종합 모델 개선 &amp; 오탐 0건 검증</h3>
              <p className="info-card-desc">
                IEEE-754 방열 경계 수정, 6종 연속 마진 피처 및 Optuna 5-Fold 계층화 OOF 앙상블로 최적화된 프로덕션 LightGBM v2 모델의 성능입니다.
              </p>

              <div className="stats-mini-grid">
                <div className="stat-box">
                  <span className="stat-label">// ACCURACY</span>
                  <span className="stat-number highlight">99.45%</span>
                  <span className="stat-sub">1,989 / 2,000 정상 분류</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">// PRECISION</span>
                  <span className="stat-number">100.0%</span>
                  <span className="stat-sub">오탐(FP) 0건 달성</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">// F1-SCORE</span>
                  <span className="stat-number">0.912</span>
                  <span className="stat-sub">베이스라인 대비 +0.050 향상</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">// PR-AUC</span>
                  <span className="stat-number">0.909</span>
                  <span className="stat-sub">Brier Score 0.006 (확률 보정)</span>
                </div>
              </div>

              <div className="info-features-list">
                <span className="features-title">// ENGINEERED FEATURES (23):</span>
                <div className="feature-tags">
                  <code>type_encoded</code>
                  <code>air_temp</code>
                  <code>proc_temp</code>
                  <code>rot_speed</code>
                  <code>torque</code>
                  <code>tool_wear</code>
                  <code>temp_diff</code>
                  <code>power_w</code>
                  <code>strain</code>
                  <code>wear_critical</code>
                  <code>hdf_risk</code>
                  <code>pwf_risk</code>
                  <code>osf_risk</code>
                  <code>twf_zone</code>
                  <code>physical_risk_sum</code>
                  <code>temp_ratio</code>
                  <code>torque_rpm_ratio</code>
                  <code>osf_margin</code>
                  <code>osf_strain_ratio</code>
                  <code>hdf_temp_margin</code>
                  <code>hdf_rpm_margin</code>
                  <code>pwf_margin</code>
                  <code>twf_wear_excess</code>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section: History Table */}
        <section className="section-history">
          <HistoryTable
            history={history}
            onSelectHistory={handleSelectHistory}
            onClearHistory={handleClearHistory}
          />
        </section>
        </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          AI4I 2020 Predictive Maintenance MLOps System &bull; Week 06 Hyperparameter Tuning &amp; Web Serving
        </p>
        <p className="footer-sub">
          FastAPI Backend (Port 8000) &bull; React + Vite + TypeScript Frontend (Port 5173)
        </p>
      </footer>
    </div>
  );
};

export default App;
