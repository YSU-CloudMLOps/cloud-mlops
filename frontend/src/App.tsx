import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Presets } from './components/Presets';
import { SensorForm } from './components/SensorForm';
import { DerivedMetrics } from './components/DerivedMetrics';
import { PredictionResult } from './components/PredictionResult';
import { HistoryTable } from './components/HistoryTable';
import { PRESET_SCENARIOS } from './constants/presets';
import { fetchHealth, predictFailure } from './services/api';
import type { HealthResponse, HistoryItem, PredictionResponse, PresetScenario, SensorInput } from './types';
import './App.css';

const DEFAULT_INPUT: SensorInput = PRESET_SCENARIOS[0].data;

export const App: React.FC = () => {
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
      />

      <main className="main-content">
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
              <h3 className="info-card-title">Week 06 하이퍼파라미터 튜닝 성능</h3>
              <p className="info-card-desc">
                Optuna 베이지안 최적화(TPE Sampler)와 5-Fold 계층화 교차 검증을 통해 최적화된 LightGBM 모델의 성능입니다.
              </p>

              <div className="stats-mini-grid">
                <div className="stat-box">
                  <span className="stat-label">테스트 정확도 (Accuracy)</span>
                  <span className="stat-number highlight">99.1%</span>
                  <span className="stat-sub">1,982 / 2,000 정상 분류</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">정밀도 (Precision)</span>
                  <span className="stat-number">93.1%</span>
                  <span className="stat-sub">오탐(FP) 4건으로 축소</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">F1-Score</span>
                  <span className="stat-number">0.857</span>
                  <span className="stat-sub">최적 임계값 적용 시 0.859</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">PR-AUC</span>
                  <span className="stat-number">0.898</span>
                  <span className="stat-sub">베이스라인 대비 +0.010 향상</span>
                </div>
              </div>

              <div className="info-features-list">
                <span className="features-title">주요 입력 및 파생 피처 (10개):</span>
                <div className="feature-tags">
                  <code>type_encoded</code>
                  <code>air_temperature_k</code>
                  <code>process_temperature_k</code>
                  <code>rotational_speed_rpm</code>
                  <code>torque_nm</code>
                  <code>tool_wear_min</code>
                  <code>temp_diff_k</code>
                  <code>power_w</code>
                  <code>strain_min_nm</code>
                  <code>tool_wear_critical</code>
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
