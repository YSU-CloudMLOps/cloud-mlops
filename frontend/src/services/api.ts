import type { HealthResponse, PredictionResponse, SensorInput } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export async function fetchHealth(): Promise<HealthResponse> {
  const url = `${API_BASE_URL}/health`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`서버 응답 오류 (HTTP ${response.status})`);
    }
    return await response.json();
  } catch (error) {
    // If proxy failed, try direct connection to localhost:8000
    if (!API_BASE_URL) {
      const fallbackUrl = 'http://127.0.0.1:8000/health';
      const response = await fetch(fallbackUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) throw error;
      return await response.json();
    }
    throw error;
  }
}

export async function predictFailure(payload: SensorInput): Promise<PredictionResponse> {
  const url = `${API_BASE_URL}/predict`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `추론 요청 실패 (HTTP ${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (Array.isArray(errorJson.detail)) {
            detail = errorJson.detail.map((d: any) => `${d.loc?.join('.')}: ${d.msg}`).join(', ');
          } else {
            detail = String(errorJson.detail);
          }
        }
      } catch {
        // ignore json parse error
      }
      throw new Error(detail);
    }

    return await response.json();
  } catch (error) {
    // Direct fallback if relative proxy is not accessible
    if (!API_BASE_URL) {
      const fallbackUrl = 'http://127.0.0.1:8000/predict';
      const fallbackResp = await fetch(fallbackUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      if (fallbackResp.ok) {
        return await fallbackResp.json();
      }
    }
    throw error;
  }
}
