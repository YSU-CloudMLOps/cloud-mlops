export type ProductType = 'L' | 'M' | 'H';

export interface SensorInput {
  type: ProductType;
  air_temperature_k: number;
  process_temperature_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
}

export interface PredictionResponse {
  machine_failure: 0 | 1;
  failure_probability: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}

export interface PresetScenario {
  id: string;
  title: string;
  badge: '정상' | '고장 위험' | '경고';
  badgeColor: 'green' | 'red' | 'yellow';
  description: string;
  data: SensorInput;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  input: SensorInput;
  result: PredictionResponse;
}

export type UserRole = 'engineer' | 'supervisor';

export type RiskTier = 'SAFE' | 'WARNING' | 'CRITICAL';

export type MaintenanceAction =
  | 'CONTINUE_NORMAL_OPERATION'
  | 'PREVENTIVE_INSPECTION'
  | 'IMMEDIATE_SHUTDOWN_MAINTENANCE';

export interface EquipmentSnapshot {
  id: string;
  name: string;
  line: string;
  currentJob: string;
  sensors: SensorInput;
}
