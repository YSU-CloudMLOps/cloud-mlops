import type { PresetScenario } from '../types';

export const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'normal',
    title: '정상 가동 설비',
    badge: '정상',
    badgeColor: 'green',
    description: '공정 온도차와 부하 토크가 정상 범위 내에서 안정적으로 동작 중인 표준 공정입니다.',
    data: {
      type: 'L',
      air_temperature_k: 298.1,
      process_temperature_k: 308.6,
      rotational_speed_rpm: 1551,
      torque_nm: 42.8,
      tool_wear_min: 15,
    },
  },
  {
    id: 'hdf_risk',
    title: '열 방출 실패 (HDF)',
    badge: '고장 위험',
    badgeColor: 'red',
    description: '공정 온도와 대기 온도의 차이가 8.6K 미만이고 회전 속도가 낮아 과열 발생 위험이 높은 상태입니다.',
    data: {
      type: 'M',
      air_temperature_k: 302.5,
      process_temperature_k: 310.8,
      rotational_speed_rpm: 1320,
      torque_nm: 58.0,
      tool_wear_min: 140,
    },
  },
  {
    id: 'osf_twf_risk',
    title: '과부하 및 마모 (OSF/TWF)',
    badge: '고장 위험',
    badgeColor: 'red',
    description: '공구 마모가 200분을 초과하고 고토크가 인가되어 누적 스트레인이 한계치를 초과한 상태입니다.',
    data: {
      type: 'L',
      air_temperature_k: 300.2,
      process_temperature_k: 310.5,
      rotational_speed_rpm: 1410,
      torque_nm: 65.4,
      tool_wear_min: 215,
    },
  },
  {
    id: 'pwf_risk',
    title: '전력 이상 (PWF)',
    badge: '고장 위험',
    badgeColor: 'red',
    description: '고속 회전축과 높은 토크의 결합으로 순간 모터 소모 전력이 9,000W를 초과하는 과부하 상태입니다.',
    data: {
      type: 'H',
      air_temperature_k: 299.0,
      process_temperature_k: 309.2,
      rotational_speed_rpm: 2820,
      torque_nm: 36.5,
      tool_wear_min: 85,
    },
  },
];
