import type { EquipmentSnapshot } from '../types';

// 데모용 라인 설비 스냅샷. 실제 환경에서는 MES/SCADA 수집값으로 대체한다.
export const FLEET: EquipmentSnapshot[] = [
  {
    id: 'CNC-MILL-01',
    name: '1호기',
    line: 'A라인',
    currentJob: '브래킷 (고정밀)',
    sensors: { type: 'H', air_temperature_k: 298.1, process_temperature_k: 308.6, rotational_speed_rpm: 1551, torque_nm: 42.8, tool_wear_min: 15 },
  },
  {
    id: 'CNC-MILL-02',
    name: '2호기',
    line: 'A라인',
    currentJob: '하우징 (표준)',
    sensors: { type: 'M', air_temperature_k: 299.0, process_temperature_k: 309.4, rotational_speed_rpm: 1500, torque_nm: 40.0, tool_wear_min: 120 },
  },
  {
    id: 'CNC-MILL-03',
    name: '3호기',
    line: 'A라인',
    currentJob: '플랜지 (보급형)',
    sensors: { type: 'L', air_temperature_k: 300.2, process_temperature_k: 310.5, rotational_speed_rpm: 1410, torque_nm: 65.4, tool_wear_min: 215 },
  },
  {
    id: 'CNC-MILL-04',
    name: '4호기',
    line: 'B라인',
    currentJob: '기어박스 (표준)',
    sensors: { type: 'M', air_temperature_k: 302.5, process_temperature_k: 310.8, rotational_speed_rpm: 1320, torque_nm: 58.0, tool_wear_min: 140 },
  },
  {
    id: 'CNC-MILL-05',
    name: '5호기',
    line: 'B라인',
    currentJob: '샤프트 (고정밀)',
    sensors: { type: 'H', air_temperature_k: 299.0, process_temperature_k: 309.2, rotational_speed_rpm: 2820, torque_nm: 36.5, tool_wear_min: 85 },
  },
  {
    id: 'CNC-MILL-06',
    name: '6호기',
    line: 'B라인',
    currentJob: '커버 (보급형)',
    sensors: { type: 'L', air_temperature_k: 298.5, process_temperature_k: 308.9, rotational_speed_rpm: 1480, torque_nm: 45.0, tool_wear_min: 180 },
  },
];
