"""Feature engineering service for predictive maintenance."""

from math import pi
import pandas as pd

from API.schemas import SensorInput

TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}
OSF_THRESHOLD_MAPPING = {0: 11000.0, 1: 12000.0, 2: 13000.0}


def make_features(sensor: SensorInput) -> pd.DataFrame:
    """Transform raw sensor input into engineered feature DataFrame for model inference.

    Generates 23 domain-engineered features:
    - 10 Base features (type_encoded, 5 raw sensors, temp_diff_k, power_w, strain_min_nm, tool_wear_critical)
    - 7 Physical boundary & interaction features (exact IEEE-754 hdf_risk, pwf_risk, osf_risk, twf_zone,
      physical_risk_sum, temp_ratio, torque_rpm_ratio)
    - 6 Continuous boundary margin features (osf_margin, osf_strain_ratio, hdf_temp_margin,
      hdf_rpm_margin, pwf_margin, twf_wear_excess)
    """
    type_enc = TYPE_MAPPING[sensor.type]
    raw_temp_diff = sensor.process_temperature_k - sensor.air_temperature_k
    temp_diff = round(raw_temp_diff, 2)
    power = round(sensor.torque_nm * (sensor.rotational_speed_rpm * 2 * pi / 60), 2)
    strain = round(sensor.tool_wear_min * sensor.torque_nm, 2)
    tool_critical = int(sensor.tool_wear_min >= 200)

    # Exact physical failure boundary indicators
    osf_thresh = OSF_THRESHOLD_MAPPING[type_enc]
    hdf_risk = int((raw_temp_diff < 8.6) and (sensor.rotational_speed_rpm < 1380))
    pwf_risk = int((power < 3500.0) or (power > 9000.0))
    osf_risk = int(strain > osf_thresh)
    twf_zone = int((sensor.tool_wear_min >= 200) and (sensor.tool_wear_min <= 240))
    physical_risk_sum = hdf_risk + pwf_risk + osf_risk

    # Ratios
    temp_ratio = round(sensor.process_temperature_k / sensor.air_temperature_k, 4)
    torque_rpm_ratio = round(sensor.torque_nm / sensor.rotational_speed_rpm, 6)

    # Continuous boundary margins
    osf_margin = round(strain - osf_thresh, 2)
    osf_strain_ratio = round(strain / osf_thresh, 4)
    hdf_temp_margin = round(8.6 - raw_temp_diff, 4)
    hdf_rpm_margin = round(1380.0 - sensor.rotational_speed_rpm, 2)
    pwf_margin = round(min(power - 3500.0, 9000.0 - power), 2)
    twf_wear_excess = round(max(0.0, float(sensor.tool_wear_min) - 198.0), 2)

    return pd.DataFrame([{
        "type_encoded": type_enc,
        "air_temperature_k": sensor.air_temperature_k,
        "process_temperature_k": sensor.process_temperature_k,
        "rotational_speed_rpm": sensor.rotational_speed_rpm,
        "torque_nm": sensor.torque_nm,
        "tool_wear_min": sensor.tool_wear_min,
        "temp_diff_k": temp_diff,
        "power_w": power,
        "strain_min_nm": strain,
        "tool_wear_critical": tool_critical,
        "hdf_risk": hdf_risk,
        "pwf_risk": pwf_risk,
        "osf_risk": osf_risk,
        "twf_zone": twf_zone,
        "physical_risk_sum": physical_risk_sum,
        "temp_ratio": temp_ratio,
        "torque_rpm_ratio": torque_rpm_ratio,
        "osf_margin": osf_margin,
        "osf_strain_ratio": osf_strain_ratio,
        "hdf_temp_margin": hdf_temp_margin,
        "hdf_rpm_margin": hdf_rpm_margin,
        "pwf_margin": pwf_margin,
        "twf_wear_excess": twf_wear_excess,
    }])
