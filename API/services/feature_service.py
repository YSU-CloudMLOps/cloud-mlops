"""Feature engineering service for predictive maintenance."""

from math import pi
import pandas as pd

from API.schemas import SensorInput

TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}


def make_features(sensor: SensorInput) -> pd.DataFrame:
    """Transform raw sensor input into engineered feature DataFrame for model inference.

    Features generated:
    - type_encoded: Categorical encoding of quality variant (L:0, M:1, H:2)
    - air_temperature_k: Ambient air temperature [K]
    - process_temperature_k: Operating process temperature [K]
    - rotational_speed_rpm: Spindle rotational speed [rpm]
    - torque_nm: Spindle torque [Nm]
    - tool_wear_min: Cumulative tool wear time [min]
    - temp_diff_k: Process temperature - Air temperature [K]
    - power_w: Mechanical rotational power [W] (Torque * Speed * 2 * pi / 60)
    - strain_min_nm: Cumulative strain factor (Tool wear * Torque) [min*Nm]
    - tool_wear_critical: Binary indicator for critical tool wear (>= 200 min)
    """
    type_enc = TYPE_MAPPING[sensor.type]
    temp_diff = round(sensor.process_temperature_k - sensor.air_temperature_k, 2)
    power = round(sensor.torque_nm * sensor.rotational_speed_rpm * 2 * pi / 60, 2)
    strain = round(sensor.tool_wear_min * sensor.torque_nm, 2)
    tool_critical = int(sensor.tool_wear_min >= 200)

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
    }])
