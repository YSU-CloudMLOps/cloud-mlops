"""CLI utility and verification pipeline for CloudWatch RL/Retraining logging and S3 export.

Usage:
    # 1. Verify model, API, CloudWatch logging, and export a Log Stream to S3:
    PYTHONPATH=. python scripts/export_logs_to_s3.py --simulate

    # 2. Export an existing CloudWatch Log Stream to S3:
    PYTHONPATH=. python scripts/export_logs_to_s3.py \
        --log-group /mlops/predictive-maintenance/inference \
        --log-stream api-inference-stream-v2 \
        --s3-bucket cloud-mlops-pdm-logs \
        --s3-prefix cloudwatch-exports/retraining-rl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from API.api import app
from API.schemas import S3ExportRequest


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "Scenario 1: Normal Operation (TN)",
        "equipment_id": "CNC-MILL-01",
        "sensor": {
            "type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 100.0,
        },
        "feedback": {
            "operator_action": "CONTINUE",
            "actual_failure": 0,
            "actual_failure_mode": "NONE",
            "downtime_minutes": 0.0,
            "maintenance_cost_usd": 0.0,
            "next_state_sensors": {
                "type": "L",
                "air_temperature_k": 300.1,
                "process_temperature_k": 310.2,
                "rotational_speed_rpm": 1495.0,
                "torque_nm": 40.5,
                "tool_wear_min": 105.0,
            },
            "notes": "정상 가공 범위 유지, 무정지 연속 생산 완료.",
        },
    },
    {
        "name": "Scenario 2: Heat Dissipation Failure - HDF (TP)",
        "equipment_id": "CNC-MILL-02",
        "sensor": {
            "type": "L",
            "air_temperature_k": 302.5,
            "process_temperature_k": 310.5,
            "rotational_speed_rpm": 1320.0,
            "torque_nm": 52.0,
            "tool_wear_min": 110.0,
        },
        "feedback": {
            "operator_action": "COOLING_REPAIR",
            "actual_failure": 1,
            "actual_failure_mode": "HDF",
            "downtime_minutes": 12.0,
            "maintenance_cost_usd": 80.0,
            "next_state_sensors": {
                "type": "L",
                "air_temperature_k": 299.8,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1510.0,
                "torque_nm": 41.0,
                "tool_wear_min": 110.0,
            },
            "notes": "온도차 8.0K (<8.6K) 및 저속 회전 감지 후 냉각수 노즐 막힘 즉시 해소, 주축 과열 손상 예방.",
        },
    },
    {
        "name": "Scenario 3: Overstrain & Tool Wear Failure - OSF/TWF (TP)",
        "equipment_id": "CNC-MILL-01",
        "sensor": {
            "type": "L",
            "air_temperature_k": 300.2,
            "process_temperature_k": 310.4,
            "rotational_speed_rpm": 1350.0,
            "torque_nm": 58.5,
            "tool_wear_min": 215.0,
        },
        "feedback": {
            "operator_action": "REPLACE_TOOL",
            "actual_failure": 1,
            "actual_failure_mode": "OSF",
            "downtime_minutes": 15.0,
            "maintenance_cost_usd": 120.0,
            "next_state_sensors": {
                "type": "L",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.1,
                "rotational_speed_rpm": 1550.0,
                "torque_nm": 38.0,
                "tool_wear_min": 0.0,
            },
            "notes": "누적 스트레인 12,577.5 min·Nm (>11,000 한계) 경보 확인 후 공구 즉시 교체, 실제 절삭날 균열 확인.",
        },
    },
    {
        "name": "Scenario 4: Power Failure - PWF (TP)",
        "equipment_id": "CNC-MILL-03",
        "sensor": {
            "type": "M",
            "air_temperature_k": 299.8,
            "process_temperature_k": 309.5,
            "rotational_speed_rpm": 2650.0,
            "torque_nm": 35.0,
            "tool_wear_min": 80.0,
        },
        "feedback": {
            "operator_action": "POWER_CHECK",
            "actual_failure": 1,
            "actual_failure_mode": "PWF",
            "downtime_minutes": 10.0,
            "maintenance_cost_usd": 60.0,
            "next_state_sensors": {
                "type": "M",
                "air_temperature_k": 299.8,
                "process_temperature_k": 309.6,
                "rotational_speed_rpm": 1800.0,
                "torque_nm": 35.0,
                "tool_wear_min": 80.0,
            },
            "notes": "기계 전력 9,712.76W (>9,000W 상한) 감지 후 인버터 출력 조정 완료.",
        },
    },
]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute end-to-end API inference, RL feedback logging, and S3 Log Stream export."""
    root_dir = Path(__file__).resolve().parents[1]
    logs_dir = root_dir / "logs"
    if args.simulate and logs_dir.exists():
        for existing in logs_dir.glob("*.jsonl"):
            existing.unlink()
        s3_dir = logs_dir / "s3_exports"
        if s3_dir.exists():
            for existing in s3_dir.glob("*.jsonl"):
                existing.unlink()

    with TestClient(app) as client:
        if args.simulate:
            print("=== [1/3] Running Model Inference & Emitting CloudWatch Trajectory Logs ===")
            for item in SCENARIOS:
                pred_res = client.post(
                    "/predict",
                    json=item["sensor"],
                    headers={"X-Equipment-ID": item["equipment_id"]},
                )
                pred_res.raise_for_status()
                pred_id = pred_res.headers["X-Prediction-ID"]
                body = pred_res.json()
                print(
                    f"  • {item['name']} -> prediction_id={pred_id} | "
                    f"machine_failure={body['machine_failure']} | "
                    f"prob={body['failure_probability']:.6f}"
                )

                fb_payload = {
                    "prediction_id": pred_id,
                    "equipment_id": item["equipment_id"],
                    **item["feedback"],
                }
                fb_res = client.post("/feedback", json=fb_payload)
                fb_res.raise_for_status()
                fb_body = fb_res.json()
                print(
                    f"    ↳ Feedback Logged: event_id={fb_body['event_id']} | "
                    f"outcome={fb_body['outcome_type']} | "
                    f"RL_reward={fb_body['computed_reward']:+.2f}"
                )

        print("\n=== [2/3] Exporting CloudWatch Log Stream to Amazon S3 ===")
        export_req = S3ExportRequest(
            log_group_name=args.log_group,
            log_stream_name=args.log_stream,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            export_mode=args.export_mode,
        )
        export_res = client.post("/logs/export-s3", json=export_req.model_dump(exclude_none=True))
        export_res.raise_for_status()
        result = export_res.json()

        # Write canonical deterministic S3 export file for repository verification
        if result.get("local_export_path"):
            src = Path(result["local_export_path"])
            canonical = src.parent / "api-inference-stream-v2-s3-export.jsonl"
            if src.exists() and src != canonical:
                src.replace(canonical)
                result["local_export_path"] = str(canonical)

        print(f"  • Status           : {result['status']}")
        print(f"  • Export Mode      : {result['export_mode']}")
        print(f"  • Export Task ID   : {result['task_id']}")
        print(f"  • Source Log Group : {result['log_group']}")
        print(f"  • Source Log Stream: {result['log_stream']}")
        print(f"  • Target S3 URI    : {result['s3_uri']}")
        print(f"  • Exported Records : {result['exported_records']}")
        print(f"  • Local Snapshot   : {result.get('local_export_path')}")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CloudWatch Retraining/RL Log Stream to Amazon S3."
    )
    parser.add_argument("--simulate", action="store_true", help="Simulate inference and RL feedback events before export")
    parser.add_argument("--log-group", default="/mlops/predictive-maintenance/inference", help="CloudWatch Log Group name")
    parser.add_argument("--log-stream", default="api-inference-stream-v2", help="CloudWatch Log Stream name")
    parser.add_argument("--s3-bucket", default="cloud-mlops-pdm-logs", help="Destination S3 bucket name")
    parser.add_argument("--s3-prefix", default="cloudwatch-exports/retraining-rl", help="Destination S3 prefix")
    parser.add_argument(
        "--export-mode",
        default="auto",
        choices=["auto", "cloudwatch_export_task", "s3_direct_upload", "local_archive"],
        help="Export execution strategy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
