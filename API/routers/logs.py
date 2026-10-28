"""Endpoints for CloudWatch retraining/RL feedback logging and S3 log stream export."""

from typing import Any, Optional
from fastapi import APIRouter, Query, Request

from API.schemas import (
    FeedbackInput,
    FeedbackResponse,
    S3ExportRequest,
    S3ExportResponse,
)
from API.services.cloudwatch_service import CloudWatchLogService

router = APIRouter(tags=["MLOps Logging & S3 Export"])


def _get_cw_service(request: Request) -> CloudWatchLogService:
    cw_service = getattr(request.app.state, "cloudwatch_service", None)
    if cw_service is None:
        cw_service = CloudWatchLogService(settings=getattr(request.app.state, "settings", None))
        request.app.state.cloudwatch_service = cw_service
    return cw_service


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Log Ground-Truth Feedback & RL Reward",
    description=(
        "Record field operator action, ground-truth failure outcome (for supervised retraining), "
        "and calculated Reinforcement Learning reward transition (S_t, A_t, R_t, S_{t+1}) to CloudWatch."
    ),
)
def log_feedback(feedback: FeedbackInput, request: Request) -> FeedbackResponse:
    """Record field maintenance outcome and compute RL reward for CloudWatch log stream."""
    cw_service = _get_cw_service(request)
    return cw_service.log_feedback_event(feedback)


@router.get(
    "/logs/recent",
    summary="Inspect Recent Retraining & RL Log Events",
    description="Retrieve recent structured inference and feedback log records from the active log stream.",
)
def get_recent_logs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200, description="Maximum number of recent log records"),
) -> dict[str, Any]:
    """Return recent structured MLOps logs from the active CloudWatch Log Stream buffer."""
    cw_service = _get_cw_service(request)
    records = cw_service.get_recent_logs(limit=limit)
    return {
        "log_group": cw_service.log_group,
        "log_stream": cw_service.log_stream,
        "count": len(records),
        "records": records,
    }


@router.post(
    "/logs/export-s3",
    response_model=S3ExportResponse,
    summary="Export CloudWatch Log Stream to Amazon S3",
    description=(
        "Export a CloudWatch Log Stream containing Retraining and Reinforcement Learning logs "
        "to an Amazon S3 bucket as structured JSONL."
    ),
)
def export_logs_to_s3(
    request: Request,
    payload: Optional[S3ExportRequest] = None,
) -> S3ExportResponse:
    """Export the active CloudWatch Log Stream to Amazon S3 for offline retraining & RL."""
    cw_service = _get_cw_service(request)
    return cw_service.export_log_stream_to_s3(payload)
