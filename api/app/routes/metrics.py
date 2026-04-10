from fastapi import APIRouter

router = APIRouter(tags=["metrics"])

# Prometheus metrics are exposed by the instrumentator middleware at /metrics.
# This router exists as a placeholder if custom metrics are needed beyond
# what prometheus-fastapi-instrumentator provides automatically.
