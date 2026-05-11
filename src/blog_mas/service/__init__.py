"""Production service layer — Ch10 §1.2: FastAPI orchestration layer.

Exposes the Context Engine as a network-accessible service with:
  • async task dispatch (decoupled from execution)
  • trace_id-keyed result retrieval
  • Prometheus metrics
  • OpenTelemetry distributed tracing
  • API-key auth and pre-flight moderation at the API edge
"""
