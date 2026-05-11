# Chapter 10 → blog-mas: Implementation Index

> Every topic from Chapter 10 "The Blueprint for Production-Ready AI" accounted
> for in the blog-mas codebase.  Use this table as the handshake between the
> book and the project.
>
> **Key insight:** Ch10 is entirely wrapper architecture.  No engine module was
> touched (`engine_graph.py`, `executor.py`, `planner.py`, `validators.py`,
> `context_engine.py`, `tracer.py`, `registry.py`, `resolver.py`, all agents).
> This is the same architectural discipline as Ch8 (meta-controller) and Ch9
> (domain independence) — production readiness lives in the surrounding layers,
> not the reasoning core.

---

## Phase 1 — Productionising the Glass-Box Engine

### §1.1 Environment Configuration and Secrets Management

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Twelve-Factor App: separate config from code | `src/blog_mas/config.py` | `Settings` class reads 100% from `os.environ`; `get_settings()` cached with `lru_cache` |
| `GENERATION_MODEL = os.environ.get(...)` | `src/blog_mas/config.py::LLMSettings` | Book's exact pattern, extended to all settings |
| Fail fast if essential API keys missing | `src/blog_mas/config.py::Settings.validate()` | Called at startup in production; lists every missing var in one error |
| python-dotenv for local dev | `src/blog_mas/config.py::_load_dotenv()` | Loads `.env` before Settings reads env; K8s injects directly |
| Documented env vars | `.env.example` | Every var with comment, safe default, and purpose |
| Centralised secrets management | `src/blog_mas/secrets.py` | `SecretsBackend` protocol + 4 implementations |
| HashiCorp Vault | `src/blog_mas/secrets.py::VaultBackend` | KV v2 HTTP fetch; caches all secrets in one call |
| AWS Secrets Manager | `src/blog_mas/secrets.py::AWSSecretsManagerBackend` | boto3 JSON-object fetch |
| File backend (K8s Secret volume / init-container) | `src/blog_mas/secrets.py::FileBackend` | Reads from `/var/run/secrets/blog-mas/` |
| Env backend (default) | `src/blog_mas/secrets.py::EnvBackend` | What K8s injects after sidecar runs |
| `load_secrets_into_env()` at startup | `src/blog_mas/secrets.py::load_secrets_into_env` | Bridges any backend to `os.environ`; called from `api.py` lifespan |
| K8s ConfigMap for non-sensitive config | `deploy/k8s/configmap.yaml` | All non-secret settings declared |
| K8s Secret for credentials | `deploy/k8s/secret.yaml` | Placeholder + ExternalSecrets operator pattern documented |

---

### §1.2 Building the Production API (Orchestration Layer)

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| FastAPI as production framework | `src/blog_mas/service/api.py` | `FastAPI(title="Context Engine Service")` — book's exact title |
| Async/await, Pydantic validation, OpenAPI docs | `src/blog_mas/service/api.py` | Built into FastAPI; `/docs` and `/redoc` auto-generated |
| `GoalRequest(BaseModel)` | `src/blog_mas/service/schemas.py::GoalRequest` | Book's exact shape + async lifecycle fields |
| `ExecutionResponse(BaseModel)` | `src/blog_mas/service/schemas.py::GoalAcceptedResponse + StatusResponse` | Split for async lifecycle (202 vs. poll) |
| `POST /api/v1/execute` | `src/blog_mas/service/api.py::execute_goal_endpoint` | Returns 202 Accepted + trace_id immediately |
| API does NOT execute the engine itself | `src/blog_mas/service/api.py` | `_enqueue_task()` dispatches to Celery; engine runs in worker |
| Lifespan: initialise clients at startup | `src/blog_mas/service/api.py::lifespan` | LLM, QdrantStore, embedder, registry created once |
| API gateway: auth (API keys / JWT) | `src/blog_mas/service/auth.py` | `require_api_key` dependency + `HTTPBearer` stub |
| API gateway: rate limiting | `deploy/k8s/ingress.yaml` annotations | `nginx.ingress.kubernetes.io/limit-rps: "20"` |
| API gateway: SSL termination | `deploy/k8s/ingress.yaml` | cert-manager TLS block; the Ingress terminates HTTPS |
| Status polling endpoint | `src/blog_mas/service/api.py::GET /api/v1/status/{trace_id}` | Returns pending/running/ok/blocked_pre/etc. |
| Full trace endpoint (compliance) | `src/blog_mas/service/api.py::GET /api/v1/trace/{trace_id}` | Returns `TraceResponse` with full ExecutionTrace |
| Webhook endpoint | `src/blog_mas/service/api.py::POST /api/v1/webhook/register` | HMAC-signed outbound POST on task completion |
| `uv run blog-mas serve` | `src/blog_mas/cli.py::_cmd_serve` + `build_parser` | Starts uvicorn on the FastAPI app from the CLI |

---

### §1.3 Asynchronous Execution and Task Queues

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| API as dispatcher, not executor | `src/blog_mas/service/api.py::execute_goal_endpoint` | Validates → enqueues → returns 202 |
| Step 1: Request reception | `src/blog_mas/service/api.py::execute_goal_endpoint` | Pydantic validates body |
| Step 2: Dispatch to queue | `src/blog_mas/service/api.py::_enqueue_task` | `execute_goal.apply_async(...)` |
| Step 3: 202 Accepted + trace_id | `src/blog_mas/service/api.py` | `GoalAcceptedResponse(status="accepted", trace_id=...)` |
| Step 4: Worker pulls and executes | `src/blog_mas/workers/tasks.py::execute_goal` | Celery task, `acks_late=True`, `reject_on_worker_lost=True` |
| Step 5: Result stored by trace_id | `src/blog_mas/workers/tasks.py` calls `result_store.save(trace_id, ...)` | Both `RedisResultStore` and `FilesystemResultStore` |
| Step 6: Poll or webhook | `GET /api/v1/status/{trace_id}` + `POST /api/v1/webhook/register` | Both patterns implemented |
| Redis as broker | `src/blog_mas/workers/celery_app.py` | `broker=settings.queue.celery_broker_url` |
| Celery task queue | `src/blog_mas/workers/celery_app.py` + `tasks.py` | Full Celery application with retry, acks_late, time limits |
| `acks_late=True` (reliability) | `src/blog_mas/workers/celery_app.py` + `tasks.py` | Broker re-queues on crash |
| `task_reject_on_worker_lost=True` | `src/blog_mas/workers/celery_app.py` | Guards against silent loss on worker kill |
| Exponential backoff retries | `src/blog_mas/workers/tasks.py::execute_goal` | `countdown=30 * (2 ** self.request.retries)` |
| Result store: Redis (primary) | `src/blog_mas/service/result_store.py::RedisResultStore` | TTL-aware; SETEX with configured TTL |
| Result store: filesystem (fallback) | `src/blog_mas/service/result_store.py::FilesystemResultStore` | Dev/test without Redis |
| Factory picks backend from config | `src/blog_mas/service/result_store.py::get_result_store` | Pings Redis; falls back to filesystem |
| Initialise resources once per worker process | `src/blog_mas/workers/tasks.py::_get_worker_resources` | Module-level dict; LLM/store/registry built on first task |

---

### §1.4 Centralised Logging and Observability

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Machine-readable structured logs (JSON) | `src/blog_mas/logging_json.py` | structlog with `JSONRenderer()` in production |
| `trace_id` in every log line | `src/blog_mas/logging_json.py::bind_trace_context` | `structlog.contextvars` binds trace_id, request_id, service |
| Console renderer for development | `src/blog_mas/logging_json.py` | `LOG_FORMAT=console` → `ConsoleRenderer` |
| `setup_logging()` backward compat | `src/blog_mas/logging_config.py` | Shim that delegates to `logging_json.setup_logging` |
| Centralized log aggregation (ELK/Splunk) | `deploy/docker-compose.yml` + `deploy/otel-collector-config.yaml` | OTel collector receives logs; real ELK would add a Fluentd sidecar |
| System metrics (CPU, memory, network) | K8s node-exporter (mentioned in `deploy/k8s/deployment-api.yaml` annotations) | Deployed as a DaemonSet in enterprise K8s clusters |
| Application metrics: request latency | `src/blog_mas/service/metrics.py::REQUEST_LATENCY` | Histogram, labels: route, method |
| Application metrics: error rates | `src/blog_mas/service/metrics.py::REQUESTS_TOTAL` | Labels: status_code |
| Application metrics: throughput | `src/blog_mas/service/metrics.py::TASKS_SUBMITTED` | Counter |
| Application metrics: task queue length | `src/blog_mas/service/metrics.py::TASK_QUEUE_LENGTH` | Gauge, updated at /metrics scrape |
| AI-specific: token consumption | `src/blog_mas/service/metrics.py::LLM_TOKENS_TOTAL` | Counter, labels: type (prompt/completion), model |
| AI-specific: estimated cost | `src/blog_mas/service/metrics.py::LLM_COST_USD_TOTAL` | Counter; computed via `tokens.estimate_cost` |
| AI-specific: LLM call latency | `src/blog_mas/service/metrics.py::LLM_CALL_LATENCY` | Histogram; `measure_llm_call()` context manager |
| AI-specific: vector DB query latency | `src/blog_mas/service/metrics.py::VECTOR_QUERY_LATENCY` | Histogram; `measure_vector_query()` context manager |
| Moderation / sanitizer counters | `src/blog_mas/service/metrics.py::MODERATION_BLOCKS_TOTAL`, `SANITIZER_REJECTIONS_TOTAL` | Security observability |
| Prometheus + Grafana | `deploy/prometheus.yml` + `deploy/docker-compose.yml::grafana` | Prometheus scrapes /metrics; Grafana dashboards over it |
| /metrics endpoint | `src/blog_mas/service/api.py::GET /metrics` | `prometheus_client.generate_latest()` |
| Alerting rules skeleton | `deploy/prometheus.yml` (commented section) | High token rate, moderation spike, queue backlog |
| Distributed tracing (Jaeger) | `src/blog_mas/service/tracing.py` | OTel SDK → OTLP → otel-collector → Jaeger |
| OTLP exporter | `src/blog_mas/service/tracing.py::setup_tracing` | `OTLPSpanExporter`, `BatchSpanProcessor` |
| Auto-instrument FastAPI + httpx | `src/blog_mas/service/tracing.py::setup_tracing` | `FastAPIInstrumentor`, `HTTPXClientInstrumentor` |
| W3C TraceContext propagation to Celery | `src/blog_mas/service/tracing.py::get_current_span_context` + `extract_span_context_from_headers` | API span → task header → worker span (same trace in Jaeger) |
| OTel Collector config | `deploy/otel-collector-config.yaml` | Receives OTLP on :4317; exports to Jaeger and Prometheus |
| Jaeger UI | `deploy/docker-compose.yml::jaeger` | :16686 |

---

### §1.5 Infrastructure and Containerization

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Dockerfile (multi-stage, slim) | `deploy/Dockerfile.api` | `python:3.13-slim`, uv install, non-root user |
| Same image principle for API | `deploy/Dockerfile.api` | `CMD ["uvicorn", "blog_mas.service.api:app", ...]` — book's exact pattern |
| Same image for workers | `deploy/Dockerfile.worker` | Same base + deps; different CMD |
| `CMD ["celery", "-A", "tasks", "worker", ...]` | `deploy/Dockerfile.worker` | Book's exact command, adapted for blog-mas module path |
| Liveness probe (API) | `deploy/Dockerfile.api::HEALTHCHECK` + `deploy/k8s/deployment-api.yaml::livenessProbe` | HTTP GET /healthz |
| Readiness probe (API) | `deploy/k8s/deployment-api.yaml::readinessProbe` | HTTP GET /readyz |
| Liveness probe (worker) | `deploy/k8s/deployment-worker.yaml::livenessProbe` | `celery inspect ping` |
| Environment variables injected at runtime | Both Dockerfiles + docker-compose.yml | No credentials in image; `--env-file .env` or K8s Secret |
| Docker Compose: full local stack | `deploy/docker-compose.yml` | api + worker + redis + qdrant + prometheus + grafana + otel-collector + jaeger |

---

### §1.6 Kubernetes Orchestration

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Kubernetes cluster (namespace) | `deploy/k8s/namespace.yaml` | `blog-mas-prod` |
| Deployments (API: 3 replicas, Worker: 5 replicas) | `deploy/k8s/deployment-api.yaml`, `deployment-worker.yaml` | Book's exact example numbers |
| Services | `deploy/k8s/service-api.yaml` | ClusterIP :80 → :8000 |
| ConfigMaps | `deploy/k8s/configmap.yaml` | All non-secret config |
| Secrets | `deploy/k8s/secret.yaml` | Credentials; ExternalSecrets pattern documented |
| Ingress (external access + load balancer) | `deploy/k8s/ingress.yaml` | nginx annotations for rate limiting + TLS |
| API gateway concerns at Ingress | `deploy/k8s/ingress.yaml` | cert-manager TLS, nginx rate limits, auth-url annotation |
| HPA (API): CPU + memory | `deploy/k8s/hpa-api.yaml` | `averageUtilization: 70` CPU |
| HPA (worker): task queue length custom metric | `deploy/k8s/hpa-worker.yaml` | External metric `task_queue_length`; `averageValue: 4` |
| Cluster Autoscaler | Documented in `deploy/k8s/hpa-worker.yaml` | Managed service concern (EKS/AKS/GKE); provisions new nodes when cluster is full |
| Managed K8s services (EKS/AKS/GKE) | Mentioned in `deploy/k8s/*.yaml` comments | "focus on workloads rather than infrastructure" |
| Rolling updates (zero downtime) | `deploy/k8s/deployment-api.yaml::strategy` | `maxUnavailable: 0` |
| Graceful shutdown (in-flight tasks) | `deploy/k8s/deployment-worker.yaml::terminationGracePeriodSeconds: 120` | Lets Celery warm-shutdown finish tasks |

---

## Phase 2 — Enterprise Capabilities and Production Guardrails

### §2.1 Proactive Context Reduction (cost management)

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| `count_tokens` as production-readiness signal | `src/blog_mas/tokens.py::count_tokens` | Already in place from Ch5 |
| Auto-invoke Summarizer when input > budget | `src/blog_mas/workers/tasks.py::_maybe_prepend_summarizer` | Production policy: fires before run_with_policy |
| `SUMMARIZER_TRIGGER_TOKENS` config | `src/blog_mas/config.py::LLMSettings.summarizer_trigger_tokens` | Default 4 000; override via env var |
| Token savings metrics | `src/blog_mas/service/metrics.py::SUMMARIZER_TRIGGERS_TOTAL`, `SUMMARIZER_TOKEN_SAVINGS` | Prometheus counter + summary |
| Treat context reduction as a policy, not an option | `src/blog_mas/workers/tasks.py` | Runs unconditionally when threshold exceeded; no manual trigger needed |

---

### §2.2 High-Fidelity RAG: Auditability (trust and compliance)

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Researcher agent with page-level citations | `src/blog_mas/agents/researcher_hifi.py` | Already in place from Ch7 |
| ExecutionTrace as immutable audit record | `src/blog_mas/engine/tracer.py::ExecutionTrace` | `trace_id` + all steps + `metadata` dict |
| `metadata` field for queue/worker correlation | `src/blog_mas/engine/tracer.py` | Added by Ch10; `task_id`, `worker_id` injected by task layer |
| Persist trace to result store | `src/blog_mas/workers/tasks.py` → `result_store.save(trace_dict=...)` | Stored by trace_id; TTL-configurable |
| Audit retrieval for compliance officers | `src/blog_mas/service/api.py::GET /api/v1/trace/{trace_id}` | Returns `TraceResponse` with plan + all steps + citations |
| Immutable JSONL audit log | `src/blog_mas/meta_controller.py::default_audit_logger` | Every moderation checkpoint appended to `data/audit/audit.jsonl` |
| `require_audit_trace: bool` on request | `src/blog_mas/service/schemas.py::GoalRequest` | When True, saves trace JSON to `TRACE_STORE_DIR` |

---

### §2.3 Defending the Data Pipeline

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Sanitisation at ingest (ring 1) | `src/blog_mas/rag/ingestion_graph.py` | `sanitize_chunk()` called on every chunk before embedding; already in place from Ch7 |
| Sanitisation at runtime retrieval (ring 2) | `src/blog_mas/agents/researcher_hifi.py` | Chunks sanitised again before LLM sees them; already in place |
| `INGEST_SANITIZATION_ENABLED` toggle | `src/blog_mas/config.py::RAGSettings.ingest_sanitization_enabled` | Allows disabling for fully-trusted corpora |
| `sanitizer_rejections_total` metric | `src/blog_mas/service/metrics.py` | Counter labels: stage (ingest/retrieval) |
| Defence-in-depth: two independent rings | Both files above | Attacker must bypass ingest AND retrieval sanitiser independently |

---

### §2.4 Automated Guardrails: Pre/Post-Flight Moderation

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Pre-flight moderation at API EDGE (before queue) | `src/blog_mas/service/api.py::execute_goal_endpoint` | `helper_moderate_content(goal)` → HTTP 400 if flagged; no worker slot consumed |
| Pre-flight moderation in worker (ring 2) | `src/blog_mas/meta_controller.py::run_with_policy` | Already in place from Ch8 |
| Post-flight moderation | `src/blog_mas/meta_controller.py::run_with_policy` | Already in place from Ch8 |
| `API_PREFLIGHT_MODERATION` toggle | `src/blog_mas/config.py::SecuritySettings.api_preflight_moderation` | Can disable for integration tests |
| `moderation_blocks_total` metric | `src/blog_mas/service/metrics.py` | Labels: stage (api_pre_flight / pre_flight / post_flight), category |
| Fail-closed moderation (provider error → block) | `src/blog_mas/security/moderation.py::OpenAIModerationProvider` | Already in place from Ch8 |
| Audit log entry per checkpoint | `src/blog_mas/meta_controller.py::default_audit_logger` | Already in place from Ch8 |

---

### §2.5 Brand Governance via Semantic Blueprints

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Librarian agent retrieves brand voice blueprints | `src/blog_mas/agents/librarian.py` | Already in place from Ch3 |
| ContextLibrary namespace = centrally managed blueprints | `src/blog_mas/rag/blueprints.py` + `data/blueprints/` | 6 JSON blueprint files; Qdrant `blueprints` collection |
| Marketing blueprint (witty, playful social media) | `data/blueprints/` | Ch9 demonstrated with real marketing corpus |
| Blueprints version-controlled in the repo | `data/blueprints/*.json` | No code change needed to update brand voice; redeploy the blueprint file |
| Brand governance across departments | `src/blog_mas/control_decks.py` | `template_competitive_analysis`, `template_product_marketing_copy`, `template_persuasive_pitch_on_brand` apply different intents over the same blueprints |

---

## Phase 3 — Business Value

| Ch10 topic | blog-mas file | Details |
|---|---|---|
| Value Multiplier Flywheel (Fig. 10.2) | `BUSINESS_VALUE.md §1` | ASCII flywheel, per-agent ROI framing, concrete metric citations |
| Reduce Costs: Summarizer | `BUSINESS_VALUE.md §1` + `service/metrics.py::SUMMARIZER_TRIGGERS_TOTAL` | Dollar-level framing for project manager |
| Increase Productivity: Researcher | `BUSINESS_VALUE.md §1` | Paralegal/researcher analogy |
| Accelerate Revenue: Writer | `BUSINESS_VALUE.md §1` | Time-to-market framing |
| Pillar of Trust (Fig. 10.3) | `BUSINESS_VALUE.md §2` | ASCII pillar, per-stakeholder table |
| Auditability dividend (compliance) | `BUSINESS_VALUE.md §2` + `GET /api/v1/trace/{id}` | Retrieval procedure documented |
| Security guarantee as brand protection | `BUSINESS_VALUE.md §2` + `moderation_blocks_total` | Cost-of-prevention framing |
| Knowledge Moat Cycle (Fig. 10.4) | `BUSINESS_VALUE.md §3` | ASCII moat diagram, three strategic lenses |
| Proprietary intelligence from public LLMs | `BUSINESS_VALUE.md §3` | Trace archive = company's unique reasoning dataset |
| Compounding knowledge effect | `BUSINESS_VALUE.md §3` | Analytics on trace archive over time |
| Fine-tuning future proprietary models | `BUSINESS_VALUE.md §3` | `(goal, plan, final_output)` triples as training data |

---

## Files changed summary

### New files (30 total)

| File | Ch10 section |
|---|---|
| `src/blog_mas/config.py` | §1.1 |
| `src/blog_mas/secrets.py` | §1.1 |
| `src/blog_mas/logging_json.py` | §1.4 |
| `src/blog_mas/service/__init__.py` | §1.2 |
| `src/blog_mas/service/schemas.py` | §1.2 |
| `src/blog_mas/service/auth.py` | §1.2 |
| `src/blog_mas/service/metrics.py` | §1.4 |
| `src/blog_mas/service/tracing.py` | §1.4 |
| `src/blog_mas/service/result_store.py` | §1.3 |
| `src/blog_mas/service/api.py` | §1.2 + §1.3 + §2.4 |
| `src/blog_mas/workers/__init__.py` | §1.3 |
| `src/blog_mas/workers/celery_app.py` | §1.3 |
| `src/blog_mas/workers/tasks.py` | §1.3 + §2.1 |
| `deploy/Dockerfile.api` | §1.5 |
| `deploy/Dockerfile.worker` | §1.5 |
| `deploy/docker-compose.yml` | §1.4 + §1.5 |
| `deploy/prometheus.yml` | §1.4 |
| `deploy/otel-collector-config.yaml` | §1.4 |
| `deploy/k8s/namespace.yaml` | §1.6 |
| `deploy/k8s/configmap.yaml` | §1.1 + §1.6 |
| `deploy/k8s/secret.yaml` | §1.1 + §1.6 |
| `deploy/k8s/deployment-api.yaml` | §1.6 |
| `deploy/k8s/deployment-worker.yaml` | §1.6 |
| `deploy/k8s/service-api.yaml` | §1.6 |
| `deploy/k8s/ingress.yaml` | §1.2 + §1.6 |
| `deploy/k8s/hpa-api.yaml` | §1.6 |
| `deploy/k8s/hpa-worker.yaml` | §1.6 |
| `.env.example` | §1.1 (extended) |
| `BUSINESS_VALUE.md` | Phase 3 |
| `CHAPTER10-IMPLEMENTATION.md` | Cross-cutting |

### Edited files (5 total, all additive)

| File | Change |
|---|---|
| `src/blog_mas/engine/tracer.py` | Added `metadata: dict` field to `ExecutionTrace`; included in `to_dict()` |
| `src/blog_mas/logging_config.py` | Converted to shim that delegates to `logging_json.py` (backward compat) |
| `src/blog_mas/cli.py` | Added `serve` subcommand + `_cmd_serve()` function |
| `pyproject.toml` | Added `[prod]` optional dependency group (fastapi, uvicorn, celery, redis, prometheus-client, otel) |
| `.env.example` | Replaced minimal file with comprehensive 12-factor documentation |

### Engine files NOT touched (proving wrapper architecture)

`engine_graph.py`, `executor.py`, `planner.py`, `validators.py`,
`context_engine.py`, `registry.py`, `resolver.py`, `engine_state.py`,
`mcp_envelope.py`, `agent_adapters.py`,
`agents/researcher.py`, `agents/researcher_hifi.py`, `agents/librarian.py`,
`agents/summarizer.py`, `agents/writer.py`, `agents/validator.py`,
`agents/intake.py`, `meta_controller.py` (logic unchanged; only the task
layer calls it differently).

---

## How to run the Ch10 stack locally

```bash
# 1. Set secrets in your shell (never in docker-compose.yml)
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
export API_KEY="my-local-key"

# 2. Start the full observability + service stack
cd blog-mas
docker compose -f deploy/docker-compose.yml up --build

# 3. Ingest knowledge base (one-time)
uv run blog-mas ingest --path data/knowledge

# 4. Submit a goal
curl -X POST http://localhost:8000/api/v1/execute \
  -H "x-api-key: my-local-key" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Write a cited blog post about the Juno Jupiter mission.", "require_audit_trace": true}'
# → {"status":"accepted","trace_id":"<id>","message":"..."}

# 5. Poll for result
curl http://localhost:8000/api/v1/status/<id> -H "x-api-key: my-local-key"

# 6. Retrieve the full audit trace
curl http://localhost:8000/api/v1/trace/<id> -H "x-api-key: my-local-key"

# 7. View dashboards
open http://localhost:9090    # Prometheus
open http://localhost:3000    # Grafana (admin/admin)
open http://localhost:16686   # Jaeger
```
