# Understudy

**Teacher–student distillation for a drift-aware edge–server assistant**
DA3408 · AI Operations · end-semester project

> *Answer the familiar questions instantly, hand the hard ones to a bigger mind, and keep
> learning which is which, without anyone babysitting the system.*

A small quantized **student** at the edge answers frequent questions in milliseconds from a
bank of human-approved answers. Anything it is not confident about goes to a self-hosted
**teacher** LLM, whose answer streams back while the user sees a "while you wait" card. When
the questions people ask drift, the student's coverage drops, Prometheus raises an alert, and
an automated loop (**discover → distil → human approval → retrain → gate → promote / roll
back**) teaches the student the new frequent questions. Everything, including training and the
LLM, runs on one local CPU machine. There is no cloud.

```
browser ──TLS──▶ nginx ──▶ edge (student, 1 CPU / 1 GB) ──fallback──▶ gateway ──▶ teacher (Ray Serve + llama.cpp)
                              │ query logs (PII-scrubbed)                  ▲
                              ▼                                            │ Ray actors
         DVC pipeline: ingest → validate → Spark preprocess → Spark mining → distil → [approve]
                       → build_dataset → Ray Tune train → evaluate_gate → MLflow register → edge reload
         Prometheus ─▶ Alertmanager ─▶ console webhook (drift ⇒ discovery) · Grafana dashboards
```

Full diagrams: [docs/architecture.md](docs/architecture.md).

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations: env, package, config, logging, DVC, git hooks, MLflow | ✅ |
| 1 | Data pipeline: ingest → validate → Spark preprocess → Spark mining, EDA | ✅ |
| 2 | Teacher (Qwen2.5-3B GGUF, llama.cpp, Ray Serve) + streaming gateway with guardrails | ✅ |
| 3 | Distillation (Ray actors), approvals, dataset, Ray Tune training, gate, MLflow registry | ✅ |
| 4 | Edge service (ONNX 8-bit student, micro-batching), Docker images, Compose | ✅ |
| 5 | Web UI (chat with streaming + waiting card) and pipeline console | ✅ |
| 6 | Prometheus / Alertmanager / Grafana / Pushgateway, drift-triggered discovery, rollback | ✅ |
| 7 | Unit / integration / e2e / load tests, drills, documentation | ✅ (Docker Hub push: `make push REGISTRY=…`) |

Measured results: [docs/test_report.md](docs/test_report.md).

## Quick start

Prerequisites: Docker with Compose v2, conda (Miniforge), about 10 GB of disk.

```bash
# 1. Environment (Python 3.11, Java 17 for Spark, CPU-only torch, llama.cpp)
conda env create -f environment.yml && conda activate understudy
pip install -e . --no-deps
make hooks                                  # pre-commit: ruff + unit tests; post-commit: DVC CI

# 2. Data, teacher weights, first student (cold start)
make teacher-weights                        # pinned Qwen2.5-3B-Instruct Q4_K_M (2.1 GB, SHA-256 checked)
dvc repro ingest_dataset                    # pinned CLINC150
python -m understudy.data.simulate_traffic --weeks 0-1   # two weeks of simulated questions
docker compose up -d mlflow                 # tracking + registry on :5000 (or: make mlflow)
dvc repro distil                            # mine frequent fallbacks, teacher drafts FAQs
python -m understudy.pipeline.approve list  # review drafts (or use the console)
python -m understudy.pipeline.approve auto  # …or let the simulated reviewer decide (demo only)
dvc repro                                   # dataset → Ray Tune → gate → register champion v1

# 3. Whole system
make up                                     # docker compose up -d --build
```

| URL | What |
|---|---|
| https://localhost:8443 | Chat (self-signed certificate) |
| https://localhost:8443/console.html | Pipeline console: DAG, approvals, models, rollback |
| http://localhost:3000 | Grafana (anonymous view; admin / `understudy`) |
| http://localhost:5000 | MLflow tracking and model registry |
| http://localhost:9090 · :9093 | Prometheus · Alertmanager |

Without the 2 GB model: `make up-fake` starts a deterministic fake teacher.

## Operating it

```bash
make demo        # drift week 3 → alert → discovery → simulated review → retrain → week 4
make drill       # roll the champion back under load, then restore it
make load        # Locust: 90% frequent / 10% streamed fallbacks
make bench       # edge latency (1 CPU) and teacher TTFT baselines
make test        # unit + integration (+ e2e when the stack is up)
python -m understudy.pipeline.register status|rollback|promote <v>
```

User and operator guide: [docs/user_manual.md](docs/user_manual.md).

## Pipeline

| Stage | Tool | What it does | Output |
|---|---|---|---|
| `ingest_dataset` | requests | CLINC150 from a pinned commit, SHA-256 verified | `data/raw/clinc150` |
| `ingest_logs` | — | Snapshot of the query logs (edge + simulated) | `data/snapshots` |
| `validate` | pydantic | Schema/value checks; **stops the pipeline** on bad data | `reports/validation.json` |
| `preprocess` | Spark | Normalise, PII-scrub again, de-dup by request id, no test leakage | `data/processed/*` |
| `mine_frequent` | Spark ML | MiniLM embeddings, cosine KMeans, merge, rank fallback clusters | `data/candidates/` |
| `distil` | Ray + llama.cpp | Teacher drafts question, answer and paraphrases (JSON schema), cached | `data/drafts/` |
| *(approval)* | console / CLI | Human approves, edits, rejects or merges, one Git-tracked file per FAQ | `data/approvals/` |
| `export_encoder` | ONNX Runtime | MiniLM → ONNX with pooling, per-channel 8-bit (22 MB, ~5 ms) | `models/encoder` |
| `build_dataset` | pandas | Answer bank + positives, OOS and log negatives, held-out test | `data/dataset` |
| `train` | Ray Tune + MLflow | Head search, τ tuned on validation, pyfunc logged with lineage | `models/student` |
| `evaluate_gate` | — | Precision / OOS / latency / behavioural checks, no regression vs champion | `reports/gate.json` |
| `register` | MLflow registry | New version → `challenger` → `champion`, edge hot-reload | `reports/register.json` |
| `eda` | matplotlib | Popularity, domain mix by week, question length | `reports/eda/` |

Every stage writes `reports/stages/<stage>.json` and pushes duration / rows / success to the
Pushgateway. `dvc dag` shows the graph. The console renders it live.

## Repository layout

```
params.yaml  dvc.yaml  dvc.lock  MLproject       pipeline parameters, DAG, MLflow entry points
configs/      services.yaml (all REST endpoints) · prompts/teacher_v1.yaml · filler/filler_v1.yaml
src/understudy/
  common/     config, JSON logs, schemas, PII scrub, text, encoders, SSE, metrics, tracking, Spark
  data/       ingest, validate, traffic simulator (offline/online), EDA
  pipeline/   preprocess/mine (Spark), distil (Ray), approve, build_dataset, export_encoder,
              train (Ray Tune), evaluate_gate, register
  student/    ONNX encoder, heads, StudentPredictor + MLflow pyfunc, Router, AnswerBank, registry
  teacher/    llama.cpp engine (+ fake), prompts, Ray Serve app, client, weight download
  gateway/    guarded streaming gateway        edge/  edge service, gateway client, micro-batcher
  console/    console API + pipeline runner    querylog/  QueryLogger
ui/           chat + console pages, nginx (TLS)
docker/       Dockerfiles (edge, gateway, teacher, pipeline/console, ui, mlflow) + requirements
monitoring/   prometheus.yml, alert_rules.yml, alertmanager.yml, Grafana provisioning + dashboard
scripts/      CI, MLflow server, benchmarks, drift demo, rollback drill, dashboard/OpenAPI/report
tests/        unit/ integration/ e2e/ load/
docs/         architecture, HLD, LLD (+ OpenAPI), test plan, test report, user manual, decision log
```

## Documents

| Document | Content |
|---|---|
| [architecture.md](docs/architecture.md) | Block and sequence diagrams, ports |
| [HLD.md](docs/HLD.md) | Design and rationale |
| [LLD.md](docs/LLD.md) | Modules, endpoints, schemas ([OpenAPI](docs/openapi/)) |
| [test_plan.md](docs/test_plan.md) · [test_report.md](docs/test_report.md) | Criteria and measured results |
| [user_manual.md](docs/user_manual.md) | Using and operating the system |
| [decision_log.md](docs/decision_log.md) | Every non-obvious decision and problem hit |
