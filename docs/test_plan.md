# Test plan

Understudy is tested at five levels. Each level has a pytest marker or script, and the acceptance
criteria below are the proposal's Section 6 targets, now fixed against the measured hardware
baseline (16-core CPU, 31 GB RAM, no GPU; edge container limited to 1 CPU / 1 GB).

| Level | What it covers | How to run | Where it runs |
|---|---|---|---|
| Unit | Pure functions, stage logic, every service in-process (FastAPI `TestClient`, fake teacher over ASGI) | `pytest -m unit` | pre-commit hook (every commit) |
| Integration | Real Spark and Ray chains on synthetic data, real llama.cpp model, MLflow registry (SQLite) incl. promote and rollback | `pytest -m integration` | `scripts/ci.sh` (post-commit) |
| End-to-end | The running Docker Compose stack: TLS proxy, edge, gateway, real teacher, console, Prometheus | `pytest -m e2e` (skips if the stack is down) | after `make up` |
| Load | Locust: 90% frequent questions, 10% streamed fallbacks | `make load` | after `make up` |
| Operational drills | Drift demo (alert → discovery → approval → retrain → promote), rollback under load | `make demo`, `make drill` | after `make up` |

Lint (ruff: PEP 8, pyflakes, isort, bugbear, naming, docstrings) runs before the tests in both hooks.

## Acceptance criteria

| # | Area | Criterion | Target | How it is measured |
|---|---|---|---|---|
| A1 | Speed | Student instant-answer latency, p95, edge container at 1 CPU / 1 GB | < 100 ms | `scripts/bench_edge.py` (client-side, sequential and 8 concurrent) → `reports/edge_latency.json`. Also live in Grafana (`EdgeLatencyHigh` alert). |
| A2 | Speed | Time until the user sees something useful on a fallback | < 1 s to the "while you wait" card; first token reported | `scripts/bench_edge.py` (`fallback.filler_card_ms_*`, `first_token_ms_*`); `reports/teacher_baseline.json` |
| A3 | Routing | Precision of instant answers on held-out queries | ≥ 95% | `evaluate_gate` on the CLINC test split (never replayed as traffic), weighted by recent traffic mix (decision 25) → `reports/gate.json`. The uniform-mix figure is reported as a stress test. |
| A4 | Routing | Out-of-scope questions answered by the student | ≤ 5% | `evaluate_gate` on CLINC `oos_test` (1,000 questions) |
| A5 | Efficiency | Coverage tracked over time and recovers after retraining | recovers | Grafana "Fallback ratio"; `reports/drift_demo.json` (fallback ratio in the drift week vs the week after retraining) |
| A6 | Operations | Time from a new topic appearing to a drift alert | within one simulated week | `scripts/drift_demo.py`: time from the start of the drift week's replay to `FallbackRatioHigh` firing |
| A7 | Operations | Rollback to the previous student | < 1 min, no failed requests | `scripts/rollback_drill.py` under constant load → `reports/rollback_drill.json` |
| A8 | Operations | API error rate | < 5%, alerted | `EdgeErrorRateHigh` alert rule; Locust failure ratio (`reports/load/`) |
| A9 | Reproducibility | Rebuild the student from a fresh clone | one command: `dvc repro` | Pinned data/model SHA-256, `dvc.lock`; `scripts/ci.sh` runs `dvc repro` |
| A10 | Safety | PII never reaches disk; prompt injection never reaches the teacher | 0 leaks | unit tests (`test_pii`, `test_querylog`, `test_edge`, `test_guard_sse_prompts`), integration (`test_spark_pipeline`), e2e (`test_prompt_injection_rejected_end_to_end`) |
| A11 | Degradation | Frequent questions still answered with the teacher down | 100% of instant answers still served | `test_edge.py::test_server_down_degrades_gracefully` |

## Test inventory

| File | Level | Focus |
|---|---|---|
| `tests/unit/test_pii.py` | unit | PII detection (email, phone, Luhn card, roll number, IP, account), no false positives |
| `tests/unit/test_text_and_embeddings.py` | unit | text normalisation, hashing encoder |
| `tests/unit/test_config_schemas_metrics.py` | unit | config + env overrides, data contracts, stage reports |
| `tests/unit/test_querylog.py` | unit | scrub-before-write, rotation, feedback file |
| `tests/unit/test_simulate_traffic.py` | unit | drift schedule, determinism, no test-split replay, oracle routing |
| `tests/unit/test_ingest_validate.py` | unit | checksum failure, log snapshots, validation failure stops pipeline |
| `tests/unit/test_mine_summarise.py` | unit | cluster merge, ranking, known-but-missed |
| `tests/unit/test_guard_sse_prompts.py` | unit | prompt-injection rules incl. false positives, SSE codec, prompt files |
| `tests/unit/test_teacher_gateway.py` | unit | teacher core, gateway streaming, 400/503/504 paths, metrics |
| `tests/unit/test_student_head_eval.py` | unit | heads (logreg/MLP/centroid) round-trip, metrics, τ choice, answer bank |
| `tests/unit/test_approve_distil_dataset.py` | unit | paraphrase cleaning, draft cache, approve/reject/merge, dataset negatives, traffic weights |
| `tests/unit/test_edge.py` | unit | edge with a real ONNX student: instant answers, streaming fallback + filler, degradation, injection, PII in logs, reload, admin token |
| `tests/unit/test_console.py` | unit | DAG, approval API, alert webhook (start/cooldown/ignore), pipeline runner |
| `tests/integration/test_spark_pipeline.py` | integration | simulate → ingest → validate → Spark preprocess → Spark mining |
| `tests/integration/test_distil_train_register.py` | integration | Ray distil (fake teacher, cache) → approval → dataset → Ray Tune → gate → registry → promote → rollback |
| `tests/integration/test_teacher_model.py` | integration | the real Qwen2.5-3B GGUF: streaming, schema-valid distil JSON |
| `tests/e2e/test_stack.py` | e2e | health, TLS proxy, real teacher stream, admin not exposed, injection, Prometheus targets + rules, console vs edge version |
| `tests/load/locustfile.py` | load | mixed traffic against the edge |

## Entry and exit

* **Entry**: `ruff` clean; pipeline reproduces (`dvc status` clean or `dvc repro` succeeds).
* **Exit**: every unit and integration test passes; e2e passes on the running stack; criteria
  A1–A11 met or the gap documented with its cause in `docs/test_report.md`.

## Metric names: proposal vs implementation

The proposal sketched `understudy_*` names. The implementation uses one prefix per service, and
derives ratios with Prometheus recording rules instead of storing them as gauges (a gauge would
reset on restart and cannot be re-aggregated).

| Proposal | Implemented |
|---|---|
| `understudy_requests_total{tier,route,status}` | `edge_requests_total{route}`, `gateway_requests_total{endpoint,outcome}` |
| `understudy_request_latency_seconds{tier}` | `edge_request_latency_seconds{route}`, `gateway_request_duration_seconds` |
| `understudy_student_confidence` | `edge_student_confidence` |
| `understudy_fallback_ratio` | recording rule `understudy:fallback_ratio:rate2m` |
| `understudy_teacher_ttft_seconds` | `gateway_teacher_ttft_seconds` (+ `edge_teacher_ttft_seconds`) |
| `understudy_teacher_tokens_per_second` | `gateway_teacher_tokens_per_second` |
| `understudy_feedback_total{helpful}` | `edge_feedback_total{helpful}` |
| `understudy_model_info{version}` | `edge_model_info{version,bank_version,tau,n_faqs}` |
| `pipeline_stage_duration_seconds{stage}` | same (Pushgateway) |
| `pipeline_rows_processed_total{stage}` | same (Pushgateway) |
