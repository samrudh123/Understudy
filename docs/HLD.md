# High-level design

Rationale for individual choices is recorded once, in [`decision_log.md`](decision_log.md);
this document refers to entries as **D<n>** (and problems as **P<n>**).
Component diagram and ports: [`architecture.md`](architecture.md). Interfaces: [`LLD.md`](LLD.md).

## 1. Problem

A help desk receives many questions. A few dominate, and what dominates changes over time:
a new service launches, a term starts, a policy changes. A large LLM can answer anything,
but it is slow and expensive on every request. A small classifier is instant, but only
knows what it was trained on, and goes stale.

Understudy combines the two:

* an edge **student** answers the frequent questions instantly from a bank of
  human-approved answers;
* a server **teacher** LLM answers the rest, streaming;
* an AIOps loop notices when fallbacks grow, mines the newly frequent questions, has the
  teacher draft FAQ entries, puts them in front of a human, retrains the student, gates
  it, promotes it and hot-reloads the edge, all with lineage and monitoring.

## 2. Goals and non-goals

| Goals | Targets (Section 6 of the proposal, `params.yaml: gate`) |
|---|---|
| Instant answers are right | traffic-weighted instant-answer precision ≥ 0.95 |
| Out-of-scope questions are not answered by the student | OOS false-accept ≤ 0.05 |
| Instant answers are fast on a constrained device | p95 ≤ 100 ms on 1 CPU / 1 GB |
| Fallbacks start quickly | teacher TTFT p95 alert at 5 s |
| The loop is automatic but supervised | drift alert starts discovery; a human approves; retrain → gate → promote is one action |
| Bad models can be undone fast | rollback = alias move + edge reload, no downtime |
| Reproducible, local-only | every stage under DVC; all training on this machine (D3) |

Non-goals: a production RAG system (the teacher answers from its own knowledge), multi-host
deployment, GPU inference, authentication of end users.

## 3. Data

* **CLINC150** (D1): 150 intents in 10 domains plus an out-of-scope set. Pinned to an
  upstream commit and SHA-256 (D2).
* **Simulated traffic** (D7): weekly batches replayed from CLINC train/val utterances with
  Zipf intent popularity, 5 % OOS and 2 % injected PII. Whole domains are held back and
  arrive later (`auto_and_commute` in week 3, `home` in week 5) with a two-week surge,
  which is the drift event. Test utterances are never replayed, so they stay clean for
  evaluation.
  * *Offline* mode writes `data/querylogs/sim-week-XX.jsonl` and uses oracle routing (D8).
    It is used to bootstrap before any model exists.
  * *Online* mode sends each question to the real edge `POST /ask` with the simulation
    labels attached. Only `--teacher-rate` of the fallbacks are actually sent to the
    teacher. The rest are routed and logged without generation.
* **Query logs**: every edge request is appended as a PII-scrubbed `QueryLogRecord`.
  These logs are the only input to mining.

## 4. ML pipeline (DVC)

`dvc.yaml` defines the stages. Each stage runs inside `common/metrics.stage_run`, which
writes `reports/stages/<stage>.json` and pushes duration, rows and success to the
Pushgateway (D11).

| Stage | Engine | Consumes | Produces |
|---|---|---|---|
| `ingest_dataset` | requests | `params.data` URLs + SHA-256 | `data/raw/clinc150` |
| `ingest_logs` | python | `data/querylogs/*.jsonl` | `data/snapshots/{querylog,feedback}.jsonl` + manifest |
| `validate` | pydantic | raw dataset, snapshots | `reports/validation.json`; **fails the pipeline** on schema errors or > 1 % malformed log rows |
| `preprocess` | **Spark** | raw dataset, snapshots | `data/processed/clinc`, `data/processed/querylog` (Parquet, partitioned by week): normalised, PII scrubbed again (D6), de-duplicated by `request_id` (D5), test/train leakage removed |
| `mine_frequent` | **Spark** (`mapInPandas` MiniLM embeddings, Spark ML KMeans) | processed logs, approvals | `data/candidates/candidates.json`, `reports/mine_metrics.json`. Over-cluster + merge (D9, D14); clusters near an approved FAQ are `known_but_missed` (D10) |
| `distil` | **Ray** actor pool, llama.cpp | candidates, approvals, prompts | `data/drafts/drafts.json`: canonical question, answer, paraphrases per new cluster, grammar-constrained JSON (D18), cached (D22) |
| human approval | console / CLI | drafts | `data/approvals/faq-*.json` (git-tracked; approve, reject, merge) (D4, D23) |
| `export_encoder` | torch → ONNX Runtime | `params.encoder`, pinned MiniLM revision | `models/encoder` (per-channel UINT8 ONNX, pooling and L2 norm in the graph) (D21) |
| `build_dataset` | pandas | approvals, processed data, encoder | `data/dataset/{train,val,test}.parquet`, `bank.json`; positives, OOS and log negatives (D24); test rows carry `traffic_weight` (D25) |
| `train` | **Ray Tune**, scikit-learn, MLflow | dataset, encoder | `models/student`, MLflow run + pyfunc model (D27, D28), `reports/train_metrics.json`, threshold curve and confusion plots |
| `evaluate_gate` | numpy | student, dataset, current champion | `reports/gate.json` |
| `register` | MLflow registry | gate, train run | registry version, aliases, edge reload; `reports/register.json` |
| `eda` | matplotlib | processed data | `reports/eda/*` |

The DAG is acyclic because approvals are an input that no stage produces (D4).
Changing an approval re-runs mining. Distillation is cached, so it does not regenerate drafts.

### Student model

`all-MiniLM-L6-v2` is exported once to ONNX with dynamic per-channel UINT8 weights (22 MB,
cosine 0.99 to fp32). A light head is trained on top: logistic regression, MLP or
centroid, with an optional `__fallback__` class. Training embeds with the same INT8 graph
the edge runs, so there is no train/serve skew. The head runs in numpy, so the edge needs
neither torch nor scikit-learn. The confidence threshold τ is chosen on validation as the
lowest value meeting stricter targets than the gate: precision ≥ 0.97, false-accept ≤ 0.03,
coverage > 0 (D26).

The artifact directory (`encoder/`, `head/`, `bank.json`, `student_meta.json`) is identical
locally and inside the MLflow model, so model, answers and τ are versioned together.

## 5. Serving design

### Routing

1. The edge strips the question and rejects empty input or input over 500 characters (`422`).
2. It scrubs PII, then the student predicts the top FAQ and its probability.
3. **Instant answer** when the head does not vote `__fallback__` and the confidence is ≥ τ.
   The answer comes from the bank (no generation), so it cannot be prompt-injected.
4. Otherwise it **falls back**: the edge immediately sends a `meta` event with a "while you
   wait" card from the versioned filler file and up to three related FAQs (confidence ≥ 0.25).
   It then relays the gateway's token stream.

### Degradation modes

| Failure | Behaviour |
|---|---|
| No champion yet / registry unreachable at start | `/ready` returns 503. The edge retries every `model_retry_s` and meanwhile sends every question to the teacher. |
| Registry down later | The edge keeps the loaded model. `/admin/reload` fails with 500 and keeps the old model. |
| Teacher or gateway down | Instant answers continue. Fallbacks get `error` with a friendly "unavailable" message and are logged as route `unavailable`. |
| Teacher slow | Gateway returns `504 teacher_timeout` if no first token arrives within 15 s. After the first token, an `error` event if the total time exceeds 60 s. |
| Prompt injection | Gateway `400 prompt_injection`; edge logs route `rejected` and shows a rephrase message. |

### Throughput on 1 CPU

Concurrent requests are routed by `edge/batcher.MicroBatcher`. It is opportunistic: a lone
question runs immediately, and questions arriving while a batch runs form the next batch,
encoded in one ONNX call on one worker thread. This avoids several threads fighting for a
single CFS CPU quota (`reports/edge_latency.json`).

### Teacher

Ray Serve runs one llama.cpp model per replica with `max_ongoing_requests=1` (D16), so extra
requests queue in Ray. The gateway holds the HTTP status until the first token (D17).
The baseline TTFT is about 0.5 s warm and about 4 s cold (D15, `reports/teacher_baseline.json`).

## 6. MLOps

* **Tracking**: each `train` run logs to MLflow experiment `understudy-student`:
  * head config, τ, seeds and tune settings;
  * lineage tags: git commit and dirty flag, data hash (`data_version`), bank version, prompt
    versions, teacher models, encoder revision;
  * validation and test metrics, latency, model size;
  * the threshold curve, confusion matrix, Tune trial table and dataset stats;
  * one nested run per Ray Tune trial.
* **Model**: logged as an MLflow pyfunc from code (`student/pyfunc_entry.py`).
* **Gate** (`params.yaml: gate`, `pipeline/evaluate_gate.py`). The challenger must meet every
  rule:

  | Rule | Threshold |
  |---|---|
  | traffic-weighted precision | ≥ 0.95 |
  | traffic-weighted coverage | ≥ 0.30 |
  | OOS false-accept (uniform OOS test) | ≤ 0.05 |
  | p95 latency, one thread | ≤ 100 ms |
  | canonical recall (each FAQ's own question routes to it) | ≥ 0.90 |
  | PII invariance (appended scrubbed e-mail does not change the answer) | ≥ 0.90 |
  | gibberish / empty input deferred | ≥ 1.0 |
  | vs champion on the same test set | precision drop ≤ 0.02, coverage drop ≤ 0.02 |

  The uniform-CLINC precision is reported as a stress metric but not gated (D25).
* **Registry**: `register` creates a version, tags it with build id and gate metrics, and sets
  `challenger`. With `register.auto_promote: true` it moves `champion`, records
  `previous_champion` on the new version, and calls the edge's `POST /admin/reload`.
* **Rollback**: `registry.rollback` points `champion` at `previous_champion` (or a given
  version), tags `rolled_back_from`, and the edge reloads. The edge swaps the router
  atomically after warming the new model, so in-flight requests are not dropped.
* **CI** (D13): the pre-commit hook runs ruff and the unit tests. The post-commit hook runs
  `scripts/ci.sh` (lint, all tests, `dvc repro`).

## 7. Monitoring and alerting

| Service | Metrics (Prometheus names) |
|---|---|
| edge | `edge_requests_total{route}`, `edge_request_latency_seconds{route}`, `edge_student_inference_seconds`, `edge_student_confidence`, `edge_teacher_ttft_seconds`, `edge_feedback_total{helpful}`, `edge_shadow_grades_total{verdict}`, `edge_errors_total{kind}`, `edge_model_reloads_total{outcome}`, `edge_teacher_skipped_total`, `edge_model_info{version,bank_version,tau,n_faqs}`, `edge_model_loaded` |
| gateway | `gateway_requests_total{endpoint,outcome}`, `gateway_teacher_ttft_seconds`, `gateway_teacher_tokens_per_second`, `gateway_teacher_tokens_total`, `gateway_request_duration_seconds`, `gateway_inflight_requests`, `gateway_teacher_up` |
| teacher | Ray / Ray Serve runtime metrics on port 8080 |
| console | `console_pipeline_runs_total{target,outcome}`, `console_pipeline_running`, `console_approvals_total{decision}`, `console_pending_drafts`, `console_rollbacks_total{outcome}`, `console_last_rollback_seconds`, `console_alert_triggers_total{alertname,action}` |
| pipeline (Pushgateway) | `pipeline_stage_duration_seconds{stage}`, `pipeline_rows_processed_total{stage}`, `pipeline_stage_success{stage}`, `pipeline_stage_last_run_timestamp_seconds{stage}` |
| every Python service | `process_cpu_seconds_total`, `process_resident_memory_bytes` (edge budget: 1 CPU / 1 GiB) |

Recording rules and alerts (`monitoring/alert_rules.yml`):

| Alert | Condition | Action |
|---|---|---|
| `FallbackRatioHigh` | fallback ratio (2 m) > 0.55 and > 0.5 req/s, for 1 m | **drift**: Alertmanager webhook makes the console start discovery (cooldown 600 s) |
| `EdgeErrorRateHigh` | unavailable ratio > 5 %, for 2 m | operator checks gateway/teacher |
| `TeacherTTFTHigh` | p95 TTFT > 5 s, for 2 m | add replicas / threads |
| `EdgeLatencyHigh` | instant-answer p95 > 100 ms, for 5 m | check edge resources |
| `StudentPrecisionLow` | teacher judged > 10 % of sampled instant answers irrelevant (≥ 20 grades in 30 m) | consider rollback |
| `PipelineStageFailed` | `pipeline_stage_success == 0` | inspect stage report |
| `EdgeNoModel` | `edge_model_loaded == 0`, for 2 m | train/register a champion |
| `ServiceDown` | `up == 0` for edge/gateway/teacher/console, for 1 m | restart service |

**Shadow grading**: the edge sends a random `shadow_rate` (5 %) of live instant answers
(not simulated ones) to the gateway's `/grade`, where the teacher judges relevance. This
estimates live precision without labels.

The Grafana dashboard is generated by `scripts/build_dashboard.py` and provisioned
read-only. It has rows for routing, latency, answer quality, pipeline/model operations and
infrastructure. Firing alerts are shown as annotations.

## 8. Security and privacy

* **PII** is scrubbed at write time (`QueryLogger`) and again in Spark preprocessing (D6).
  It covers e-mail, phone, Luhn-valid card numbers, account numbers, IPv4 and roll numbers.
  The edge also scrubs before prediction and before forwarding to the gateway, so the teacher
  never sees raw PII. Teacher paraphrases containing PII are dropped in `distil`.
* **Prompt injection**: heuristic guard in the gateway (D19) and a system prompt that treats
  the message as a question only. Chat-template tokens are rejected outright.
* **Admin endpoint**: `/admin/reload` requires header `X-Admin-Token` when
  `UNDERSTUDY_ADMIN_TOKEN` is set. nginx never proxies `/api/edge/admin/*`.
* **TLS** at nginx. Port 80/8080 redirects to HTTPS. Security headers: `nosniff`,
  `X-Frame-Options DENY`, `Referrer-Policy`.
* **Input limits**: 500 characters at the edge, 16 KB request body at nginx, pydantic models
  that forbid extra fields on log records.

## 9. Reproducibility

* Dataset and teacher weights are pinned by URL/commit + SHA-256. The MiniLM revision is
  pinned in `params.yaml`.
* `params.yaml` is the single source of tunables. DVC tracks the params each stage declares.
* `dvc.lock` records the hashes of every input and output. `dvc repro` rebuilds only what
  changed.
* Approvals are git-tracked JSON. The answer bank is a pure function of them (D4).
* MLflow runs record git commit, dirty flag, data hash and prompt versions. Registry versions
  carry the build id.
* Environment: `environment.yml` + `requirements.lock.txt`. Docker images pin the same versions.

## 10. Resource constraints

Everything runs on one machine: 16 cores, 31 GB RAM, no CUDA (D3).

| Component | Budget |
|---|---|
| edge | Compose limit 1 CPU, 1 GB (about 90 MB used) |
| teacher | 1 replica x 8 threads, about 2.5 GB RAM |
| distil | 2 Ray actors x 7 threads (about 4 drafts/min) |
| train | 8 Ray Tune trials in parallel (about 25 s tuning) |
| preprocess / mine | Spark `local[*]` with 4 GB driver memory |
