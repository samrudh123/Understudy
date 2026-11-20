# AI usage disclosure

This project used AI coding assistants (large language models) for the parts listed below.
Every AI-generated file was reviewed, edited where needed and **verified by running it**: the
commands, the tests and the full drift loop. The verification is described in the second half
of this document.

## Where AI was used

| Area | Files | How AI was used |
|---|---|---|
| **User interface** | `ui/index.html`, `ui/console.html`, `ui/assets/chat.js`, `ui/assets/console.js`, `ui/assets/sse.js`, `ui/assets/style.css`, `ui/nginx.conf`, `docker/ui-entrypoint.sh` | Generated the chat page (streaming answers over SSE, "while you wait" card, feedback buttons) and the pipeline console (live DVC DAG, approval queue, model registry table with rollback/promote, run history), plus the nginx TLS reverse-proxy config. |
| **YAML configuration** | `docker-compose.yml`, `dvc.yaml`, `params.yaml`, `environment.yml`, `configs/services.yaml`, `configs/prompts/teacher_v1.yaml`, `configs/filler/filler_v1.yaml`, `monitoring/prometheus.yml`, `monitoring/alert_rules.yml`, `monitoring/alertmanager.yml`, `monitoring/grafana/provisioning/*/*.yml` | Drafted the Compose services, the DVC stage definitions, the parameter file, the conda environment, Prometheus recording and alert rules (e.g. `FallbackRatioHigh`), Alertmanager routing to the console webhook, and Grafana provisioning. |
| **Dockerfiles** | `docker/Dockerfile.edge`, `.gateway`, `.teacher`, `.pipeline`, `.ui`, `.mlflow`, `docker/requirements/*.txt` | Wrote the per-service images: a slim edge image (run with a 1 CPU / 1 GB limit in Compose), the teacher with llama.cpp and Ray Serve, the pipeline/console image with Java for Spark, nginx for the UI and the MLflow server. |
| **Tests** | `tests/conftest.py`, `tests/unit/*`, `tests/integration/*`, `tests/e2e/test_stack.py`, `tests/load/locustfile.py` | Generated pytest unit tests (112), integration tests (Spark pipeline, distil → train → register, teacher model), the end-to-end stack test and the Locust load test. |
| **Design documents** | `docs/architecture.md`, `docs/HLD.md`, `docs/LLD.md` | Drafted the architecture (block and sequence diagrams, ports), the high-level design (rationale, trade-offs) and the low-level design (modules, endpoints, schemas). I checked these against the code and corrected them where they disagreed. |

AI was also used after the main development:

* **Approval-queue fix** (`ui/assets/console.js`). The console reloads the approval queue every
  20 s, and this used to rebuild every draft card and discard an answer the reviewer was still
  editing. An AI assistant rewrote the queue rendering so that cards are kept per draft and
  only added or removed when drafts appear or are decided. Unsaved edits, the open
  "Real user questions" list, keyboard focus and the merge target now survive a refresh. This
  was verified in a browser against both the old code (edits lost) and the new code (edits kept).
* **Demo video.** AI automation was used to drive the terminal and browser while recording
  the demonstration video, and to cut and speed up the waiting parts. Every command in the
  video was actually run on the system shown.

## How I verified the AI-generated code

I did not accept generated code on trust. I ran it, end to end, and checked the results
against what the design says should happen.

**1. Every command, from a fresh clone.** I cloned the repository from GitHub onto a clean
machine and ran each setup step:

* `conda env create -f environment.yml`, `pip install -e . --no-deps` and `dvc dag`
* `python -m understudy.teacher.download`
* `docker compose build --no-cache` and `docker compose up -d`, with all 10 services healthy

**2. The first student is built and served.** `dvc repro register` ran every stage:

* CLINC150 download and validation
* Spark preprocessing
* ONNX encoder export
* a 32-trial Ray Tune search, logged to MLflow
* the gate
* registration of **v1** as `@champion`

The edge hot-reloaded it, and the chat showed instant answers for known questions.

**3. Tests.** `pytest -m unit` gave 112 passed. These are the same tests the pre-commit hook
runs.

**4. Data drift is detected, and fallback really happens.** I sent questions about a topic
the student had never seen (library hours, an assignment deadline, campus Wi-Fi) to the
edge's `/ask` endpoint, about two per second. I then checked each link of the monitoring
chain:

* **The student falls back.** Every one of these questions was routed to the teacher.
  Prometheus showed `understudy:fallback_ratio:rate2m` at **100 %**, and Grafana's routing
  panel switched to "teacher".
* **The alert fires.** `FallbackRatioHigh` went **pending** after about 40 s and **firing**
  after about 1 min 40 s.
* **Alertmanager notifies the console.** Alertmanager received the alert and called the
  console webhook, and the console started discovery (`dvc repro distil`) with no manual
  trigger.

I also measured how the simulated CLINC weeks behave against the current student. Week 3
gives only 33 % fallback, because its domain is already covered by approved FAQs, so the
alert correctly does *not* fire.

**5. The pipeline runs end to end.**

* **Spark.** During discovery the Spark UI showed the cleaning, embedding and KMeans jobs.
* **Ray.** The Ray dashboard showed the parallel `DistilWorker` teacher actors.
* **Mining.** Mining produced one draft per new topic.
* **Review.** I reviewed the drafts in the console, replaced the teacher's generic answers
  with the real ones, and approved them.
* **Retrain.** *Retrain & promote* rebuilt the dataset, retrained with Ray Tune, passed the
  gate against the current champion, registered **v2** as `@champion` with **39** FAQs, and
  hot-reloaded the edge.
* **Recovery.** Asked again, the same questions, and new phrasings of them, were answered
  instantly with the approved answers. Replaying the traffic dropped the fallback ratio to
  about **6 %**, and the alert returned to inactive.
* **Rollback.** Rolling the champion back took **0.29 s**. The rolled-back model sent the new
  questions to the teacher again, and re-promoting v2 restored them.

**6. Problems found by running the code.** Running the full loop exposed issues that review
alone would not have caught:

* **Approval queue discarding edits.** Fixed, as described above.
* **Clustering merging two topics.** With too few distinct new questions, mining sometimes put
  two topics into one draft. A larger, more varied set of drift questions avoids this.
* **A gate failure after a bad merge.** Merging an unrelated draft into an existing FAQ made
  the retrained model fail the gate's coverage check, so it was not promoted. This is the gate
  working as intended. Undoing the merge and retraining passed.
