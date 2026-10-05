# SLM Studio demo

A standalone wizard demo of the five-stage product workflow. **No pipeline imports,
Azure connections, GPT calls, training, uploads, or live deployments occur at
application runtime.** The Evaluation tab also includes a read-only snapshot from
a completed Azure evaluation run; it never connects back to Azure.
Production mode is visible but unavailable. Recipe choices and parameter
recommendations are illustrative fixtures, not implemented algorithms or LLM output.
The base-model selector includes simulated Llama, Qwen, Phi, and Mistral family
options; choosing one does not download or invoke that model.

## Run locally

From the repository root:

```powershell
python -m pip install -r app\requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Keep the app bound to localhost; it has no
multi-user authentication. Do not expose it publicly.

## Explore the demo

- Select a completed passing run, a blocked candidate, or a training failure.
  Future steps stay locked until their prerequisite succeeds; completed steps
  remain available for review with Back and Next navigation.
- Choose **New run**, a source-folder label, and a recipe. Optional
  **Example workflow options** select a scenario.
  Folder labels are never read from disk.
- Simulate preparation, accept or edit training suggestions, then simulate
  training and evaluation. Progress is a timer, not a loss chart.
- Inspect the five-metric comparison and the separate registration gate.
  All three models use the same fictional sealed test split.
- Explore the real Azure evaluation snapshot: summary faithfulness, relevancy,
  correctness, mean latency, and p95 latency appear above a searchable/filterable
  grid of queries, reference CoT answers, GPT answers, original-model answers,
  and fine-tuned answers.
- A blocked gate requires **OVERRIDE** and a reason. Pending or failed evidence
  cannot be overridden.
- Approve the candidate release to simulate model registration, zero-traffic
  endpoint deployment, and drift-reference creation.
- Run the separate deployment gate, select a canary or full traffic percentage,
  and review the resulting traffic map and drift-monitoring schedule.
- Expand **Inspect the complete run record** to see the full evidence
  and decision audit. Optional JSON downloads are browser-generated copies.

The interface uses production-style action names. A single top banner says
**Demo only. No Azure services are called.** Simulation markers remain in the
API responses and raw YAML records, not repeated throughout the workflow screens.

### Recipe names

- **Context-grounded (RAFT):** the intended training-example recipe uses relevant
  and distractor documents to teach context-grounded answers.
- **Context-assisted (RAG):** the intended retrieval-assisted path. RAG retrieves
  supporting context at answer time; it is not a fine-tuning method by itself.
  Its production dataset-generation behavior remains undefined.

These are display labels for the existing `context-grounded` and
`context-assisted` API values; neither runs real generation in this app.

## State and isolation

Rich YAML records live only in [runs/demo/](../runs/demo/). Each record carries
`mode: demo`, `simulated: true`, workflow state, data-card metadata,
training configuration, candidate/test lineage, three-model evaluation,
offline gate evidence, release receipts, deployment-gate evidence, traffic
decisions, monitoring configuration, recommendations, and audit.

The app does not use the training run registry or scan production records.
Fake resource references use `demo://` and are not live Azure links.
Workflow metrics and costs are explicitly simulated. The separate Azure evaluation
showcase is generated from a hash-verified predictions CSV by
`scripts/export_evaluation_showcase.py` and served read-only from
`app/data/evaluation_showcase.json`. Missing evidence is not a zero score.

Refresh the bundled showcase after downloading a completed Stage 03 job:

```powershell
python scripts\export_evaluation_showcase.py `
  --predictions <downloaded-job>\named-outputs\predictions\<run>_predictions.csv `
  --evidence <downloaded-job>\named-outputs\predictions\evaluation_evidence.json `
  --output app\data\evaluation_showcase.json
```

The exporter verifies the predictions SHA-256 against the evaluation manifest
before writing application data. It retains only the query, reference CoT answer,
three generated answers, per-record scores/latencies, and aggregate summaries;
retrieved context and judge-reason text are not bundled into the UI.
Operations retain their state after refresh; interrupted operations are reported
as interrupted/failed on restart rather than silently marked successful.

Run state is already gitignored by the repository. A clean, empty demo store is
automatically seeded with the same three scenarios from the app's fixtures;
existing records are never overwritten.

The implementation is confined to [app/](./) and [runs/demo/](../runs/demo/).
The product specification outside these folders is intentionally unchanged.

## Tests

```powershell
python -m pytest app\tests -q
```

API tests use isolated temporary stores, not the checked-in demo records.
The demo app dependencies are independent of the GPU training environment.
