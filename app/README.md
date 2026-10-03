# SLM Studio demo

A standalone demo of the five-stage product workflow. **No pipeline imports,
Azure connections, GPT calls, training, uploads, or live deployments occur.**
Production mode is visible but unavailable. Recipe choices and parameter
recommendations are illustrative fixtures, not implemented algorithms or LLM output.

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
- Choose **New run**, a source-folder label, and a recipe. Optional
  **Example workflow options** select a scenario.
  Folder labels are never read from disk.
- Simulate preparation, accept or edit training suggestions, then simulate
  training and evaluation. Progress is a timer, not a loss chart.
- Inspect the five-metric comparison and the separate registration gate.
  All three models use the same fictional sealed test split.
- A blocked gate requires **OVERRIDE** and a reason. Pending or failed evidence
  cannot be overridden.
- Register first, then explicitly confirm simulated deployment and monitoring.
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
gate evidence, recommendations, user decisions, deployment outcomes, and audit.

The app does not use the training run registry or scan production records.
Fake resource references use `demo://` and are not live Azure links.
Metrics and costs are explicitly simulated. Missing evidence is not a zero score.
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
