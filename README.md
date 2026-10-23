# SLM Studio

Prepare data, fine-tune, evaluate, register candidates, and deploy approved Azure ML endpoints.
All operations use real services. GPU and GPT charges apply.

## 1. Start the app

1. Open a terminal in the repository root.
2. Install the app dependencies:

   ```powershell
   python -m pip install -r app\requirements.txt
   ```

3. Start SLM Studio:

   ```powershell
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

4. Open <http://127.0.0.1:8000>.

Keep the app local. It has no multi-user authentication.

## 2. Prepare data

### Set up Azure access

1. Install the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt -r app\requirements.txt
   ```

2. Sign in with an account that can access your Azure OpenAI deployment and Azure ML workspace:

   ```powershell
   az login
   ```

3. Set these values in a local `.env` file:
   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_RESOURCE_GROUP`
   - `AZUREML_WORKSPACE_NAME`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_VERSION`
   - `AZURE_OPENAI_GPT_DEPLOYMENT`
4. Start or restart the app using the command in Step 1.

Do not commit credentials. Production preparation makes billable GPT calls and publishes an Azure ML data asset.

### Generate a dataset

1. Open **Data**.
2. Use the default `data/` folder, or enter another folder containing PDFs that the app's computer can access.
3. Choose **Context-grounded (RAFT)**. Context-assisted generation is not available in production.
4. Set **Number of samples (total)**:
   - This is the exact total across train, validation, and test.
   - Default: **300 total**, targeting about 30 test examples with an 80/10/10 source-grouped split.
   - Release requires **30 actual sealed-test examples**. Group sizes can change the counts; 300 total is not a guarantee.
   - Request at least 3 samples.
   - Provide enough PDF content for at least 5 distinct usable chunks.
5. Adjust **Questions per generation call**, **Oracle probability**, and **Azure ML data asset name** if needed.
6. Confirm the billable operation and click **Prepare data**.
7. Watch **Preparation job status** for the latest activity; it refreshes every 10 seconds.
8. Expand **Recent job logs** to inspect page extraction, sample generation, validation, and publication messages.
9. After completion, review:
   - Generated sample count and split counts.
   - **Release sample-count check:** fewer than 30 test examples requires more data before release evaluation. A passing sample-count check does not mean the quality gates have passed.
   - Validation result and dataset fingerprint.
   - Azure ML data asset reference.
   - The **Sample data** table: every generated field, including context, oracle context, CoT answer, and instruction, with up to two samples per split.

### Review or retry a run

1. Select a run from **Recent runs**.
2. Click **Refresh** to see its latest status.
3. Review the saved results or displayed error.
4. If preparation failed or was interrupted, resolve the error before submitting a new run.

- Run one production operation at a time using one local app instance.
- While any task is running, submission forms are hidden across all stages. Use **Cancel task** to stop local work and monitoring and restore the forms. Cancellation is saved in the run history.
- Cancelling does not guarantee remote Azure jobs stop or delete allocated resources. For training and evaluation, check the Azure ML job link and cancel the cloud job there; for releases, inspect saved models and endpoints for billable resources before retrying.
- Completed results remain available after refreshing or restarting.
- Each new production submission creates a new run and data-asset version.
- Studio leaves the dataset name unchanged and uses `data-DDMMYYHHMMSS` (UTC) as its version.
- Versions are 17 characters, within Azure ML's 30-character limit; duplicate name/version submissions are rejected.

If only publication failed, reuse the prepared dataset without generating it again:

```powershell
python scripts\01_prepare_data.py --skip-generation `
  --dataset-dir "runs\production\<run-id>\dataset" `
  --data-asset-name "slm-studio-data" `
  --data-asset-version "1"
```

Replace the run ID and asset name with your failed run's values.
Choose an unused version of 1 to 30 characters.
This publishes the data asset; the original Studio run remains marked failed.

## 3. Fine-tune a model

1. Open **Fine-tuning**.
2. Choose a successfully published dataset from a previous preparation run.
3. To use another Azure dataset, choose **Existing Azure dataset ID** and enter `azureml:dataset-name:version`.
4. Review the format popup. RAFT assets need `train.jsonl` and `validation.jsonl` with the required grounded fields; keep `test.jsonl` for evaluation. RAG training is not available.
5. Choose a Foundry base model and review all trainer settings:
   - Sequence-length fallback, learning rate, epochs, batch size, and gradient accumulation.
   - LoRA rank, adapter name, chat template, seed, and 4-bit loading.
6. Confirm the dataset, settings, and GPU charges, then click **Start training**.
7. Watch **Fine-tuning job status**; the UI polls the run log every 10 seconds.
8. Expand **Recent job logs** for submission, streamed training logs, Azure job status heartbeats, completion, or errors. Status is logged every 10 seconds while waiting for Azure.
9. Open the saved Azure ML job link for detailed training progress and GPU trainer logs.

- Selecting or reloading a fine-tuning run with an unresolved Azure job status fetches its latest status directly from Azure ML, even if the local monitor failed or was interrupted. Polling continues while Azure reports the job active and stops when it finishes. Confirmed Azure completions/failures and user-cancelled runs are not queried again. A local monitoring error does not mean the Azure job failed.
- Azure compute, managed identity, and environment use the existing backend configuration.
- Each submission creates a separate training run and preserves the source preparation run.
- If monitoring is interrupted, check the saved Azure job before retrying. Stopping the app does not cancel the remote job.

## 4. Evaluate a trained model

1. Open **Evaluation**.
2. Choose a **Completed training job** saved in the runs folder.
3. Review its training date, model, dataset version, sample count, epochs, and adapter name.
   - Older jobs may show **Not recorded** for unavailable parameters.
4. Keep the judge configuration on the backend:
   - `AZURE_OPENAI_GPT_DEPLOYMENT` selects the GPT comparison model and default judge.
   - Optionally set `AZURE_OPENAI_JUDGE_DEPLOYMENT` for a separate judge.
5. Confirm the job, sealed test split, GPU costs, and judge calls; click **Run evaluation**.
6. The app reuses the job's recorded GPT baseline when available; otherwise it generates a new baseline first.
7. Watch **Evaluation job status** and **Recent job logs**; the UI polls every 10 seconds.
8. Review the metrics table for the fine-tuned model, original SLM, and GPT baseline:
   - Faithfulness, relevancy, correctness, mean latency, and p95 latency.
   - Input/output token counts; model-specific frontier API cost estimates when configured.
9. Search or filter the response-level comparison, or open the themed **Azure ML** job link.

Evaluation creates a separate run and preserves the selected training run.
Baseline dataset and judge mismatches fail visibly; they are not silently replaced.
If monitoring is interrupted, check the saved Azure job before retrying.

### Configure frontier token prices

1. Open `token_prices.json`.
2. Enter USD per million input/output tokens for the exact model, such as `gpt-4.1`, `gpt-4.1-mini`, or `gpt-5.2`.
3. Add another explicit frontier model ID if needed. Do not use deployment aliases or SLM names.
4. Leave unknown prices as `null`. Unknown model identities, prices, or usage show unavailable, not zero.
5. Run a new evaluation. Its recorded API model identities select the matching prices; dated snapshots can match their exact model family.
6. Review inference and judge usage separately. Reused GPT baseline usage is historical, not newly incurred.

Self-hosted SLMs show token counts but no per-token tariff. GPU hourly costs are separate.
Estimates use standard input/output rates without cached-input discounts. Each evaluation saves its price snapshot.

## 5. Release a candidate

1. Open **Candidate release**.
2. Select a recent completed evaluation.
3. Review the **Release policy** table and the separate **Data preparation**, **Training**, and **Evaluation** lineage tables.
4. Every quality metric—faithfulness, relevancy and correctness—must meet all conditions:
   - At least **30** sealed-test examples.
   - At least **2 percentage points** mean uplift over the original SLM.
   - Strictly **less than 5%** relative shortfall from GPT.
   - A paired-bootstrap **95% uplift interval strictly above zero**: 5,000 resamples, seed 42.
   - **Perfect-baseline exception:** if the original SLM scores exactly 100% for a metric, the fine-tuned model must also score exactly 100% on every test example. Only that metric's uplift and positive-confidence requirements are waived. The UI labels this **PASS (PERFECT TIE)**; all other requirements still apply.
5. Check the badges: **green PASS**, **red FAIL**, or **gray NOT RECORDED**. Each metric has separate uplift, GPT-gap, and confidence gates. Sample count has its own gate.
   - Any failed requirement blocks release. Latency and token costs are informational.
   - The policy cannot be edited or overridden.
6. Choose **Register candidate only** or **Register and deploy**, then **Review release plan**.
7. For deployment, provide a new dedicated endpoint name, an existing CPU monitoring cluster, and a maintained governed production-request folder.
   - Monitoring requests must be JSONL with `instruction` and `question` strings.
   - Training/test data and raw collector telemetry are not production monitoring streams.
8. Review GPT's GPU SKU recommendation, regional Azure retail hourly estimate, fp16 serving budget and plan expiry.
9. Enter approver and reason; confirm the exact plan. For deployment, also accept the displayed ongoing GPU cost.
10. Click **Approve release**. Watch status/logs every 10 seconds.
11. Registration-only creates no endpoint or GPU. In the same **Candidate release** tab, review deployment status, scoring validation, monitoring and the authenticated endpoint URI. There is no separate fifth tab.

Plans expire after 30 minutes. GPU charges continue while resources are allocated; monitoring/storage cost extra.
If deployment fails or monitoring stops, inspect the saved model and endpoint before retrying. Resources are not automatically deleted.
The tool does not manage canary traffic or rollout policies.

## 6. Use command-line stages

1. Use the same Azure setup above.
2. Check the required options for each stage:

   ```powershell
   python scripts\01_prepare_data.py --help
   python scripts\02_azureml_fine_tuning.py --help
   python scripts\03_azureml_slm_vs_gpt_evaluation.py --help
   python scripts\04_release_candidate.py --help
   python scripts\05_deployment_gate.py --help
   ```

3. Prepare data, or use the Azure ML data asset created in the UI.
4. Submit fine-tuning with the selected base model and dataset.
5. Run evaluation with `--mode llm_baseline`, then `--mode slm_evaluation` using that baseline version.
6. Stage 04 now accepts a completed Studio `--evaluation-record` and registers only after fixed-policy validation and explicit approval.
7. Stage 05 first writes a deployment plan without allocating GPUs. Review its SKU, hourly estimate and hash.
8. Approve with `--plan-file`, `--accept-plan-sha256`, `--approved-by`, `--approval-reason` and `--approve`.

Stage 04/05 intentionally replace the former zero-traffic/promotion CLI. Traffic-percentage and editable quality-policy arguments are no longer accepted.

Review Azure costs, permissions, compute, and release settings before running these stages.

## 7. Check the installation

Run the fast, credential-free production integration tests:

```powershell
python -m pytest app\tests\test_production_api.py app\tests\test_training_api.py app\tests\test_evaluation_api.py app\tests\test_candidate_api.py -q
```

For all app tests:

```powershell
python -m pytest app\tests -q
```

To check the 10-second UI polling behavior (requires Node.js):

```powershell
node --test app\tests\test_ui_polling.mjs
```
