# SLM Studio

Prepare training data, explore fine-tuning, compare models, and review deployment decisions.

- **Demo:** explore the full workflow without Azure calls or charges.
- **Production:** prepare real datasets using GPT, publish them to Azure ML, and submit fine-tuning.
- Production evaluation and deployment currently use command-line tools, not the UI.

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

## 2. Try the demo

1. Select **Demo** from the mode selector.
2. Select an existing run, or click **+ New run**.
3. Enter a data-folder label, choose a recipe, and set the sample count.
4. Optionally select a passing, blocked, or failed scenario under **Example workflow options**.
5. Confirm and click **Prepare data**.
6. Review each step using **Back** and **Next**:
   - **Data:** inspect the data card and sample previews.
   - **Fine-tuning:** edit parameters and start simulated training.
   - **Evaluation:** compare the candidate, original model, and GPT baseline.
   - **Candidate release:** approve the candidate; a blocked gate requires `OVERRIDE` and a reason.
   - **Deployment gate:** select traffic and confirm the simulated deployment.
7. Expand **Inspect the complete run record** to review or download the details.

Demo folders are labels only. Models, workflow results, and costs are simulated.
The Azure evaluation showcase is a saved, read-only result.

## 3. Prepare real data

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

1. Select **Production**.
2. Use the default `data/` folder, or enter another folder containing PDFs that the app's computer can access.
3. Choose **Context-grounded (RAFT)**. Context-assisted generation is not available in production.
4. Set **Number of samples (total)**:
   - This is the exact total across train, validation, and test.
   - Request at least 3 samples.
   - Provide enough PDF content for at least 5 distinct usable chunks.
5. Adjust **Questions per generation call**, **Oracle probability**, and **Azure ML data asset name** if needed.
6. Confirm the billable operation and click **Prepare data**.
7. Watch **Preparation job status** for the latest activity; it refreshes every 10 seconds.
8. Expand **Recent job logs** to inspect page extraction, sample generation, validation, and publication messages.
9. After completion, review:
   - Generated sample count and split counts.
   - Validation result and dataset fingerprint.
   - Azure ML data asset reference.
   - The **Sample data** table: every generated field, including context, oracle context, CoT answer, and instruction, with up to two samples per split.

### Review or retry a run

1. Select a run from **Recent runs**.
2. Click **Refresh** to see its latest status.
3. Review the saved results or displayed error.
4. If preparation failed or was interrupted, resolve the error before submitting a new run.

- Run one preparation at a time using one local app instance.
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

## 4. Fine-tune a model

1. Select **Production**, then **Configure fine-tuning** or the **Fine-tuning** step.
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

- Azure compute, managed identity, and environment use the existing backend configuration.
- Each submission creates a separate training run and preserves the source preparation run.
- If monitoring is interrupted, check the saved Azure job before retrying. Stopping the app does not cancel the remote job.

## 5. Continue with the command-line workflow

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
6. Review the evidence and approve candidate release. This deploys at zero traffic.
7. Run the deployment gate before routing traffic. Configure drift monitoring if needed.

Review Azure costs, permissions, compute, and release settings before running these stages.

## 6. Check the installation

Run the fast, credential-free production integration tests:

```powershell
python -m pytest app\tests\test_production_api.py -q
```

For all app tests:

```powershell
python -m pytest app\tests -q
```

To check the 10-second UI polling behavior (requires Node.js):

```powershell
node --test app\tests\test_ui_polling.mjs
```
