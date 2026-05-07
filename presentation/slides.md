# RAFT for Domain-Specific RAG

Fine-tuning a small language model to handle noisy retrieval

**Tech Talk with Sriks**  
Senior Data Scientist @ Microsoft

<span class="duration">30 minute walkthrough</span>

<aside class="notes">
Can a small model be trained to behave better inside a real RAG pipeline, especially when retrieval is imperfect?
</aside>

---

## Regular RAG: The Promise

Retrieval Augment Generation (RAG) turns the model into an **open-book system**.

1. Embed the question
2. Retrieve top-k chunks
3. Put chunks into the prompt
4. Ask the model to answer from context

**In theory:** the model does not need to memorize private knowledge.  
**In practice:** the model is only as good as the retrieved context.

<aside class="notes">
Use familiar data science language: RAG is a pipeline, not a single model. Retrieval is an upstream feature-generation step for the generator.
</aside>

---

## Where RAG Breaks


The RAFT paper frames the hard case as **in-domain open-book QA with distractors**.

| Retrieval state | What the generator sees | Typical failure |
|-----------------|-------------------------|-----------------|
| Oracle chunk present | Relevant evidence + noise | Model may cite or use the wrong chunk |
| Oracle chunk absent | Only distractors | Model may still produce a confident answer |
| Partial evidence | Fragmented facts | Model blends context with prior knowledge |

**Retrieval error becomes generation error.**

<aside class="notes">
Mention the paper's key term: distractor documents. In production, these are often stale pages, adjacent concepts, policy text, repeated boilerplate, or chunks with similar embeddings.
</aside>

---

## Why Prompting Alone Is Brittle
<!-- .slide: class="compact-slide" -->

Adding "answer only from context" helps, but it does not fully change model behavior.

- The base model was not trained specifically on your retrieval distribution 
- RAG (aka. open-book exam) relies on perfect retrieval during test times
- When the oracle is missing, the model may over-trust distractors or rely on memorized domain knowledge
- Standard SFT often trains on clean context by memorizing, they do not consider retrieval
- The model learns answer style, but not necessarily **evidence selection**

From a data science perspective, this is a **distribution mismatch** problem.

<aside class="notes">
Position the problem like train/test skew. At inference time, the model sees messy retrieval. If training only used clean oracle documents, we should expect brittleness.
IN general RAFT is a kind of supervised instruction fine tuning where we are teaching the model to adapt to imperfect retrievals and align to the right documents
</aside>

---

## What RAFT Solves
<!-- .slide: class="compact-slide" -->
**RAFT = Retrieval-Augmented Fine-Tuning**

RAFT promotes combines open-book exam (RAG) + Supervised Fine-tuning (SFT) in a fixed domain setting. 
The model is trained on the kind of context it will see at inference time (imperfect):

- A question
- A retrieved document set
- One relevant oracle document when available
- Multiple distractor documents
- A response that cites useful evidence and ignores noise in chain-of-thought style

The goal is not just memorization.  The goal is **retrieval-aware answer behavior**.

<aside class="notes">
Use the paper title directly: adapting a language model to domain-specific RAG. The important shift is training the model to operate inside the RAG setting.
If you are wondering why do not we perfect the retrieval, it is not always possible - specially when driven by user query dynamics even in fixed domain settings
</aside>

---

## Example: Same Question, Three Models
<!-- .slide: class="compact-slide" -->

<p class="question-callout"><strong>Question:</strong> What features does the AI Agent Studio offer for managing AI Agents?</p>

<div class="answer-grid">
  <article>
    <span>Frontier model</span>
    <h3>GPT-4.1</h3>
    <p>The retrieved documents do not contain information about features offered by the AI Agent Studio for managing AI Agents.</p>
    <p>GPT-4.1 correctly refuses to answer from the retrieved documents because the AI Agent Studio details are absent from the context. </p>
  </article>
  <article>
    <span>Baseline SLM</span>
    <h3>Llama-3.2-1B-Instruct</h3>
    <p>The AI Agent Studio is a tool that enables customers to manage AI Agents, including defining system purpose, establishing procedures for human oversight, and implementing contestability to guard against unintended outcomes.<p>
    <p>Starts a step-by-step response, but drifts into unrelated implementation-support details instead of directly answering the trademark question.</p>
  </article>
  <article>
    <span>RAFT fine-tuned SLM</span>
    <h3>Llama-3.2-1B RAFT</h3>
    <p>##begin_quote## "The AI Agent Studio offers features such as AI Agent Management, AI Agent Monitoring, and AI Agent Reporting." ##end_quote##
    The AI Agent Studio offers features such as AI Agent Management, AI Agent Monitoring, and AI Agent Reporting for managing AI Agents.</p>
    <p>The fine-tuned SLM is concise, aligns the answer to the context it has seen during the training phase, though not part of the context</p>
  </article>
</div>

<aside class="notes">
The point is not that RAFT beats GPT-4.1 here. The point is that the small model becomes more direct and domain-aligned than its baseline.
</aside>

---


## End-to-End RAFT Design

```text
ServiceNow PDF
  |
  v
pdf_to_chunks.py  ->  page/chunk text
  |
  v
raft_datagen.py   ->  Q + context + CoT answer [GPT 4o]
  |                 p: oracle + distractors
  |              1-p: distractors only, target A*
  v
train / validation / test JSONL
  |
  |
  |
  +--> llm_inference on Held-out set [GPT 4.1] -> llm_predictions.csv
  |
  |
  +--> Unsloth + LoRA fine-tune Llama-3.2-1B [Kaggle, T4]
  |        |
  |        v
  |    RAFT SLM predictions on Held-out set [Kaggle, T4]
           |
           +--> held-out test prompts
                |
                +--> baseline Llama-3.2-1B --> slm_predictions.csv
                +--> RAFT fine-tuned Llama-3.2-1B --> slm_baseline_predictions.csv
                +--> Download both the files
  |-->  raft_llama_evaluate.py -> merged_predictions.csv
```

Source domain: **ServiceNow security best-practice document**. Each record stores `question`, `context`, `oracle_context`, `cot_answer`, `instruction`, and `type`.

<aside class="notes">
This is the paper translated into the project workflow. The core asset is the dataset: questions, mixed retrieved contexts, and oracle-derived answers. The experiment compares GPT-4.1, the baseline small model, and the RAFT fine-tuned small model on the same held-out questions.
</aside>

---

## When To Use RAFT

Use RAFT when these are true:

- You have a stable domain corpus
- Retrieval returns semantically close but noisy chunks
- You need a smaller or cheaper model in production
- You care about grounded answers under imperfect retrieval
- You can generate or label domain QA pairs

RAFT is strongest when the retrieval distribution at training time resembles production retrieval.

<aside class="notes">
This is the decision slide. RAFT is useful when your domain and retrieval patterns are repeatable.
</aside>

---

## Limitations And Risks
<!-- .slide: class="compact-slide" -->

- Quality depends on the labelled QA and chunking strategy
- Bad oracle selection teaches bad evidence behavior
- Models will become stale if the domain evolves, it still retains the RAG feature provided retrieval is perfect.
- Fine-tuning does not replace retrieval evaluation
- Long context and tables may still be hard for a 1B model
- Domain adaptation can reduce generality outside the domain
- RAFT does not solve irrelevant yet semantically correlated retrieval

RAFT improves the generator, but the full RAG system still needs retrieval monitoring.

<aside class="notes">
Keep this honest. The title is not "RAG is solved". The message is "train the model for the retrieval conditions it will actually face".
</aside>

---

## Key Takeaways
<!-- .slide: class="compact-slide" -->

1. Regular RAG often fails because retrieved context is noisy or incomplete. 
2. Standard SFT (typically finetuning only with golden dataset) performs poorly.
3. RAFT trains the model to learn evidence selection or use domain information in high-noise retrieval. 
4. A small domain-tuned model can become practical for production RAG workflows in fixed domain settings.
5. Use RAFT when the domain is stable, retrieval is noisy, and cost matters. 
6. Number of distractor is domain dependent - treat it as hyperparameter during experimentation

**Data science framing:** RAFT reduces train-inference mismatch for RAG.

<aside class="notes">
Close by connecting to the channel audience: this is a practical ML systems pattern, not just a paper summary.
</aside>
