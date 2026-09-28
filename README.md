# micro-distill

**Sub-Millisecond Knowledge Distillation Engine for Production AI Agent & RAG Query Routing**

Routing each prompt with a large language model can add hundreds of milliseconds and a paid inference call before the useful work begins. This project explores a small CPU model that chooses among four destinations: RAG document search, SQL database query, general chit-chat, and sensitive action requiring confirmation.

The experiment implements temperature-scaled knowledge distillation in PyTorch. The teacher is a four-layer MLP; the student is a two-layer MLP. Both consume 384-dimensional vectors. **The dataset generates synthetic clustered vectors, not real text embeddings.** This measures model mechanics and CPU inference only. It does not establish production routing accuracy or safe handling of sensitive actions.

## How it works

```text
Query text --[external embedding model; outside this benchmark]--> 384 floats
                                                            |
                                          +-----------------+-----------------+
                                          |                                   |
                                  TeacherNet (training)              StudentNet
                                  384-256-128-64-4                   384-32-4
                                          |                                   |
                                  soft targets (T=4)                   logits
                                          +---------------+-------------------+
                                                          |
                                     0.3 * cross-entropy(labels, logits)
                                   + 0.7 * T^2 * KL(teacher targets || student)
                                                          |
                                             trained student routes queries
                                                0 RAG  1 SQL  2 chat  3 action
```

The implementation uses `KL(teacher targets || student predictions)` in PyTorch's `kl_div` argument convention. The teacher runs only during training. Its logits are detached so the student loss cannot update it. `T²` restores the approximate gradient scale lost when both logit distributions are softened by temperature.

## Measured results

One run on this machine, PyTorch 2.14.0, single CPU thread, 1,000 timed single-query forwards after 100 warm-up forwards. Accuracy is on 400 held-out synthetic vectors with fresh noise; size is float32 parameter storage, excluding interpreter and runtime overhead. Numbers vary by CPU and run.

| Model | Parameters | Weights (KiB) | CPU p50 (ms) | CPU p95 (ms) | Accuracy |
|---|---:|---:|---:|---:|---:|
| TeacherNet | 139,972 | 546.8 | 0.0183 | 0.0192 | 100.0% |
| StudentNet, baseline | 12,452 | 48.6 | 0.0068 | 0.0070 | 100.0% |
| StudentNet, distilled | 12,452 | 48.6 | 0.0069 | 0.0071 | 100.0% |

The distilled student uses **91.1% fewer parameters** and its measured median forward pass is **2.7× faster** than the teacher. Both students reached 100% on this easy synthetic task, so this run shows compression but **no accuracy improvement from distillation**. The forward pass is under 0.01 ms; embedding generation, request handling, and downstream routing are outside the timing. The target of roughly 2.9× depends on hardware and measurement conditions.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m pip install notebook
jupyter notebook notebooks/01_learning_distillation_walkthrough.ipynb
```

Run the small logic check with `python -m pytest tests/test_distillation.py` if pytest is available. The notebook walks through logits, temperature, model sizes, the blended loss, training, and p50/p95 profiling.

For production evaluation, replace `AgentQueryDataset` with embeddings from real labeled queries, use a held-out distribution, measure end-to-end latency, and separately audit false negatives for sensitive actions. A sensitive action still requires confirmation independent of the classifier's confidence.
