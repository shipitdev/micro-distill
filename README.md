# micro-distill

**A small PyTorch customer-support intent classifier, with a measured distillation comparison.**

This project predicts one of the **27 original intents** in [Bitext's customer-support dataset](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants). It compares a small classifier, a larger teacher, and a fresh student trained with the teacher's softened scores. The central result is honest: **the tested distillation recipe did not improve the small classifier**. The ordinary student was selected using validation before the final test was opened.

The model predicts a Bitext intent name, such as `track_order`. A prediction does not perform or authorize an action.

## Architecture

```text
Original customer request
          |
          v
Pretrained all-MiniLM-L6-v2 sentence encoder (fixed, CPU)
          |
          v
384-number sentence embedding
          |
          +----------------------------+-----------------------------+
          |                            |                             |
          v                            v                             v
Ordinary student              Teacher (training)              Distilled student
Linear 384 -> 27              MLP 384 -> 256 -> 128 -> 27     Linear 384 -> 27
Labels only                   Labels only                    Labels + teacher scores
          |                            |                             |
          +----------------------------+-----------------------------+
                                       |
                                       v
                        Compare on the same held-out requests
```

For the distilled student, the loss is `0.3 × cross-entropy + 0.7 × T² × KL`, with temperature `T=4`. The teacher supplies 27 softened scores during training and receives no student gradients. At inference, the student runs without the teacher. The encoder still runs first for **every** request.

## Data and evaluation

- The Kaggle CSV has **21,534 rows** and **27 intent labels**. Bitext describes the data as authored/generated customer-support examples; these are not measured production requests. Download the CSV from the [Bitext dataset page](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants) and keep its README and license locally. These source files are not committed here.
- All original wording is kept. A comparison-only key groups capitalization and punctuation variants before a seeded split. There are **17,253 training**, **2,135 validation**, and **2,146 test** rows. Related paraphrases with different keys may still cross splits.
- Training adds one missing-letter practice copy per eligible training request. Generated copies matching any original request are excluded; **17,127** typo copies remain. The source labels are unchanged.
- Model choice was fixed from validation. The final test split was then evaluated once. Generated missing-letter and swapped-letter test variants are diagnostic stress tests, not real-world typo benchmarks. One swapped-letter test key matches an original training key; the original test set has no exact comparison-key overlap with training.

### Accuracy

One reproducible run with seed 42 and 30 epochs per model:

| Model | Original validation | Missing-letter validation | Swapped-letter validation | Original test | Missing-letter test | Swapped-letter test |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary student **(selected)** | 99.95% | 99.67% | 98.64% | 99.77% | 99.21% | 98.84% |
| Teacher | 99.95% | 99.48% | 98.92% | 99.95% | 99.67% | 99.39% |
| Distilled student | 99.95% | 98.78% | 97.66% | 99.63% | 98.42% | 97.76% |

The ordinary student made **5 errors** among 2,146 original test requests; the teacher made **1**; the distilled student made **8**. The rare `cancel_order` intent had only **3 test examples**, and the selected student got **2** correct. High aggregate accuracy should not be read as strong evidence for every intent or for new customer language. The test result did not change the validation-based selection.

### CPU size and latency

Measured on macOS arm64, PyTorch 2.14.0, one CPU thread. Single short request, warmed model; classifier timing uses 1,000 passes, encoder and full-path timing use 50 passes. Values vary with hardware and request length. Weight sizes count float32 parameter storage only, not tokenizer, Python, or runtime memory.

| Component | Parameters | Weights (KiB) | Classifier p50 / p95 (ms) | Full text-to-intent p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| Encoder | 22,713,216 | 88,723.5 | — | 2.256 / 2.301 for encoding only |
| Ordinary student | 10,395 | 40.6 | 0.0032 / 0.0033 | 2.272 / 2.342 |
| Teacher | 134,939 | 527.1 | 0.0145 / 0.0149 | 2.314 / 2.402 |
| Distilled student | 10,395 | 40.6 | 0.0032 / 0.0033 | 2.275 / 2.333 |

The small classifier has **92.3% fewer parameters** than the teacher classifier. End-to-end latency is around **2.3 ms**, because the 22.7-million-parameter encoder dominates. This project does **not** claim sub-millisecond text-to-intent routing.

## Reproduce

Download the dataset from the link above and place its CSV at `data/raw/bitext/requests.csv`. Keep the source README and license alongside it locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py                 # train and compare on validation
python train.py --evaluate-test # also reproduce the frozen final test report
python predict.py "Where is my order?" # print one predicted intent
jupyter notebook notebooks/intent_experiment.ipynb # inspect the experiment
```

Install Jupyter separately if needed (`pip install notebook`). The public notebook is a short, runnable record of the dataset, softened teacher scores, and the measured comparison. The training script saves three classifier state dictionaries plus `results.json` and `splits.json` under `artifacts/` (ignored by Git). These record the source CSV hash, exact split row IDs, labels, hardware, per-intent results, model sizes, and timings. For a quick logic check, install `pytest` and run `python -m pytest tests/test_distillation.py`.

## Limits and next use

The source data uses many related request templates, so even the grouped split may be easier than genuinely new customer traffic. Our artificial typo checks cover two error types and are not a substitute for a reviewed sample of real misspellings. The trained intent model never executes actions or chooses a safe tool workflow. A production agent would need separate authorization, confirmation, out-of-scope handling, and evaluation on representative requests.
