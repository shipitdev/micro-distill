"""Train and compare Bitext teacher, ordinary student, and distilled student."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from random import Random

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark import (
    accuracy,
    common_confusions,
    count_parameters,
    measure_latency,
    measure_model_size_kib,
    per_intent_recall,
    predict,
)
from src.dataset import load_requests, remove_one_letter, split_requests, swap_two_letters, training_typo_copies
from src.losses import DistillationLoss
from src.models import StudentNet, TeacherNet

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode(encoder, texts: list[str]) -> torch.Tensor:
    """Make one 384-number row per sentence, ready for classifier training."""
    return encoder.encode(texts, batch_size=128, convert_to_tensor=True, show_progress_bar=False).clone()


def train_classifier(model, inputs, answers, epochs, learning_rate, seed, teacher=None):
    """The notebook's five-step PyTorch training loop, shared by all models."""
    batches = DataLoader(
        TensorDataset(inputs, answers),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = DistillationLoss(temperature=4.0) if teacher is not None else nn.CrossEntropyLoss()
    if teacher is not None:
        teacher.eval()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in batches:
            optimizer.zero_grad()
            scores = model(batch_X)
            if teacher is None:
                loss = loss_fn(scores, batch_y)
            else:
                # The teacher is a fixed target; only student weights receive gradients.
                with torch.no_grad():
                    teacher_scores = teacher(batch_X)
                loss = loss_fn(scores, teacher_scores, batch_y)
            loss.backward()
            optimizer.step()
        if epoch in (1, 10, epochs):
            print(f"{model.__class__.__name__}: epoch {epoch}/{epochs}")
    model.eval()
    return model


def evaluate_models(models, inputs, answers, names):
    results = {}
    correct = answers.tolist()
    for model_name, model in models.items():
        guesses = predict(model, inputs)
        results[model_name] = {
            "accuracy": accuracy(guesses, correct),
            "per_intent_recall": per_intent_recall(guesses, correct, names),
            "common_confusions": common_confusions(guesses, correct, names),
        }
    return results


def print_accuracy_table(title, results):
    print(f"\n{title}")
    print(f"{'Model':20} {'Original':>10} {'Missing':>10} {'Swapped':>10}")
    for name in ("ordinary_student", "teacher", "distilled_student"):
        print(
            f"{name:20} {results['original'][name]['accuracy']:>9.2%} "
            f"{results['missing_letter'][name]['accuracy']:>9.2%} "
            f"{results['swapped_letters'][name]['accuracy']:>9.2%}"
        )


def benchmark(models, encoder, request: str):
    """Measure warmed single-request CPU paths, including the encoder."""
    embedded = encode(encoder, [request])
    sizes = {}
    timings = {}
    for name, model in models.items():
        sizes[name] = {
            "parameters": count_parameters(model),
            "weights_kib": measure_model_size_kib(model),
        }
        with torch.inference_mode():
            p50, p95 = measure_latency(lambda: model(embedded), warmups=100, passes=1000)
        timings[name] = {"classifier_p50_ms": p50, "classifier_p95_ms": p95}

    sizes["encoder"] = {
        "parameters": count_parameters(encoder),
        "weights_kib": measure_model_size_kib(encoder),
    }
    encoder_p50, encoder_p95 = measure_latency(lambda: encoder.encode([request]), passes=50)
    timings["encoder"] = {"p50_ms": encoder_p50, "p95_ms": encoder_p95}
    for name, model in models.items():
        with torch.inference_mode():
            p50, p95 = measure_latency(
                lambda: model(encoder.encode([request], convert_to_tensor=True)), passes=50
            )
        timings[name]["full_p50_ms"] = p50
        timings[name]["full_p95_ms"] = p95
    return sizes, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/bitext/requests.csv"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--evaluate-test", action="store_true", help="open the final test split once choices are fixed")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if not args.data.exists():
        parser.error(f"missing {args.data}; download the Bitext Kaggle CSV first")

    torch.set_num_threads(1)
    rows = load_requests(args.data)
    names, train_ids, validation_ids, test_ids = split_requests(rows, args.seed)
    if len(names) != 27:
        parser.error(f"expected 27 Bitext intents, found {len(names)}")
    label_to_id = {name: number for number, name in enumerate(names)}
    typo_copies = training_typo_copies(rows, train_ids)
    print(
        f"Rows: {len(rows)} | train: {len(train_ids)} | validation: {len(validation_ids)} "
        f"| test: {len(test_ids)} | kept typo copies: {len(typo_copies)}"
    )

    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    original_train = encode(encoder, [rows[index]["utterance"] for index in train_ids])
    typo_train = encode(encoder, [text for _, text in typo_copies])
    X_train = torch.cat([original_train, typo_train])
    y_train = torch.tensor(
        [label_to_id[rows[index]["intent"]] for index in train_ids]
        + [label_to_id[rows[index]["intent"]] for index, _ in typo_copies],
        dtype=torch.long,
    )

    torch.manual_seed(args.seed)
    ordinary = train_classifier(StudentNet(), X_train, y_train, args.epochs, 0.003, args.seed)
    torch.manual_seed(args.seed)
    teacher = train_classifier(TeacherNet(), X_train, y_train, args.epochs, 0.001, args.seed)
    torch.manual_seed(args.seed)
    distilled = train_classifier(StudentNet(), X_train, y_train, args.epochs, 0.003, args.seed, teacher)
    models = {"ordinary_student": ordinary, "teacher": teacher, "distilled_student": distilled}

    def evaluate_split(ids, missing_seed, swap_seed):
        texts = [rows[index]["utterance"] for index in ids]
        answers = torch.tensor([label_to_id[rows[index]["intent"]] for index in ids], dtype=torch.long)
        variants = {"original": texts}
        missing_rng = Random(missing_seed)
        swap_rng = Random(swap_seed)
        variants["missing_letter"] = [remove_one_letter(text, missing_rng) for text in texts]
        variants["swapped_letters"] = [swap_two_letters(text, swap_rng) for text in texts]
        return {kind: evaluate_models(models, encode(encoder, samples), answers, names) for kind, samples in variants.items()}

    validation = evaluate_split(validation_ids, 29, 31)
    print_accuracy_table("Validation (used for model choice)", validation)
    # This choice was fixed from notebook validation before opening test labels.
    selected = "ordinary_student"
    print(f"Selected classifier: {selected}; distillation did not improve validation robustness.")

    test = None
    if args.evaluate_test:
        test = evaluate_split(test_ids, 43, 47)
        print_accuracy_table("Final held-out test", test)
        selected_result = test["original"][selected]
        print("Worst five test intents (correct / total):")
        recalls = selected_result["per_intent_recall"]
        for name, (correct, total) in sorted(recalls.items(), key=lambda item: item[1][0] / item[1][1])[:5]:
            print(f"  {name}: {correct}/{total}")
        print("Most common test confusions (correct → guessed):")
        for correct, guessed, count in selected_result["common_confusions"]:
            print(f"  {correct} → {guessed}: {count}")

    sizes, timings = benchmark(models, encoder, "Where is my order?")
    print("\nWarmed CPU latency (p50 / p95 ms)")
    print(f"Encoder: {timings['encoder']['p50_ms']:.3f} / {timings['encoder']['p95_ms']:.3f}")
    for name in models:
        value = timings[name]
        print(
            f"{name}: classifier {value['classifier_p50_ms']:.4f} / {value['classifier_p95_ms']:.4f} "
            f"| full text-to-intent {value['full_p50_ms']:.3f} / {value['full_p95_ms']:.3f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    split_manifest = {
        "data_sha256": data_sha256,
        "seed": args.seed,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
    }
    (args.output_dir / "splits.json").write_text(json.dumps(split_manifest) + "\n")
    report = {
        "source": str(args.data),
        "data_sha256": data_sha256,
        "encoder": ENCODER_NAME,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "seed": args.seed,
        "epochs": args.epochs,
        "split_rows": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "intent_names": names,
        "selected_classifier": selected,
        "validation": validation,
        "test": test,
        "sizes": sizes,
        "timings": timings,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved weights, split manifest, and results to {args.output_dir}")


if __name__ == "__main__":
    main()
