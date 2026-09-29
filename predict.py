"""Predict a Bitext intent for one request using a saved local classifier."""

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.models import StudentNet, TeacherNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="one customer-support request")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=("ordinary_student", "teacher", "distilled_student"), default="ordinary_student")
    args = parser.parse_args()
    if not args.request.strip():
        parser.error("request must not be empty")

    report_path = args.artifacts / "results.json"
    if not report_path.exists():
        parser.error(f"missing {report_path}; run train.py first")
    report = json.loads(report_path.read_text())
    names = report["intent_names"]
    model = TeacherNet(len(names)) if args.model == "teacher" else StudentNet(len(names))
    weights_path = args.artifacts / f"{args.model}.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    encoder = SentenceTransformer(report["encoder"], device="cpu")
    with torch.inference_mode():
        embedding = encoder.encode([args.request], convert_to_tensor=True)
        intent_number = model(embedding).argmax(dim=1).item()
    print(names[intent_number])


if __name__ == "__main__":
    main()
