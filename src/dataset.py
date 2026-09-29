"""Load Bitext requests and keep matching sentences in the same split."""

import csv
import re
from collections import defaultdict
from pathlib import Path
from random import Random


def load_requests(path: str | Path) -> list[dict[str, str]]:
    """Read the original CSV without changing customer wording or intent labels."""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"utterance", "intent"}.issubset(reader.fieldnames or ()):
            raise ValueError("CSV needs utterance and intent columns")
        rows = list(reader)
    if not rows or any(not row["utterance"].strip() or not row["intent"].strip() for row in rows):
        raise ValueError("CSV has no rows or contains an empty utterance/intent")
    return rows


def comparison_key(sentence: str) -> str:
    """A comparison-only key; the encoder always sees the original sentence."""
    return " ".join(re.findall(r"[a-z0-9']+", sentence.lower()))


def split_requests(rows: list[dict[str, str]], seed: int = 42):
    """Return intent names and grouped, stratified train/validation/test row IDs."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[comparison_key(row["utterance"])].append(index)

    names = sorted({row["intent"] for row in rows})
    by_intent = defaultdict(list)
    for ids in groups.values():
        name = rows[ids[0]]["intent"]
        if any(rows[index]["intent"] != name for index in ids):
            raise ValueError(f"matching requests have conflicting labels: {rows[ids[0]]['utterance']}")
        by_intent[name].append(ids)

    rng = Random(seed)
    train_ids, validation_ids, test_ids = [], [], []
    for name in names:
        bundles = by_intent[name][:]
        if len(bundles) < 3:
            raise ValueError(f"intent {name} needs at least three distinct requests")
        rng.shuffle(bundles)
        count = max(1, round(len(bundles) * 0.10))
        for group in bundles[:count]:
            test_ids.extend(group)
        for group in bundles[count : 2 * count]:
            validation_ids.extend(group)
        for group in bundles[2 * count :]:
            train_ids.extend(group)
    return names, train_ids, validation_ids, test_ids


def remove_one_letter(sentence: str, rng: Random) -> str:
    """Make one controlled practice typo in a word with at least five letters."""
    words = list(re.finditer(r"[A-Za-z]{5,}", sentence))
    if not words:
        return sentence
    word = rng.choice(words)
    position = rng.randrange(word.start() + 1, word.end() - 1)
    return sentence[:position] + sentence[position + 1 :]


def swap_two_letters(sentence: str, rng: Random) -> str:
    """Make a different typo for an evaluation stress test."""
    words = list(re.finditer(r"[A-Za-z]{5,}", sentence))
    if not words:
        return sentence
    word = rng.choice(words)
    position = rng.randrange(word.start() + 1, word.end() - 2)
    letters = list(sentence)
    letters[position], letters[position + 1] = letters[position + 1], letters[position]
    return "".join(letters)


def training_typo_copies(rows: list[dict[str, str]], train_ids: list[int], seed: int = 17):
    """Return source row IDs and typo text, excluding copies of any original row."""
    original_keys = {comparison_key(row["utterance"]) for row in rows}
    rng = Random(seed)
    copies = []
    for index in train_ids:
        typo = remove_one_letter(rows[index]["utterance"], rng)
        # This prevents an augmented training row from copying held-out text.
        if comparison_key(typo) not in original_keys:
            copies.append((index, typo))
    return copies
