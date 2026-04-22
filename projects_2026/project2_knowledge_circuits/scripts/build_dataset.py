"""Build data/facts.json from the CounterFact dataset.

Filters entries so that target_true and target_false each tokenize to a
single GPT-2 token (with leading space). Saves up to MAX_ENTRIES to disk.
Filtering for "model predicts correct" happens later in the notebook.
"""
import json
from pathlib import Path

from datasets import load_dataset
from transformers import GPT2Tokenizer

MAX_ENTRIES = 1000  # oversample; notebook will further filter by "model knows it"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "facts.json"


def main() -> None:
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")

    kept = []
    for row in ds:
        true_ids = tok.encode(row["target_true"])
        false_ids = tok.encode(row["target_false"])
        if len(true_ids) != 1 or len(false_ids) != 1:
            continue
        kept.append({
            "prompt": row["prompt"],
            "correct": row["target_true"],
            "counterfactual": row["target_false"],
            "relation": row["relation_id"],
            "subject": row["subject"].strip(),
        })
        if len(kept) >= MAX_ENTRIES:
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(kept, f, indent=2)
    print(f"Wrote {len(kept)} entries to {OUT_PATH}")
    if kept:
        print("First entry:", kept[0])


if __name__ == "__main__":
    main()
