#!/usr/bin/env python3
"""Convert local JSONL {problem, answer} rows to verl's math parquet schema."""
import argparse
import json
from pathlib import Path


def convert(source, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    with Path(source).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("problem"), str) or not item["problem"].strip() or "answer" not in item:
                raise ValueError(f"Line {number}: expected nonempty problem and answer.")
            rows.append({
                "data_source": "math",
                "prompt": [{"role": "user", "content": item["problem"] + "\nPlease reason step by step and put your final answer within \\boxed{}."}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                "extra_info": {"index": number - 1},
            })
    if not rows:
        raise ValueError("The JSONL input is empty.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    print(f"Wrote {len(rows)} examples to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
