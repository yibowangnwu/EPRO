import json

import pyarrow.parquet as pq
import pytest

from prepare_data import convert


def test_jsonl_becomes_verl_math_parquet(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"problem": "What is 2 + 3?", "answer": 5}) + "\n")
    target = tmp_path / "data/train.parquet"
    convert(source, target)
    record = pq.read_table(target).to_pylist()[0]
    assert record["reward_model"] == {"style": "rule", "ground_truth": "5"}
    assert record["prompt"][0]["role"] == "user"
    assert record["data_source"] == "math"
    assert "\\boxed{}" in record["prompt"][0]["content"]


def test_incomplete_data_is_rejected(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"problem": "Missing answer"}\n')
    with pytest.raises(ValueError, match="answer"):
        convert(source, tmp_path / "out.parquet")
