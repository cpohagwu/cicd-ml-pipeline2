from pathlib import Path

import train


def test_validation_thresholds_match_assignment():
    assert train.MIN_ACCURACY == 0.60
    assert train.MAX_INFERENCE_COST == 0.003


def test_parse_inference_cost_uses_environment(monkeypatch):
    monkeypatch.setenv("INFERENCE_COST_PER_1000", "0.002")

    assert train.parse_inference_cost() == 0.002


def test_parse_inference_cost_falls_back_for_invalid_value(monkeypatch):
    monkeypatch.setenv("INFERENCE_COST_PER_1000", "not-a-number")

    assert train.parse_inference_cost(default=0.001) == 0.001


def test_resolve_data_path_prefers_explicit_data_path(tmp_path, monkeypatch):
    data_path = tmp_path / "reviews.csv"
    data_path.write_text("text,sentiment\nexcellent,1\nbad,0\n", encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", str(data_path))

    assert train.resolve_data_path() == data_path


def test_resolve_data_path_supports_repo_root_reviews(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_PATH", raising=False)
    Path("reviews.csv").write_text("text,sentiment\nexcellent,1\nbad,0\n", encoding="utf-8")

    assert train.resolve_data_path() == Path("reviews.csv")
