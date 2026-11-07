"""JSON persistence contracts, collected explicitly by the CI test step."""

import json

from evaluation.utils import load_json_file, save_dataset_to_json, save_json_file


def test_save_json_round_trip(tmp_path):
    path = tmp_path / "record.json"
    data = {"label": "caf\u00e9", "nested": {"enabled": True, "missing": None}}

    save_json_file(str(path), data)

    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert load_json_file(str(path)) == data


def test_save_json_replaces_previous_contents(tmp_path):
    path = tmp_path / "record.json"
    save_json_file(str(path), {"obsolete": "a longer previous value"})
    save_json_file(str(path), {"current": True})

    assert load_json_file(str(path)) == {"current": True}


def test_save_dataset_drops_columns_without_mutating_input(tmp_path):
    path = tmp_path / "records.json"
    data = [{"label": "example", "internal": "omit"}]

    save_dataset_to_json(data, str(path), columns_to_drop=["internal"])

    assert load_json_file(str(path)) == [{"label": "example"}]
    assert data == [{"label": "example", "internal": "omit"}]
