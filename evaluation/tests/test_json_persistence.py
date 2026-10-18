"""JSON transport fixtures; values are not clinical findings."""
import json

import pytest

from evaluation import utils


@pytest.mark.parametrize("save", [utils.save_json_file, lambda p, d: utils.save_dataset_to_json(d, p)])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_invalid_payload_preserves_existing_file(tmp_path, save, value):
    destination = tmp_path / "data.json"
    previous = b'{"saved":"software fixture"}'
    destination.write_bytes(previous)
    with pytest.raises((ValueError, TypeError)):
        save(destination, [{"value": value}])
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("payload", ['{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}', '{"value":1e999}', '{"value":1,"value":2}'])
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, payload):
    source = tmp_path / "source.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        utils.load_json_file(source)


def test_unicode_roundtrip_and_column_filtering(tmp_path):
    destination = tmp_path / "data.json"
    data = [{"label": "software fixture — δοκιμή", "drop": "unused", "value": 1}]
    utils.save_dataset_to_json(data, destination, columns_to_drop=["drop"])
    assert utils.load_json_file(destination) == [{"label": data[0]["label"], "value": 1}]
    assert data[0]["drop"] == "unused"
    assert not destination.read_bytes().startswith(b'\xef\xbb\xbf')


def test_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "data.json"
    previous = b'{"saved":"software fixture"}'
    destination.write_bytes(previous)
    def fail(*args):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr(utils.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        utils.save_json_file(destination, {"new": "fixture"})
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]


def test_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "data.json"
    destination.write_text('{"original":true}', encoding="utf-8")
    def fail(data, file, **kwargs):
        file.write('{"partial":')
        raise OSError("simulated disk failure")
    monkeypatch.setattr(json, "dump", fail)
    with pytest.raises(OSError, match="disk failure"):
        utils.save_json_file(destination, {"new": "fixture"})
    assert destination.read_text(encoding="utf-8") == '{"original":true}'
    assert list(tmp_path.iterdir()) == [destination]
