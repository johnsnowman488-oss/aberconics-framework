"""Tests for bAbI dataset downloading and validation without network access."""

from __future__ import annotations

import io
import json

import pytest

from d2c.digital import download_babi


class FakeResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stream.close()

    def read(self, size=-1):
        return self._stream.read(size)


def test_downloads_and_validates_all_splits_atomically(tmp_path, monkeypatch) -> None:
    payload = (
        json.dumps({
            "passage": "Mary moved to kitchen.",
            "question": "Where is Mary?",
            "answer": "kitchen",
            "task": 1,
        }) + "\n"
    ).encode("utf-8")
    requested = []

    def fake_urlopen(url, timeout):
        requested.append((url, timeout))
        return FakeResponse(payload)

    monkeypatch.setattr(download_babi, "urlopen", fake_urlopen)
    result = download_babi.download_babi_dataset(tmp_path, timeout=3.0)

    assert set(result) == {"train", "test", "valid"}
    assert len(requested) == 3
    assert all(row["records"] == 1 and row["downloaded"] for row in result.values())
    assert all((tmp_path / f"babi_{split}.jsonl").exists() for split in result)


def test_existing_valid_data_is_reused_without_network(tmp_path, monkeypatch) -> None:
    row = {
        "passage": "John went to office.",
        "question": "Where is John?",
        "answer": "office",
        "task": 1,
    }
    for split in ("train", "test", "valid"):
        (tmp_path / f"babi_{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    def unexpected_urlopen(*args, **kwargs):
        raise AssertionError("valid existing files should not be downloaded")

    monkeypatch.setattr(download_babi, "urlopen", unexpected_urlopen)
    result = download_babi.download_babi_dataset(tmp_path)
    assert all(not value["downloaded"] for value in result.values())


def test_invalid_download_is_not_installed_and_temporary_file_is_removed(tmp_path, monkeypatch) -> None:
    def fake_urlopen(url, timeout):
        return FakeResponse(b"not-json\n")

    monkeypatch.setattr(download_babi, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="invalid JSON"):
        download_babi.download_babi_dataset(tmp_path)

    assert not (tmp_path / "babi_train.jsonl").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_selected_splits_do_not_touch_unselected_test_split(tmp_path, monkeypatch) -> None:
    payload = (
        json.dumps({
            "passage": "Mary moved to kitchen.",
            "question": "Where is Mary?",
            "answer": "kitchen",
            "task": 1,
        }) + "\n"
    ).encode("utf-8")
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return FakeResponse(payload)

    # An invalid test file is a sentinel: selected train/valid preparation
    # must neither validate, replace, nor otherwise read it.
    test_path = tmp_path / "babi_test.jsonl"
    test_path.write_text("deliberately unread test sentinel", encoding="utf-8")
    monkeypatch.setattr(download_babi, "urlopen", fake_urlopen)

    result = download_babi.download_babi_dataset(
        tmp_path, timeout=3.0, splits=("train", "valid")
    )

    assert set(result) == {"train", "valid"}
    assert len(requested) == 2
    assert all("babi_test.jsonl" not in url for url in requested)
    assert test_path.read_text(encoding="utf-8") == "deliberately unread test sentinel"


def test_selected_splits_reject_empty_unknown_or_duplicate_names(tmp_path) -> None:
    for invalid in ((), ("missing",), ("train", "train")):
        with pytest.raises(ValueError, match="splits must be"):
            download_babi.download_babi_dataset(tmp_path, splits=invalid)
