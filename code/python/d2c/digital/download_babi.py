"""Download and validate the bAbI JSONL splits used by Aberconics E0/E-F0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence
from urllib.request import urlopen


_BABI_URLS = {
    "train": "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl",
    "test": "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_test.jsonl",
    "valid": "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_valid.jsonl",
}
_REQUIRED_FIELDS = {"passage", "question", "answer", "task"}


def validate_babi_jsonl(path: str | Path) -> dict[str, Any]:
    """Validate one split and return record/task counts."""
    source = Path(path)
    records = 0
    task_ids: set[int] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or not _REQUIRED_FIELDS.issubset(row):
                raise ValueError(f"{source}:{line_number}: missing required bAbI fields")
            if not all(isinstance(row[key], str) for key in ("passage", "question", "answer")):
                raise ValueError(f"{source}:{line_number}: passage/question/answer must be strings")
            try:
                task_id = int(row["task"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: task must be an integer") from exc
            if task_id <= 0:
                raise ValueError(f"{source}:{line_number}: task must be positive")
            records += 1
            task_ids.add(task_id)
    if records == 0:
        raise ValueError(f"{source}: split contains no records")
    return {"path": str(source), "records": records, "task_ids": sorted(task_ids)}


def _download(url: str, target: Path, *, timeout: float) -> None:
    """Stream a download to disk without holding the full split in memory."""
    with urlopen(url, timeout=timeout) as response, target.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def download_babi_dataset(
    output_dir: str | Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Download train/test/valid splits atomically and validate before install.

    Existing valid files are kept by default. Downloads use temporary files in
    the destination directory and are atomically renamed only after validation.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    for split, url in _BABI_URLS.items():
        target = destination / f"babi_{split}.jsonl"
        if target.exists() and not force:
            results[split] = {**validate_babi_jsonl(target), "downloaded": False}
            continue

        fd, temp_name = tempfile.mkstemp(
            prefix=f".babi_{split}.",
            suffix=".part",
            dir=destination,
        )
        os.close(fd)
        temporary = Path(temp_name)
        try:
            _download(url, temporary, timeout=timeout)
            validation = validate_babi_jsonl(temporary)
            os.replace(temporary, target)
            results[split] = {
                **validation,
                "path": str(target),
                "downloaded": True,
            }
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return results


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    default_dir = repo_root / "code" / "python" / "d2c" / "progress" / "babi_data"
    parser = argparse.ArgumentParser(description="Download and validate bAbI JSONL data.")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--force", action="store_true", help="replace existing valid split files")
    parser.add_argument(
        "--splits", default="train,test,valid",
        help="comma-separated subset of train,valid,test (default: all three)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    results = download_babi_dataset(
        args.output_dir,
        force=args.force,
        timeout=args.timeout,
        splits=tuple(value.strip() for value in args.splits.split(",") if value.strip()),
    )
    for split, result in results.items():
        print(
            f"{split}: {result['records']} records, "
            f"tasks={result['task_ids']}, downloaded={result['downloaded']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
