"""Portable, integrity-checked Experiment Lab review bundles.

The exporter deliberately excludes datasets, virtual environments and local logs.
It records dataset fingerprints and normalises host-specific paths so a bundle
can be moved between Windows, Linux and macOS without changing its meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA_VERSION = 1
ALLOWED_NAMES = {
    "summary.json", "summary.md", "config.json", "metadata.json",
    "manifest.json", "experiment_lab_register.json", "per_example.csv",
    "per_example.jsonl", "per_seed_metrics.csv", "fixtures.jsonl",
    "comparison.json", "comparison.csv", "G0_REPORT.md",
}
ALLOWED_SUFFIXES = {".png", ".svg"}
EXCLUDED_DIRS = {".venv", "venv", "__pycache__", "babi_data", "datasets", "node_modules"}
MAX_FILE_BYTES = 32 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-._")
    return result[:120] or "run"


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_metadata(summary_dir: Path, input_root: Path) -> Path | None:
    for parent in (summary_dir, *summary_dir.parents):
        if parent == input_root.parent:
            break
        candidate = parent / "metadata.json"
        if candidate.is_file():
            return candidate
        if parent == input_root:
            break
    return None


def _candidate_file(summary_dir: Path, metadata_path: Path | None, name: str) -> Path | None:
    candidates = [summary_dir / name]
    if metadata_path:
        candidates.extend([metadata_path.parent / name, metadata_path.parent.parent / name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _normalise_command(command: Any) -> list[str] | None:
    if not isinstance(command, list):
        return None
    result: list[str] = []
    hide_next = False
    for index, item in enumerate(command):
        text = str(item)
        if hide_next:
            if index == 0:
                result.append("python")
            elif PureWindowsPath(text).is_absolute() or Path(text).is_absolute():
                result.append(".")
            else:
                result.append(text)
            hide_next = False
            continue
        if text in {"--output-dir", "--out", "--output"}:
            result.append(text)
            hide_next = True
            continue
        if text in {"--data-dir", "--dataset-dir"}:
            result.extend([text, "dataset://recorded-fingerprint"])
            hide_next = True
            continue
        if index == 0:
            result.append("python")
        elif PureWindowsPath(text).is_absolute() or Path(text).is_absolute():
            result.append(Path(text.replace("\\", "/")).name or ".")
        else:
            result.append(text)
    return result


def _normalise_object(value: Any) -> Any:
    """Remove machine-specific paths while preserving experiment semantics."""
    if isinstance(value, list):
        return [_normalise_object(item) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, str):
            win = PureWindowsPath(value)
            if win.is_absolute() or Path(value).is_absolute():
                return win.name or "."
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if lowered in {"directory", "output_dir", "working_directory", "cwd"}:
            output[key] = "."
        elif lowered in {"command", "argv"}:
            output[key] = _normalise_command(item)
        else:
            output[key] = _normalise_object(item)
    return output


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep interpretation and diagnostics; keep per-example rows as files."""
    omitted = {
        "per_example", "per_example_rows", "predictions", "examples",
        "fixtures", "per_example_jsonl",
    }
    return {key: value for key, value in summary.items() if key not in omitted}


def _experiment_name(input_root: Path, summary_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        relative = summary_path.relative_to(input_root)
        first = relative.parts[0].lower() if len(relative.parts) > 1 else input_root.name.lower()
    except ValueError:
        first = input_root.name.lower()
    for name in ("g1l", "g2", "g3", "g3-validation", "e0", "ef0", "g1", "g0"):
        if first.startswith(name):
            return name
    return safe_id(first)


def _files_for_run(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(run_dir).parts):
            continue
        if path.name in ALLOWED_NAMES or path.suffix.lower() in ALLOWED_SUFFIXES:
            if path.stat().st_size <= MAX_FILE_BYTES:
                files.append(path)
    return files


def export_bundle(
    input_root: Path,
    output_dir: Path,
    experiment: str | None = None,
    require_runs: bool = True,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if output_dir == input_root or input_root in output_dir.parents:
        raise ValueError("output directory must be outside the input root")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summaries = sorted(
        path for path in input_root.rglob("summary.json")
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.parts)
    )
    if require_runs and not summaries:
        raise RuntimeError(f"no summary.json files found under {input_root}")

    runs: list[dict[str, Any]] = []
    included_files: dict[str, str] = {}
    seen: set[str] = set()

    for summary_path in summaries:
        run_dir = summary_path.parent
        metadata_path = _candidate_metadata(run_dir, input_root)
        metadata = _read_json(metadata_path)
        summary = _read_json(summary_path)
        exp = _experiment_name(input_root, summary_path, experiment)
        raw_run_id = metadata.get("run_id") or run_dir.name
        run_id = safe_id(str(raw_run_id))
        key = safe_id(f"{exp}-{run_id}")
        if key in seen:
            key = safe_id(f"{key}-{hashlib.sha256(str(summary_path).encode()).hexdigest()[:8]}")
        seen.add(key)

        destination = runs_root / key
        destination.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for source in _files_for_run(run_dir):
            relative = source.relative_to(run_dir).as_posix()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            bundle_relative = target.relative_to(output_dir).as_posix()
            included_files[bundle_relative] = sha256_file(target)
            copied.append(bundle_relative)

        # Metadata may live outside the result directory. Copy a sanitised,
        # self-contained version into the portable run root.
        config_path = _candidate_file(run_dir, metadata_path, "config.json")
        register_path = _candidate_file(run_dir, metadata_path, "experiment_lab_register.json")
        manifest_path = _candidate_file(run_dir, metadata_path, "manifest.json")
        portable = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": exp,
            "status": metadata.get("status", "completed"),
            "created_at": metadata.get("created_at") or metadata.get("started_at"),
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at"),
            "exit_code": metadata.get("exit_code"),
            "code_revision": metadata.get("code_revision")
                or metadata.get("commit_sha")
                or summary.get("code_revision"),
            "python_version": metadata.get("python_version"),
            "configuration": _normalise_object(
                _read_json(config_path) or metadata.get("config") or summary.get("configuration") or {}
            ),
            "dataset_fingerprints": (
                metadata.get("dataset_fingerprints")
                or metadata.get("config", {}).get("dataset_fingerprints")
                or summary.get("dataset")
                or {}
            ),
            "source_directory": run_dir.relative_to(input_root).as_posix(),
            "source_metadata_sha256": sha256_file(metadata_path) if metadata_path else None,
            "result_files": copied,
        }
        (destination / "metadata.json").write_text(
            json.dumps(portable, indent=2, sort_keys=True), encoding="utf-8"
        )
        included_files[f"runs/{key}/metadata.json"] = sha256_file(destination / "metadata.json")

        if manifest_path:
            original_manifest = _read_json(manifest_path)
            safe_manifest = _normalise_object(original_manifest)
            safe_manifest["portable_bundle"] = True
            safe_manifest["source_manifest_sha256"] = sha256_file(manifest_path)
            (destination / "manifest.json").write_text(
                json.dumps(safe_manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            included_files[f"runs/{key}/manifest.json"] = sha256_file(destination / "manifest.json")

        if config_path:
            config = _normalise_object(_read_json(config_path))
            (destination / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
            )
            included_files[f"runs/{key}/config.json"] = sha256_file(destination / "config.json")

        if register_path:
            register = _normalise_object(_read_json(register_path))
            (destination / "experiment_lab_register.json").write_text(
                json.dumps(register, indent=2, sort_keys=True), encoding="utf-8"
            )
            included_files[f"runs/{key}/experiment_lab_register.json"] = sha256_file(
                destination / "experiment_lab_register.json"
            )

        summary_target = next(
            (destination / Path(item).relative_to(destination).as_posix()
             for item in copied if Path(item).name == "summary.json"),
            None,
        )
        summary_rel = summary_target.relative_to(output_dir).as_posix() if summary_target else None
        entry = {
            "id": key,
            "run_id": run_id,
            "experiment": exp,
            "status": portable["status"],
            "created_at": portable["created_at"],
            "code_revision": portable["code_revision"],
            "evaluation_split": (
                portable["configuration"].get("evaluation_split")
                or portable["configuration"].get("eval_split")
                or summary.get("protocol", {}).get("evaluation_split")
                or summary.get("evaluation_split")
            ),
            "claim_boundary": summary.get("claim_boundary") or summary.get("interpretation"),
            "summary_path": summary_rel,
            "summary": _compact_summary(summary),
            "metadata_path": f"runs/{key}/metadata.json",
            "manifest_path": f"runs/{key}/manifest.json" if manifest_path else None,
            "config_path": f"runs/{key}/config.json" if config_path else None,
            "register_path": f"runs/{key}/experiment_lab_register.json" if register_path else None,
            "files": sorted(set(copied)),
            "dataset_fingerprints": portable["dataset_fingerprints"],
        }
        runs.append(entry)

    index = {
        "bundle_schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source": "Aberconics Experiment Lab",
        "runs": runs,
    }
    (output_dir / "review-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    included_files["review-index.json"] = sha256_file(output_dir / "review-index.json")
    manifest = {
        "bundle_schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_count": len(runs),
        "files": included_files,
    }
    (output_dir / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"run_count": len(runs), "file_count": len(included_files), "runs": runs}


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest = _read_json(bundle_dir / "bundle-manifest.json")
    if manifest.get("bundle_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing review bundle schema version")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("bundle manifest contains no file hashes")
    checked = 0
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe bundle path: {relative}")
        path = (bundle_dir / rel).resolve()
        if bundle_dir not in path.parents:
            raise ValueError(f"bundle path escapes root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"bundle file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        checked += 1
    index = _read_json(bundle_dir / "review-index.json")
    if not isinstance(index.get("runs"), list):
        raise ValueError("review index has no run list")
    return {"valid": True, "checked_files": checked, "run_count": len(index["runs"])}


def write_zip(bundle_dir: Path, zip_path: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(bundle_dir).as_posix())


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="export a portable review bundle")
    export.add_argument("--input-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--experiment")
    export.add_argument("--allow-empty", action="store_true")
    export.add_argument("--zip", type=Path)
    verify = subparsers.add_parser("verify", help="verify an exported bundle")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        result = export_bundle(
            args.input_root, args.output_dir, args.experiment,
            require_runs=not args.allow_empty,
        )
        if args.zip:
            write_zip(args.output_dir, args.zip)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(verify_bundle(args.bundle_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
