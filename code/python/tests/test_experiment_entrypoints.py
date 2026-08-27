from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_symbolic_induction_script_runs_directly_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "code/python/d2c/experiments/symbolic_induction.py"), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Run the Milestone D multi-pair symbolic retrieval probe." in completed.stdout


def test_thinking_between_tokens_script_runs_directly_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "code/python/d2c/experiments/thinking_between_tokens.py"), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Run the Milestone D2 free-evolution probe." in completed.stdout


def test_learned_symbolic_retrieval_script_runs_directly_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "code/python/d2c/experiments/learned_symbolic_retrieval.py"), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Train a query-conditioned readout over frozen D2C memory." in completed.stdout


def test_temporal_logic_script_runs_directly_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "code/python/d2c/experiments/temporal_logic.py"), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Run the D3 delayed-feedback temporal-rule probe." in completed.stdout
    assert "--soft-eligibility" in completed.stdout
    assert "--nonlinear-eligibility" in completed.stdout
    assert "--predictive-credit-eligibility" in completed.stdout
    assert "--persistent-predictive-credit-eligibility" in completed.stdout
