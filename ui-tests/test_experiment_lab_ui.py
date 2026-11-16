from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("LAB_UI_ARTIFACT_DIR", "artifacts/experiment-lab-ui"))


@pytest.fixture(scope="module")
def lab_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = os.environ.copy()
    env["D2C_LAB_PORT"] = str(port)
    env["PYTHONPATH"] = str(ROOT / "code" / "python")
    process = subprocess.Popen(
        [sys.executable, "-m", "d2c.experiment_lab.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Experiment Lab exited during startup:\n{output}")
            try:
                with urllib.request.urlopen(f"{url}/api/experiments", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.25)
        else:
            raise TimeoutError("Experiment Lab did not become ready within 30 seconds")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_experiment_lab_browser_review(lab_url):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trace_path = ARTIFACTS / "experiment-lab-browser-trace.zip"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
        )
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        try:
            page.goto(lab_url, wait_until="networkidle")
            expect(page.get_by_role("heading", name="Aberconics Experiment Lab")).to_be_visible()
            expect(page.locator("#experiment option")).to_have_count(6)
            expect(page.locator("#experiment option[value='g3']")).to_have_count(1)
            expect(page.get_by_role("button", name="Launch experiment")).to_be_visible()
            page.screenshot(path=str(ARTIFACTS / "01-configure-desktop.png"), full_page=True)

            page.locator("#experiment").select_option("g1l")
            expect(page.locator("#g1l-stage-wrap")).to_be_visible()
            page.locator("#g1l-stage").select_option("all")
            expect(page.locator("#confirmation-seeds-wrap")).to_be_visible()
            page.locator("#experiment").select_option("g3")
            expect(page.locator("#bootstrap-wrap")).to_be_visible()
            page.screenshot(path=str(ARTIFACTS / "02-configure-g3.png"), full_page=True)

            page.get_by_role("link", name="Results & compare").click()
            expect(page.locator("#page-results")).to_be_visible()
            expect(page.get_by_role("heading", name="Results explorer")).to_be_visible()
            page.screenshot(path=str(ARTIFACTS / "03-results-desktop.png"), full_page=True)

            page.get_by_role("link", name="Datasets & protocols").click()
            expect(page.locator("#page-datasets")).to_be_visible()
            expect(page.get_by_role("heading", name="Registered dataset profiles")).to_be_visible()
            page.screenshot(path=str(ARTIFACTS / "04-datasets-desktop.png"), full_page=True)

            mobile = context.new_page()
            mobile.set_viewport_size({"width": 390, "height": 844})
            mobile.goto(lab_url, wait_until="networkidle")
            expect(mobile.get_by_role("heading", name="Aberconics Experiment Lab")).to_be_visible()
            mobile.screenshot(path=str(ARTIFACTS / "05-configure-mobile.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(ARTIFACTS / "failure-desktop.png"), full_page=True)
            raise
        finally:
            context.tracing.stop(path=str(trace_path))
            browser.close()
