.PHONY: all venv cpp-static cpp-shared cpp-test python-test clean help

PYTHON   ?= python3
VENV     := .venv
VENV_BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
  PYTHON   := python
endif

# ── Default target ────────────────────────────────────────────────────
all: venv cpp-static cpp-shared python-test

help:
	@echo "Targets:"
	@echo "  all          – venv + C++ build + Python tests (full pipeline)"
	@echo "  venv         – create venv and install Python deps"
	@echo "  cpp-static   – build static library + tests + apps"
	@echo "  cpp-shared   – build shared library for wrappers"
	@echo "  cpp-test     – run ctest"
	@echo "  python-test  – run pytest"
	@echo "  clean        – remove build dirs and .venv"

# ── Python venv ───────────────────────────────────────────────────────
venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip --quiet
	$(VENV_BIN)/pip install -r requirements.txt --quiet

# ── C++ builds ────────────────────────────────────────────────────────
CPP_DIR  := code/c++ core
STATIC   := $(CPP_DIR)/build
SHARED   := $(CPP_DIR)/build-shared

cpp-static:
	cmake -S "$(CPP_DIR)" -B "$(STATIC)" -DCMAKE_BUILD_TYPE=Release
	cmake --build "$(STATIC)" -j

cpp-shared:
	cmake -S "$(CPP_DIR)" -B "$(SHARED)" -DBUILD_SHARED_LIBS=ON -DGFE_BUILD_TESTS=OFF -DGFE_BUILD_APPS=OFF
	cmake --build "$(SHARED)" -j

cpp-test: cpp-static
	ctest --test-dir "$(STATIC)" --output-on-failure

# ── Python tests ──────────────────────────────────────────────────────
python-test: venv
	PYTHONPATH=code/python $(VENV_BIN)/pytest code/python/tests -q

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	rm -rf "$(STATIC)" "$(SHARED)" "$(VENV)" .pytest_cache
	find code/python -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true