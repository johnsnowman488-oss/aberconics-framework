#!/usr/bin/env bash
# Source this file from the repository root to set up environment variables
# for running local Python experiments that depend on the in-repo Python wrapper
# and the built shared library.

# Resolve repo root (assumes this script is in ./scripts)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Add local Python package sources to PYTHONPATH when sourced
export PYTHONPATH="${REPO_ROOT}/code/python:${PYTHONPATH:-}"

# Point the ctypes loader to the shared library build location by default
export GFE_CORE_LIB="${REPO_ROOT}/code/c++ core/build-shared/libgfe_core.so"

echo "[abersoe env] PYTHONPATH set to: ${REPO_ROOT}/code/python"
echo "[abersoe env] GFE_CORE_LIB set to: ${GFE_CORE_LIB}"

return 0 2>/dev/null || true
