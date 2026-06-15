#!/usr/bin/env bash
# Runs once after the container is created. Verifies the Python stack and points
# you at the SDK setup step. Kept light so container creation stays fast — the
# multi-GB SDK clones are opt-in via setup-sdks.sh.
set -euo pipefail

echo "==> Verifying Python toolchain"
python - <<'PY'
import numpy
print(f"  numpy        {numpy.__version__}")
try:
    import tensorflow as tf
    print(f"  tensorflow   {tf.__version__}")
except Exception as e:  # pragma: no cover - diagnostic only
    print(f"  tensorflow   FAILED: {e}")
try:
    from rknn.api import RKNN  # noqa: F401
    print("  rknn-toolkit2 import OK")
except Exception as e:  # pragma: no cover - diagnostic only
    print(f"  rknn-toolkit2 FAILED: {e}")
PY

echo
echo "==> Claude Code"
if command -v claude >/dev/null 2>&1; then
  echo "  $(claude --version 2>/dev/null || echo 'installed')"
  echo "  Run 'claude' and log in once; auth persists in the ~/.claude volume."
else
  echo "  claude CLI not found on PATH"
fi

echo
if [ -d "${SDK_ROOT:-/opt/sdks}/luckfox-pico" ] && [ -d "${SDK_ROOT:-/opt/sdks}/rknn-toolkit2" ]; then
  echo "==> SDKs already present in ${SDK_ROOT:-/opt/sdks}"
else
  echo "==> Cross-compile SDKs not installed yet."
  echo "    To build the board app, fetch them once with:"
  echo "        bash .devcontainer/setup-sdks.sh"
fi
echo
echo "Ready. See README.md 'Step 1 — Train & export' to start."
