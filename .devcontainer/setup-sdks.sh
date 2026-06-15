#!/usr/bin/env bash
# Fetch the two external SDKs needed to cross-compile the board app:
#   - luckfox-pico  : provides the arm-rockchip830 uClibc cross toolchain
#   - rknn-toolkit2 : provides librknnmrt.so + rknn_api.h (RKNN runtime)
#
# They are cloned into the persistent /opt/sdks volume (see devcontainer.json),
# so this only needs to run once per volume. Shallow clones keep the download
# manageable. The TOOLCHAIN / RKNN_RT env vars already point here.
set -euo pipefail

SDK_ROOT="${SDK_ROOT:-/opt/sdks}"
mkdir -p "$SDK_ROOT"

clone_if_missing() {
  local url="$1" dir="$2"
  if [ -d "$SDK_ROOT/$dir/.git" ]; then
    echo "==> $dir already cloned, skipping"
  else
    echo "==> Cloning $dir (shallow)"
    git clone --depth 1 "$url" "$SDK_ROOT/$dir"
  fi
}

clone_if_missing https://github.com/LuckfoxTECH/luckfox-pico.git luckfox-pico
clone_if_missing https://github.com/airockchip/rknn-toolkit2.git rknn-toolkit2

echo
echo "==> Checking expected paths"
CC="${TOOLCHAIN}gcc"
if [ -x "$CC" ]; then
  echo "  toolchain OK: $CC"
else
  echo "  WARNING: cross gcc not found at $CC"
  echo "  (the luckfox-pico layout may have changed; update TOOLCHAIN in devcontainer.json)"
fi

if [ -d "$RKNN_RT" ]; then
  echo "  rknn runtime OK: $RKNN_RT"
else
  echo "  WARNING: RKNN runtime not found at $RKNN_RT"
fi

echo
echo "Done. You can now run 'make' to cross-compile kws."
