#!/usr/bin/env bash
# Set up rknn-toolkit2 connected-board debugging for the Luckfox (RV1103/RV1106)
# over the USB *network* gadget (adb-over-TCP), from inside the dev container.
#
# Why: the dev container has no USB passthrough, so the toolkit's normal USB-adb
# transport is unavailable. The board's adbd can listen on tcp:5555, and the
# toolkit's bundled adb can `adb connect` to it over the gadget network.
#
# Usage:  bash tools/rv1106_diag/setup_hw_debug.sh [BOARD_IP]
# Then in the 1.5.2 venv:  init_runtime(target='rv1106', device_id='<IP>:5555')
#
# Caveat: rknn.accuracy_analysis() does NOT work over this transport — it runs
# `adb root`, which drops the TCP connection. accuracy_analysis needs USB-adb
# (run the toolkit from a Linux host with usbipd/USB passthrough). For per-layer
# work over TCP, use the dump_*.c harnesses here instead.
set -euo pipefail
BOARD_IP="${1:-172.32.0.93}"
PORT=5555
SSH="sshpass -p luckfox ssh -o StrictHostKeyChecking=accept-new root@${BOARD_IP}"
SCP="sshpass -p luckfox scp -o StrictHostKeyChecking=accept-new"

# The toolkit's bundled adb (must be used so the device lands on the toolkit's
# adb server; version mismatches with the system adb cause it to kill servers).
BADB=$(find / -path '*rknn/3rdparty/platform-tools/adb/linux-x86_64/adb' 2>/dev/null | head -1)
RKNPU2="${RKNPU2:-/opt/sdks/rknpu2-152}"   # airockchip/rknpu2 @ v1.5.2 checkout

echo "==> board adbd: ensure it listens on tcp:${PORT}"
$SSH "pgrep adbd >/dev/null || (setsid adbd >/tmp/adbd.log 2>&1 < /dev/null &); sleep 1; \
      grep -q ':15B3' /proc/net/tcp && echo 'adbd listening on 5555' || echo 'WARN: 5555 not listening'"

echo "==> push + start rknn_server 1.5.2 (RV1106 mini runtime)"
SRV="$RKNPU2/runtime/RV1106/Linux/rknn_server/arm/usr/bin"
MRT="$RKNPU2/runtime/RV1106/Linux/librknn_api/armhf/librknnmrt.so"
$SCP "$SRV/rknn_server" "$SRV/start_rknn.sh" "$SRV/restart_rknn.sh" root@${BOARD_IP}:/usr/bin/
$SCP "$MRT" root@${BOARD_IP}:/usr/lib/
$SSH "chmod +x /usr/bin/rknn_server /usr/bin/start_rknn.sh /usr/bin/restart_rknn.sh; \
      killall rknn_server 2>/dev/null; sleep 1; \
      setsid start_rknn.sh >/tmp/rknn_server.log 2>&1 < /dev/null & sleep 2; \
      head -2 /tmp/rknn_server.log"

echo "==> connect the toolkit's adb to the board over TCP"
"$BADB" disconnect >/dev/null 2>&1 || true
"$BADB" connect "${BOARD_IP}:${PORT}"
"$BADB" devices
echo "==> done. In the 1.5.2 venv: rknn.init_runtime(target='rv1106', device_id='${BOARD_IP}:${PORT}')"
