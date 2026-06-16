# Phase 4 — RV1106 on-device bring-up & the INT8 hardware-divergence investigation

This is the full record of deploying the wake-word model to a real **Luckfox
Pico (RV1103, 64 MB)** and the still-open bug that blocks it. It covers what was
made to work, every fix that was tried, and what's left.

**Status:** the end-to-end *pipeline* runs on the NPU; the *model* collapses to a
single class on hardware due to an **rknn-toolkit2 1.5.2 / RV1106 graph-lowering
or NPU-execution bug** that the rknn simulator does not reveal. Diagnostic tools
and the hardware-debug setup live in [`tools/rv1106_diag/`](../tools/rv1106_diag/).

---

## Part A — Systems bring-up (what now works)

Getting from "cross-compiles" to "runs on the NPU" required unwinding several
wrong assumptions baked into the original design.

1. **Board OS reality.** The board was first flashed with **Ubuntu 22.04 armhf
   (glibc 2.35)**, not the **Buildroot/uClibc** image the project assumed. The
   committed uClibc binary won't even load there (`/lib/ld-uClibc.so.0` absent).

2. **RV1106 needs the "mini" runtime.** The NPU is driven only by
   **`librknnmrt.so`** (the RV1103/RV1106 mini runtime), *not* the full
   `librknnrt.so` (RK356X/RK3588) — the full runtime rejects RV1106 models
   ("Invalid RKNN format"), even Rockchip's own. And `librknnmrt.so` is
   **uClibc-linked**, so RV1106 NPU work fundamentally wants the Buildroot/uClibc
   environment.

3. **Reflash to Buildroot.** Switching the board to the **Buildroot/uClibc**
   image (root@172.32.0.93) fixed both the libc and a critical memory issue:

   | | Ubuntu image | Buildroot image |
   |---|---|---|
   | libc | glibc 2.35 | uClibc 1.0.31 |
   | CMA (NPU DMA memory) | **1 MB** (model alloc fails, ENOMEM) | **24 MB** (ok) |
   | RAM headroom (64 MB part) | ~1 MB free | enough |

4. **Version matching.** The board's NPU **driver is v0.9.2 (2023-08-25)**.
   rknn-toolkit2 / runtime **2.3.2** (and 1.6.0) reject the model on it (needs a
   newer driver + `/dev/dma_heap`). The match is **rknpu2 v1.5.2** (mini runtime
   dated 2023-08-23) + a model built with **rknn-toolkit2 1.5.2**. Reconstructing
   the 1.5.2 toolkit needed an isolated venv (TF 2.8 + numpy 1.23.4 + onnx 1.14 +
   onnxruntime 1.14.1 + onnxoptimizer 0.3.8 + opencv-headless 4.8).

5. **Native INT8 zero-copy I/O.** The mini runtime rejects FP32 tensors, so
   [`board/kws.c`](../board/kws.c) quantizes the spectrogram to INT8 (scale/zp
   from the input attr, honouring the padded `w_stride`) and dequantizes the INT8
   logits itself. The NPU is **root-only** (`/dev/rknpu` is `crw------- root`), so
   `kws` runs with `sudo`/as root.

6. **Width alignment.** The spectrogram width was changed **129 → 128** (drop the
   Nyquist bin) so the NPU's 16-element input stride needs no padding.

**Net result:** `rknn_init` returns 0, `rknn_run` executes, and the **NPU is
confirmed healthy** — Rockchip's stock `mobilenet_v1` classifies correctly on the
board (dog→156, cat→283).

---

## Part B — The correctness bug (still open)

The model runs but **predicts one class for every input** on hardware, while the
rknn simulator is correct. Root-causing this took most of Phase 4.

### The trap: the rknn "simulator" runs ~float

`init_runtime(target=None)` (the PC simulator used by `pc/convert_rknn.py`)
produces logits that match the **float** TFLite to within **0.14**. It does **not**
exercise INT8 — so its "5/5 parity" and confident predictions are meaningless for
predicting hardware. **Only the board tells the truth.**

### The model and the NPU are both fine

- A **true full-INT8 TFLite** (`tf.lite`, int8 in/out), run on CPU, scores **~0.83**
  with a balanced prediction histogram — the model survives INT8 cleanly.
  (See [`tools/rv1106_diag/check_int8.py`](../tools/rv1106_diag/check_int8.py).)
- The NPU runs mobilenet correctly (above).

So the fault is specifically in **rknn's conversion/execution of our graph on the
RV1106**.

### Hardware introspection

Set up connected-board debugging from the dev container over the USB **network**
gadget (no USB passthrough available):
[`tools/rv1106_diag/setup_hw_debug.sh`](../tools/rv1106_diag/setup_hw_debug.sh)
makes the board's `adbd` listen on tcp:5555, starts **`rknn_server` 1.5.2**, and
`adb connect`s the toolkit's *bundled* adb. `init_runtime(target='rv1106',
device_id='172.32.0.93:5555')` then returns 0.

- `accuracy_analysis()` (the official per-layer tool) **does not work over
  TCP** — it runs `adb root`, which drops the TCP transport (dump path → `None`).
  It needs **USB-adb** (run the toolkit from a Linux/WSL2 host with USB
  passthrough).
- So we did per-layer comparison **manually**: single-output truncated models,
  read on-device as **native INT8** with NC1HWC2 de-tiling for 4-D tensors
  (`nhwc[h,w,c]=nc1hwc2[c//16,h,w,c%16]`), compared by cosine to the float
  "golden" activations. Tools: `dump_nc.c`, `dump_i8.c`, `make_multi.py`.

### Finding: divergence starts at the first conv, and is bit-width-independent

- The **first conv layer** already diverges on hardware: cosine ≈ 0 vs float,
  output **saturated** (many values pinned at the INT8 floor; compressed range)
  and spatially uncorrelated. It cascades into the one-class collapse.
- **INT16** (`quantized_dtype='asymmetric_quantized-16'`) builds and runs on the
  NPU but **collapses identically**. With 16 bits of headroom, saturation /
  requant precision **cannot** be the cause → the bug is **bit-width-independent**,
  i.e. graph-lowering / NPU-execution, not quantization quality.

---

## Part C — Everything tried (and the result)

| # | Attempt | Result |
|---|---------|--------|
| 1 | Full runtime `librknnrt.so` (glibc) for RV1106 | ✗ rejects model ("Invalid RKNN format") — RV1106 needs the mini runtime |
| 2 | rknn-toolkit2 / runtime 2.3.2, then 1.6.0 | ✗ driver 0.9.2 too old (DMA-heap / model-verify) — needs 1.5.2 |
| 3 | Ubuntu image | ✗ 1 MB CMA → NPU model alloc ENOMEM; reflashed to Buildroot (24 MB CMA) |
| 4 | FP32 zero-copy I/O | ✗ mini runtime requires native INT8 — switched to manual quant/dequant |
| 5 | 129-wide input | ✗ stride-padded to 144; aligned to **128** (no padding) |
| 6 | Drop `Resizing(32,32)` | ✓ required — mini runtime can't run Resize (CPU offload unsupported) |
| 7 | Drop baked `Normalization`, Conv-first | partial — input then reaches NPU, but still collapses |
| 8 | `quantized_algorithm='mmse'` | ✗ marginal; still collapses |
| 9 | `build(do_quantization=False)` from PTQ INT8 TFLite | ✗ collapse → varied-but-wrong (different failure, still useless) |
| 10 | UINT8-input TFLite | ✗ rknn forces INT8 input for rv1106 regardless |
| 11 | `BatchNorm` (Conv→BN→ReLU) | ✗ normalizes activations but still collapses |
| 12 | Stride-1 `'valid'` convs + `MaxPool` (drop strided/'same') | ✗ still collapses |
| 13 | `GlobalAveragePooling → Dense` head | ✗ output saturates (different failure) |
| 14 | **INT16** quantization | ✗ collapses identically → **bug is bit-width-independent** |

Confirmed-good baselines throughout: float model, true INT8 TFLite (CPU), rknn
simulator, and the NPU itself (mobilenet).

---

## Part D — Recommended next steps

1. **Authoritative layer ID:** run rknn `accuracy_analysis` over **USB-adb** from
   a Linux/WSL2 host (WSL2: attach the board with `usbipd attach --wsl`; then
   `pip install rknn-toolkit2==1.5.2`; `config(target_platform='rv1106')` →
   `load_tflite` → `build` → `accuracy_analysis(inputs=[<calib_nchw.npy>],
   target='rv1106')`). Read `snapshot/error_analysis.txt` — first low-`cos` layer
   = the op to report.
2. **Escalate to Rockchip/Luckfox** with the repro: a small `Conv→…→Dense` KWS
   net that is correct in tf.lite-int8 + rknn-sim but collapses on the RV1106 NPU
   **at any bit width**, diverging from the first conv.
3. **Mirror the shipped Luckfox RV1106 audio-KWS model's op set** (it runs on the
   same NPU/driver) to dodge whatever pattern rknn 1.5.2 mis-lowers.
4. **Update the board to a newer NPU driver (≥0.9.6)** and use **rknn-toolkit
   2.x** (already in the dev image at 2.3.2), which may have fixed this.

The only untested *structural* idea with a rationale is a **fully-convolutional
head** (no `Flatten→Dense`; GlobalAvgPool → 1×1-conv classifier, like mobilenet),
since the `Flatten→Dense` transition is the main structural difference from the
known-good mobilenet.
