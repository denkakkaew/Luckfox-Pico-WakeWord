# RV1106 INT8 divergence — investigation & hardware-debug tooling

## TL;DR

The KWS CNN is **correct in float and in true INT8** (TFLite int8 ≈ 0.83 acc,
balanced predictions), and the **NPU itself is fine** (Rockchip's stock
`mobilenet_v1` classifies correctly on the board). But after conversion with
**rknn-toolkit2 1.5.2** and execution on the **RV1103/RV1106 NPU**, the model
**collapses to a single class**. The rknn *simulator* does **not** catch this —
its logits match the float TFLite to within ~0.14, i.e. it runs ~float and never
exercises int8. **Hardware is the only place the bug appears.**

Per-layer hardware introspection localizes the failure to the **first conv
layer**: its on-device output is **saturated and spatially uncorrelated**
(cosine ≈ 0 vs the float reference) and cascades into the collapse.

This is an **rknn-toolkit-1.5.2 / RV1106 conversion-or-execution bug for this
graph**, not a model or quantization-quality problem.

## Evidence

| Check | Result |
|---|---|
| Keras float test acc | ~0.74–0.85 |
| **Full INT8 TFLite** (tf.lite, int8 in/out), CPU | **~0.83, balanced** — see `check_int8.py` |
| Stock `mobilenet_v1` on the board NPU | correct (dog=156, cat=283) |
| rknn **simulator** (`init_runtime(target=None)`) | matches float TFLite within 0.14 → runs ~float |
| rknn export on **RV1106 hardware** | collapses to one class |
| First diverging layer on hardware | **conv1**: saturated, cosine ≈ 0 vs float |

Things tried that did **not** fix the hardware collapse:
- BatchNorm (`Conv→BN→ReLU`) to normalize activation scale
- removing strided / `'same'`-padded convs (stride-1 `'valid'` + `MaxPool`)
- `quantized_algorithm='mmse'`
- `build(do_quantization=False)` from a PTQ INT8 TFLite (changed the failure
  from a hard one-class collapse to varied-but-wrong output)
- NB: rknn forces **INT8 input** for rv1106 regardless of the source TFLite's
  `inference_input_type` (a uint8-input model still exports as int8 input).

## Hardware-debug setup (adb-over-TCP, from the dev container)

The container can't do USB passthrough, so use the USB *network* gadget:

```bash
bash tools/rv1106_diag/setup_hw_debug.sh 172.32.0.93
```

This makes the board's `adbd` listen on tcp:5555, pushes + starts
`rknn_server` 1.5.2, and `adb connect`s the toolkit's *bundled* adb. Then, in the
rknn-toolkit2 **1.5.2** venv:

```python
rknn.init_runtime(target="rv1106", device_id="172.32.0.93:5555")  # returns 0
```

**Limitation:** `rknn.accuracy_analysis(..., target='rv1106', device_id=...)`
fails over TCP — it runs `adb root`, which drops the TCP transport (dump path
comes back `None`). The official per-layer tool needs **USB-adb**: run the
toolkit from a Linux host (or WSL2 with `usbipd` USB passthrough) with the board
on USB, then `accuracy_analysis` works and writes `snapshot/error_analysis.txt`
(per-layer float-vs-hardware similarity) — the authoritative way to name the
diverging op.

## Per-layer dump over TCP (what we used instead)

Because `accuracy_analysis` is blocked over TCP, compare layers manually:

1. `make_multi.py` — trains the model, saves the float per-layer "golden"
   activations (`/tmp/golden_k.npy`), one labelled test input
   (`/tmp/spec_in.f32`, raw float32, frame-major 124×128), and **single-output
   truncated TFLites** (`/tmp/tap_k.tflite`, one Keras layer = model output).
2. Convert a `tap_k.tflite` to rknn (`do_quantization=True`, mmse, calib =
   `artifacts/_dataset_nchw.txt`).
3. On the board, dump the layer with **native INT8** reading:
   - `dump_nc.c` — sets the 4-D output attr `fmt = RKNN_TENSOR_NC1HWC2` and
     reads native int8; de-tile in NumPy:
     `nhwc[h,w,c] = nc1hwc2[c//16, h, w, c%16]` (C=16/32/64 → no padding).
   - `dump_i8.c` — for 1-D outputs (dense), plain int8 read + dequant.
   - `test_infer2.c` — full-model classifier (int8 or uint8 input), prints argmax.
4. Compare cosine(golden_k, hardware_k). The first low-cosine layer is the break.

Gotchas:
- **Don't** request FLOAT32 output for intermediate tensors — the conversion is
  unreliable and returns NaN/1e22. Read native INT8 and dequant/de-tile yourself.
- Single-output truncated models for *spatial* layers can hit
  `unsupport cpu Transpose op` (the mini runtime can't run the output transpose
  the truncation introduces). NC1HWC2 native reads avoid this for most layers.
- Match outputs to layers by `n_elems` (sizes are usually distinct).

## Recommended next steps

1. **Authoritative localization:** run `accuracy_analysis` from a Linux/WSL2 host
   over USB-adb (see above) to get `snapshot/error_analysis.txt`.
2. **Escalate to Rockchip/Luckfox** with the crisp repro: a small
   `Conv→…→Dense` KWS CNN that is correct in tf.lite-int8 + rknn-sim but whose
   **first conv saturates** on the RV1106 NPU and collapses the model.
3. **Mirror the shipped Luckfox RV1106 audio KWS model's op set** (it runs on
   the same NPU/driver) to dodge whatever op/pattern rknn 1.5.2 mis-lowers.
4. **Update the board to a newer NPU driver (≥0.9.6)** and use **rknn-toolkit
   2.x** (already in the dev image at 2.3.2), which may have fixed this.

## Files

- `setup_hw_debug.sh` — adb-over-TCP + rknn_server bring-up.
- `check_int8.py` — local true-INT8 TFLite accuracy (proves the model survives int8).
- `make_multi.py` — golden per-layer activations + truncated tap TFLites.
- `conv_no_requant.py` — convert an INT8 TFLite to rknn without re-quantizing.
- `dump_nc.c` / `dump_i8.c` / `test_infer2.c` / `dump_outs.c` — on-device dump/probe
  harnesses (cross-compile with the uClibc toolchain against the 1.5.2 RV1106
  `librknnmrt.so`).
