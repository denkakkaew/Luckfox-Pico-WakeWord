#!/usr/bin/env python3
"""Phase 2 — convert the trained CNN core to an INT8 RKNN model for the NPU.

Reads the float TFLite + calibration set produced by pc/train_export.py and
emits ``artifacts/kws.rknn`` (INT8, target rv1106 — also valid for rv1103),
then runs the RKNN **simulator** on a few calibration samples so you can see
class probabilities before touching the board. A best-effort parity check
compares the simulator's predictions against the float TFLite.

CAVEAT (learned in Phase 4): this simulator runs ~float — its logits match the
float TFLite to within ~0.14, so the parity check does NOT validate INT8
behaviour on the real NPU. Only on-board testing does that.

No normalization is configured here (no mean/std): train_export.py already
conditions the feature (log magnitude mapped into [0,255]) and the board feeds
it as raw uint8, so the [0,255] log-spectrogram .npy goes straight in and the
NPU's uint8 input quantizer scales it.

Run from the repo root:  python pc/convert_rknn.py
Inputs (from Phase 1):   artifacts/{kws_core.tflite, dataset.txt, calib/*.npy, labels.txt}
Output:                  artifacts/kws.rknn
"""

import sys
import pathlib

import numpy as np
from rknn.api import RKNN

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
IN_TFLITE = ARTIFACTS / "kws_core.tflite"
DATASET_TXT = ARTIFACTS / "dataset.txt"
LABELS_TXT = ARTIFACTS / "labels.txt"
OUT_RKNN = ARTIFACTS / "kws.rknn"

TARGET = "rv1106"     # also valid for rv1103
N_CHECK = 5           # calibration samples to run through the simulator


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def prepare_calib_dataset() -> pathlib.Path:
    """Build the calibration dataset file rknn.build() consumes.

    Two transforms vs the Phase-1 calib files:
      * Layout: rknn loads the NHWC TFLite into a channel-first graph and wants
        calibration data in NCHW. We transpose each HWC (124,128,1) sample to
        CHW (1,124,128); rknn prepends the batch dim -> (1,1,124,128).
      * Paths: dataset.txt holds repo-root-relative paths, but rknn resolves
        them against the CWD, so we emit absolute paths.
    """
    nchw_dir = ARTIFACTS / "_calib_nchw"
    nchw_dir.mkdir(parents=True, exist_ok=True)
    for old in nchw_dir.glob("*.npy"):
        old.unlink()

    rel = [ln for ln in DATASET_TXT.read_text().split() if ln.strip()]
    abs_paths: list[str] = []
    for ln in rel:
        hwc = np.load(REPO_ROOT / ln)              # (124, 128, 1)
        chw = np.transpose(hwc, (2, 0, 1))         # (1, 124, 128)
        dst = nchw_dir / pathlib.Path(ln).name
        np.save(dst, chw.astype(np.float32))
        abs_paths.append(str(dst.resolve()))

    out = ARTIFACTS / "_dataset_nchw.txt"
    out.write_text("\n".join(abs_paths) + "\n")
    return out


def parity_check(samples: list[pathlib.Path], rknn_preds: list[int],
                 labels: list[str]) -> None:
    """Best-effort: run the float TFLite on the same samples and report how
    often the INT8 simulator agrees. Skipped (not fatal) if TF isn't loadable."""
    try:
        import tensorflow as tf
    except Exception as e:  # noqa: BLE001 - parity is a nicety, never fatal
        print(f"\n(parity check skipped — could not import tensorflow: {e})")
        return

    interp = tf.lite.Interpreter(model_path=str(IN_TFLITE))
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    agree = 0
    print("\nParity vs float TFLite:")
    for fp, rk in zip(samples, rknn_preds):
        x = np.load(fp)[np.newaxis, ...].astype(np.float32)
        interp.set_tensor(in_idx, x)
        interp.invoke()
        tfl = int(np.argmax(interp.get_tensor(out_idx)[0]))
        ok = "ok" if tfl == rk else "MISMATCH"
        agree += tfl == rk
        print(f"  {fp.name}: rknn={labels[rk]:6s} tflite={labels[tfl]:6s} [{ok}]")
    print(f"  agreement: {agree}/{len(samples)}")


def main() -> None:
    for path in (IN_TFLITE, DATASET_TXT, LABELS_TXT):
        if not path.exists():
            sys.exit(f"error: {path} not found — run `python pc/train_export.py` first")

    labels = [ln for ln in LABELS_TXT.read_text().split() if ln.strip()]
    samples = sorted(ARTIFACTS.glob("calib/*.npy"))[:N_CHECK]
    if not samples:
        sys.exit("error: no artifacts/calib/*.npy — run train_export.py first")

    rknn = RKNN(verbose=False)
    # INT8 (w8a8) is the default quantized_dtype; quantized_method='channel'
    # (per-channel weights) is the default. quantized_algorithm='mmse' minimizes
    # quantization MSE (vs 'normal' min/max) — important here because on real
    # RV1106 hardware the 'normal' int8 quantization collapsed the model toward
    # one class (the x86 simulator runs ~float and does NOT reveal this).
    rknn.config(target_platform=TARGET, quantized_algorithm="mmse")

    if rknn.load_tflite(model=str(IN_TFLITE)) != 0:
        sys.exit("error: load_tflite failed")

    if rknn.build(do_quantization=True, dataset=str(prepare_calib_dataset())) != 0:
        sys.exit("error: rknn build failed")

    if rknn.export_rknn(str(OUT_RKNN)) != 0:
        sys.exit("error: export_rknn failed")
    print(f"\nWrote {OUT_RKNN}")

    # Simulator check (target=None runs on the PC, no board needed).
    if rknn.init_runtime() != 0:
        sys.exit("error: init_runtime (simulator) failed")

    print(f"\nSimulator predictions (target {TARGET}):")
    rknn_preds: list[int] = []
    for fp in samples:
        x = np.load(fp)[np.newaxis, ...].astype(np.float32)  # (1, 124, 128, 1)
        out = rknn.inference(inputs=[x], data_format=["nhwc"])
        probs = softmax(np.asarray(out[0][0], dtype=np.float32))
        idx = int(np.argmax(probs))
        rknn_preds.append(idx)
        print(f"  {fp.name}: {labels[idx]} (p={probs[idx]:.3f})")

    rknn.release()
    parity_check(samples, rknn_preds, labels)
    print("\nPhase 2 complete. Next: build the board app (board/kws.c + board/Makefile).")


if __name__ == "__main__":
    main()
