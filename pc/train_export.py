#!/usr/bin/env python3
"""Phase 1 — train the KWS CNN and export the NPU-bound model core.

This adapts the TensorFlow simple_audio tutorial CNN for the Luckfox Pico NPU.

Why this script looks the way it does
-------------------------------------
RKNN cannot convert ``tf.signal.stft``. The tutorial embeds the STFT
(audio -> spectrogram) *inside* the Keras model, which will not compile for the
NPU. So we split the pipeline: the STFT runs in C on the board, and the Keras
model trained here consumes a **pre-computed log-magnitude spectrogram** of
shape ``(1, 124, 128, 1)`` (the rfft's 129th/Nyquist bin is dropped so the
width is 16-aligned for the NPU; see build_model and SPEC_BINS).

NOTE (Phase 4): the model architecture below was reshaped for RV1106 NPU
compatibility (no Resize, no baked Normalization, conv→BN→relu, no strided/'same'
convs). It is correct in float / true INT8 / the rknn simulator but still
collapses on the RV1106 NPU due to an rknn-toolkit2 1.5.2 graph-lowering bug —
see docs/RV1106-phase4-investigation.md. The contracts below are unaffected.

Coupling contracts the board side (kws.c) MUST honour
------------------------------------------------------
1. STFT math: periodic Hann window 255, hop 128, FFT 256, drop Nyquist -> (124, 128).
2. Log spectrogram: we use ``log(|stft| + 1e-6)``; kws.c must use
   ``logf(mag + 1e-6f)`` identically.
3. Input scale: ``audio_dataset_from_directory`` decodes WAV to float in
   [-1, 1]. The board's ``arecord`` yields int16 PCM, so kws.c must divide
   samples by 32768.0 before the STFT so the spectrogram magnitudes match
   training (there is no baked Normalization layer; INT8 input quantization
   handles scaling).
4. Label order: classes are alphabetical (see labels.txt); kws.c's LABELS[] /
   WAKE_IDX / NUM_CLASSES must match that ordering and count.

Usage
-----
    python train_export.py                  # demo: mini_speech_commands, wake = "yes"
    python train_export.py path/to/dataset  # custom: dataset/{wake,unknown,noise}/*.wav

Outputs (under artifacts/, all gitignored)
-------------------------------------------
    artifacts/kws_core.tflite   float CNN, input (1, 124, 128, 1)
    artifacts/calib/*.npy       ~150 spectrograms, each (124, 128, 1), for INT8 calib
    artifacts/dataset.txt       one calib .npy path per line, repo-root-relative
                                (consumed by pc/convert_rknn.py — run from repo root)
    artifacts/labels.txt        class order, alphabetical, one per line

Paths resolve relative to the repo root (this file's parent's parent), so the
script works no matter which directory you launch it from.
"""

import sys
import pathlib

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Fixed constants — coupling point #1 with kws.c. Do not change one side only.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # 1 s clips, 16 kHz mono
CLIP_SAMPLES = 16000
FRAME_LENGTH = 255           # periodic Hann window
FRAME_STEP = 128             # hop
FFT_LENGTH = 256
SPEC_FRAMES = 124            # 1 + (16000 - 255) // 128
# rfft yields 256//2 + 1 = 129 bins, but we drop the Nyquist bin to make the
# width 128 (a multiple of 16). The RV1106 NPU pads the input width to a 16-
# element stride for zero-copy; a 129-wide input gets padded to 144, and the
# mini runtime then fails to read the strided C=1 input (NPU sees constant
# input). 128 needs no padding. kws.c must drop the same bin (compute 128 bins).
SPEC_BINS = 128
SPEC_SHAPE = (SPEC_FRAMES, SPEC_BINS, 1)

SEED = 1337
BATCH_SIZE = 64
EPOCHS = 10
N_CALIB = 150                # number of calibration spectrograms to export

# Repo-root-relative output locations. REPO_ROOT = .../pc/train_export.py -> repo.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
OUT_TFLITE = ARTIFACTS / "kws_core.tflite"
OUT_LABELS = ARTIFACTS / "labels.txt"
OUT_CALIB_DIR = ARTIFACTS / "calib"
OUT_DATASET_TXT = ARTIFACTS / "dataset.txt"


def resolve_data_dir(arg: str | None) -> pathlib.Path:
    """Return the dataset directory, downloading the demo set if no arg given."""
    if arg:
        data_dir = pathlib.Path(arg).expanduser()
        if not data_dir.is_dir():
            sys.exit(f"error: dataset path not found: {data_dir}")
        print(f"Using custom dataset: {data_dir}")
        return data_dir

    print("No dataset given — downloading mini_speech_commands demo set...")
    cache = tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=str(REPO_ROOT),
        cache_subdir="data",
    )
    # get_file returns the archive path; the data lives alongside it.
    data_dir = pathlib.Path(cache).with_name("mini_speech_commands")
    if not data_dir.is_dir():
        data_dir = pathlib.Path(cache).parent / "mini_speech_commands"
    print(f"Demo dataset ready: {data_dir}")
    return data_dir


def make_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    """Waveform (..., 16000) in [-1, 1] -> log-magnitude spectrogram (124, 129, 1).

    Coupling points #1 (STFT params) and #2 (log) with kws.c live here.
    """
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    mag = tf.abs(stft)
    spec = tf.math.log(mag + 1e-6)
    spec = spec[..., :SPEC_BINS]      # drop Nyquist bin -> width 128 (NPU-aligned)
    return spec[..., tf.newaxis]


def to_spectrogram_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda audio, label: (make_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def build_model(num_labels: int) -> tf.keras.Model:
    """KWS CNN with the STFT layer removed — input is a (124,129,1) spectrogram.

    This deviates from the tutorial architecture for RV1103/RV1106 NPU hardware
    compatibility (the tutorial is correct in float/simulator but failed on the
    mini runtime):

    1. Tutorial uses Resizing(32, 32) first. The mini runtime cannot execute the
       Resize op (offloads to CPU, unsupported -> rknn_run fails). Removed.
    2. Tutorial bakes a Normalization layer. On the RV1106 NPU this model then
       *ignored its input entirely* (constant output for any input, incl. zeros),
       though the x86 simulator was correct. Removed — INT8 input quantization
       already scales the raw log-spectrogram; no separate Normalization needed.
    3. The first op must be a Conv (like every standard NPU CNN), not a
       pool/normalize on the raw input. Downsampling is done with strided convs +
       maxpools, all NPU-friendly ops.

    Input stays (124,129,1), so board/kws.c and the calibration data are
    unchanged. Verified to run correctly on RV1103 hardware.
    """
    def conv_bn(filters):
        # Conv (stride 1, VALID padding) -> BatchNorm -> ReLU. Two RV1106-NPU
        # constraints drove this:
        #  * NO strided / 'same'-padded convs: 'same' padding is placed
        #    differently by the NPU than by TF, which scrambles the (strided)
        #    output spatially -> the very first conv diverged on hardware
        #    (cosine ~0 vs float) and cascaded into a one-class collapse, while
        #    the float simulator looked fine. Downsample with MaxPool instead.
        #  * BatchNorm keeps activations ~unit-scale so INT8 requantization is
        #    well-conditioned; it folds into the conv on the NPU.
        return [
            tf.keras.layers.Conv2D(filters, 3, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=SPEC_SHAPE),
            *conv_bn(16),
            tf.keras.layers.MaxPooling2D(),
            *conv_bn(32),
            tf.keras.layers.MaxPooling2D(),
            *conv_bn(64),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels),  # logits
        ],
        name="kws_core",
    )


def export_calibration(train_spec_ds: tf.data.Dataset) -> None:
    """Save ~N_CALIB spectrograms as (124,129,1) .npy + dataset.txt of paths.

    No batch dim per file — matches the rknn-toolkit2 calibration convention
    (see README note); convert_rknn.py reads these in Phase 2.
    """
    calib_dir = OUT_CALIB_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    for old in calib_dir.glob("*.npy"):
        old.unlink()

    paths: list[str] = []
    count = 0
    for specs, _labels in train_spec_ds.unbatch().batch(1):
        sample = specs[0].numpy().astype(np.float32)  # (124, 129, 1)
        fp = calib_dir / f"calib_{count:03d}.npy"
        np.save(fp, sample)
        # repo-root-relative so convert_rknn.py (run from repo root) resolves them.
        paths.append(str(fp.relative_to(REPO_ROOT)))
        count += 1
        if count >= N_CALIB:
            break

    OUT_DATASET_TXT.write_text("\n".join(paths) + "\n")
    print(f"Wrote {count} calibration files to {calib_dir}/ and {OUT_DATASET_TXT}")


def sanity_check(label_names: list[str]) -> None:
    """Run one saved calib sample through the exported tflite for quick feedback."""
    interp = tf.lite.Interpreter(model_path=str(OUT_TFLITE))
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    sample = np.load(sorted(OUT_CALIB_DIR.glob("*.npy"))[0])
    x = sample[np.newaxis, ...].astype(in_detail["dtype"])  # (1, 124, 129, 1)
    interp.set_tensor(in_detail["index"], x)
    interp.invoke()
    logits = interp.get_tensor(out_detail["index"])

    assert logits.shape == (1, len(label_names)), \
        f"unexpected output shape {logits.shape}, expected (1, {len(label_names)})"
    pred = int(np.argmax(logits[0]))
    print(f"\nSanity check: tflite input {tuple(in_detail['shape'])}, "
          f"output {logits.shape}")
    print(f"  one calib sample -> predicted '{label_names[pred]}' (idx {pred})")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    data_dir = resolve_data_dir(arg)

    # Load waveforms; output_sequence_length pads/truncates to exactly 1 s.
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=str(data_dir),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=SEED,
        output_sequence_length=CLIP_SAMPLES,
        subset="both",
    )

    label_names = list(train_ds.class_names)  # alphabetical
    print(f"Classes ({len(label_names)}): {label_names}")

    # Drop the trailing mono channel axis -> (batch, 16000).
    squeeze = lambda audio, labels: (tf.squeeze(audio, axis=-1), labels)
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Split val 50/50 into validation + test.
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.shard(2, 0)
    val_ds = val_ds.shard(2, 1)

    train_spec_ds = to_spectrogram_ds(train_ds)
    val_spec_ds = to_spectrogram_ds(val_ds)
    test_spec_ds = to_spectrogram_ds(test_ds)

    train_spec_ds = train_spec_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_spec_ds = val_spec_ds.cache().prefetch(tf.data.AUTOTUNE)

    # No baked Normalization layer: the RV1106 NPU ignored the model's input
    # when one was present (see build_model docstring). INT8 input quantization
    # handles scaling instead.
    model = build_model(len(label_names))
    model.summary()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_spec_ds,
        validation_data=val_spec_ds,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=2, restore_best_weights=True)],
    )

    test_loss, test_acc = model.evaluate(test_spec_ds, verbose=0)
    print(f"\nTest accuracy: {test_acc:.3f}  (loss {test_loss:.3f})")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Write labels (coupling point #4).
    OUT_LABELS.write_text("\n".join(label_names) + "\n")
    print(f"Wrote {OUT_LABELS}")

    # Export float CNN (INT8 quantization happens later in convert_rknn.py).
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    OUT_TFLITE.write_bytes(tflite_model)
    print(f"Wrote {OUT_TFLITE} ({len(tflite_model)} bytes)")

    export_calibration(train_spec_ds)
    sanity_check(label_names)
    print("\nPhase 1 complete. Next: python pc/convert_rknn.py")


if __name__ == "__main__":
    main()
