# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Only `README.md` exists so far. The files below are specified in the README but not yet written; treat the README as the design spec and this file as the architectural summary when implementing them.

## What this is

On-device wake-word detection (keyword spotting) for the Luckfox Pico (RV1103 / RV1106). It adapts the TensorFlow [simple_audio](https://www.tensorflow.org/tutorials/audio/simple_audio) tutorial CNN to run on the board's NPU via RKNN.

## The core architectural constraint

**RKNN cannot convert `tf.signal.stft`.** The tutorial embeds the STFT (audio → spectrogram) inside the Keras model, which will not compile for the NPU. The model is therefore split across two processors:

| Stage | Runs on | Detail |
|-------|---------|--------|
| Mic capture + STFT spectrogram | CPU (C on the board) | 1 s of PCM → 124×129 magnitude spectrogram |
| CNN core (Resizing → Norm → Conv → Dense) | NPU (`.rknn`, INT8) | spectrogram → class logits |

**Consequence that drives most bugs:** the STFT math must be byte-for-byte consistent between training (Python) and inference (C). The C reimplementation uses periodic Hann window 255, hop 128, FFT 256. If you change spectrogram math on one side (e.g. switching to log-spectrograms), you must change the other identically.

## Pipeline / file responsibilities

```
PC (x86_64 Linux)                          Luckfox Pico (RV1103/RV1106)
train_export.py  → kws_core.tflite         arecord (S16_LE 16k mono) → stdin
                   + calib/ + labels.txt   kws (C): STFT → INT8 quant → NPU
convert_rknn.py  → kws.rknn                kws.rknn on NPU → logits → softmax
                                           → "WAKE: yes (p=0.97)"
```

| File | Runs on | Purpose |
|------|---------|---------|
| `train_export.py` | PC | Train CNN, export `kws_core.tflite` (no STFT), `calib/*.npy` + `dataset.txt`, `labels.txt` |
| `convert_rknn.py` | PC | TFLite → `kws.rknn`, INT8, target `rv1106` (also valid for rv1103); runs simulator check |
| `kws.c` | Board | Mic → STFT → NPU → wake-word logic |
| `Makefile` | PC (cross-compile) | Build `kws` for ARM uClibc target |

**Three coupling points that must stay in sync:**
1. STFT parameters (window/hop/FFT) between `train_export.py` and `kws.c`.
2. `labels.txt` ordering (**alphabetical**, not folder order) vs. the `LABELS[]` array and `WAKE_IDX` in `kws.c`.
3. `NUM_CLASSES` in `kws.c` vs. the number of model outputs / lines in `labels.txt`.

## Commands

### PC setup
```bash
pip install "tensorflow<2.16" numpy rknn-toolkit2   # rknn-toolkit2 is x86_64-only
```

### Train & export
```bash
python train_export.py                  # demo: mini_speech_commands (8 words), wake word = "yes"
python train_export.py path/to/dataset  # custom dataset
```
Custom dataset layout (1 s clips, 16 kHz mono WAV): `dataset/wake/`, `dataset/unknown/` (hard negatives), `dataset/noise/`.

### Convert to RKNN
```bash
python convert_rknn.py                  # → kws.rknn, INT8, target rv1106
```

### Cross-compile board app
Edit the two paths at the top of `Makefile` (`TOOLCHAIN`, `RKNN_RT`), update `LABELS[]`/`WAKE_IDX`/`NUM_CLASSES` in `kws.c` to match `labels.txt`, then:
```bash
make
```

### Deploy & run (board)
```bash
scp kws kws.rknn root@<board-ip>:/root/
scp ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so root@<board-ip>:/usr/lib/
# on board:
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | ./kws kws.rknn
```

### External SDKs (for cross-compiling)
- Toolchain: `git clone https://github.com/LuckfoxTECH/luckfox-pico.git ~/luckfox-pico`
- RKNN runtime (`librknnmrt.so` + `rknn_api.h`): `git clone https://github.com/airockchip/rknn-toolkit2.git ~/rknn-toolkit2`

## Platform-specific gotchas

- **RV1103/RV1106 support only the zero-copy RKNN API** (`rknn_create_mem` / `rknn_set_io_mem`). Do **not** use `rknn_inputs_set` from generic RK3588 examples.
- Detector tuning constants live in a marked block in `kws.c`: `THRESHOLD` (0.85), `CONSEC_NEEDED` (2), `COOLDOWN_HOPS` (8), `HOP_SAMPLES` (4000 = 250 ms inference cadence).
- INT8 quantization of linear-magnitude spectrograms has harsh dynamic range; the README recommends log-spectrograms (`tf.math.log(tf.abs(...)+1e-6)` / `logf(mag+1e-6f)`) for better accuracy — apply to both sides together.
