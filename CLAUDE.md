# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phase 1 (training) is implemented: `pc/train_export.py` trains the CNN and exports the artifacts. The remaining source files (`pc/convert_rknn.py`, `board/kws.c`, `board/Makefile`) are specified in the README but not yet written; treat the README as the design spec and this file as the architectural summary when implementing them.

## Repository layout

```
pc/         PC-side Python (train_export.py; convert_rknn.py later)
board/      board-side C (kws.c + Makefile later)
artifacts/  generated outputs (kws_core.tflite, kws.rknn, labels.txt, dataset.txt, calib/) — gitignored
data/       demo dataset downloads — gitignored
.devcontainer/
```

`pc/train_export.py` resolves its output paths relative to the repo root (via `__file__`), so it can be launched from any directory. Run PC scripts from the repo root (e.g. `python pc/train_export.py`) so `dataset.txt`'s repo-root-relative calib paths resolve.

## Dev environment

All PC-side work is meant to run inside the dev container (`.devcontainer/`), which gives a clean, reproducible x86_64 Linux env separate from the host/WSL:

- `Dockerfile` — `mcr.microsoft.com/devcontainers/python:3.10-bookworm`, pinned to `linux/amd64` (rknn-toolkit2 and the cross toolchain are x86_64-only). Installs `requirements.txt` (`tensorflow-cpu<2.16`, `numpy`, `rknn-toolkit2`) — CPU-only, no CUDA. `CUDA_VISIBLE_DEVICES=-1` is set so TF never probes for a GPU.
- `devcontainer.json` — mounts a persistent `luckfox-sdks` volume at `/opt/sdks`, and exports `TOOLCHAIN` + `RKNN_RT` env vars pointing into it. **The `Makefile` must read these via `?=`** so it works in-container without edits.
- `setup-sdks.sh` — opt-in, shallow-clones `luckfox-pico` + `rknn-toolkit2` into `/opt/sdks` (kept out of the image to stay small/fast). Run once per volume before cross-compiling.
- `post-create.sh` — light verification only; does not clone the heavy SDKs.

Claude Code (`@anthropic-ai/claude-code`) is installed in the image (via Node 20) and runs inside the container; its `~/.claude` config/login is persisted in a `luckfox-claude-config` volume, so you authenticate once.

When changing PC-side deps, update `.devcontainer/requirements.txt` and the README "Prerequisites" together. The external SDKs (`luckfox-pico/`, `rknn-toolkit2/`) and build artifacts are gitignored.

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
PC (x86_64 Linux)                             Luckfox Pico (RV1103/RV1106)
pc/train_export.py → artifacts/kws_core.tflite  arecord (S16_LE 16k mono) → stdin
                     + calib/ + labels.txt      kws (C): STFT → INT8 quant → NPU
pc/convert_rknn.py → artifacts/kws.rknn         kws.rknn on NPU → logits → softmax
                                                → "WAKE: yes (p=0.97)"
```

| File | Runs on | Purpose |
|------|---------|---------|
| `pc/train_export.py` | PC | Train CNN, export `artifacts/kws_core.tflite` (no STFT), `artifacts/calib/*.npy` + `dataset.txt`, `labels.txt` |
| `pc/convert_rknn.py` | PC | TFLite → `artifacts/kws.rknn`, INT8, target `rv1106` (also valid for rv1103); runs simulator check |
| `board/kws.c` | Board | Mic → STFT → NPU → wake-word logic |
| `board/Makefile` | PC (cross-compile) | Build `kws` for ARM uClibc target |

**Coupling points that must stay in sync (Python ↔ C):**
1. STFT parameters (window/hop/FFT) between `pc/train_export.py` and `board/kws.c`.
2. **Log spectrogram**: training uses `log(|stft| + 1e-6)`, so `board/kws.c` must use `logf(mag + 1e-6f)`.
3. **Input scale**: WAV decodes to [-1, 1] in training; `board/kws.c` must divide int16 PCM by 32768 before the STFT (the baked-in Normalization layer depends on it).
4. `labels.txt` ordering (**alphabetical**, not folder order) vs. the `LABELS[]` array and `WAKE_IDX` in `board/kws.c`.
5. `NUM_CLASSES` in `board/kws.c` vs. the number of model outputs / lines in `labels.txt`.

## Commands

### PC setup
Preferred: open in the dev container (`.devcontainer/`) — deps are preinstalled. Then, before cross-compiling, fetch the SDKs once:
```bash
bash .devcontainer/setup-sdks.sh                    # clones SDKs into /opt/sdks volume
```
Local alternative:
```bash
pip install "tensorflow<2.16" numpy rknn-toolkit2   # rknn-toolkit2 is x86_64-only
```

### Train & export
```bash
python pc/train_export.py                  # demo: mini_speech_commands (8 words), wake word = "yes"
python pc/train_export.py path/to/dataset  # custom dataset
```
Custom dataset layout (1 s clips, 16 kHz mono WAV): `dataset/wake/`, `dataset/unknown/` (hard negatives), `dataset/noise/`. Outputs land in `artifacts/`.

### Convert to RKNN
```bash
python pc/convert_rknn.py               # → artifacts/kws.rknn, INT8, target rv1106
```

### Cross-compile board app
In the dev container `TOOLCHAIN`/`RKNN_RT` are already exported (no Makefile edit needed); locally, edit the two paths at the top of `board/Makefile`. Either way, update `LABELS[]`/`WAKE_IDX`/`NUM_CLASSES` in `board/kws.c` to match `artifacts/labels.txt`, then:
```bash
make -C board
```

### Deploy & run (board)
```bash
scp board/kws artifacts/kws.rknn root@<board-ip>:/root/
scp ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so root@<board-ip>:/usr/lib/
# on board:
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | ./kws kws.rknn
```

### External SDKs (for cross-compiling)
- Toolchain: `git clone https://github.com/LuckfoxTECH/luckfox-pico.git ~/luckfox-pico`
- RKNN runtime (`librknnmrt.so` + `rknn_api.h`): `git clone https://github.com/airockchip/rknn-toolkit2.git ~/rknn-toolkit2`

## Platform-specific gotchas

- **RV1103/RV1106 support only the zero-copy RKNN API** (`rknn_create_mem` / `rknn_set_io_mem`). Do **not** use `rknn_inputs_set` from generic RK3588 examples.
- Detector tuning constants live in a marked block in `board/kws.c`: `THRESHOLD` (0.85), `CONSEC_NEEDED` (2), `COOLDOWN_HOPS` (8), `HOP_SAMPLES` (4000 = 250 ms inference cadence).
- `pc/train_export.py` already uses **log-spectrograms** (`tf.math.log(tf.abs(...)+1e-6)`); `board/kws.c` must match with `logf(mag+1e-6f)`. Both sides must always change together.
