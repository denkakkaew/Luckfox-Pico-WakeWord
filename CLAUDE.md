# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phases 1–2 (PC side) are implemented and verified in the dev container:
- `pc/train_export.py` trains the CNN and exports `artifacts/{kws_core.tflite, calib/*.npy, dataset.txt, labels.txt}` (test acc ~0.82 on the demo set).
- `pc/convert_rknn.py` quantizes to `artifacts/kws.rknn` (INT8, rv1106) and runs a simulator check. **Caveat learned in Phase 4: this simulator runs ~float — its logits match the float TFLite to within ~0.14, so the "parity" check does NOT validate INT8 behaviour on the NPU.**

Phase 3 (`board/kws.c`, `board/Makefile`) is implemented: mic/stdin → STFT → INT8 zero-copy NPU inference → wake logic. Cross-compiles for the ARM uClibc target.

**Phase 4 (on-board) — pipeline runs, model is blocked by an rknn/RV1106 bug.**
The full chain runs on a real Luckfox RV1103 (Buildroot/uClibc) and the NPU is
confirmed healthy (stock mobilenet classifies correctly), but the wake-word
model **collapses to one class on hardware** while being correct in float, true
INT8 (tf.lite), and the rknn simulator. Root-caused to an rknn-toolkit2 1.5.2 /
RV1106 graph-lowering/execution bug that is **bit-width-independent** (INT8 and
INT16 fail identically; first conv diverges). Full write-up, every fix tried, and
the hardware-debug tooling: **[docs/RV1106-phase4-investigation.md](docs/RV1106-phase4-investigation.md)**
and **[tools/rv1106_diag/](tools/rv1106_diag/)**. The board needs the **rknpu2
v1.5.2 mini runtime** (`librknnmrt.so`) to match its NPU driver v0.9.2.

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
Default Luckfox login over the USB gadget is `pico@172.32.0.70` (password `luckfox`); adjust user/IP for your board. The `pico` user can't write to `/root` or `/usr/lib`, so copy to its home and move the lib with `sudo`.
```bash
scp board/kws artifacts/kws.rknn pico@172.32.0.70:~/
scp ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so pico@172.32.0.70:~/
ssh pico@172.32.0.70 'sudo mv ~/librknnmrt.so /usr/lib/'
# on board:
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | ./kws kws.rknn
```

### External SDKs (for cross-compiling)
- Toolchain: `git clone https://github.com/LuckfoxTECH/luckfox-pico.git ~/luckfox-pico`
- RKNN runtime (`librknnmrt.so` + `rknn_api.h`): `git clone https://github.com/airockchip/rknn-toolkit2.git ~/rknn-toolkit2`

## Platform-specific gotchas

- **RV1103/RV1106 support only the zero-copy RKNN API** (`rknn_create_mem` / `rknn_set_io_mem`). Do **not** use `rknn_inputs_set` from generic RK3588 examples.
- **`onnx` version pin**: rknn-toolkit2 pulls `onnx>=1.16.1`, but `onnx>=1.18` references `ml_dtypes.float4_e2m1fn` (absent in the `ml_dtypes 0.3.x` that `tensorflow-cpu<2.16` pins), which crashes `from rknn.api import RKNN`. `requirements.txt` therefore caps `onnx>=1.16.1,<1.18`.
- **RKNN calibration layout**: after `load_tflite` (NHWC), rknn's internal graph is **NCHW** and expects calibration `.npy` in NCHW. Phase-1 calib files are HWC `(124,129,1)`; `convert_rknn.py` transposes copies to `(1,124,129)` (rknn prepends batch → `(1,1,124,129)`) into `artifacts/_calib_nchw/`. Inference can stay NHWC via `data_format=["nhwc"]`.
- Detector tuning constants live in a marked block in `board/kws.c`: `THRESHOLD` (0.85), `CONSEC_NEEDED` (2), `COOLDOWN_HOPS` (8), `HOP_SAMPLES` (4000 = 250 ms inference cadence).
- `pc/train_export.py` already uses **log-spectrograms** (`tf.math.log(tf.abs(...)+1e-6)`); `board/kws.c` must match with `logf(mag+1e-6f)`. Both sides must always change together.
