# Luckfox-Pico-WakeWord

On-device keyword spotting based on the TensorFlow
[simple_audio](https://www.tensorflow.org/tutorials/audio/simple_audio) tutorial,
deployed to the Luckfox Pico NPU via RKNN.

The model detects a single **wake word** from a short list of keywords and prints
a trigger you can hook up to GPIO, MQTT, an HTTP call, or anything else.

---

## The one thing you must understand first

**RKNN cannot convert `tf.signal.stft`.** The tutorial wraps the STFT
(audio → spectrogram) *inside* the Keras model. That part will not compile for
the NPU.

So the model is split in two:

| Stage | Runs on | What it does |
|-------|---------|--------------|
| Mic capture + **STFT spectrogram** | **CPU** (C code on the board) | 1 s of PCM → 124×129 magnitude spectrogram |
| **CNN core** (Resizing → Norm → Conv → Dense) | **NPU** (`.rknn`, INT8) | spectrogram → class logits |

The C program reimplements the exact STFT math the tutorial uses
(periodic Hann window 255, hop 128, FFT 256) followed by a **log-magnitude**
(`logf(mag + 1e-6f)`), so training and inference stay consistent.

---

## Pipeline overview

```
   PC (x86_64 Linux)                         Luckfox Pico (RV1103/RV1106)
   ────────────────────                      ────────────────────────────
   pc/train_export.py                        arecord  (S16_LE 16k mono)
      │  trains tutorial CNN                       │  raw PCM on stdin
      │  exports CNN core (no STFT)                ▼
      ▼                                        kws  (C program)
   artifacts/kws_core.tflite                    │  STFT in C  → log-spectrogram
      │   + calib/ + labels.txt                  │  INT8 quantize
   pc/convert_rknn.py                            ▼
      │  INT8 quantize for rv1106            kws.rknn on NPU → logits
      ▼                                          │  softmax + threshold
   artifacts/kws.rknn  ─────  scp  ──────►       ▼
                                             "WAKE: yes (p=0.97)"
```

---

## Files in this project

| File | Where it runs | Purpose |
|------|---------------|---------|
| `pc/train_export.py` | PC | Train the KWS CNN, export `kws_core.tflite` + calibration data + `labels.txt` |
| `pc/convert_rknn.py` | PC | Convert TFLite → `kws.rknn` (INT8, target `rv1106`) |
| `board/kws.c` | Board | Mic capture → STFT → NPU inference → wake-word logic |
| `board/Makefile` | PC (cross-compile) | Build `kws` for the board's ARM uClibc target |
| `README.md` | — | This guide |

All generated outputs land in `artifacts/` (gitignored); dataset downloads land in `data/`.

---

## Prerequisites

### Option A — Dev container (recommended)

The repo ships a [`.devcontainer/`](.devcontainer/) that builds a clean,
reproducible **x86_64 Linux** environment (Python 3.10, `tensorflow<2.16`,
`rknn-toolkit2`, the cross-compile toolchain libs) so you don't have to pollute
your WSL/host setup.

1. Open the folder in **VS Code** → *Reopen in Container* (needs the Dev
   Containers extension + Docker).
2. On first start it verifies the Python stack. To cross-compile (Step 3), fetch
   the external SDKs once — they land in a persistent volume, not your repo:

   ```bash
   bash .devcontainer/setup-sdks.sh
   ```

The container exports `TOOLCHAIN` and `RKNN_RT` pointing at those SDKs, so the
`Makefile` picks them up automatically — **no Makefile edits needed**.

**Claude Code** is preinstalled in the container — just run `claude` and log in
once (your login persists across rebuilds in a dedicated volume).

> The container is pinned to `linux/amd64` because `rknn-toolkit2` and the
> prebuilt toolchain are x86_64-only. On Apple Silicon it runs under emulation.

### Option B — Local install (x86_64 Linux)

```bash
pip install "tensorflow<2.16" numpy rknn-toolkit2
```

- `rknn-toolkit2` is x86_64-only; it will not install on the board or on Apple Silicon.
- If you only have macOS/Windows, use the dev container or a Linux VM.

Then grab the two SDKs for cross-compiling (Step 3):

```bash
# 1) Luckfox SDK — provides the cross toolchain
git clone https://github.com/LuckfoxTECH/luckfox-pico.git ~/luckfox-pico

# 2) RKNN runtime — provides librknnmrt.so + rknn_api.h
git clone https://github.com/airockchip/rknn-toolkit2.git ~/rknn-toolkit2
```

### Hardware

- A **Luckfox Pico** board. Prefer an **RV1106** variant (256 MB RAM) over the
  64 MB RV1103 boards — more headroom for audio buffers.
- A **microphone**. The base board has none. Use a USB audio mic or an I2S MEMS
  mic such as the **INMP441**. Confirm with `arecord -l` on the board.

---

## Step 1 — Train & export (PC)

```bash
python pc/train_export.py                  # demo: mini_speech_commands (8 words)
# or
python pc/train_export.py path/to/dataset  # your own wake-word dataset
```

Run from the repo root. First run with the default dataset to validate the
whole pipeline end-to-end, using **"yes"** as a stand-in wake word.

**Outputs (in `artifacts/`):**

- `artifacts/kws_core.tflite` — float CNN, input spectrogram `(1, 124, 129, 1)`
- `artifacts/calib/*.npy` + `artifacts/dataset.txt` — calibration set for INT8 quantization
- `artifacts/labels.txt` — class order (**alphabetical** — you'll need this in Step 3)

### Your own dataset layout

1-second clips, **16 kHz, mono, WAV**:

```
dataset/
  wake/      *.wav   200+ recordings of your wake word (varied speakers, distances, rooms)
  unknown/   *.wav   other words & random speech  (hard negatives)
  noise/     *.wav   background noise (clinic ambience, fans, A/C, chatter)
```

More variety in `wake/` and good hard negatives in `unknown/` matter far more
than model size for real-world reliability.

---

## Step 2 — Convert to RKNN (PC)

```bash
python pc/convert_rknn.py
```

Produces **`artifacts/kws.rknn`** (INT8, target `rv1106` — also valid for
`rv1103`) and runs a quick simulator check so you can see class probabilities,
plus an INT8-vs-float parity check, before touching the board.

> The calibration `.npy` are stored HWC `(124, 129, 1)`; rknn-toolkit2's graph
> is channel-first, so `convert_rknn.py` transposes copies to NCHW
> (`artifacts/_calib_nchw/`) automatically before building.

---

## Step 3 — Cross-compile the board app (PC)

**In the dev container this is already wired up** — `TOOLCHAIN` and `RKNN_RT`
are exported as env vars, so you can skip straight to editing the labels below.

For a local install, edit the two paths at the top of `board/Makefile` if your
clones live elsewhere (both use `?=`, so an env var or `make TOOLCHAIN=...` wins):

```make
TOOLCHAIN ?= $(HOME)/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
RKNN_RT   ?= $(HOME)/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
```

Then **match the labels to your training run**. Open `board/kws.c` and update
this block from your `artifacts/labels.txt` (remember: alphabetical order):

```c
#define NUM_CLASSES   8
static const char *LABELS[NUM_CLASSES] =
    { "down", "go", "left", "no", "right", "stop", "up", "yes" };
#define WAKE_IDX      7      /* index of YOUR wake word in LABELS[] */
```

Build:

```bash
make -C board
```

You'll also need `librknnmrt.so` on the board (see Step 4).

---

## Step 4 — Deploy & run (board)

Copy the binary, the model, and the runtime library. This assumes the default
Luckfox login over the USB gadget (`pico@172.32.0.70`, password `luckfox` — adjust
user/IP for your board). The `pico` user can't write to `/root` or `/usr/lib`
directly, so copy to its home dir and move the library with `sudo`:

```bash
scp board/kws artifacts/kws.rknn pico@172.32.0.70:~/
scp ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so \
    pico@172.32.0.70:~/
# In the dev container the runtime lives under $RKNN_RT:
# scp "$RKNN_RT/armhf-uclibc/librknnmrt.so" pico@172.32.0.70:~/

# then on the board, install the runtime lib:
ssh pico@172.32.0.70 'sudo mv ~/librknnmrt.so /usr/lib/'
```

On the board:

```bash
arecord -l    # confirm a capture device exists (else fix your mic first)

arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | ./kws kws.rknn
```

On detection it prints:

```
WAKE: yes (p=0.97)
```

Hook your action into the `TODO` block near the bottom of `kws.c`.

---

## Tuning the detector

All in the marked block in `kws.c`:

| Constant | Effect |
|----------|--------|
| `THRESHOLD` (0.85) | Raise to cut false triggers; lower if it misses the word |
| `CONSEC_NEEDED` (2) | Hits in a row before firing — higher = stricter |
| `COOLDOWN_HOPS` (8) | Refractory period after a trigger (~2 s), stops repeats |
| `HOP_SAMPLES` (4000) | Inference cadence (250 ms). Smaller = snappier, more CPU |

---

## Troubleshooting

**`rknn_inputs_set` undefined / runtime errors.**
RV1103/RV1106 support **only the zero-copy API**. This code already uses
`rknn_create_mem` / `rknn_set_io_mem`. Don't copy generic RK3588 examples that
use `rknn_inputs_set`.

**Detects the wrong word.**
`LABELS[]` / `WAKE_IDX` don't match `labels.txt`. The order is alphabetical, not
the order folders appear on disk.

**`arecord -l` lists no devices.**
No mic detected. Wire up a USB or I2S mic; the bare board has none.

**Poor real-world accuracy.**
This build already uses **log-spectrograms** for better INT8 dynamic range:

- training (`pc/train_export.py`): `spectrogram = tf.math.log(tf.abs(...) + 1e-6)`
- C (`board/kws.c`, in `compute_spectrogram`): `out[k] = logf(mag + 1e-6f);`

Both sides **must** stay identical — if you ever change one, change the other.
If accuracy is still poor, invest in more/better training data (see below).

**Too many false triggers in the clinic.**
Add more `noise/` and `unknown/` data, augment training with noise mixing and
small time shifts, then raise `THRESHOLD` / `CONSEC_NEEDED`.

**`model shape mismatch` on startup.**
`NUM_CLASSES` in `kws.c` doesn't match the trained model's output. Update it to
the number of lines in `labels.txt` and rebuild.

---

## Realistic expectations

The tutorial model is a **demo-grade** classifier. It's perfect for proving the
pipeline, but a production wake word ("Hey Clinic") needs a few hundred
recordings of your actual phrase plus solid hard negatives. Treat the
mini_speech_commands run as your smoke test, then invest in a real dataset.
