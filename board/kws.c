/* kws.c — on-device wake-word detector for the Luckfox Pico (RV1103/RV1106).
 *
 * Pipeline (see README / CLAUDE.md):
 *
 *     arecord (S16_LE 16k mono) --stdin--> kws
 *         |  slide a 1 s window, every HOP_SAMPLES
 *         v
 *     STFT in C  ->  log-magnitude spectrogram (124 x 129)
 *         |
 *         v
 *     kws.rknn on the NPU (INT8)  ->  class logits
 *         |
 *         v
 *     softmax + consecutive/cooldown logic  ->  "WAKE: yes (p=0.97)"
 *
 * Why the STFT runs here in C: RKNN cannot convert tf.signal.stft, so the
 * tutorial model was split — the spectrogram is computed on the CPU and only
 * the CNN core runs on the NPU. The STFT math below MUST stay byte-for-byte
 * consistent with pc/train_export.py. The coupling contracts are:
 *
 *   1. STFT params: periodic Hann window 255, hop 128, FFT 256 -> (124, 129).
 *   2. Log spectrogram: training uses log(|stft| + 1e-6); we use
 *      logf(mag + 1e-6f) identically.
 *   3. Input scale: training decodes WAV to float in [-1, 1]; arecord gives
 *      int16 PCM, so we divide each sample by 32768.0f before the STFT (the
 *      Normalization layer baked into the model was fit on the [-1, 1] scale).
 *   4. Label order: LABELS[] / WAKE_IDX / NUM_CLASSES must match
 *      artifacts/labels.txt (alphabetical) and the model's output count.
 *
 * Build:  make -C board   (cross-compiles for the ARM uClibc target)
 * Run :   arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | ./kws kws.rknn
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "rknn_api.h"

/* =====================================================================
 * Coupling constants — must match pc/train_export.py. Do NOT change one
 * side without changing the other (see contracts 1-4 in the header).
 * ===================================================================== */
#define SAMPLE_RATE   16000
#define CLIP_SAMPLES  16000        /* 1 s window fed to the model           */
#define FRAME_LENGTH  255          /* periodic Hann window length           */
#define FRAME_STEP    128          /* STFT hop within the 1 s window        */
#define FFT_LENGTH    256          /* zero-padded FFT size                  */
#define SPEC_FRAMES   124          /* 1 + (16000 - 255) / 128               */
#define SPEC_BINS     129          /* 256 / 2 + 1                           */
#define SPEC_SIZE     (SPEC_FRAMES * SPEC_BINS)

/* Label set — coupling contract #4. Mirror artifacts/labels.txt (alphabetical)
 * and set WAKE_IDX to your wake word's position in this array. */
#define NUM_CLASSES   8
static const char *LABELS[NUM_CLASSES] =
    { "down", "go", "left", "no", "right", "stop", "up", "yes" };
#define WAKE_IDX      7            /* index of YOUR wake word in LABELS[]    */

/* =====================================================================
 * Detector tuning — adjust to trade false triggers vs. misses.
 * ===================================================================== */
#define THRESHOLD       0.85f      /* min wake-word probability to count a hit */
#define CONSEC_NEEDED   2          /* consecutive hits before firing           */
#define COOLDOWN_HOPS   8          /* refractory hops after a trigger (~2 s)   */
#define HOP_SAMPLES     4000       /* inference cadence: 4000 / 16000 = 250 ms */
/* ===================================================================== */

/* ---------------------------------------------------------------------
 * FFT — iterative radix-2 Cooley-Tukey, in place, size FFT_LENGTH (256).
 * re/im are length FFT_LENGTH; on return they hold the DFT.
 * ------------------------------------------------------------------- */
static void fft(float *re, float *im)
{
    const int n = FFT_LENGTH;

    /* bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }

    /* butterflies */
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        float wlen_re = (float)cos(ang);
        float wlen_im = (float)sin(ang);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k;
                int b = i + k + len / 2;
                float v_re = re[b] * w_re - im[b] * w_im;
                float v_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - v_re;
                im[b] = im[a] - v_im;
                re[a] = re[a] + v_re;
                im[a] = im[a] + v_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                float nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

/* periodic Hann window of length FRAME_LENGTH, computed once. */
static float g_hann[FRAME_LENGTH];

static void init_hann(void)
{
    /* periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n / FRAME_LENGTH), n=0..L-1.
     * (matches tf.signal.hann_window(255, periodic=True)) */
    for (int n = 0; n < FRAME_LENGTH; n++)
        g_hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n /
                                       (float)FRAME_LENGTH);
}

/* ---------------------------------------------------------------------
 * compute_spectrogram: window (CLIP_SAMPLES floats in [-1,1]) ->
 * log-magnitude spectrogram out[SPEC_FRAMES * SPEC_BINS], frame-major
 * (HWC with C=1), matching the model's (1,124,129,1) input.
 *
 * Mirrors tf.signal.stft(frame_length=255, frame_step=128, fft_length=256):
 * frame i covers samples [i*128, i*128+255), windowed, zero-padded to 256.
 * ------------------------------------------------------------------- */
static void compute_spectrogram(const float *window, float *out)
{
    for (int f = 0; f < SPEC_FRAMES; f++) {
        float re[FFT_LENGTH];
        float im[FFT_LENGTH];
        int start = f * FRAME_STEP;

        for (int n = 0; n < FRAME_LENGTH; n++) {
            re[n] = window[start + n] * g_hann[n];
            im[n] = 0.0f;
        }
        /* zero-pad the tail (FRAME_LENGTH .. FFT_LENGTH-1) */
        for (int n = FRAME_LENGTH; n < FFT_LENGTH; n++) {
            re[n] = 0.0f;
            im[n] = 0.0f;
        }

        fft(re, im);

        float *row = out + (size_t)f * SPEC_BINS;
        for (int k = 0; k < SPEC_BINS; k++) {
            float mag = sqrtf(re[k] * re[k] + im[k] * im[k]);
            row[k] = logf(mag + 1e-6f);   /* coupling contract #2 */
        }
    }
}

/* ---------------------------------------------------------------------
 * softmax over NUM_CLASSES logits (numerically stable).
 * ------------------------------------------------------------------- */
static void softmax(const float *logits, float *probs)
{
    float maxv = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++)
        if (logits[i] > maxv)
            maxv = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++)
        probs[i] /= sum;
}

/* ---------------------------------------------------------------------
 * RKNN context wrapper. RV1103/RV1106 support ONLY the zero-copy API
 * (rknn_create_mem / rknn_set_io_mem) — never rknn_inputs_set.
 *
 * We hand the runtime FLOAT32 input (NHWC) and request FLOAT32 output, so
 * the runtime performs the INT8 quantize/dequantize internally. This keeps
 * the C side identical to the float path the simulator validated in
 * pc/convert_rknn.py (float spectrogram in, float logits out) and avoids
 * re-deriving scale/zero-point here.
 * ------------------------------------------------------------------- */
typedef struct {
    rknn_context     ctx;
    rknn_tensor_mem *in_mem;
    rknn_tensor_mem *out_mem;
    rknn_tensor_attr in_attr;
    rknn_tensor_attr out_attr;
} kws_model;

static void *read_file(const char *path, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "error: cannot open model '%s'\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "error: model '%s' is empty\n", path);
        fclose(fp);
        return NULL;
    }
    void *buf = malloc((size_t)sz);
    if (buf && fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fprintf(stderr, "error: short read on model '%s'\n", path);
        free(buf);
        buf = NULL;
    }
    fclose(fp);
    if (buf)
        *out_size = (size_t)sz;
    return buf;
}

static int model_init(kws_model *m, const char *model_path)
{
    memset(m, 0, sizeof(*m));

    size_t model_size = 0;
    void *model_data = read_file(model_path, &model_size);
    if (!model_data)
        return -1;

    int ret = rknn_init(&m->ctx, model_data, (uint32_t)model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        fprintf(stderr, "error: rknn_init failed (%d)\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input != 1 || io_num.n_output != 1) {
        fprintf(stderr, "error: expected 1 input / 1 output, got %u / %u\n",
                io_num.n_input, io_num.n_output);
        return -1;
    }

    m->in_attr.index = 0;
    ret = rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &m->in_attr,
                     sizeof(m->in_attr));
    if (ret < 0) {
        fprintf(stderr, "error: query input attr failed (%d)\n", ret);
        return -1;
    }

    m->out_attr.index = 0;
    ret = rknn_query(m->ctx, RKNN_QUERY_OUTPUT_ATTR, &m->out_attr,
                     sizeof(m->out_attr));
    if (ret < 0) {
        fprintf(stderr, "error: query output attr failed (%d)\n", ret);
        return -1;
    }

    /* Shape sanity — catches a LABELS[]/NUM_CLASSES vs. model mismatch
     * ("model shape mismatch" in the README troubleshooting). */
    if (m->in_attr.n_elems != SPEC_SIZE) {
        fprintf(stderr, "error: model input has %u elems, expected %d "
                "(%dx%d spectrogram)\n",
                m->in_attr.n_elems, SPEC_SIZE, SPEC_FRAMES, SPEC_BINS);
        return -1;
    }
    if (m->out_attr.n_elems != NUM_CLASSES) {
        fprintf(stderr, "error: model has %u outputs, but NUM_CLASSES=%d\n",
                m->out_attr.n_elems, NUM_CLASSES);
        return -1;
    }

    /* Feed FLOAT32 NHWC, receive FLOAT32 — runtime handles INT8 conversion.
     * pass_through must be 0 so rknn_set_io_mem honours type/fmt and converts
     * (pass_through=1 would feed the buffer straight to the node, un-quantized). */
    m->in_attr.type  = RKNN_TENSOR_FLOAT32;
    m->in_attr.fmt   = RKNN_TENSOR_NHWC;
    m->in_attr.pass_through = 0;
    m->out_attr.type = RKNN_TENSOR_FLOAT32;
    m->out_attr.pass_through = 0;

    m->in_mem = rknn_create_mem(m->ctx, m->in_attr.n_elems * sizeof(float));
    m->out_mem = rknn_create_mem(m->ctx, m->out_attr.n_elems * sizeof(float));
    if (!m->in_mem || !m->out_mem) {
        fprintf(stderr, "error: rknn_create_mem failed\n");
        return -1;
    }

    if (rknn_set_io_mem(m->ctx, m->in_mem, &m->in_attr) < 0 ||
        rknn_set_io_mem(m->ctx, m->out_mem, &m->out_attr) < 0) {
        fprintf(stderr, "error: rknn_set_io_mem failed\n");
        return -1;
    }

    return 0;
}

static void model_release(kws_model *m)
{
    if (m->in_mem)
        rknn_destroy_mem(m->ctx, m->in_mem);
    if (m->out_mem)
        rknn_destroy_mem(m->ctx, m->out_mem);
    if (m->ctx)
        rknn_destroy(m->ctx);
}

/* Run one spectrogram through the NPU; fills logits[NUM_CLASSES]. */
static int model_infer(kws_model *m, const float *spec, float *logits)
{
    memcpy(m->in_mem->virt_addr, spec, SPEC_SIZE * sizeof(float));

    int ret = rknn_run(m->ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "error: rknn_run failed (%d)\n", ret);
        return -1;
    }

    memcpy(logits, m->out_mem->virt_addr, NUM_CLASSES * sizeof(float));
    return 0;
}

/* ---------------------------------------------------------------------
 * Read exactly `count` int16 samples from stdin (fd 0). Returns the number
 * actually read; < count means EOF/short stream.
 * ------------------------------------------------------------------- */
static size_t read_samples(int16_t *buf, size_t count)
{
    uint8_t *p = (uint8_t *)buf;
    size_t want = count * sizeof(int16_t);
    size_t got = 0;
    while (got < want) {
        ssize_t n = read(STDIN_FILENO, p + got, want - got);
        if (n <= 0)
            break;                /* EOF or error */
        got += (size_t)n;
    }
    return got / sizeof(int16_t);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.rknn>\n", argv[0]);
        fprintf(stderr, "  feed raw PCM on stdin, e.g.:\n");
        fprintf(stderr, "  arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw | "
                "%s kws.rknn\n", argv[0]);
        return 1;
    }

    init_hann();

    kws_model model;
    if (model_init(&model, argv[1]) != 0)
        return 1;

    fprintf(stderr, "kws ready: listening for '%s' "
            "(threshold %.2f, %d consecutive)\n",
            LABELS[WAKE_IDX], (double)THRESHOLD, CONSEC_NEEDED);

    static float   window[CLIP_SAMPLES];   /* sliding 1 s of scaled audio */
    static float   spec[SPEC_SIZE];
    int16_t        pcm[HOP_SAMPLES];
    float          logits[NUM_CLASSES];
    float          probs[NUM_CLASSES];

    /* Prime the window with a full second before the first inference. */
    {
        int16_t prime[CLIP_SAMPLES];
        if (read_samples(prime, CLIP_SAMPLES) != CLIP_SAMPLES) {
            fprintf(stderr, "error: not enough audio to fill the first window\n");
            model_release(&model);
            return 1;
        }
        for (int i = 0; i < CLIP_SAMPLES; i++)
            window[i] = (float)prime[i] / 32768.0f;   /* coupling contract #3 */
    }

    int consec = 0;
    int cooldown = 0;

    for (;;) {
        compute_spectrogram(window, spec);
        if (model_infer(&model, spec, logits) != 0)
            break;
        softmax(logits, probs);

        float wake_p = probs[WAKE_IDX];
        if (wake_p >= THRESHOLD)
            consec++;
        else
            consec = 0;

        if (cooldown > 0) {
            cooldown--;
        } else if (consec >= CONSEC_NEEDED) {
            printf("WAKE: %s (p=%.2f)\n", LABELS[WAKE_IDX], (double)wake_p);
            fflush(stdout);

            /* -----------------------------------------------------------
             * TODO: hook your action here — toggle a GPIO, publish MQTT,
             * fire an HTTP request, etc. Runs once per detection.
             * --------------------------------------------------------- */

            consec = 0;
            cooldown = COOLDOWN_HOPS;
        }

        /* Slide the window left by HOP_SAMPLES and append fresh audio. */
        memmove(window, window + HOP_SAMPLES,
                (CLIP_SAMPLES - HOP_SAMPLES) * sizeof(float));
        size_t got = read_samples(pcm, HOP_SAMPLES);
        for (size_t i = 0; i < got; i++)
            window[CLIP_SAMPLES - HOP_SAMPLES + i] = (float)pcm[i] / 32768.0f;
        if (got < HOP_SAMPLES)
            break;                /* end of stream */
    }

    fprintf(stderr, "kws: input stream ended\n");
    model_release(&model);
    return 0;
}
