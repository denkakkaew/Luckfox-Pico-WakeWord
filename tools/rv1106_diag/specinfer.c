/* specinfer — feed a known-good float32 spectrogram straight into the NPU,
 * bypassing the C STFT, to isolate runtime/model collapse from STFT issues.
 *
 *   specinfer <model.rknn> <spec.f32>
 *
 * <spec.f32> is SPEC_FRAMES*SPEC_BINS little-endian float32 (one channel,
 * frame-major) — e.g. a training calib spectrogram. It is replicated into all
 * input channels, quantized to the model's native input dtype (int8/uint8,
 * honouring w_stride), run once, and the per-class probabilities + argmax are
 * printed. Cross-compile against the RV1106 mini runtime (see Makefile).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "rknn_api.h"

#define SPEC_FRAMES 124
#define SPEC_BINS   128
#define SPEC_SIZE   (SPEC_FRAMES * SPEC_BINS)

static void *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(s); if (b && fread(b, 1, s, f) != (size_t)s) { free(b); b = NULL; }
    fclose(f); if (b) *n = (size_t)s; return b;
}

static int quant_q(float v, float sc, int zp, int uns) {
    int q = (int)lrintf(v / sc) + zp, lo = uns ? 0 : -128, hi = uns ? 255 : 127;
    return q < lo ? lo : (q > hi ? hi : q);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.rknn> <spec.f32>\n", argv[0]); return 2; }

    size_t msz = 0; void *md = read_file(argv[1], &msz);
    if (!md) { fprintf(stderr, "cannot read model\n"); return 1; }
    rknn_context ctx;
    if (rknn_init(&ctx, md, msz, 0, NULL) < 0) { fprintf(stderr, "rknn_init failed\n"); return 1; }
    free(md);

    rknn_tensor_attr in = {0}, out = {0};
    in.index = 0;  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,  &in,  sizeof(in));
    out.index = 0; rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out, sizeof(out));
    int H = in.dims[1], W = in.dims[2], C = in.dims[3];
    int ws = in.w_stride ? in.w_stride : W;
    int uin = (in.type == RKNN_TENSOR_UINT8);
    fprintf(stderr, "in dims=[%u,%u,%u,%u] type=%d ws=%d scale=%g zp=%d | out n=%u type=%d scale=%g zp=%d\n",
            in.dims[0], in.dims[1], in.dims[2], in.dims[3], in.type, ws, in.scale, in.zp,
            out.n_elems, out.type, out.scale, out.zp);

    size_t fsz = 0; float *spec = read_file(argv[2], &fsz);
    if (!spec || fsz < SPEC_SIZE * sizeof(float)) { fprintf(stderr, "bad spec file (%zu bytes)\n", fsz); return 1; }

    rknn_tensor_mem *im = rknn_create_mem(ctx, in.size_with_stride);
    rknn_tensor_mem *om = rknn_create_mem(ctx, out.size_with_stride);
    rknn_set_io_mem(ctx, im, &in);
    rknn_set_io_mem(ctx, om, &out);

    uint8_t *ib = (uint8_t *)im->virt_addr;
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            uint8_t q = (uint8_t)quant_q(spec[h * W + w], in.scale, in.zp, uin);
            for (int c = 0; c < C; c++) ib[((size_t)h * ws + w) * C + c] = q;
        }

    if (rknn_run(ctx, NULL) < 0) { fprintf(stderr, "rknn_run failed\n"); return 1; }

    int n = out.n_elems, uout = (out.type == RKNN_TENSOR_UINT8), best = 0;
    float logits[64], mx;
    const uint8_t *ob = (const uint8_t *)om->virt_addr;
    for (int i = 0; i < n; i++) {
        int qv = uout ? (int)ob[i] : (int)(int8_t)ob[i];
        logits[i] = (qv - out.zp) * out.scale;
        if (i == 0 || logits[i] > mx) { mx = logits[i]; best = i; }
    }
    printf("argmax=%d logits:", best);
    for (int i = 0; i < n; i++) printf(" %.3f", logits[i]);
    printf("\n");
    rknn_destroy(ctx);
    return 0;
}
