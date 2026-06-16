/* specinfer2 — like specinfer, but uses the STANDARD rknn API
 * (rknn_inputs_set / rknn_outputs_get) instead of the zero-copy API. The
 * runtime then handles input quantization + native layout itself, which is what
 * the rknn-toolkit "connected" path does (100% on calib). Use this to confirm
 * the standard API works on the RV1106 mini runtime.
 *
 *   specinfer2 <model.rknn> <spec.f32>   (spec.f32 = 124*128 float, frame-major)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

#define SPEC_SIZE (124 * 128)

static void *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(s); if (b && fread(b, 1, s, f) != (size_t)s) { free(b); b = NULL; }
    fclose(f); if (b) *n = (size_t)s; return b;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.rknn> <spec.f32>\n", argv[0]); return 2; }
    size_t msz = 0; void *md = read_file(argv[1], &msz);
    if (!md) { fprintf(stderr, "cannot read model\n"); return 1; }
    rknn_context ctx;
    if (rknn_init(&ctx, md, msz, 0, NULL) < 0) { fprintf(stderr, "rknn_init failed\n"); return 1; }
    free(md);

    rknn_tensor_attr ia; memset(&ia, 0, sizeof(ia)); ia.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof(ia));
    float sc = ia.scale; int zp = ia.zp;
    fprintf(stderr, "in type=%d scale=%g zp=%d\n", ia.type, sc, zp);

    size_t fsz = 0; float *spec = read_file(argv[2], &fsz);
    if (!spec || fsz < SPEC_SIZE * sizeof(float)) { fprintf(stderr, "bad spec file\n"); return 1; }

    /* Quantize float -> int8 (q = round(v/scale)+zp), then let rknn_inputs_set
     * handle the native layout conversion. This mirrors the toolkit path. */
    int8_t *q = malloc(SPEC_SIZE);
    for (int i = 0; i < SPEC_SIZE; i++) {
        int v = (int)(spec[i] / sc + (spec[i] >= 0 ? 0.5f : -0.5f)) + zp;
        if (v < -128) v = -128; if (v > 127) v = 127;
        q[i] = (int8_t)v;
    }
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_INT8;      /* runtime requested int8 */
    in.size = SPEC_SIZE;
    in.fmt = RKNN_TENSOR_NHWC;
    in.buf = q;
    in.pass_through = 0;
    if (rknn_inputs_set(ctx, 1, &in) < 0) { fprintf(stderr, "rknn_inputs_set(int8) failed\n"); return 1; }

    if (rknn_run(ctx, NULL) < 0) { fprintf(stderr, "rknn_run failed\n"); return 1; }

    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.index = 0;
    out.want_float = 1;              /* dequantized float logits */
    if (rknn_outputs_get(ctx, 1, &out, NULL) < 0) { fprintf(stderr, "rknn_outputs_get failed\n"); return 1; }

    float *logits = (float *)out.buf;
    int n = (int)(out.size / sizeof(float)), best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    printf("argmax=%d logits:", best);
    for (int i = 0; i < n; i++) printf(" %.3f", logits[i]);
    printf("\n");

    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
    return 0;
}
