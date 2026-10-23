/* FP32 port of the frozen channel encoder; geometry/pooling are downstream.
 * A causal rolling cache stores the four previous normalized inputs per block.
 * No retraining, pruning, reduced precision, or nonlinear approximation. */
#include "common.h"
#include "weight_offsets.h"
#include <math.h>

static void linear(const float *x, float *y, unsigned inputs, unsigned outputs,
                   unsigned weight_offset, unsigned bias_offset) {
    const float *w = learned_weights + weight_offset;
    const float *b = learned_weights + bias_offset;
    for (unsigned o = 0; o < outputs; ++o) {
        float value = 0.f;
        for (unsigned i = 0; i < inputs; ++i) value += x[i] * w[o * inputs + i];
        y[o] = value + b[o];
    }
}

static float gelu(float x) { return .5f * x * (1.f + erff(x * .7071067811865475244f)); }

static void layer_norm(const float *x, float *y, unsigned w, unsigned b) {
    float mean = 0.f, variance = 0.f;
    for (unsigned d = 0; d < DIM; ++d) mean += x[d];
    mean /= (float)DIM;
    for (unsigned d = 0; d < DIM; ++d) { float delta = x[d] - mean; variance += delta * delta; }
    float inverse = 1.f / sqrtf(variance / (float)DIM + 1.e-5f);
    for (unsigned d = 0; d < DIM; ++d)
        y[d] = (x[d] - mean) * inverse * learned_weights[w + d] + learned_weights[b + d];
}

void learned_step(const float window[PATCH], float history[4][4][DIM], float out[DIM]) {
    float hidden[1024], x[DIM], normalized[DIM], depthwise[DIM], update[DIM];
    linear(window, hidden, PATCH, 32, PATCH_W, PATCH_B);
    for (unsigned d = 0; d < 32; ++d) hidden[d] = gelu(hidden[d]);
    linear(hidden, x, 32, DIM, PROJECT_W, PROJECT_B);
    for (unsigned l = 0; l < 4; ++l) {
        const unsigned *o = BLOCK_OFFSETS[l];
        layer_norm(x, normalized, o[0], o[1]);
        for (unsigned d = 0; d < DIM; ++d) {
            float v = 0.f;
            for (unsigned t = 0; t < 4; ++t) v += history[l][t][d] * learned_weights[o[2] + d * 5 + t];
            v += normalized[d] * learned_weights[o[2] + d * 5 + 4];
            depthwise[d] = v + learned_weights[o[3] + d];
        }
        for (unsigned t = 0; t < 3; ++t)
            for (unsigned d = 0; d < DIM; ++d) history[l][t][d] = history[l][t + 1][d];
        for (unsigned d = 0; d < DIM; ++d) history[l][3][d] = normalized[d];
        linear(depthwise, hidden, DIM, 1024, o[4], o[5]);
        for (unsigned d = 0; d < 1024; ++d) hidden[d] = gelu(hidden[d]);
        linear(hidden, update, 1024, DIM, o[6], o[7]);
        for (unsigned d = 0; d < DIM; ++d) x[d] += update[d];
    }
    layer_norm(x, out, FINAL_W, FINAL_B);
}
