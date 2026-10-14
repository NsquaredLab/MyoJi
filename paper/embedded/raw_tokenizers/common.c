#include "common.h"
#include <math.h>
#include <float.h>

/* Deterministic in-place selection; same lower median as torch.sort()[499]. */
static float select_lower(float *x, int n) {
    int lo = 0, hi = n - 1, target = (n - 1) / 2;
    while (lo < hi) {
        float pivot = x[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        do {
            while (x[i] < pivot) ++i;
            while (x[j] > pivot) --j;
            if (i <= j) { float tmp = x[i]; x[i++] = x[j]; x[j--] = tmp; }
        } while (i <= j);
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }
    return x[target];
}

void calibrate_channel(const float *raw, float *centre, float *scale) {
    float scratch[CALIBRATION];
    for (unsigned i = 0; i < CALIBRATION; ++i) scratch[i] = raw[i];
    *centre = select_lower(scratch, CALIBRATION);
    for (unsigned i = 0; i < CALIBRATION; ++i) scratch[i] = fabsf(raw[i] - *centre);
    *scale = fmaxf(1.4826f * select_lower(scratch, CALIBRATION), 1.e-4f);
}

void normalize_samples(const float *raw, float *out, unsigned n, float centre, float scale) {
    for (unsigned i = 0; i < n; ++i) {
        float v = (raw[i] - centre) / scale;
        out[i] = fminf(12.f, fmaxf(-12.f, v));
    }
}

uint16_t nearest_code(const float *x, const float *book, unsigned dim) {
    float best = FLT_MAX;
    uint16_t id = 0;
    for (unsigned k = 0; k < CODES; ++k) {
        float distance = 0.f;
        for (unsigned j = 0; j < dim; ++j) {
            float delta = x[j] - book[k * dim + j];
            distance += delta * delta;
        }
        if (distance < best) { best = distance; id = (uint16_t)k; }
    }
    return id;
}
