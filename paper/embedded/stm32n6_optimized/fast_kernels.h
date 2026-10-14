#ifndef EMG_FAST_KERNELS_H
#define EMG_FAST_KERNELS_H
#include "common.h"
#define FAST_BATCH 16
#define FAST_CODES 16
#define FAST_VQ_CHUNK 64
typedef struct {
    float input[PATCH][FAST_BATCH];
    float x[DIM][FAST_BATCH], norm[DIM][FAST_BATCH];
    float depthwise[DIM][FAST_BATCH], update[DIM][FAST_BATCH];
    float hidden[1024][FAST_BATCH];
} FastWorkspace;
void fast_prepare_book(const float *source, unsigned dim, float *packed);
uint16_t fast_nearest_code(const float *x, const float *packed, unsigned dim);
void fast_nearest_code_batch(const float *x, unsigned stride, unsigned count,
                             const float *packed, unsigned dim, uint16_t *ids);
void fast_spar_spectra(const float *window, Complex out[7][32]);
void fast_spar_features(const Complex (*spectra)[7][32], float (*features)[SPAR_DIM],
                        unsigned channels, const uint16_t (*members)[4], const uint8_t (*mask)[3]);
void fast_learned_batch(FastWorkspace *w, float (*history)[4][4][DIM], unsigned count);
#endif
