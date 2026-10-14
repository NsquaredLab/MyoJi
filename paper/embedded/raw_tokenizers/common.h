#ifndef RAW_TOKENIZERS_COMMON_H
#define RAW_TOKENIZERS_COMMON_H
#include <stdint.h>
#include <stddef.h>
#define CHANNELS 320
#define RAW_SAMPLES 2120
#define PATCH 320
#define HOP 40
#define CALIBRATION 1000
#define DIM 256
#define CODES 1024
#define SPAR_DIM 18
typedef struct { float re, im; } Complex;
void calibrate_channel(const float *raw, float *centre, float *scale);
void normalize_samples(const float *raw, float *out, unsigned n, float centre, float scale);
uint16_t nearest_code(const float *x, const float *book, unsigned dim);
void matrix_log_entries(const Complex input[4][4], float out[3]);
void spar_spectra(const float *window, Complex out[7][32]);
void spar_features(const Complex spectra[CHANNELS][7][32], float features[CHANNELS][SPAR_DIM]);
void learned_step(const float window[PATCH], float history[4][4][DIM], float out[DIM]);
extern const float learned_weights[];
extern const float spar_codebook[CODES * SPAR_DIM], spar_mean[SPAR_DIM], spar_scale[SPAR_DIM];
extern const float hann128[128], fft_cos[64], fft_sin[64];
extern const uint16_t spar_members[CHANNELS][4];
extern const uint8_t spar_mask[CHANNELS][3];
#endif
