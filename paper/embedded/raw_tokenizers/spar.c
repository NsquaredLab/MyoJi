/* Portable radix-2 FFT + local complex-Hermitian Jacobi matrix logarithm.
 * Frozen six-band SPAR_SIMPLE montage/weights/statistics; no fitted changes. */
#include "common.h"
#include <math.h>

static Complex add(Complex a, Complex b) { return (Complex){a.re+b.re, a.im+b.im}; }
static Complex sub(Complex a, Complex b) { return (Complex){a.re-b.re, a.im-b.im}; }
static Complex mul(Complex a, Complex b) { return (Complex){a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re}; }
static Complex scale(Complex a, float b) { return (Complex){a.re*b, a.im*b}; }
static Complex conjx(Complex a) { return (Complex){a.re, -a.im}; }

void spar_spectra(const float *window, Complex out[7][32]) {
    for (unsigned segment = 0; segment < 7; ++segment) {
        Complex data[128];
        for (unsigned i = 0; i < 128; ++i) {
            unsigned reverse = 0, v = i;
            for (unsigned j = 0; j < 7; ++j) { reverse = (reverse << 1) | (v & 1); v >>= 1; }
            data[reverse] = (Complex){window[segment * 32 + i] * hann128[i], 0.f};
        }
        for (unsigned length = 2; length <= 128; length *= 2) {
            unsigned half = length / 2, step = 128 / length;
            for (unsigned start = 0; start < 128; start += length)
                for (unsigned j = 0; j < half; ++j) {
                    Complex twiddle = {fft_cos[j * step], fft_sin[j * step]};
                    Complex b = mul(data[start+j+half], twiddle), a = data[start+j];
                    data[start+j] = add(a,b); data[start+j+half] = sub(a,b);
                }
        }
        for (unsigned bin = 0; bin < 32; ++bin) out[segment][bin] = data[bin];
    }
}

void matrix_log_entries(const Complex input[4][4], float out[3]) {
    Complex a[4][4], vectors[4][4];
    float largest = 1.e-12f;
    for (unsigned p = 0; p < 4; ++p) largest = fmaxf(largest, input[p][p].re);
    for (unsigned p = 0; p < 4; ++p)
        for (unsigned q = 0; q < 4; ++q) {
            a[p][q] = input[p][q];
            vectors[p][q] = (Complex){p == q ? 1.f : 0.f, 0.f};
        }
    for (unsigned p = 0; p < 4; ++p) a[p][p].re += 1.e-8f * largest;
    for (unsigned sweep = 0; sweep < 24; ++sweep) {
        float off = 0.f;
        for (unsigned p = 0; p < 3; ++p)
            for (unsigned q = p + 1; q < 4; ++q) {
                float r = hypotf(a[p][q].re, a[p][q].im);
                off = fmaxf(off, r);
                if (r < 1.e-20f || r < 1.e-8f * fminf(fabsf(a[p][p].re), fabsf(a[q][q].re))) continue;
                float tau = (a[q][q].re - a[p][p].re) / (2.f*r);
                float t = copysignf(1.f / (fabsf(tau) + hypotf(1.f, tau)), tau);
                float c = 1.f / sqrtf(1.f + t*t), s = t*c;
                Complex phase = scale(a[p][q], 1.f/r), sp = scale(phase, s), sm = conjx(sp);
                a[p][p].re -= t*r; a[q][q].re += t*r;
                a[p][q] = a[q][p] = (Complex){0.f,0.f};
                for (unsigned k = 0; k < 4; ++k) {
                    if (k != p && k != q) {
                        Complex ap = a[k][p], aq = a[k][q];
                        a[k][p] = sub(scale(ap,c),mul(sm,aq));
                        a[k][q] = add(mul(sp,ap),scale(aq,c));
                        a[p][k] = conjx(a[k][p]); a[q][k] = conjx(a[k][q]);
                    }
                    Complex vp = vectors[k][p], vq = vectors[k][q];
                    vectors[k][p] = sub(scale(vp,c),mul(sm,vq));
                    vectors[k][q] = add(mul(sp,vp),scale(vq,c));
                }
            }
        if (off <= 1.e-10f * largest) break;
    }
    out[0] = out[1] = out[2] = 0.f;
    for (unsigned k = 0; k < 4; ++k) {
        float l = logf(fmaxf(a[k][k].re,1.e-12f));
        Complex v = vectors[0][k];
        out[0] += (v.re*v.re + v.im*v.im)*l;
        out[1] += mul(v,conjx(vectors[2][k])).re*l;
        out[2] += mul(v,conjx(vectors[3][k])).re*l;
    }
}

void spar_features(const Complex spectra[CHANNELS][7][32], float features[CHANNELS][SPAR_DIM]) {
    const unsigned edges[7] = {2,3,4,7,11,16,32};
    float power[32] = {0.f};
    for (unsigned c = 0; c < CHANNELS; ++c)
        for (unsigned bin = 2; bin < 32; ++bin) {
            float sum = 0.f;
            for (unsigned s = 0; s < 7; ++s) {
                Complex v = spectra[c][s][bin]; sum += v.re*v.re+v.im*v.im;
            }
            power[bin] += sum / 7.f;
        }
    for (unsigned band = 0; band < 6; ++band) {
        float total = 0.f, weight[32];
        for (unsigned bin = edges[band]; bin < edges[band+1]; ++bin) total += power[bin];
        total = fmaxf(total,1.e-12f);
        for (unsigned bin = edges[band]; bin < edges[band+1]; ++bin) weight[bin] = sqrtf(power[bin]/total);
        for (unsigned c = 0; c < CHANNELS; ++c) {
            Complex a[4][4];
            for (unsigned p = 0; p < 4; ++p)
                for (unsigned q = p; q < 4; ++q) {
                    Complex v = {0.f,0.f};
                    for (unsigned s = 0; s < 7; ++s)
                        for (unsigned bin = edges[band]; bin < edges[band+1]; ++bin) {
                            Complex x = scale(spectra[spar_members[c][p]][s][bin],weight[bin]);
                            Complex y = scale(spectra[spar_members[c][q]][s][bin],weight[bin]);
                            v = add(v,mul(x,conjx(y)));
                        }
                    a[p][q] = scale(v,1.f/7.f); a[q][p] = conjx(a[p][q]);
                }
            float largest = 1.e-12f, entries[3];
            for (unsigned p = 0; p < 4; ++p) largest = fmaxf(largest,a[p][p].re);
            for (unsigned p = 0; p < 4; ++p) a[p][p].re += 1.e-6f*largest;
            matrix_log_entries(a,entries);
            for (unsigned d = 0; d < 3; ++d) {
                unsigned j = band * 3 + d;
                features[c][j] = spar_mask[c][d] ? (entries[d]-spar_mean[j])/spar_scale[j] : 0.f;
            }
        }
    }
}
