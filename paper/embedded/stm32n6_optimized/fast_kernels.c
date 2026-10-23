/* Independent lanes retain each scalar reduction's original order. No FMA,
 * reassociation, reduced precision, approximate activation or retraining. */
#include "fast_kernels.h"
#include "benchmark.h"
#include "weight_offsets.h"
#include <math.h>
#include <float.h>
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
#include <arm_mve.h>
#define USE_MVE 1
#else
#define USE_MVE 0
#endif

void fast_prepare_book(const float *source, unsigned dim, float *packed) {
    for (unsigned k=0;k<CODES;k+=FAST_CODES)
        for (unsigned d=0;d<dim;++d)
            for (unsigned lane=0;lane<FAST_CODES;++lane)
                packed[k*dim+d*FAST_CODES+lane]=source[(k+lane)*dim+d];
}

/* One code group's 16 squared distances. Each lane keeps the reference
 * accumulation order over dim, so results are bit-identical to the scalar
 * reference regardless of how callers order channels and groups. */
static void code_group_sums(const float *x, const float *group, unsigned dim, float *sums) {
#if USE_MVE
    float32x4_t s0=vdupq_n_f32(0.f),s1=s0,s2=s0,s3=s0;
    for (unsigned d=0;d<dim;++d) {
        const float *b=group+d*FAST_CODES;
        float32x4_t v=vdupq_n_f32(x[d]);
        float32x4_t d0=vsubq_f32(v,vldrwq_f32(b));
        float32x4_t d1=vsubq_f32(v,vldrwq_f32(b+4));
        float32x4_t d2=vsubq_f32(v,vldrwq_f32(b+8));
        float32x4_t d3=vsubq_f32(v,vldrwq_f32(b+12));
        s0=vaddq_f32(s0,vmulq_f32(d0,d0));
        s1=vaddq_f32(s1,vmulq_f32(d1,d1));
        s2=vaddq_f32(s2,vmulq_f32(d2,d2));
        s3=vaddq_f32(s3,vmulq_f32(d3,d3));
    }
    vstrwq_f32(sums,s0);vstrwq_f32(sums+4,s1);
    vstrwq_f32(sums+8,s2);vstrwq_f32(sums+12,s3);
#else
    for (unsigned lane=0;lane<FAST_CODES;++lane) sums[lane]=0.f;
    for (unsigned d=0;d<dim;++d)
        for (unsigned lane=0;lane<FAST_CODES;++lane) {
            float delta=x[d]-group[d*FAST_CODES+lane];
            sums[lane]+=delta*delta;
        }
#endif
}

uint16_t fast_nearest_code(const float *x, const float *packed, unsigned dim) {
    float best=FLT_MAX; uint16_t id=0;
    for (unsigned k=0;k<CODES;k+=FAST_CODES) {
        float sums[FAST_CODES];
        code_group_sums(x,packed+k*dim,dim,sums);
        for (unsigned lane=0;lane<FAST_CODES;++lane)
            if (sums[lane]<best) {best=sums[lane];id=(uint16_t)(k+lane);}
    }
    return id;
}

/* Same search, with the code group outside the channel loop: a group stays in
 * the data cache across every channel, so the book is read once per chunk
 * instead of once per channel. Each channel still visits codes in increasing
 * order with the same strict comparison, so both the distances and the
 * earliest-index tie rule match fast_nearest_code exactly. */
void fast_nearest_code_batch(const float *x, unsigned stride, unsigned count,
                             const float *packed, unsigned dim, uint16_t *ids) {
    for (unsigned base=0;base<count;base+=FAST_VQ_CHUNK) {
        unsigned lanes=count-base<FAST_VQ_CHUNK ? count-base : FAST_VQ_CHUNK;
        float best[FAST_VQ_CHUNK];
        for (unsigned c=0;c<lanes;++c) {best[c]=FLT_MAX;ids[base+c]=0;}
        for (unsigned k=0;k<CODES;k+=FAST_CODES) {
            const float *group=packed+k*dim;
            for (unsigned c=0;c<lanes;++c) {
                float sums[FAST_CODES];
                code_group_sums(x+(base+c)*stride,group,dim,sums);
                for (unsigned lane=0;lane<FAST_CODES;++lane)
                    if (sums[lane]<best[c]) {best[c]=sums[lane];ids[base+c]=(uint16_t)(k+lane);}
            }
        }
    }
}

/* Inputs/outputs are feature-major. Every weight is loaded once for a batch
 * of simultaneous electrode channels, with no batching across time. */
#if USE_MVE
#define LINEAR_MVE(GROUPS) do { \
    for (unsigned o=0;o<outputs;++o) { \
        const float *weights=learned_weights+wo+o*inputs; \
        float32x4_t a=vdupq_n_f32(0.f),b=a,c=a,d=a; \
        for (unsigned i=0;i<inputs;++i) { \
            float weight=weights[i]; \
            const float *in=x+i*FAST_BATCH; \
            a=vaddq_f32(a,vmulq_n_f32(vldrwq_f32(in),weight)); \
            if ((GROUPS)>1) b=vaddq_f32(b,vmulq_n_f32(vldrwq_f32(in+4),weight)); \
            if ((GROUPS)>2) c=vaddq_f32(c,vmulq_n_f32(vldrwq_f32(in+8),weight)); \
            if ((GROUPS)>3) d=vaddq_f32(d,vmulq_n_f32(vldrwq_f32(in+12),weight)); \
        } \
        float bias=learned_weights[bo+o]; \
        vstrwq_f32(y+o*FAST_BATCH,vaddq_n_f32(a,bias)); \
        if ((GROUPS)>1) vstrwq_f32(y+o*FAST_BATCH+4,vaddq_n_f32(b,bias)); \
        if ((GROUPS)>2) vstrwq_f32(y+o*FAST_BATCH+8,vaddq_n_f32(c,bias)); \
        if ((GROUPS)>3) vstrwq_f32(y+o*FAST_BATCH+12,vaddq_n_f32(d,bias)); \
    } \
} while (0)
#endif

static void linear_batch(const float *x,float *y,unsigned inputs,unsigned outputs,
                         unsigned wo,unsigned bo,unsigned count) {
#if USE_MVE
    /* Nine to twelve lanes used to get their own three-group expansion, which
     * measured slower than the four-group one despite doing less arithmetic, so
     * they now take the four-group path: lanes past `count` compute values the
     * caller never reads. Lanes are independent, so every used lane keeps its
     * own reduction order and result. One and two groups are still cheaper. */
    if (count<=4) LINEAR_MVE(1);
    else if (count<=8) LINEAR_MVE(2);
    else LINEAR_MVE(4);
#else
    for (unsigned o=0;o<outputs;++o) {
        const float *weights=learned_weights+wo+o*inputs;
        for (unsigned c=0;c<count;++c) {
            float sum=0.f;
            for (unsigned i=0;i<inputs;++i) sum+=x[i*FAST_BATCH+c]*weights[i];
            y[o*FAST_BATCH+c]=sum+learned_weights[bo+o];
        }
    }
#endif
}

static float gelu(float x) {return .5f*x*(1.f+erff(x*.7071067811865475244f));}

static void norm_batch(const float *x,float *y,unsigned wo,unsigned bo,unsigned count) {
    for (unsigned c=0;c<count;++c) {
        float mean=0.f,variance=0.f;
        for (unsigned d=0;d<DIM;++d) mean+=x[d*FAST_BATCH+c];
        mean/=(float)DIM;
        for (unsigned d=0;d<DIM;++d) {float delta=x[d*FAST_BATCH+c]-mean;variance+=delta*delta;}
        float inverse=1.f/sqrtf(variance/(float)DIM+1.e-5f);
        for (unsigned d=0;d<DIM;++d)
            y[d*FAST_BATCH+c]=(x[d*FAST_BATCH+c]-mean)*inverse*learned_weights[wo+d]+learned_weights[bo+d];
    }
}

void fast_learned_batch(FastWorkspace *w,float (*history)[4][4][DIM],unsigned count) {
    linear_batch(&w->input[0][0],&w->hidden[0][0],PATCH,32,PATCH_W,PATCH_B,count);
    for (unsigned d=0;d<32;++d) for (unsigned c=0;c<count;++c) w->hidden[d][c]=gelu(w->hidden[d][c]);
    linear_batch(&w->hidden[0][0],&w->x[0][0],32,DIM,PROJECT_W,PROJECT_B,count);
    for (unsigned l=0;l<4;++l) {
        const unsigned *o=BLOCK_OFFSETS[l];
        norm_batch(&w->x[0][0],&w->norm[0][0],o[0],o[1],count);
        for (unsigned c=0;c<count;++c) {
            for (unsigned d=0;d<DIM;++d) {
                float v=0.f;
                for (unsigned t=0;t<4;++t) v+=history[c][l][t][d]*learned_weights[o[2]+d*5+t];
                v+=w->norm[d][c]*learned_weights[o[2]+d*5+4];
                w->depthwise[d][c]=v+learned_weights[o[3]+d];
            }
            for (unsigned t=0;t<3;++t) for (unsigned d=0;d<DIM;++d) history[c][l][t][d]=history[c][l][t+1][d];
            for (unsigned d=0;d<DIM;++d) history[c][l][3][d]=w->norm[d][c];
        }
        linear_batch(&w->depthwise[0][0],&w->hidden[0][0],DIM,1024,o[4],o[5],count);
        for (unsigned d=0;d<1024;++d) for (unsigned c=0;c<count;++c) w->hidden[d][c]=gelu(w->hidden[d][c]);
        linear_batch(&w->hidden[0][0],&w->update[0][0],1024,DIM,o[6],o[7],count);
        for (unsigned d=0;d<DIM;++d) for (unsigned c=0;c<count;++c) w->x[d][c]+=w->update[d][c];
    }
    norm_batch(&w->x[0][0],&w->norm[0][0],FINAL_W,FINAL_B,count);
}


/* Same six-band SPAR features as the frozen port, reordered so a channel's
 * four member spectra are visited once for all six bands instead of the whole
 * spectra array being swept once per band, and so each member bin is scaled by
 * its band weight once instead of once per covariance pair. Every surviving
 * floating-point expression, and the order of every reduction, is unchanged. */
void fast_spar_features(const Complex (*spectra)[7][32], float (*features)[SPAR_DIM],
                        unsigned channels, const uint16_t (*members)[4], const uint8_t (*mask)[3]) {
    const unsigned edges[7] = {2,3,4,7,11,16,32};
    float power[32] = {0.f}, weight[32];
    for (unsigned c = 0; c < channels; ++c)
        for (unsigned bin = 2; bin < 32; ++bin) {
            float sum = 0.f;
            for (unsigned s = 0; s < 7; ++s) {
                Complex v = spectra[c][s][bin]; sum += v.re*v.re+v.im*v.im;
            }
            power[bin] += sum / 7.f;
        }
    /* The six bands partition bins 2..31, so one weight per bin covers all. */
    for (unsigned band = 0; band < 6; ++band) {
        float total = 0.f;
        for (unsigned bin = edges[band]; bin < edges[band+1]; ++bin) total += power[bin];
        total = fmaxf(total,1.e-12f);
        for (unsigned bin = edges[band]; bin < edges[band+1]; ++bin)
            weight[bin] = sqrtf(power[bin]/total);
    }
    for (unsigned c = 0; c < channels; ++c) {
        (void)bench_clock_poll(); /* Extend 32-bit hardware cycle count. */
        /* Weight every member bin once per channel in one streaming pass, with
         * the member index innermost: the four members of a (segment,bin) then
         * load as one vector below, and the four stores fill one aligned chunk
         * instead of striding. The bands partition bins 2..31, so each bin is
         * still scaled exactly once. Real and imaginary parts are kept apart to
         * avoid de-interleaving in the reduction. */
        float scaled_re[7][32][4], scaled_im[7][32][4];
        const Complex (*source[4])[32];
        for (unsigned p = 0; p < 4; ++p) source[p] = spectra[members[c][p]];
        for (unsigned s = 0; s < 7; ++s)
            for (unsigned bin = 2; bin < 32; ++bin) {
                float w = weight[bin];
                for (unsigned p = 0; p < 4; ++p) {
                    Complex v = source[p][s][bin];
                    scaled_re[s][bin][p] = v.re*w; scaled_im[s][bin][p] = v.im*w;
                }
            }
        for (unsigned band = 0; band < 6; ++band) {
            unsigned low = edges[band], width = edges[band+1]-low;
            float sum_re[4][4], sum_im[4][4];
            /* One member p against all four at once. Each lane is a separate
             * accumulator running the reference's segment-then-bin order, so no
             * reduction is reassociated; the lanes below the diagonal repeat the
             * mirrored entries and are dropped. x.re*y.re - x.im*(-y.im) and
             * x.re*(-y.im) + x.im*y.re are written as the identical products
             * added in the identical order, so every value matches the scalar
             * reference bit-for-bit. */
            for (unsigned p = 0; p < 4; ++p) {
#if USE_MVE
                float32x4_t accumulate_re = vdupq_n_f32(0.f), accumulate_im = accumulate_re;
                for (unsigned s = 0; s < 7; ++s)
                    for (unsigned b = 0; b < width; ++b) {
                        float32x4_t yr = vld1q_f32(scaled_re[s][low+b]);
                        float32x4_t yi = vld1q_f32(scaled_im[s][low+b]);
                        float32x4_t xr = vdupq_n_f32(scaled_re[s][low+b][p]);
                        float32x4_t xi = vdupq_n_f32(scaled_im[s][low+b][p]);
                        accumulate_re = vaddq_f32(accumulate_re,
                            vaddq_f32(vmulq_f32(xr,yr),vmulq_f32(xi,yi)));
                        accumulate_im = vaddq_f32(accumulate_im,
                            vsubq_f32(vmulq_f32(xi,yr),vmulq_f32(xr,yi)));
                    }
                vst1q_f32(sum_re[p],accumulate_re); vst1q_f32(sum_im[p],accumulate_im);
#else
                for (unsigned q = 0; q < 4; ++q) {
                    float re = 0.f, im = 0.f;
                    for (unsigned s = 0; s < 7; ++s)
                        for (unsigned b = 0; b < width; ++b) {
                            float xr = scaled_re[s][low+b][p], xi = scaled_im[s][low+b][p];
                            float yr = scaled_re[s][low+b][q], yi = scaled_im[s][low+b][q];
                            re += xr*yr - xi*(-yi);
                            im += xr*(-yi) + xi*yr;
                        }
                    sum_re[p][q] = re; sum_im[p][q] = im;
                }
#endif
            }
            Complex a[4][4];
            for (unsigned p = 0; p < 4; ++p)
                for (unsigned q = p; q < 4; ++q) {
                    a[p][q] = (Complex){sum_re[p][q]*(1.f/7.f), sum_im[p][q]*(1.f/7.f)};
                    a[q][p] = (Complex){a[p][q].re, -a[p][q].im};
                }
            float largest = 1.e-12f, entries[3];
            for (unsigned p = 0; p < 4; ++p) largest = fmaxf(largest,a[p][p].re);
            for (unsigned p = 0; p < 4; ++p) a[p][p].re += 1.e-6f*largest;
            matrix_log_entries(a,entries);
            for (unsigned d = 0; d < 3; ++d) {
                unsigned j = band * 3 + d;
                features[c][j] = mask[c][d] ? (entries[d]-spar_mean[j])/spar_scale[j] : 0.f;
            }
        }
    }
}

/* Two transforms live here. EMG_EXACT_TRANSFORM selects the frozen radix-2 one,
 * which reproduces the released port bit-for-bit. The default selects CMSIS-DSP's
 * real FFT, which is faster and NOT bit-exact; see this experiment's README. */
#ifdef EMG_EXACT_TRANSFORM
/* Same radix-2 decimation-in-time FFT as the frozen port. Butterflies are
 * independent, so evaluating four of them per instruction changes no value and
 * no reduction order. The twiddle table is restaged into contiguous per-stage
 * copies of the very same floats, only so four can be loaded at once. The
 * final stage stops at bin 32 because nothing reads the rest. */
#define FFT_VECTOR_STAGES 5
static float fft_stage_re[FFT_VECTOR_STAGES][64], fft_stage_im[FFT_VECTOR_STAGES][64];
static uint8_t fft_reverse[128];
static int fft_tables_ready;

static void fft_tables_init(void) {
    for (unsigned i=0;i<128;++i) {
        unsigned reverse=0, v=i;
        for (unsigned j=0;j<7;++j) {reverse=(reverse<<1)|(v&1);v>>=1;}
        fft_reverse[i]=(uint8_t)reverse;
    }
    unsigned s=0;
    for (unsigned length=8;length<=128;length*=2,++s) {
        unsigned half=length/2, step=128/length;
        for (unsigned j=0;j<half;++j) {
            fft_stage_re[s][j]=fft_cos[j*step];
            fft_stage_im[s][j]=fft_sin[j*step];
        }
    }
    fft_tables_ready=1;
}

void fast_spar_spectra(const float *window, Complex out[7][32]) {
    if (!fft_tables_ready) fft_tables_init();
    for (unsigned segment=0;segment<7;++segment) {
        Complex data[128];
        for (unsigned i=0;i<128;++i)
            data[fft_reverse[i]]=(Complex){window[segment*32+i]*hann128[i],0.f};
        /* Stages with fewer than four butterflies per block stay scalar. */
        for (unsigned length=2;length<=4;length*=2) {
            unsigned half=length/2, step=128/length;
            for (unsigned start=0;start<128;start+=length)
                for (unsigned j=0;j<half;++j) {
                    float c=fft_cos[j*step], s=fft_sin[j*step];
                    Complex v=data[start+j+half], a=data[start+j];
                    Complex b={v.re*c-v.im*s, v.re*s+v.im*c};
                    data[start+j]=(Complex){a.re+b.re,a.im+b.im};
                    data[start+j+half]=(Complex){a.re-b.re,a.im-b.im};
                }
        }
        unsigned stage=0;
        for (unsigned length=8;length<=128;length*=2,++stage) {
            unsigned half=length/2;
            /* Only bins 0..31 leave the transform, so the widest stage stops. */
            unsigned span=length==128 ? 32u : half;
            const float *tr=fft_stage_re[stage], *ti=fft_stage_im[stage];
            for (unsigned start=0;start<128;start+=length)
#if USE_MVE
                for (unsigned j=0;j<span;j+=4) {
                    float *low=(float *)&data[start+j], *high=(float *)&data[start+j+half];
                    float32x4x2_t a=vld2q_f32(low), v=vld2q_f32(high), l, h;
                    float32x4_t c=vld1q_f32(tr+j), s=vld1q_f32(ti+j);
                    float32x4_t br=vsubq_f32(vmulq_f32(v.val[0],c),vmulq_f32(v.val[1],s));
                    float32x4_t bi=vaddq_f32(vmulq_f32(v.val[0],s),vmulq_f32(v.val[1],c));
                    l.val[0]=vaddq_f32(a.val[0],br); l.val[1]=vaddq_f32(a.val[1],bi);
                    h.val[0]=vsubq_f32(a.val[0],br); h.val[1]=vsubq_f32(a.val[1],bi);
                    vst2q_f32(low,l); vst2q_f32(high,h);
                }
#else
                for (unsigned j=0;j<span;++j) {
                    float c=tr[j], s=ti[j];
                    Complex v=data[start+j+half], a=data[start+j];
                    Complex b={v.re*c-v.im*s, v.re*s+v.im*c};
                    data[start+j]=(Complex){a.re+b.re,a.im+b.im};
                    data[start+j+half]=(Complex){a.re-b.re,a.im-b.im};
                }
#endif
        }
        for (unsigned bin=0;bin<32;++bin) out[segment][bin]=data[bin];
    }
}
#else
/* CMSIS-DSP real FFT. Half the transform work of the frozen complex radix-2 on
 * real input, and Helium-vectorized by the library. Results differ from the
 * frozen port in the last few bits: this path is not bit-exact and the README
 * records the measured deviation and the validation it still owes.
 * spar_features reads only bins 2..31 (its power loop and every band edge start
 * at bin 2), so the packed DC/Nyquist slot in out[0] never reaches a feature. */
#include "arm_math.h"
static arm_rfft_fast_instance_f32 rfft_instance;
static int rfft_ready;

void fast_spar_spectra(const float *window, Complex out[7][32]) {
    if (!rfft_ready) {
        arm_rfft_fast_init_128_f32(&rfft_instance);
        rfft_ready=1;
    }
    for (unsigned segment=0;segment<7;++segment) {
        float windowed[128], spectrum[128];
        for (unsigned i=0;i<128;++i) windowed[i]=window[segment*32+i]*hann128[i];
        arm_rfft_fast_f32(&rfft_instance,windowed,spectrum,0);
        out[segment][0]=(Complex){0.f,0.f};
        out[segment][1]=(Complex){0.f,0.f};
        for (unsigned bin=2;bin<32;++bin)
            out[segment][bin]=(Complex){spectrum[2*bin],spectrum[2*bin+1]};
    }
}
#endif
