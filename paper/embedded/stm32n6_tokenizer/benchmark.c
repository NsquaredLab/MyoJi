/* Hardware workload driver. Numerical tokenizer kernels are linked unchanged;
 * only SPAR's channel loop/montage arguments are generalized by build.py.
 * Synthetic sample generation and console/debugger transfers are not timed.
 * Window updates, normalization, feature generation, VQ and packing are timed.
 */
#include "benchmark.h"
#include "weight_offsets.h"
#include <string.h>

#ifdef BENCH_HOST
#define REGION(name)
#else
#define REGION(name) __attribute__((section(name),aligned(32)))
#endif

volatile BenchConfig bench_config = {0, 0, 16, 16, 16, 260923};
volatile BenchResult bench_result;
BenchFrame bench_frames[BENCH_MAX_FRAMES];
uint16_t bench_first_ids[BENCH_MAX_CHANNELS], bench_last_ids[BENCH_MAX_CHANNELS];
static float raw[BENCH_MAX_CHANNELS][PATCH] REGION(".input_ram");
static float incoming[BENCH_MAX_CHANNELS][HOP] REGION(".input_ram");
static union {
    Complex spectra[BENCH_MAX_CHANNELS][7][32];
    float history[BENCH_MAX_LEARNED_CHANNELS][4][4][DIM];
} state REGION(".state_ram");
static float spar_output[BENCH_MAX_CHANNELS][SPAR_DIM];
static float stats[BENCH_MAX_CHANNELS][2], previous[BENCH_MAX_CHANNELS];
static uint32_t rng_state[BENCH_MAX_CHANNELS], sample_index[BENCH_MAX_CHANNELS];
static uint16_t members[BENCH_MAX_CHANNELS][4], ids[BENCH_MAX_CHANNELS];
static uint8_t masks[BENCH_MAX_CHANNELS][3], packed[(BENCH_MAX_CHANNELS*10+7)/8];

void bench_spar_features(const Complex (*spectra)[7][32], float (*features)[SPAR_DIM],
                        unsigned channels, const uint16_t (*members)[4],
                        const uint8_t (*mask)[3]);

static uint32_t next_random(unsigned c) {
    uint32_t x=rng_state[c];
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state[c]=x;
}

/* Deterministic broadband, amplitude-modulated synthetic load. This is not a
 * physiological EMG simulator and supplies no accuracy/generalization claim. */
static float next_sample(unsigned c) {
    float u=0.f;
    for (unsigned i=0;i<3;++i) u += (float)(next_random(c)>>8)*(1.f/16777216.f)-.5f;
    unsigned t=sample_index[c]++;
    float envelope=.35f+.65f*(float)((t/40u+c*7u)%100u)/99.f;
    float value=(u-.5f*previous[c])*envelope*100.f;
    previous[c]=u;
    return value+13.f*(float)((int)(c%7u)-3);
}

static void montage(unsigned channels) {
    /* Repeat the existing five-grid, 320-channel montage as a synthetic load.
     * A partial final tile masks edges whose neighbour is absent. At N=320
     * membership and masks are byte-for-byte the original paper montage. */
    for (unsigned c=0;c<channels;++c) {
        unsigned base=(c/CHANNELS)*CHANNELS, local=c%CHANNELS;
        for (unsigned j=0;j<4;++j) {
            unsigned neighbour=base+spar_members[local][j];
            members[c][j]=(uint16_t)(neighbour<channels ? neighbour : c);
        }
        masks[c][0]=spar_mask[local][0];
        masks[c][1]=(uint8_t)(spar_mask[local][1] && base+spar_members[local][2]<channels);
        masks[c][2]=(uint8_t)(spar_mask[local][2] && base+spar_members[local][3]<channels);
    }
}

static uint32_t token_checksum(unsigned channels) {
    uint32_t h=2166136261u;
    for (unsigned c=0;c<channels;++c) {
        h=(h^(ids[c]&255u))*16777619u;
        h=(h^(ids[c]>>8))*16777619u;
    }
    for (unsigned b=0;b<(channels*10u+7u)/8u;++b) h=(h^packed[b])*16777619u;
    return h;
}

unsigned bench_pack10(const uint16_t *input, unsigned n, uint8_t *out) {
    uint32_t accumulator=0; unsigned bits=0, used=0;
    for (unsigned i=0;i<n;++i) {
        accumulator |= (uint32_t)(input[i]&1023u)<<bits;
        bits+=10;
        while (bits>=8) {out[used++]=(uint8_t)accumulator; accumulator>>=8; bits-=8;}
    }
    if (bits) out[used++]=(uint8_t)accumulator;
    return used;
}

void bench_run(void) {
    unsigned n=bench_config.channels, route=bench_config.route;
    unsigned frames=bench_config.frames, warmup=bench_config.warmup;
    bench_result.status=2;
    if (!n || n>BENCH_MAX_CHANNELS || !frames || frames>BENCH_MAX_FRAMES ||
        route>1 || (route==1 && (n>BENCH_MAX_LEARNED_CHANNELS || warmup<16))) {
        bench_result.status=0xe001u; return;
    }
    bench_result.route=route; bench_result.channels=n; bench_result.frames=frames;
    bench_result.seed=bench_config.seed;
    bench_result.input_bytes=n*(PATCH+HOP)*sizeof(float);
    bench_result.state_bytes=route ? n*4u*4u*DIM*sizeof(float) : n*7u*32u*sizeof(Complex);
    bench_result.model_bytes=route ? (BOOK+CODES*DIM)*sizeof(float) :
        sizeof(spar_codebook)+sizeof(spar_mean)+sizeof(spar_scale);
    bench_result.calibration_cycles=0;
    memset(bench_frames,0,sizeof bench_frames);
    montage(n);
    float calibration[CALIBRATION], window[PATCH], output[DIM];
    for (unsigned c=0;c<n;++c) {
        rng_state[c]=(bench_config.seed ^ (0x9e3779b9u*(c+1u)))|1u;
        previous[c]=0.f; sample_index[c]=0;
        for (unsigned i=0;i<CALIBRATION;++i) calibration[i]=next_sample(c);
        uint64_t begin=bench_clock_poll();
        calibrate_channel(calibration,&stats[c][0],&stats[c][1]);
        bench_result.calibration_cycles+=bench_clock_poll()-begin;
        for (unsigned i=0;i<PATCH;++i) raw[c][i]=next_sample(c);
        if (route) memset(state.history[c],0,sizeof state.history[c]);
    }
    for (unsigned step=0;step<warmup+frames;++step) {
        /* Generate only new samples, before the timed raw-window update. */
        for (unsigned c=0;c<n;++c)
            for (unsigned i=0;i<HOP;++i) incoming[c][i]=next_sample(c);
        BenchFrame row={0};
        uint64_t begin=bench_clock_poll();
        for (unsigned c=0;c<n;++c) {
            uint64_t t=bench_clock_poll();
            memmove(raw[c],raw[c]+HOP,(PATCH-HOP)*sizeof(float));
            memcpy(raw[c]+PATCH-HOP,incoming[c],HOP*sizeof(float));
            normalize_samples(raw[c],window,PATCH,stats[c][0],stats[c][1]);
            uint64_t next=bench_clock_poll(); row.preprocessing+=next-t; t=next;
            if (route) learned_step(window,state.history[c],output);
            else spar_spectra(window,state.spectra[c]);
            next=bench_clock_poll(); row.frontend+=next-t; t=next;
            if (route) {
                ids[c]=nearest_code(output,learned_weights+BOOK,DIM);
                row.quantizer+=bench_clock_poll()-t;
            }
        }
        if (!route) {
            uint64_t t=bench_clock_poll();
            bench_spar_features(state.spectra,spar_output,n,members,masks);
            row.frontend+=bench_clock_poll()-t;
            for (unsigned c=0;c<n;++c) {
                t=bench_clock_poll();
                ids[c]=nearest_code(spar_output[c],spar_codebook,SPAR_DIM);
                row.quantizer+=bench_clock_poll()-t;
            }
        }
        uint64_t t=bench_clock_poll();
        (void)bench_pack10(ids,n,packed);
        row.packing=bench_clock_poll()-t;
        row.total=bench_clock_poll()-begin;
        /* Consume all output IDs after timing, preventing dead-code removal. */
        row.checksum=token_checksum(n); row.tokens=n;
        if (step>=warmup) {
            unsigned index=step-warmup;
            bench_frames[index]=row;
            if (index==0) memcpy(bench_first_ids,ids,n*sizeof(uint16_t));
            if (index+1==frames) memcpy(bench_last_ids,ids,n*sizeof(uint16_t));
        }
    }
    bench_result.status=3;
}

#ifndef BENCH_HOST
int main(void) {
    bench_platform_init();
    while (1) {
        bench_ready();
        if (bench_config.command==1) {
            if (bench_platform_start()) {
                bench_run();
                bench_platform_stop();
            }
            bench_config.command=0;
        }
    }
}
#endif
