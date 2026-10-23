#ifndef STM32N6_TOKENIZER_BENCHMARK_H
#define STM32N6_TOKENIZER_BENCHMARK_H
#include <stdint.h>
#include "common.h"

#define BENCH_MAX_CHANNELS 1024u
#define BENCH_MAX_LEARNED_CHANNELS 112u
#define BENCH_MAX_FRAMES 64u
#define BENCH_SCHEMA 1u

/* route: 0 = paper SPAR, 1 = paper learned encoder. No neural accelerator. */
typedef struct {
    uint32_t command, route, channels, frames, warmup, seed;
} BenchConfig;

typedef struct {
    uint64_t total, preprocessing, frontend, quantizer, packing;
    uint32_t checksum, tokens;
} BenchFrame;

typedef struct {
    uint32_t schema, status, cpu_hz, route, channels, frames, seed;
    uint32_t hardware, cache_enabled, input_bytes, state_bytes, model_bytes;
    uint32_t fault_cfsr, fault_hfsr, fault_bfar, fault_mmfar;
    uint64_t calibration_cycles;
} BenchResult;

extern volatile BenchConfig bench_config;
extern volatile BenchResult bench_result;
extern BenchFrame bench_frames[BENCH_MAX_FRAMES];
extern uint16_t bench_first_ids[BENCH_MAX_CHANNELS];
extern uint16_t bench_last_ids[BENCH_MAX_CHANNELS];
uint64_t bench_clock_poll(void);
void bench_run(void);
void bench_platform_init(void);
int bench_platform_start(void);
void bench_platform_stop(void);
void bench_ready(void);
#endif
