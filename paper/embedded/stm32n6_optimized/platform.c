#include "stm32n6xx_hal.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "benchmark.h"
#include "frozen_digest.h"

void SystemClock_Config(void); /* Unmodified ST GPIO example: 600 MHz. */
static uint32_t previous_cycle;
static uint64_t elapsed_cycles;
static int weights_ready;
volatile uint32_t bench_startup_stage;

__attribute__((noinline)) void bench_progress(uint32_t stage) {
    bench_startup_stage=stage;
    __asm__ volatile("nop" ::: "memory");
}

/* Complete the BSP's weak initializer with the Nucleo example's XSPIM
 * port-2 routing. The BSP alone does not configure the manager. */
HAL_StatusTypeDef MX_XSPI_NOR_Init(XSPI_HandleTypeDef *hxspi, MX_XSPI_InitTypeDef *init) {
    hxspi->Instance=XSPI2;
    hxspi->Init.FifoThresholdByte=4;
    hxspi->Init.MemoryMode=HAL_XSPI_SINGLE_MEM;
    hxspi->Init.MemorySize=init->MemorySize;
    hxspi->Init.ChipSelectHighTimeCycle=2;
    hxspi->Init.FreeRunningClock=HAL_XSPI_FREERUNCLK_DISABLE;
    hxspi->Init.ClockMode=HAL_XSPI_CLOCK_MODE_0;
    hxspi->Init.WrapSize=HAL_XSPI_WRAP_NOT_SUPPORTED;
    hxspi->Init.ClockPrescaler=init->ClockPrescaler;
    hxspi->Init.SampleShifting=init->SampleShifting;
    hxspi->Init.ChipSelectBoundary=HAL_XSPI_BONDARYOF_NONE;
    hxspi->Init.MemoryType=HAL_XSPI_MEMTYPE_MACRONIX;
    hxspi->Init.DelayHoldQuarterCycle=HAL_XSPI_DHQC_ENABLE;
    hxspi->Init.MemorySelect=HAL_XSPI_CSSEL_NCS1;
    if (HAL_XSPI_Init(hxspi)!=HAL_OK) return HAL_ERROR;
    XSPIM_CfgTypeDef manager={0};
    manager.nCSOverride=HAL_XSPI_CSSEL_OVR_NCS1;
    manager.IOPort=HAL_XSPIM_IOPORT_2;
    manager.Req2AckTime=1;
    return HAL_XSPIM_Config(hxspi,&manager,HAL_XSPI_TIMEOUT_DEFAULT_VALUE);
}

/* The official startup calls newlib's constructor initialiser. There are no
 * constructors in this C-only benchmark. */
void _init(void) {}

uint64_t bench_clock_poll(void) {
    uint32_t now=DWT->CYCCNT;
    elapsed_cycles+=(uint32_t)(now-previous_cycle);
    previous_cycle=now;
    return elapsed_cycles;
}

void SysTick_Handler(void) { HAL_IncTick(); }
void Error_Handler(void) { bench_result.status=0xe100u; while (1) {} }
void HardFault_Handler(void) {
    bench_result.status=0xe101u;
    bench_result.fault_cfsr=SCB->CFSR; bench_result.fault_hfsr=SCB->HFSR;
    bench_result.fault_bfar=SCB->BFAR; bench_result.fault_mmfar=SCB->MMFAR;
    while (1) {}
}
void SecureFault_Handler(void) { bench_result.status=0xe102u; while (1) {} }
void MemManage_Handler(void) { HardFault_Handler(); }
void BusFault_Handler(void) { HardFault_Handler(); }
void UsageFault_Handler(void) { HardFault_Handler(); }

void bench_platform_init(void) {
    HAL_Init();
    __enable_irq();
    SystemClock_Config();
    __HAL_RCC_AXISRAM1_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM3_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM4_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM5_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM6_MEM_CLK_ENABLE();
    /* These banks reset in shutdown. A clock alone leaves writes in cache
     * without backing storage; power them up through the secure RAMCFG alias. */
    __HAL_RCC_RAMCFG_CLK_ENABLE();
    CLEAR_BIT(RAMCFG_SRAM3_AXI_S->CR, RAMCFG_CR_SRAMSD);
    CLEAR_BIT(RAMCFG_SRAM4_AXI_S->CR, RAMCFG_CR_SRAMSD);
    CLEAR_BIT(RAMCFG_SRAM5_AXI_S->CR, RAMCFG_CR_SRAMSD);
    CLEAR_BIT(RAMCFG_SRAM6_AXI_S->CR, RAMCFG_CR_SRAMSD);
    __DSB(); __ISB();
    /* Check the state arena with D-cache still off, before any workload. */
    for (unsigned offset=0;offset<1792u*1024u;offset+=4096u) {
        volatile uint32_t *address=(volatile uint32_t *)(0x34200000u+offset);
        uint32_t expected=0xa5c39e17u^offset;
        *address=expected; __DSB();
        if (*address!=expected) {
            bench_result.status=0xe105u;
            bench_result.fault_bfar=(uint32_t)address;
            while (1) {}
        }
        *address=0;
    }
    SCB_EnableICache();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT=0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t check=DWT->CYCCNT;
    for (volatile unsigned i=0;i<1024u;++i) __NOP();
    if (DWT->CYCCNT==check) { bench_result.status=0xe103u; while (1) {} }
    previous_cycle=DWT->CYCCNT; elapsed_cycles=0;
    bench_result.schema=BENCH_SCHEMA;
    bench_result.hardware=1;
    SystemCoreClockUpdate();
    bench_result.cpu_hz=SystemCoreClock;
    bench_result.status=1;
}

int bench_platform_start(void) {
    if (bench_config.route==1 && !weights_ready) {
        bench_progress(1);
        /* Official Nucleo template's 50 MHz setting, usable without changing
         * the permanent high-speed I/O fuse. CPU remains at 600 MHz. */
        RCC_PeriphCLKInitTypeDef clocks={0};
        clocks.PeriphClockSelection=RCC_PERIPHCLK_XSPI2;
        clocks.Xspi2ClockSelection=RCC_XSPI2CLKSOURCE_IC3;
        clocks.ICSelection[RCC_IC3].ClockSelection=RCC_ICCLKSOURCE_PLL1;
        clocks.ICSelection[RCC_IC3].ClockDivider=24;
        if (HAL_RCCEx_PeriphCLKConfig(&clocks)!=HAL_OK) Error_Handler();
        __HAL_RCC_XSPIM_CLK_ENABLE();
        /* Read-only mapping. This firmware never erases or programs NOR. */
        BSP_XSPI_NOR_Init_t init={BSP_XSPI_NOR_OPI_MODE,BSP_XSPI_NOR_DTR_TRANSFER};
        if (BSP_XSPI_NOR_Init(0,&init)!=BSP_ERROR_NONE) Error_Handler();
        bench_progress(2);
        if (BSP_XSPI_NOR_EnableMemoryMappedMode(0)!=BSP_ERROR_NONE) Error_Handler();
        bench_progress(3);
        /* Refuse measurements on absent or different NOR weights. This check
         * is outside all timing intervals; installation also uses SHA-256. */
        const volatile uint8_t *bytes=(const volatile uint8_t *)learned_weights;
        uint32_t digest=2166136261u;
        for (unsigned i=0;i<BENCH_WEIGHT_BYTES;++i) digest=(digest^bytes[i])*16777619u;
        if (digest!=BENCH_WEIGHT_FNV1A) { bench_result.status=0xe104u; return 0; }
        weights_ready=1;
        bench_progress(4);
    }
    SCB_InvalidateDCache();
    SCB_EnableDCache();
    bench_result.cache_enabled=1;
    previous_cycle=DWT->CYCCNT; elapsed_cycles=0;
    return 1;
}

void bench_platform_stop(void) {
    /* Debugger transfers occur only with D-cache off: host-written commands
     * cannot be hidden by a stale cache line, and all results are visible. */
    SCB_CleanInvalidateDCache();
    SCB_DisableDCache();
}

__attribute__((noinline)) void bench_ready(void) {
    __asm__ volatile("nop" ::: "memory");
}
