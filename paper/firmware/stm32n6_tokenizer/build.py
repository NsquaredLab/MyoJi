"""Build a separate NUCLEO-N657X0-Q benchmark; never connects or flashes.

The MCU build requires STM32CUBEN6_ROOT, the pinned official STM32CubeN6
checkout described in paper/embedded/stm32n6_tokenizer/README.md; the host
build (--host) does not. Host and MCU builds use the existing FP32
raw-tokenizer kernels and frozen artifacts.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "embedded/stm32n6_tokenizer"
RAW = ROOT / "embedded/raw_tokenizers"
FROZEN = ROOT / "results/raw_tokenizers_cortex_m4_v1"
OUT = ROOT / "results/stm32n6_tokenizer_v1"
CUBE_URL = "https://github.com/STMicroelectronics/STM32CubeN6"
CUBE_COMMIT = "80a3a665fccaf9a429343a7ab96a97c1b8a6467a"
FLAGS = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off", "-fno-math-errno",
         "-Wall", "-Wextra", "-fstack-usage", "-ffunction-sections", "-fdata-sections"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cube():
    """The STM32CubeN6 checkout (not shipped), from STM32CUBEN6_ROOT."""
    root = os.environ.get("STM32CUBEN6_ROOT")
    if not root or not Path(root, "Drivers").is_dir():
        raise RuntimeError(f"Set STM32CUBEN6_ROOT to a checkout of {CUBE_URL} at commit {CUBE_COMMIT} "
                           "with its submodules initialized; see paper/embedded/stm32n6_tokenizer/README.md")
    return Path(root).resolve()


def receipt_key(path):
    """Receipt name of a source: relative to paper/, else STM32CubeN6/... inside the CubeN6 checkout."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else f"STM32CubeN6/{path.relative_to(cube())}"


def receipt_path(key):
    """Inverse of receipt_key, for re-hashing the sources a receipt lists."""
    return cube() / key.removeprefix("STM32CubeN6/") if key.startswith("STM32CubeN6/") else ROOT / key


def tool(name):
    explicit = os.environ.get("STM32N6_GCC")
    if explicit:
        p = Path(explicit).with_name(name)
        if p.exists(): return p
    p = shutil.which(name)
    if p: return Path(p)
    paths = sorted(Path('/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins').glob(
        f'com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32*/tools/bin/{name}'))
    if paths: return paths[-1]
    raise RuntimeError(f"Cannot find {name}; set STM32N6_GCC to the compiler executable")


def run(args, cwd=None):
    result = subprocess.run(list(map(str,args)), cwd=cwd, text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def extract_function(text, name):
    match = re.search(r'void\s+' + re.escape(name) + r'\(void\)\s*\{', text)
    if not match: raise ValueError(f"Missing upstream {name}")
    depth=1; end=match.end()
    while depth:
        depth += (text[end]=='{')-(text[end]=='}'); end+=1
    return text[match.start():end]


def generated():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'generated'; dest.mkdir(exist_ok=True)
    source=(RAW/'spar.c').read_text()
    signature='void spar_features(const Complex spectra[CHANNELS][7][32], float features[CHANNELS][SPAR_DIM]) {'
    if source.count(signature)!=1: raise RuntimeError('Upstream SPAR signature changed; inspect before adapting')
    source=source.replace(signature,
        'void bench_spar_features(const Complex (*spectra)[7][32], float (*features)[SPAR_DIM], '
        'unsigned channels, const uint16_t (*members)[4], const uint8_t (*mask)[3]) {')
    source=source.replace('#include "common.h"','#include "benchmark.h"')
    source=source.replace('c < CHANNELS','c < channels')
    source=source.replace('spar_members[c]','members[c]').replace('spar_mask[c]','mask[c]')
    needle='for (unsigned c = 0; c < channels; ++c) {\n            Complex a[4][4];'
    if source.count(needle)!=1: raise RuntimeError('Upstream SPAR loop changed')
    source=source.replace(needle,'for (unsigned c = 0; c < channels; ++c) {\n'
        '            (void)bench_clock_poll(); /* Extend 32-bit hardware cycle count. */\n'
        '            Complex a[4][4];')
    (dest/'spar_runtime.c').write_text('/* Generated from the existing SPAR port; see build receipt. */\n'+source)
    weights=(FROZEN/'learned_weights.f32').read_bytes()
    digest=2166136261
    for byte in weights: digest=((digest^byte)*16777619)&0xffffffff
    (dest/'frozen_digest.h').write_text(
        f'#define BENCH_WEIGHT_BYTES {len(weights)}u\n#define BENCH_WEIGHT_FNV1A 0x{digest:08x}u\n')
    return dest


def build_host(dest):
    compiler=Path(shutil.which('clang') or shutil.which('cc') or '')
    if not compiler.is_file(): raise RuntimeError('Host compiler unavailable')
    prefix='_' if sys.platform=='darwin' else ''
    section='.section __TEXT,__const' if sys.platform=='darwin' else '.section .rodata'
    asm=dest/'host_weights.S'
    asm.write_text(f'{section}\n.p2align 2\n.globl {prefix}learned_weights\n{prefix}learned_weights:\n'
                   f'.incbin "{FROZEN/"learned_weights.f32"}"\n')
    clock=dest/'host_clock.c'
    clock.write_text('#include <stdint.h>\n#include <time.h>\n'
        'uint64_t bench_clock_poll(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); '
        'return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec; }\n')
    sources=[RAW/'common.c',RAW/'learned.c',SRC/'benchmark.c',dest/'spar_runtime.c',
             FROZEN/'constants.c',asm,clock]
    lib=OUT/('host.dylib' if sys.platform=='darwin' else 'host.so')
    cmd=[compiler,*FLAGS,'-DBENCH_HOST','-shared','-fPIC',f'-I{RAW}',f'-I{FROZEN}',f'-I{SRC}',
         *sources,'-lm','-o',lib]
    log=run(cmd); (OUT/'host_compile.log').write_text(log)
    (OUT/'host_build.json').write_text(json.dumps({'hardware':False,'command':list(map(str,cmd)),
        'sources':{receipt_key(p):sha(p) for p in sources},'library_sha256':sha(lib)},indent=2)+'\n')
    return lib


def build_mcu(dest):
    CUBE=cube()
    if run(['git','rev-parse','HEAD'],CUBE).strip()!=CUBE_COMMIT:
        raise RuntimeError('Unexpected CubeN6 commit; review and update the pinned dependency deliberately')
    example=CUBE/'Projects/NUCLEO-N657X0-Q/Examples/GPIO/GPIO_IOToggle'
    hal=CUBE/'Drivers/STM32N6xx_HAL_Driver'
    cmsis=CUBE/'Drivers/CMSIS'
    bsp=CUBE/'Drivers/BSP/STM32N6xx_Nucleo'
    device=cmsis/'Device/ST/STM32N6xx'
    config=(example/'FSBL/Inc/stm32n6xx_hal_conf.h').read_text()
    config=config.replace('/*#define HAL_XSPI_MODULE_ENABLED   */','#define HAL_XSPI_MODULE_ENABLED')
    (dest/'stm32n6xx_hal_conf.h').write_text(config)
    shutil.copy2(bsp/'stm32n6xx_nucleo_conf_template.h',dest/'stm32n6xx_nucleo_conf.h')
    flash_config=(CUBE/'Drivers/BSP/Components/mx25um51245g/mx25um51245g_conf_template.h').read_text()
    (dest/'mx25um51245g_conf.h').write_text(flash_config.replace('stm32xxxx_hal.h','stm32n6xx_hal.h'))
    clock=extract_function((example/'FSBL/Src/main.c').read_text(),'SystemClock_Config')
    (dest/'clock.c').write_text('/* STMicroelectronics GPIO example clock, unchanged. See pinned source/license. */\n'
        '#include "stm32n6xx_hal.h"\nvoid Error_Handler(void);\n'+clock+'\n')
    linker=(example/'STM32CubeIDE/FSBL/STM32N657X0HXQ_AXISRAM2_fsbl.ld').read_text()
    linker=linker.replace('_Min_Stack_Size = 0x800;','_Min_Stack_Size = 0x8000;')
    linker=linker.replace('  ROM    (xrw)',
        '  INPUT_RAM  (xrw) : ORIGIN = 0x34000000, LENGTH = 1536K\n'
        '  STATE  (xrw) : ORIGIN = 0x34200000, LENGTH = 1792K\n'
        '  ROM    (xrw)')
    linker=linker.replace('SECTIONS\n{','learned_weights = 0x71000000; /* Frozen weights in NOR; installed separately. */\n'
        'SECTIONS\n{\n  .input_ram (NOLOAD) : { *(.input_ram) } >INPUT_RAM\n'
        '  .state_ram (NOLOAD) : { *(.state_ram) } >STATE\n')
    linker=re.sub(r'  /DISCARD/\s*:\s*\{.*?\n  \}', '', linker, flags=re.S)
    (dest/'memory.ld').write_text(linker)
    sources=[SRC/'benchmark.c',SRC/'platform.c',RAW/'common.c',RAW/'learned.c',
        dest/'spar_runtime.c',FROZEN/'constants.c',dest/'clock.c',
        example/'FSBL/Src/system_stm32n6xx_fsbl.c',
        example/'STM32CubeIDE/FSBL/Application/Startup/startup_stm32n657x0hxq_fsbl.s',
        bsp/'stm32n6xx_nucleo_xspi.c',
        CUBE/'Drivers/BSP/Components/mx25um51245g/mx25um51245g.c']
    sources += [hal/'Src'/f'stm32n6xx_hal{name}.c' for name in
        ['', '_cortex','_rcc','_rcc_ex','_pwr','_pwr_ex','_gpio','_dma','_dma_ex','_xspi']]
    includes=[dest,SRC,RAW,FROZEN,hal/'Inc',cmsis/'Include',device/'Include',bsp]
    gcc=tool('arm-none-eabi-gcc')
    flags=[*FLAGS,'-mcpu=cortex-m55','-mthumb','-mfloat-abi=hard','-DSTM32N657xx','-DUSE_HAL_DRIVER',
           '-D__ARM_FEATURE_CMSE=3', '-g3',*[f'-I{p}' for p in includes]]
    # -mcmse sets the compiler's own CMSE feature macro without redefining it.
    flags.remove('-D__ARM_FEATURE_CMSE=3'); flags.append('-mcmse')
    objects=OUT/'objects'; objects.mkdir(exist_ok=True)
    commands=[]
    def compile_source(item):
        index,p=item
        obj=objects/f'{index:02d}_{p.stem}.o'
        cmd=[gcc,*flags,'-c',p,'-o',obj]
        return obj,cmd,run(cmd)
    compiled=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for obj,cmd,log in pool.map(compile_source,enumerate(sources)):
            compiled.append(obj);commands.append(list(map(str,cmd)))
            if log: print(log,end='')
    elf=OUT/'benchmark.elf'
    command=[gcc,'-mcpu=cortex-m55','-mthumb','-mfloat-abi=hard','-mcmse','-nostartfiles',
             *compiled,f'-T{dest/"memory.ld"}',f'-Wl,-Map={OUT/"benchmark.map"}',
             '-Wl,--gc-sections','--specs=nosys.specs','-Wl,--start-group','-lm','-lc','-lgcc',
             '-lnosys','-Wl,--end-group','-o',elf]
    (OUT/'link.log').write_text(run(command))
    sizes=run([tool('arm-none-eabi-size'),'-A',elf]);(OUT/'size.txt').write_text(sizes)
    symbols=run([tool('arm-none-eabi-nm'),'-n',elf]);(OUT/'symbols.txt').write_text(symbols)
    addresses={parts[2]:int(parts[0],16) for line in symbols.splitlines()
               if len(parts:=line.split())==3 and parts[0][0] in '0123456789abcdefABCDEF'}
    receipt={'schema':1,'board':'NUCLEO-N657X0-Q','hardware_run':False,'backend':'Cortex-M55 FP32 CPU',
        'configured_cpu_hz':600000000,'npu_used':False,'cube_commit':CUBE_COMMIT,
        'compiler':run([gcc,'--version']).splitlines()[0],'commands':commands+[list(map(str,command))],
        'elf_sha256':sha(elf),'weights_address':'0x71000000',
        'weights_sha256':sha(FROZEN/'learned_weights.f32'),
        'sources':{receipt_key(p):sha(p) for p in sources+[SRC/'benchmark.h',RAW/'common.h',
            FROZEN/'weight_offsets.h',dest/'frozen_digest.h',dest/'stm32n6xx_hal_conf.h',
            dest/'stm32n6xx_nucleo_conf.h',dest/'mx25um51245g_conf.h',dest/'memory.ld']},
        'addresses':{k:v for k,v in addresses.items() if k.startswith('bench_') or k in ['Reset_Handler','_estack']},
        'note':'Build only. No MCU timings or hardware validation until collected on the physical board.'}
    (OUT/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return elf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',action='store_true')
    args=parser.parse_args()
    dest=generated()
    print(build_host(dest) if args.host else build_mcu(dest))


if __name__=='__main__': main()
