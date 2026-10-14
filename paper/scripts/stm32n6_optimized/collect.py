"""Run one SRAM benchmark configuration and retain raw physical-board evidence.

Requires a working ST-LINK connection. Does not program external NOR or change
option bytes. The learned route additionally requires separately installed,
verified frozen weights. A successful build is not a successful hardware run.
"""
from __future__ import annotations
import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import numpy as np

from build import OUT, FROZEN, receipt_path, sha, tool
from verify_host import Frame, Result, load, run_case


def installed(name,pattern):
    override=os.environ.get(name)
    if override: return Path(override)
    found=sorted(Path('/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins').glob(pattern))
    if not found: raise RuntimeError(f'Set {name} to the required executable')
    return found[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',choices=['spar','learned'],default='spar')
    parser.add_argument('--channels',type=int,default=16)
    parser.add_argument('--sweep-channels',type=int,nargs='+',
                        help='Run these channel counts sequentially, with a fresh SRAM load for each')
    parser.add_argument('--frames',type=int,default=2)
    parser.add_argument('--warmup',type=int)
    parser.add_argument('--seed',type=int,default=260923)
    parser.add_argument('--serial')
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--port',type=int,default=61234)
    parser.add_argument('--swd-khz',type=int,default=1000,
                        help='Debug transfer clock only; does not set the MCU execution clock')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--diagnostics',action='store_true',
                        help='Read final input, calibration, spectra and SPAR features for numerical diagnosis')
    parser.add_argument('--trace-startup',action='store_true')
    parser.add_argument('--recorded',action='store_true',help='Replay existing recorded fixture, <=64 channels; use 17 warmup + 3 measured frames')
    args=parser.parse_args()
    if args.sweep_channels:
        # Child commands use exactly the same guarded collection path.
        outcomes=[]
        for channels in args.sweep_channels:
            command=[sys.executable,str(Path(__file__).resolve()),'--route',args.route,
                '--channels',str(channels),'--frames',str(args.frames),'--seed',str(args.seed),
                '--timeout',str(args.timeout),'--port',str(args.port),'--swd-khz',str(args.swd_khz)]
            if args.warmup is not None: command += ['--warmup',str(args.warmup)]
            if args.serial: command += ['--serial',args.serial]
            if args.diagnostics: command += ['--diagnostics']
            if args.trace_startup: command += ['--trace-startup']
            if args.plan_only: command += ['--plan-only']
            if args.recorded: command += ['--recorded']
            print(f'Starting {args.route}, {channels} channels',flush=True)
            completed=subprocess.run(command)
            outcomes.append({'channels':channels,'exit_code':completed.returncode})
        report=OUT/('sweep_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
        report.write_text(json.dumps({'configuration':vars(args),'outcomes':outcomes},indent=2)+'\n')
        if any(row['exit_code'] for row in outcomes):
            raise RuntimeError(f'Some runs failed verification; retained all outcomes in {report}')
        return
    route=int(args.route=='learned')
    warmup=args.warmup if args.warmup is not None else (16 if route else 1)
    if not 1<=args.channels<=(112 if route else 1024): parser.error('Channel count outside harness limits')
    if not 1<=args.frames<=64 or warmup<0 or (route and warmup<16): parser.error('Invalid frames/warmup')
    if not 0<=args.seed<=0xffffffff: parser.error('Seed must be uint32')
    if not 1<=args.swd_khz<=24000: parser.error('SWD clock must be 1..24000 kHz')
    if args.recorded and (args.channels>64 or warmup!=17 or args.frames!=3):
        parser.error('Recorded validation uses <=64 channels, --warmup 17 --frames 3')
    receipt=json.loads((OUT/'build.json').read_text())
    elf=OUT/'benchmark.elf'
    if sha(elf)!=receipt['elf_sha256']: raise RuntimeError('ELF differs from its build receipt')
    for source,digest in receipt['sources'].items():
        if sha(receipt_path(source))!=digest: raise RuntimeError(f'Source changed since MCU build: {source}')
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest=OUT/'runs'/f'{timestamp}_{args.route}_{args.channels}ch';dest.mkdir(parents=True)
    config={**vars(args),'warmup':warmup,'timestamp_utc':timestamp,
            'input':'synthetic amplitude-modulated broadband noise',
            'window_ms':160,'update_ms':20,'elf_sha256':receipt['elf_sha256']}
    if args.recorded:config['input']='Recorded raw EMG fixture; first N of original 320 channels'
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    addresses=receipt['addresses']
    def dump(symbol,size,name):
        start=addresses[symbol]
        return f'dump binary memory {dest/name} 0x{start:x} 0x{start+size:x}'
    script=['set confirm off','set pagination off','set remotetimeout 15',
        f'target remote localhost:{args.port}','monitor reset',f'load {elf}',
        'set $sp = (unsigned int)&_estack','set $pc = Reset_Handler',
        'hbreak bench_ready','hbreak HardFault_Handler','hbreak SecureFault_Handler',
        'hbreak Error_Handler','continue',dump('bench_result',ct.sizeof(Result),'boot.bin'),
        'if bench_result.status != 1','  quit 2','end']
    script.extend(f'set variable bench_config.{key} = {value}' for key,value in
                  [('route',route),('channels',args.channels),('frames',args.frames),
                   ('warmup',warmup),('seed',args.seed),('command',1)])
    fixture=None
    if args.recorded:
        fixture=np.fromfile(FROZEN/'raw_320x2120.f32',dtype='<f4').reshape(320,2120)[:args.channels].copy()
        fixture.tofile(dest/'recorded_input.bin')
        config['recorded_input_sha256']=sha(dest/'recorded_input.bin')
        script.extend([f'restore {dest/"recorded_input.bin"} binary 0x34300000',
                       'set variable bench_input_mode = 1'])
    if args.trace_startup:
        script.extend(['hbreak bench_progress','commands','silent',
            'printf "Startup stage %u\\n", $r0','p/x $primask','p uwTick','continue','end'])
    script.extend(['continue',dump('bench_result',ct.sizeof(Result),'result.bin'),
        dump('bench_frames',ct.sizeof(Frame)*args.frames,'frames.bin'),
        dump('bench_first_ids',args.channels*2,'first_ids.bin'),
        dump('bench_last_ids',args.channels*2,'last_ids.bin'),dump('bench_setup_cycles',8,'setup.bin')])
    if args.diagnostics:
        symbols={p[2]:int(p[0],16) for line in (OUT/'symbols.txt').read_text().splitlines()
                 if len(p:=line.split())==3 and p[0][0] in '0123456789abcdefABCDEF'}
        symbols['raw']=symbols['arena']
        for symbol,size in [('raw',args.channels*320*4),('stats',args.channels*2*4)]+(
                [('state',args.channels*7*32*8),('spar_output',args.channels*18*4)] if not route else []):
            start=symbols[symbol]
            script.append(f'dump binary memory {dest/(symbol+".bin")} 0x{start:x} 0x{start+size:x}')
    script.extend(['printf "N6 status: 0x%x\\n", bench_result.status','disconnect','quit'])
    gdbfile=dest/'run.gdb';gdbfile.write_text('\n'.join(script)+'\n')
    if args.plan_only:
        print(f'Prepared {gdbfile}; no connection or measurements.');return
    server=installed('STM32N6_GDB_SERVER','com.st.stm32cube.ide.mcu.externaltools.stlink-gdb-server*/tools/bin/ST-LINK_gdbserver')
    programmer=installed('STM32N6_PROGRAMMER','com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer*/tools/bin/STM32_Programmer_CLI')
    probe=subprocess.run([str(programmer),'-l','st-link'],capture_output=True,text=True,timeout=20)
    listing=re.sub(r'\x1b\[[0-9;]*m','',probe.stdout+probe.stderr)
    (dest/'probe.log').write_text(listing)
    serials=re.findall(r'ST-LINK SN[ \t]*:[ \t]*([0-9A-Fa-f]{24})',listing)
    serial=args.serial or (serials[0] if len(serials)==1 else None)
    if not serial or serial not in serials:
        raise RuntimeError(f'No unambiguous responding ST-LINK identity. See {dest/"probe.log"}')
    config['stlink_serial']=serial
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    # Refuse a occupied port instead of accidentally using another debug session.
    with socket.socket() as test: test.bind(('127.0.0.1',args.port))
    # N6 exposes the Cortex-M55 on access port 1 (ST AN6265).
    command=[str(server),'-d','-m','1','-k','--halt','-i',serial,'-p',str(args.port),'--frequency',str(args.swd_khz),
             '-cp',str(programmer.parent),'-l','1']
    (dest/'server_command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (dest/'server.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            # Watch the server log; do not consume its single GDB socket with a probe.
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if proc.poll() is not None: raise RuntimeError(f'ST-LINK server exited; see {dest/"server.log"}')
                if 'Waiting for debugger connection' in (dest/'server.log').read_text(errors='replace'): break
                time.sleep(.1)
            else: raise RuntimeError(f'No GDB readiness message; see {dest/"server.log"}')
            with (dest/'gdb.log').open('w') as gdb_log:
                run=subprocess.run([str(tool('arm-none-eabi-gdb')),'-nx','--batch','-x',str(gdbfile),str(elf)],
                    stdout=gdb_log,stderr=subprocess.STDOUT,timeout=args.timeout)
            if run.returncode: raise RuntimeError(f'GDB failed; see {dest/"gdb.log"}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
    result=Result.from_buffer_copy((dest/'result.bin').read_bytes())
    if result.status!=3 or result.hardware!=1 or result.cpu_hz!=600000000:
        raise RuntimeError(f'Invalid device result: status={result.status:#x}, hardware={result.hardware}, clock={result.cpu_hz}')
    if (result.route,result.channels,result.frames,result.seed)!=(route,args.channels,args.frames,args.seed):
        raise RuntimeError('Device returned a different workload configuration')
    frames=(Frame*args.frames).from_buffer_copy((dest/'frames.bin').read_bytes())
    if not all(row.total>0 and row.tokens==args.channels for row in frames):
        raise RuntimeError('Invalid cycle/token counters')
    libpath=OUT/('host.dylib' if os.uname().sysname=='Darwin' else 'host.so')
    validation=json.loads((OUT/'host_verification.json').read_text())
    if not validation['passed'] or validation['library_sha256']!=sha(libpath):
        raise RuntimeError('Run host functional verification before collecting publishable results')
    fast_validation=json.loads((OUT/'fast_verification.json').read_text())
    if not fast_validation['passed'] or fast_validation['library_sha256']!=sha(libpath):
        raise RuntimeError('Run verify_fast.py after building the host library')
    lib=load(libpath)
    if args.recorded:
        address=ct.c_void_p.in_dll(lib,'bench_recorded_input').value
        ct.memmove(address,fixture.ctypes.data,fixture.nbytes)
        ct.c_uint32.in_dll(lib,'bench_input_mode').value=1
    expected=run_case(lib,route,args.channels,args.frames,warmup,args.seed)
    if args.recorded and route:
        frozen=np.load(FROZEN/'host_outputs.npz')['learned_ids'][:,:args.channels]
        assert expected['first_ids']==frozen[0].tolist() and expected['last_ids']==frozen[-1].tolist()
    first=list((ct.c_uint16*args.channels).from_buffer_copy((dest/'first_ids.bin').read_bytes()))
    last=list((ct.c_uint16*args.channels).from_buffer_copy((dest/'last_ids.bin').read_bytes()))
    correct=(first==expected['first_ids'] and last==expected['last_ids'] and
             [row.checksum for row in frames]==expected['checksums'])
    (dest/'output_validation.json').write_text(json.dumps({'expected':expected,
        'actual':{'first_ids':first,'last_ids':last,'checksums':[row.checksum for row in frames]},
        'exact_match':correct},indent=2)+'\n')
    with (dest/'frames.csv').open('w',newline='') as f:
        columns=['frame','route','channels','cpu_hz','total_cycles','preprocessing_cycles',
                 'frontend_cycles','quantizer_cycles','packing_cycles','latency_ms','tokens_per_s',
                 'checksum','output_verified','hardware']
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for i,row in enumerate(frames):
            writer.writerow({'frame':i,'route':args.route,'channels':args.channels,
                'cpu_hz':result.cpu_hz,'total_cycles':row.total,'preprocessing_cycles':row.preprocessing,
                'frontend_cycles':row.frontend,'quantizer_cycles':row.quantizer,'packing_cycles':row.packing,
                'latency_ms':row.total/result.cpu_hz*1000,'tokens_per_s':args.channels*result.cpu_hz/row.total,
                'checksum':row.checksum,'output_verified':correct,'hardware':True})
    summary={'hardware':True,'output_verified':correct,'configuration':config,'build':receipt,
             'setup_cycles':int.from_bytes((dest/'setup.bin').read_bytes(),'little'),
             'host_fast_validation':fast_validation,
             'device':{key:getattr(result,key) for key,_ in Result._fields_},
             'note':'Physical Cortex-M55 CPU measurements; synthetic input; no NPU, ADC or BLE timing.'}
    (dest/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not correct: raise RuntimeError(f'Hardware IDs/checksums differ from host. Retained raw evidence in {dest}; do not plot as validated.')
    print(f'Validated physical-board run: {dest}')


if __name__=='__main__': main()
