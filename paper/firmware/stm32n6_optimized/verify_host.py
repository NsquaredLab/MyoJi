"""Functional checks for the N6 workload. Host durations are not MCU results."""
from __future__ import annotations
import ctypes as ct
import json

import numpy as np
from paper.firmware.stm32n6_optimized.build import EXACT, OUT, FROZEN, build_host, generated, sha

# In the default build the Jacobi hypot is replaced, so the frozen SPAR path is
# no longer bit-exact either. Token IDs must still be reproduced exactly; the
# feature deviation is bounded instead of asserted equal. STM32N6_EXACT=1
# asserts equality, as before.
MAX_SPAR_FEATURE_ABS = 5e-2


class Config(ct.Structure):
    _fields_=[(k,ct.c_uint32) for k in ('command','route','channels','frames','warmup','seed')]


class Frame(ct.Structure):
    _fields_=[(k,ct.c_uint64) for k in ('total','preprocessing','frontend','quantizer','packing')]+[
        ('checksum',ct.c_uint32),('tokens',ct.c_uint32)]


class Result(ct.Structure):
    _fields_=[(k,ct.c_uint32) for k in ('schema','status','cpu_hz','route','channels','frames','seed',
        'hardware','cache_enabled','input_bytes','state_bytes','model_bytes',
        'fault_cfsr','fault_hfsr','fault_bfar','fault_mmfar')]+[('calibration_cycles',ct.c_uint64)]


FP=ct.POINTER(ct.c_float)
def ptr(array): return array.ctypes.data_as(FP)


def load(path):
    lib=ct.CDLL(str(path))
    for name,n in [('calibrate_channel',3),('spar_spectra',2),('learned_step',3)]:
        getattr(lib,name).argtypes=[FP]*n
        getattr(lib,name).restype=None
    lib.normalize_samples.argtypes=[FP,FP,ct.c_uint,ct.c_float,ct.c_float]
    lib.normalize_samples.restype=None
    lib.nearest_code.argtypes=[FP,FP,ct.c_uint]
    lib.nearest_code.restype=ct.c_uint16
    lib.bench_spar_features.argtypes=[FP,FP,ct.c_uint,ct.c_void_p,ct.c_void_p]
    lib.bench_spar_features.restype=None
    lib.bench_run.argtypes=[];lib.bench_run.restype=None
    lib.bench_pack10.argtypes=[ct.POINTER(ct.c_uint16),ct.c_uint,ct.POINTER(ct.c_uint8)]
    lib.bench_pack10.restype=ct.c_uint
    return lib


def run_case(lib,route,channels,frames=2,warmup=None,seed=260923):
    cfg=Config.in_dll(lib,'bench_config')
    cfg.command=1;cfg.route=route;cfg.channels=channels;cfg.frames=frames
    cfg.warmup=(16 if route else 1) if warmup is None else warmup;cfg.seed=seed
    lib.bench_run()
    result=Result.in_dll(lib,'bench_result')
    if result.status!=3: raise ValueError(f'Benchmark status {result.status:#x}')
    rows=(Frame*64).in_dll(lib,'bench_frames')
    first=np.ctypeslib.as_array((ct.c_uint16*1024).in_dll(lib,'bench_first_ids'))[:channels].copy()
    last=np.ctypeslib.as_array((ct.c_uint16*1024).in_dll(lib,'bench_last_ids'))[:channels].copy()
    assert np.all(first<1024) and np.all(last<1024)
    assert all(row.total>0 and row.tokens==channels for row in rows[:frames])
    return {'route':route,'channels':channels,'frames':frames,'warmup':cfg.warmup,'seed':seed,
        'checksums':[int(row.checksum) for row in rows[:frames]],
        'first_ids':first.tolist(),'last_ids':last.tolist()}


def main():
    libpath=build_host(generated());lib=load(libpath)
    ref=np.load(FROZEN/'reference.npz')
    previous=np.load(FROZEN/'host_outputs.npz')
    raw=np.fromfile(FROZEN/'raw_320x2120.f32',dtype='<f4').reshape(320,2120)
    stats=np.empty((320,2),np.float32);norm=np.empty_like(raw)
    for c in range(320):
        lib.calibrate_channel(ptr(raw[c]),ptr(stats[c,:1]),ptr(stats[c,1:]))
        lib.normalize_samples(ptr(raw[c]),ptr(norm[c]),2120,*map(float,stats[c]))
    np.testing.assert_array_equal(norm,ref['normalized'])
    np.testing.assert_array_equal(stats,previous['calibration'])
    members=((ct.c_uint16*4)*320).in_dll(lib,'spar_members')
    masks=((ct.c_uint8*3)*320).in_dll(lib,'spar_mask')
    spectra=np.empty((320,7,32),np.complex64)
    features=np.empty((320,18),np.float32)
    spar_ids=[]
    max_spar_error=0.
    for f,end in enumerate([2039,2079,2119]):
        for c in range(320):
            window=np.ascontiguousarray(norm[c,end-319:end+1])
            lib.spar_spectra(ptr(window),ptr(spectra[c]))
        lib.bench_spar_features(ptr(spectra),ptr(features),320,members,masks)
        if EXACT:
            np.testing.assert_allclose(features,previous['spar_features'][f],rtol=1e-5,atol=1e-5)
        max_spar_error=max(max_spar_error,float(np.abs(features-previous['spar_features'][f]).max()))
        assert EXACT or max_spar_error<MAX_SPAR_FEATURE_ABS,max_spar_error
        ids=np.array([lib.nearest_code(ptr(v),ptr(ref['spar_book']),18) for v in features],np.uint16)
        np.testing.assert_array_equal(ids,previous['spar_ids'][f]);spar_ids.extend(ids)
    # The learned kernels are linked unchanged. Check representative recorded
    # channels after the full causal warmup under the new compiler flags.
    max_learned_error=0.
    for c in [0,319]:
        history=np.zeros((4,4,256),np.float32);value=np.empty(256,np.float32)
        for end in range(1399,2120,40):
            window=np.ascontiguousarray(norm[c,end-319:end+1])
            lib.learned_step(ptr(window),ptr(history),ptr(value))
            if end>=2039:
                f=(end-2039)//40
                np.testing.assert_allclose(value,previous['learned_features'][f,c],rtol=1e-5,atol=1e-5)
                max_learned_error=max(max_learned_error,float(np.abs(value-previous['learned_features'][f,c]).max()))
                assert lib.nearest_code(ptr(value),ptr(ref['learned_book']),256)==previous['learned_ids'][f,c]
    # Independent little-endian 10-bit packing oracle, including partial bytes.
    rng=np.random.default_rng(260923)
    packing_cases=0
    for n in [0,1,2,3,4,5,16,319,320,1023,1024]:
        ids=rng.integers(0,1024,n,dtype=np.uint16)
        packed=np.zeros((n*10+7)//8+1,np.uint8)
        used=lib.bench_pack10(ids.ctypes.data_as(ct.POINTER(ct.c_uint16)),n,
                              packed.ctypes.data_as(ct.POINTER(ct.c_uint8)))
        value=sum(int(code)<<(10*i) for i,code in enumerate(ids))
        expected=value.to_bytes((n*10+7)//8,'little')
        assert used==len(expected) and packed[:used].tobytes()==expected
        assert packed[used]==0
        packing_cases+=1
    goldens=[]
    for route,n in [(0,1),(0,16),(0,320),(0,1024),(1,1),(1,2)]:
        first=run_case(lib,route,n);repeat=run_case(lib,route,n)
        assert first==repeat, 'Synthetic input or cache state is not reproducible'
        goldens.append(first)
    assert run_case(lib,0,16,seed=260924)['checksums']!=goldens[1]['checksums']
    cfg=Config.in_dll(lib,'bench_config')
    for route,n,frames,warmup in [(0,0,2,1),(0,1025,2,1),(1,129,2,16),(1,1,2,15),(2,1,2,16),(0,1,65,1)]:
        cfg.route=route;cfg.channels=n;cfg.frames=frames;cfg.warmup=warmup
        lib.bench_run()
        assert Result.in_dll(lib,'bench_result').status==0xe001
    receipt={'schema':1,'passed':True,'hardware':False,'exact_transform':EXACT,
        'library_sha256':sha(libpath),
        'normalized_samples_exact':int(norm.size),'spar_recorded_tokens_matched':len(spar_ids),
        'spar_max_abs_difference_from_existing_C_port':max_spar_error,
        'spar_recorded_token_ids_reproduced':len(spar_ids),
        'learned_recorded_tokens_matched':6,'learned_max_abs_difference_from_existing_C_port':max_learned_error,
        'packing_cases':packing_cases,'invalid_configs_rejected':6,
        'synthetic_deterministic_cases':len(goldens),
        'note':'Functional verification on host only. No STM32 timing or throughput measurements.'}
    (OUT/'host_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (OUT/'host_golden_ids.json').write_text(json.dumps(goldens,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
