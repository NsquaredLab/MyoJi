"""Verify the complete C ports against recorded inputs and PyTorch outputs."""
from __future__ import annotations
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from prepare import ROOT, OUT, SRC, sha, write

FLAGS = ["-O3","-std=c11","-fno-fast-math","-ffp-contract=off","-Wall","-Wextra","-Werror"]
FP = ct.POINTER(ct.c_float)

def ptr(x): return x.ctypes.data_as(FP)

def build():
    compiler = shutil.which("clang") or shutil.which("cc")
    asm = OUT/"host_weights.S"
    prefix = "_" if sys.platform == "darwin" else ""
    section = ".section __TEXT,__const" if sys.platform == "darwin" else ".section .rodata"
    asm.write_text(f'{section}\n.p2align 2\n.globl {prefix}learned_weights\n{prefix}learned_weights:\n.incbin "{OUT / "learned_weights.f32"}"\n')
    libpath = OUT/("raw_tokenizers.dylib" if sys.platform == "darwin" else "raw_tokenizers.so")
    command = [compiler,*FLAGS,"-shared","-fPIC",f"-I{SRC}",f"-I{OUT}",
               *(str(SRC/p) for p in ["common.c","learned.c","spar.c"]),str(OUT/"constants.c"),str(asm),"-lm","-o",str(libpath)]
    result = subprocess.run(command,capture_output=True,text=True)
    (OUT/"host_compile.log").write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(result.stderr)
    lib = ct.CDLL(str(libpath))
    for name,count in [("calibrate_channel",3),("normalize_samples",None),("learned_step",3),("spar_spectra",2),("spar_features",2)]:
        method = getattr(lib,name)
        method.argtypes = [FP]*count if count else [FP,FP,ct.c_uint,ct.c_float,ct.c_float]
        method.restype = None
    lib.nearest_code.argtypes = [FP,FP,ct.c_uint]
    lib.nearest_code.restype = ct.c_uint16
    lib.matrix_log_entries.argtypes = [FP,FP]
    lib.matrix_log_entries.restype = None
    return lib,command

def error(a,b):
    delta = np.abs(a-b)
    return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),p99_abs=float(np.quantile(delta,.99)))

def main():
    lib,command = build()
    ref = np.load(OUT/"reference.npz")
    raw = np.fromfile(OUT/"raw_320x2120.f32",dtype="<f4").reshape(320,2120)
    stats = np.empty((320,2),np.float32)
    normalized = np.empty_like(raw)
    for c in range(320):
        lib.calibrate_channel(ptr(raw[c]),ptr(stats[c,:1]),ptr(stats[c,1:]))
        lib.normalize_samples(ptr(raw[c]),ptr(normalized[c]),2120,float(stats[c,0]),float(stats[c,1]))
    np.testing.assert_array_equal(stats[:,0],ref["centre"][:,0])
    np.testing.assert_array_equal(stats[:,1],ref["scale"][:,0])
    np.testing.assert_array_equal(normalized,ref["normalized"])
    print("C calibration and all 678400 normalized raw samples exactly match Torch.",flush=True)
    # Independent matrix-log test, including nontrivial complex phases and
    # ill-conditioned positive matrices. No EMG/codebook data select cases.
    rng = np.random.default_rng(20260914)
    matrix_errors = []
    for case in range(100):
        z = rng.standard_normal((4,4))+1j*rng.standard_normal((4,4))
        q,_ = np.linalg.qr(z)
        eig = np.geomspace(10.**(-(case%7)),1.,4)
        a = (q @ np.diag(eig) @ q.conj().T).astype(np.complex64)
        a = (a+a.conj().T)*np.float32(.5)
        expected_a = a.astype(np.complex128)+np.eye(4)*(1.e-8*max(a.diagonal().real.max(),1.e-12))
        d,v = np.linalg.eigh(expected_a)
        log = (v*np.log(d.clip(1.e-12)))@v.conj().T
        expected = log[0,[0,2,3]].real
        actual = np.empty(3,np.float32)
        lib.matrix_log_entries(ptr(a),ptr(actual))
        matrix_errors.append(float(np.max(np.abs(expected-actual))))
    features = np.empty((3,320,18),np.float32)
    spectra = np.empty((320,7,32),np.complex64)
    spar_ids = np.empty((3,320),np.uint16)
    t0 = time.monotonic()
    for f,end in enumerate([2039,2079,2119]):
        for c in range(320):
            window = np.ascontiguousarray(normalized[c,end-319:end+1])
            lib.spar_spectra(ptr(window),ptr(spectra[c]))
        lib.spar_features(ptr(spectra),ptr(features[f]))
        for c in range(320): spar_ids[f,c] = lib.nearest_code(ptr(features[f,c]),ptr(ref["spar_book"]),18)
    spar_seconds = time.monotonic()-t0
    print("C SPAR:",error(features,ref["spar_features"]),"ID disagreements",int((spar_ids!=ref["spar_ids"]).sum()),flush=True)
    learned_features = np.empty((3,320,256),np.float32)
    learned_ids = np.empty((3,320),np.uint16)
    t0 = time.monotonic()
    for c in range(320):
        history = np.zeros((4,4,256),np.float32)
        value = np.empty(256,np.float32)
        for end in range(1399,2120,40):
            window = np.ascontiguousarray(normalized[c,end-319:end+1])
            lib.learned_step(ptr(window),ptr(history),ptr(value))
            if end >= 2039:
                f = (end-2039)//40
                learned_features[f,c] = value
                learned_ids[f,c] = lib.nearest_code(ptr(value),ptr(ref["learned_book"]),256)
        if (c+1)%64 == 0: print(f"C learned encoder: {c+1}/320 channels checked",flush=True)
    learned_seconds = time.monotonic()-t0
    record = dict(schema=1,calibration_exact=True,normalized_samples_exact=678400,
        spar=dict(vectors=960,token_mismatches=int((spar_ids!=ref["spar_ids"]).sum()),
                  mismatch_indices=np.argwhere(spar_ids!=ref["spar_ids"]).tolist(),features=error(features,ref["spar_features"])),
        learned=dict(vectors=960,token_mismatches=int((learned_ids!=ref["learned_ids"]).sum()),
                  mismatch_indices=np.argwhere(learned_ids!=ref["learned_ids"]).tolist(),features=error(learned_features,ref["learned_features"])),
        matrix_log=dict(cases=100,max_abs_error=float(max(matrix_errors)),errors=matrix_errors,
                        reference="NumPy complex128 Hermitian eigh on the same float32 matrices; condition numbers 1..1e6"),
        wall_seconds_including_python_driver=dict(spar=spar_seconds,learned=learned_seconds),
        timing_note="Host verification run duration, not a controlled throughput benchmark or MCU latency",
        compiler_command=command,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in SRC.glob("*") if p.is_file()})
    np.savez(OUT/"host_outputs.npz",calibration=stats,spar_features=features,spar_ids=spar_ids,
             learned_features=learned_features,learned_ids=learned_ids)
    record["outputs_sha256"] = sha(OUT/"host_outputs.npz")
    write(OUT/"host_receipt.json",record)
    print(json.dumps({k:v for k,v in record.items() if k not in ("matrix_log","compiler_command","source_sha256")}),flush=True)

if __name__ == "__main__": main()
