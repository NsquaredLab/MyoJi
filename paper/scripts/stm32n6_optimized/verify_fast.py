"""Check the optimized kernels against the frozen C reference.

Structural changes (batched codebook search, reordered SPAR features, channel
batching) must stay bit-exact in both build modes. The default build also
replaces the transform and hypot, which is not bit-exact: for those this reports
measured deviation against thresholds and requires every released token ID to be
reproduced. STM32N6_EXACT=1 asserts equality everywhere instead.
"""
import ctypes as ct
import json
import os
import numpy as np
from build import EXACT, ROOT, OUT, FROZEN, sha

# Thresholds for the non-exact transform, a few times the deviation measured on
# the recorded fixture when it was adopted (3.7e-04 and 1.01e-02), so a real
# regression trips them. Every released token ID must still be reproduced; that
# assertion, not the deviation, is the check that matters.
MAX_SPECTRUM_REL = 2e-3
MAX_FEATURE_ABS = 5e-2
from verify_host import load, ptr, run_case, FP


def main():
    path=OUT/'host.dylib';lib=load(path)
    lib.fast_prepare_book.argtypes=[FP,ct.c_uint,FP]
    lib.fast_nearest_code.argtypes=[FP,FP,ct.c_uint]
    lib.fast_nearest_code.restype=ct.c_uint16
    lib.fast_learned_batch.argtypes=[FP,FP,ct.c_uint]
    lib.fast_nearest_code_batch.argtypes=[FP,ct.c_uint,ct.c_uint,FP,ct.c_uint,ct.POINTER(ct.c_uint16)]
    lib.fast_nearest_code_batch.restype=None
    lib.fast_spar_features.argtypes=[FP,FP,ct.c_uint,ct.c_void_p,ct.c_void_p]
    lib.fast_spar_features.restype=None
    lib.fast_spar_spectra.argtypes=[FP,FP]
    lib.fast_spar_spectra.restype=None
    data=np.load(FROZEN/'reference.npz');previous=np.load(FROZEN/'host_outputs.npz')
    books={};batch_cases={}
    for route,dim in [('learned',256),('spar',18)]:
        book=np.ascontiguousarray(data[route+'_book']);packed=np.empty_like(book)
        lib.fast_prepare_book(ptr(book),dim,ptr(packed));books[route]=packed
        # All existing recorded feature vectors, plus exact ties.
        features=previous[route+'_features'].reshape(-1,dim)
        for feature,expected in zip(features,previous[route+'_ids'].ravel()):
            assert lib.fast_nearest_code(ptr(feature),ptr(packed),dim)==expected
        # Channel batching must reproduce the per-vector search exactly, at
        # chunk boundaries as well as inside one chunk.
        batched=np.empty(len(features),np.uint16)
        lib.fast_nearest_code_batch(ptr(features.ravel()),dim,len(features),ptr(packed),dim,
                                    batched.ctypes.data_as(ct.POINTER(ct.c_uint16)))
        np.testing.assert_array_equal(batched,previous[route+'_ids'].ravel())
        batch_cases[route]=int(len(features))
        tied=np.zeros_like(book);lib.fast_prepare_book(ptr(tied),dim,ptr(packed))
        assert lib.fast_nearest_code(ptr(np.ones(dim,np.float32)),ptr(packed),dim)==0
        ties=np.empty(3,np.uint16)
        lib.fast_nearest_code_batch(ptr(np.ones((3,dim),np.float32).ravel()),dim,3,ptr(packed),dim,
                                    ties.ctypes.data_as(ct.POINTER(ct.c_uint16)))
        assert list(ties)==[0,0,0]
        lib.fast_prepare_book(ptr(book),dim,ptr(packed))
    normalized=data['normalized'];matched=0;maximum=0.
    for indices in [np.arange(16),np.arange(304,320)]:
        workspace=np.zeros((320+4*256+1024,16),np.float32)
        history=np.zeros((16,4,4,256),np.float32)
        for end in range(1399,2120,40):
            workspace[:320]=normalized[indices,end-319:end+1].T
            lib.fast_learned_batch(ptr(workspace),ptr(history),16)
            if end<2039:continue
            frame=(end-2039)//40
            values=np.ascontiguousarray(workspace[320+256:320+2*256].T)
            reference=previous['learned_features'][frame,indices]
            maximum=max(maximum,float(np.abs(values-reference).max()))
            np.testing.assert_array_equal(values,reference)
            for c,feature in enumerate(values):
                assert lib.fast_nearest_code(ptr(feature),ptr(books['learned']),256)==previous['learned_ids'][frame,indices[c]]
                matched+=1
    # SPAR path over every recorded frame and the full 320-channel montage.
    members=((ct.c_uint16*4)*320).in_dll(lib,'spar_members')
    masks=((ct.c_uint8*3)*320).in_dll(lib,'spar_mask')
    frozen=np.empty((320,7,32),np.complex64);shipped=np.empty_like(frozen)
    reference=np.empty((320,18),np.float32);reordered=np.empty_like(reference)
    spar_vectors=0;spar_bins=0;spar_ids=0
    worst_bin=0.;worst_feature=0.
    for end in [2039,2079,2119]:
        for c in range(320):
            window=np.ascontiguousarray(normalized[c,end-319:end+1])
            lib.spar_spectra(ptr(window),ptr(frozen[c]))
            lib.fast_spar_spectra(ptr(window),ptr(shipped[c]))
            spar_bins+=shipped[c].size
        if EXACT:
            np.testing.assert_array_equal(shipped,frozen)
        else:
            # Bins 0 and 1 are never read by spar_features, so the shipped
            # transform leaves the packed DC/Nyquist slot zero.
            assert not shipped[:,:,:2].any()
            rel=np.abs(shipped[:,:,2:]-frozen[:,:,2:])/np.maximum(np.abs(frozen[:,:,2:]),1e-20)
            worst_bin=max(worst_bin,float(np.nanmax(rel)))
            assert worst_bin<MAX_SPECTRUM_REL,worst_bin
        for n in [1,23,30,56,64,320]:
            # Reordering is bit-exact in both modes: same spectra in, same features out.
            lib.bench_spar_features(ptr(shipped),ptr(reference),n,members,masks)
            lib.fast_spar_features(ptr(shipped),ptr(reordered),n,members,masks)
            np.testing.assert_array_equal(reordered[:n],reference[:n])
            spar_vectors+=n
        # Whole shipped SPAR path against the released features and token IDs.
        lib.fast_spar_features(ptr(shipped),ptr(reordered),320,members,masks)
        released=previous['spar_features'][(end-2039)//40]
        worst_feature=max(worst_feature,float(np.abs(reordered-released).max()))
        assert EXACT or worst_feature<MAX_FEATURE_ABS,worst_feature
        for c,feature in enumerate(np.ascontiguousarray(reordered)):
            assert lib.fast_nearest_code(ptr(feature),ptr(books['spar']),18)==\
                previous['spar_ids'][(end-2039)//40,c]
            spar_ids+=1

    baseline=load(ROOT/'results/stm32n6_tokenizer_v1/host.dylib')
    cases=[];agreed=0
    for route,n in [(0,1),(0,23),(0,30),(0,36),(0,37),(0,38),(0,39),(0,43),(0,55),(0,56),
                    (0,64),(0,65),(0,320),(0,1024),(1,1),(1,5),(1,9),(1,12),(1,16),(1,17)]:
        expected=run_case(baseline,route,n,frames=3,warmup=16)
        actual=run_case(lib,route,n,frames=3,warmup=16)
        if EXACT or route:
            assert expected==actual,(route,n)
        match=expected==actual
        agreed+=match
        cases.append({'route':route,'channels':n,'identical_to_original_port':match})
    report={'passed':True,'hardware':False,'exact_transform':EXACT,'library_sha256':sha(path),
        'reference_library_sha256':sha(ROOT/'results/stm32n6_tokenizer_v1/host.dylib'),
        'recorded_codebook_assignments_exact':1920,'tie_cases':2,
        'batched_codebook_assignments_exact':batch_cases,'batched_tie_cases':2,
        'reordered_spar_feature_vectors_exact':spar_vectors,
        'spar_spectrum_bins_compared':spar_bins,
        'released_spar_token_ids_reproduced':spar_ids,
        'max_spectrum_relative_deviation':worst_bin,
        'max_feature_absolute_deviation':worst_feature,
        'driver_cases_identical_to_original_port':f'{agreed}/{len(cases)}',
        'batched_recorded_learned_vectors_exact':matched,'max_abs_feature_difference':maximum,
        'synthetic_driver_cases_exact':cases}
    (OUT/'fast_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
