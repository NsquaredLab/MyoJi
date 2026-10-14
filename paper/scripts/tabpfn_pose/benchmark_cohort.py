"""Measure complete 60-coordinate batch-one TabPFN updates on validation EMG.

Run after cohort fitting has finished, with the RTX 4090 otherwise idle.
This is actual single-pose latency, never batch runtime divided by row count.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import h5py
import joblib
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.base import RegressorModelSpecs
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

from ._support import Monitor
from . import run as pilot
from .cohort import continuous_history


def benchmark(args):
    root = args.root
    if not json.loads((root / "results.json").read_text())["complete"]:
        raise ValueError("Finish cohort fitting before measuring latency")
    fixed = json.loads((root / "protocol.json").read_text())
    if pilot.sha256(args.model_path) != fixed["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 4090 CUDA benchmark requires CUDA")
    # Refuse to time through a competing research process. GNOME's remote
    # desktop daemon also creates a CUDA context on this workstation; retain
    # and disclose these existing desktop processes without stopping them.
    compute_apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                                         "--format=csv,noheader"],text=True).strip()
    desktop_apps=[line for line in compute_apps.splitlines()
                  if line.split(",",1)[1].strip()=="/usr/libexec/gnome-remote-desktop-daemon"]
    others=[line for line in compute_apps.splitlines()
            if int(line.split(",")[0])!=os.getpid() and line not in desktop_apps]
    if others:
        raise RuntimeError(f"Another GPU compute process is active; defer latency benchmark: {others}")
    idle_gpu=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used",
                                     "--format=csv,noheader"],text=True).strip()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = args.source / "subject_01"
    if pilot.sha256(source/"features.h5")!=fixed["source_cache_sha256"]["1"]:
        raise ValueError("Frozen benchmark feature cache changed")
    with h5py.File(source / "features.h5") as f:
        book, mask = f["codebook"][:], f["mask"][:]
        inputs = {arm: f["val"][arm][:4] for arm in ["tokens", "continuous"]}
        identities = dict(subjects=f["val/subjects"][:4].tolist(), tasks=f["val/tasks"][:4].tolist(),
                          starts=f["val/starts"][:4].tolist())
    records = []
    with Monitor(root / "resource-usage.jsonl") as monitor, threadpool_limits(limits=4):
        for arm in ["tokens", "continuous"]:
            output = root / f"resource-{arm}.json"
            if output.exists():
                r = json.loads(output.read_text())
                if r["cohort_protocol_sha256"] != pilot.sha256(root / "protocol.json"):
                    raise ValueError("Cannot reuse a benchmark from another protocol")
                records.append(r)
                continue
            folder = root / "subject_01" / arm
            config = json.loads((folder / "config.json").read_text())
            if pilot.sha256(folder / "data.npz") != config["data_sha256"]:
                raise ValueError("Prepared benchmark dataset changed")
            if pilot.sha256(folder / "input_pca.joblib") != config["pca_sha256"]:
                raise ValueError("PCA changed")
            d = dict(np.load(folder / "data.npz"))
            projection = joblib.load(folder / "input_pca.joblib")

            def encode(i):
                x = inputs[arm][i:i+1]
                history = (pilot.spatial_history(x, book, mask, score_start=99)
                           if arm == "tokens" else continuous_history(x, mask, score_start=99))
                return projection.transform(history).astype(np.float32)

            for i in range(4):
                np.testing.assert_allclose(encode(i)[0], d["X_val"][i*50+49], rtol=2e-4, atol=2e-4)
            # Compare identical receiver inputs. A one-row FP32 PCA transform
            # differs slightly from the original batched BLAS transform, and
            # fitted preprocessing can amplify that rounding at bin edges.
            online_queries=np.concatenate([encode(i) for i in range(4)])
            pca_rounding_max=float(np.max(np.abs(online_queries-d["X_val"][[49,99,149,199]])))
            # The 219M-parameter weight set is identical for every scalar task.
            # Use TabPFN's supported ModelSpecs input to retain one physical copy;
            # each estimator still fits its own preprocessing and target context.
            options=dict(device="cuda", n_estimators=8, n_preprocessing_jobs=1,
                         inference_precision=torch.float32, fit_mode="fit_preprocessors")
            independent_validation={}
            for coordinate in [0,59]:
                monitor.phase=f"{arm}: independent parity coordinate {coordinate}"
                independent=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,
                    model_path=args.model_path,random_state=260911+coordinate,**options)
                independent.fit(d["X_train"],d["y_train"][:,coordinate])
                independent_validation[coordinate]=np.asarray(
                    independent.predict(online_queries),dtype=np.float32)
                del independent
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            bootstrap=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,
                model_path=args.model_path,random_state=260911,**options)
            inference_config=bootstrap.get_inference_config()
            initial_criterion=copy.deepcopy(bootstrap.znorm_space_bardist_)
            models, batched_validation = [], []
            torch.cuda.reset_peak_memory_stats()
            for coordinate in range(60):
                monitor.phase = f"{arm}: prepare coordinate {coordinate}/60"
                spec=RegressorModelSpecs(model=bootstrap.model_,
                    architecture_config=copy.deepcopy(bootstrap.configs_[0]),
                    inference_config=copy.deepcopy(inference_config),
                    norm_criterion=copy.deepcopy(initial_criterion))
                model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,
                    model_path=spec,random_state=260911+coordinate,**options)
                model.fit(d["X_train"], d["y_train"][:, coordinate])
                assert model.n_estimators_==8 and model.model_ is bootstrap.model_
                batched_validation.append(np.asarray(model.predict(online_queries),dtype=np.float32))
                models.append(model)
            counts = [sum(p.numel() for p in model.model_.parameters()) for model in models]
            if len(set(counts)) != 1:
                raise ValueError("Expected identical shared pretrained architecture")
            assert len({id(m.model_) for m in models})==1
            assert all(m.executor_.model_caches[0].get(m.devices_[0]) is bootstrap.model_ for m in models)
            # Recheck after all target-specific preparation: sharing weights must
            # not share fitted target state or change either endpoint coordinate.
            shared_parity=[]
            for coordinate,expected in independent_validation.items():
                actual=models[coordinate].predict(online_queries)
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
                shared_parity.append(float(np.max(np.abs(actual-expected))))
            for i in range(4):
                np.testing.assert_array_equal(encode(i),online_queries[i:i+1])

            def predict_pose(i,stage):
                query = encode(i)
                prediction=[]
                for coordinate,m in enumerate(models):
                    monitor.phase=f"{arm}: {stage}, coordinate {coordinate+1}/60"
                    prediction.append(float(np.asarray(m.predict(query)).item()))
                return np.array(prediction,dtype=np.float32)

            monitor.phase = f"{arm}: complete-pose warmup"
            start = time.monotonic()
            warm = predict_pose(0,"warmup")
            torch.cuda.synchronize()
            warmup_seconds = time.monotonic()-start
            batched_validation=np.stack(batched_validation,axis=-1)
            np.testing.assert_allclose(warm,batched_validation[0],rtol=2e-4,atol=2e-3)
            timings, predictions = [], []
            for i in range(4):
                monitor.phase = f"{arm}: measured full pose {i+1}/4"
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred = predict_pose(i,f"measurement {i+1}/4")
                torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
                predictions.append(pred)
                print(json.dumps(dict(stage="latency", arm=arm, input=i, milliseconds=timings[-1])),flush=True)
                # Preserve actual measurements before validation, so a later
                # diagnostic failure cannot discard completed observations.
                pilot.write(root/f"resource-samples-{arm}.json",dict(complete=len(timings)==4,
                    benchmark_source_sha256=pilot.sha256(Path(__file__)),
                    samples_ms=timings,predictions=np.asarray(predictions).tolist(),
                    batched_validation=batched_validation.tolist(),warmup_seconds=warmup_seconds,
                    online_queries=online_queries.tolist(),pca_batch_rounding_max_abs=pca_rounding_max))
                np.testing.assert_allclose(pred,batched_validation[i],rtol=2e-4,atol=2e-3)
            if not np.isfinite(predictions).all():
                raise ValueError("Nonfinite complete-pose prediction")
            np.testing.assert_allclose(predictions[0], warm, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(predictions,batched_validation,rtol=2e-4,atol=2e-3)
            record = dict(arm=arm, complete=True, cohort_protocol_sha256=pilot.sha256(root / "protocol.json"),
                parameter_count=counts[0], unique_pretrained_parameters=counts[0],
                loaded_parameter_copies=1, coordinate_regressor_objects=len(models),
                sum_loaded_parameter_elements=counts[0], sum_referenced_parameter_elements=sum(counts),
                shared_weight_api="tabpfn.base.RegressorModelSpecs", shared_weight_parity_max_abs_mm=max(shared_parity),
                parameter_scope="One shared frozen TabPFN backbone/heads; identical weights reused across 60 scalar regressions. "
                                "Excludes PCA coefficients and training-context buffers, recorded separately.",
                pca_coefficient_elements=int(projection.components_.size+projection.mean_.size),
                pca_tensor_bytes=int(projection.components_.nbytes+projection.mean_.nbytes),
                training_table_bytes=int(d["X_train"].nbytes+d["y_train"].nbytes),
                checkpoint_sha256=fixed["checkpoint_sha256"], warmup_calls=1, measured_calls=4,
                warmup_seconds=warmup_seconds, samples_ms=timings,
                latency_mean_ms=float(np.mean(timings)), latency_sd_ms=float(np.std(timings,ddof=1)),
                p50_ms=float(np.median(timings)), p95_ms=float(np.quantile(timings,.95)),
                updates_per_second=float(1000/np.median(timings)),
                peak_process_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                validation_predictions=np.asarray(predictions).tolist(), validation_identities=identities,
                query_batch_parity_max_abs_mm=float(np.max(np.abs(np.asarray(predictions)-batched_validation))),
                pca_batch_rounding_max_abs=pca_rounding_max,
                batch_parity_input="Identical one-row PCA outputs supplied to batched and single-query receiver calls.",
                batch_size=1, n_estimators=8, coordinate_regressors=60, seed=260911, subject=1,
                precision="FP32; TF32 disabled", fit_mode="fit_preprocessors", training_kv_cache=False,
                resource_scope="Complete 60-coordinate pose, including CPU causal summaries, frozen codebook lookup for tokens, "
                    "training-fitted PCA, all 60 scalar API predictions and their transfers. Excludes fitting/loading, "
                    "upstream SPAR extraction/standardization/token assignment. Input is one full 100-frame EMG context; "
                    "preprocessed training tables are retained; training transformer context is recomputed.")
            pilot.write(output,record)
            records.append(record)
            del models, model, spec, bootstrap, initial_criterion, inference_config, d, projection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    pilot.write(root / "resource-benchmark.json",dict(complete=True,models=records,
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        tabpfn_version=importlib.metadata.version("tabpfn"),platform=platform.platform(),cpu_threads=4,
        benchmark_source_sha256=pilot.sha256(Path(__file__)),
        source_cache_sha256=fixed["source_cache_sha256"]["1"],
        cohort_protocol_sha256=pilot.sha256(root / "protocol.json"),
        retained_desktop_compute_processes=desktop_apps, prebenchmark_gpu_usage=idle_gpu,
        caution="Same RTX 4090 as previous main-figure costs, later PyTorch 2.14 runtime. TabPFN uses four full-pose calls "
                "after one warmup, with CPU summaries/PCA/API transfers included. Previous models use 48 GPU-resident "
                "decoder calls after 12 warmups under PyTorch 2.13. These timing-protocol differences are disclosed."))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--model-path",required=True,type=Path)
    benchmark(p.parse_args())


if __name__ == "__main__":
    main()
