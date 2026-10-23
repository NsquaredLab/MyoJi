"""Audit saved cohort predictions and report the three fixed primary contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import run as pilot
from .cohort import ARMS, SEEDS, SUBJECTS


def analyze(root, source):
    # Recompute complete pose errors and check every target against the existing V20 run.
    result=json.loads((root/"results.json").read_text())
    assert result["complete"] and result["fits"]==54
    assert result["protocol_sha256"]==pilot.sha256(root/"protocol.json")
    assert len(result["scores"])==54
    fixed=json.loads((root/"protocol.json").read_text())
    runtimes=[]
    for subject in SUBJECTS:
        for arm in ARMS:
            config=json.loads((root/f"subject_{subject:02d}/{arm}/config.json").read_text())
            assert config["cohort_protocol_sha256"]==result["protocol_sha256"]
            assert config["runtime"]["checkpoint_sha256"]==fixed["checkpoint_sha256"]
            runtimes.append(config["runtime"])
    for key in ["torch","tabpfn","sklearn","gpu","code_sha256"]:
        assert len({r[key] for r in runtimes})==1,key
    expected_keys={(s,a,seed) for s in SUBJECTS for a in ARMS for seed in SEEDS}
    assert {(r["subject"],r["arm"],r["seed"]) for r in result["scores"]}==expected_keys
    reference_scores={(r["subject"],r["arm"],r["seed"]):r for r in result["references"]}
    assert set(reference_scores)==expected_keys and len(result["references"])==54
    for row in result["scores"]:
        s,arm,seed=row["subject"],row["arm"],row["seed"]
        path=root/f"subject_{s:02d}/{arm}/predictions_test_{seed}.npz"
        assert pilot.sha256(path)==row["predictions_sha256"]
        reference=source/f"subject_{s:02d}/fits/{arm}_{seed}/test_predictions.npz"
        with np.load(path) as z,np.load(reference) as ref:
            np.testing.assert_array_equal(z["target"],ref["target"])
            np.testing.assert_array_equal(z["tasks"],ref["tasks"])
            np.testing.assert_allclose(pilot.error_mm(z["prediction"],z["target"]),row["error_mm"],rtol=0,atol=2e-6)
            np.testing.assert_allclose(pilot.error_mm(ref["prediction"],ref["target"]),
                reference_scores[s,arm,seed]["error_mm"],rtol=0,atol=2e-6)
    for method,collection in [("tabpfn",result["scores"]),("v20",result["references"])]:
        for arm in ARMS:
            for subject in SUBJECTS:
                values=[r["error_mm"] for r in collection if r["arm"]==arm and r["subject"]==subject]
                assert len(values)==3
                np.testing.assert_allclose(np.mean(values),result["summary"][method][arm]["participant_means_mm"][str(subject)],rtol=0,atol=1e-12)
    controls={}
    control_hashes={}
    for arm in ARMS:
        by_method={name:[] for name in ["ridge","training_mean"]}
        for subject in SUBJECTS:
            folder=root/f"subject_{subject:02d}/{arm}"
            stored=json.loads((folder/"baselines.json").read_text())["test"]
            reference=source/f"subject_{subject:02d}/fits/{arm}_{SEEDS[0]}/test_predictions.npz"
            path=folder/"baselines_test.npz"
            control_hashes[str(path.relative_to(root))]=pilot.sha256(path)
            with np.load(path) as z,np.load(reference) as ref:
                np.testing.assert_array_equal(z["target"],ref["target"])
                np.testing.assert_array_equal(z["tasks"],ref["tasks"])
                for name,key in [("ridge","ridge"),("training_mean","mean")]:
                    score=pilot.error_mm(z[key],z["target"])
                    np.testing.assert_allclose(score,stored[name]["error_mm"],rtol=0,atol=2e-6)
                    by_method[name].append(score)
        controls[arm]={name:dict(mean_mm=float(np.mean(values)),participant_sd_mm=float(np.std(values,ddof=1)),
            participant_means_mm=dict(zip(map(str,SUBJECTS),values))) for name,values in by_method.items()}
    from paper.common import hdemg_figure_statistics as stats
    definitions=[("tokens_vs_v20_tokens","tabpfn","tokens","v20","tokens"),
                 ("features_vs_v20_features","tabpfn","continuous","v20","continuous"),
                 ("tokens_vs_features","tabpfn","tokens","tabpfn","continuous")]
    contrasts=[]
    for label,method_a,arm_a,method_b,arm_b in definitions:
        a=np.array([result["summary"][method_a][arm_a]["participant_means_mm"][str(s)] for s in SUBJECTS])
        b=np.array([result["summary"][method_b][arm_b]["participant_means_mm"][str(s)] for s in SUBJECTS])
        effect,p,d=stats.exact_paired_test(a,b)
        contrasts.append(dict(name=label,alternative=[method_a,arm_a],reference=[method_b,arm_b],
            mean_difference_mm=effect,pointwise95_mm=stats.exact_location_interval(d),p_raw=p,
            participant_differences_mm=dict(zip(map(str,SUBJECTS),d)),
            lower_error_participants=int(sum(a<b)),n_participants=9))
    for c,p in zip(contrasts,stats.holm_adjust([c["p_raw"] for c in contrasts])):
        c.update(p_holm=p,significant=p<.05)
    report=dict(complete=True,protocol_sha256=result["protocol_sha256"],results_sha256=pilot.sha256(root/"results.json"),
        method="Two-sided exact participant sign-flip with inclusive ties; Holm across three declared primary contrasts. "
               "Average algorithm seeds within participant before inference; pointwise intervals invert the univariate location test.",
        comparisons=contrasts,scope=result["scope"],controls=controls,control_prediction_sha256=control_hashes,
        control_scope="Descriptive controls fixed in preparation: Ridge alpha=1 on the identical PCA table, and training-target mean. "
                      "One deterministic fit per participant/input, no seed replication or additional significance tests.")
    pilot.write(root/"primary-statistics.json",report)
    lines=["# TabPFN 3.5: all nine HD-EMG participants","",result["scope"],"",
        "Frozen participant-specific SPAR tokenizers; unchanged V20 splits and endpoints. Both inputs use the same causal summaries, "
        "128 PCA components fitted only on training data, eight ensemble members and 60 scalar regressors. Three algorithm seeds per participant/input.","",
        "| Decoder | Input | Mean 3D error (mm) | Participant SD (mm) |",
        "|---|---|---:|---:|"]
    for method in ["tabpfn","v20"]:
        for arm in ARMS:
            r=result["summary"][method][arm]
            lines.append(f"| {'TabPFN 3.5' if method=='tabpfn' else 'Proposed spatial CNN + TCN'} | SPAR {arm} | {r['mean_mm']:.3f} | {r['participant_sd_mm']:.3f} |")
    lines += ["","## Fixed simple controls","",report["control_scope"],"",
        "| Control | Input | Mean 3D error (mm) | Participant SD (mm) |",
        "|---|---|---:|---:|"]
    for name in ["ridge","training_mean"]:
        for arm in ARMS:
            r=controls[arm][name]
            lines.append(f"| {name} | SPAR {arm} | {r['mean_mm']:.3f} | {r['participant_sd_mm']:.3f} |")
    lines += ["","## Declared primary comparisons","",report["method"],"",
        "| Comparison (alternative minus reference) | Effect (mm) | Pointwise 95% interval (mm) | Raw p | Holm p | Lower error |",
        "|---|---:|---|---:|---:|---:|"]
    for c in contrasts:
        lo,hi=c["pointwise95_mm"]
        lines.append(f"| {c['name']} | {c['mean_difference_mm']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {c['p_raw']:.6f} | {c['p_holm']:.6f} | {c['lower_error_participants']}/9 |")
    lines += ["","These three-test Holm values are separate from the main figure's expanded 76-test max-T correction. "
        "Non-rejection does not establish equivalence. The nine participants, not 27 seeds or thousands of poses, are the inference units.","",
        "## Per-participant seed means","","| Participant | TabPFN tokens | TabPFN continuous | V20 tokens | V20 continuous |",
        "|---|---:|---:|---:|---:|"]
    for s in SUBJECTS:
        vals=[result["summary"][m][a]["participant_means_mm"][str(s)] for m in ["tabpfn","v20"] for a in ARMS]
        lines.append(f"| {s} | "+" | ".join(f"{v:.3f}" for v in vals)+" |")
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    args=p.parse_args()
    analyze(args.root,args.source)


if __name__=="__main__":
    main()
