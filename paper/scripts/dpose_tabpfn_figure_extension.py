"""Add the completed fixed TabPFN cohort to the existing SPAR main figure.

Keep the previous 66 participant contrasts and extend the same joint max-T
family with the ten comparisons declared before the nine-person TabPFN run.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np

import dpose_spar_figure_extension as spar

CAMPAIGN = "tabpfn_pose_all9_20260921"
TOKEN = (CAMPAIGN, "tokens")
FEATURE = (CAMPAIGN, "continuous")


def extend(b, root, rows, formats, benchmark, analysis):
    result = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    cost = json.loads((root / "resource-benchmark.json").read_text())
    assert result["complete"] and result["fits"] == 54 and result["coordinate_regressions"] == 3240
    assert cost["complete"] and result["subjects"] == spar.PEOPLE and result["seeds"] == spar.SEEDS
    assert result["protocol_sha256"] == b.resources.digest(root / "protocol.json") == cost["cohort_protocol_sha256"]
    assert protocol["statistics"]["main_figure_family_size"] == 76
    assert cost["source_cache_sha256"] == protocol["source_cache_sha256"]["1"]
    benchmark_source=b.ROOT/"scripts/tabpfn_pose/benchmark_cohort.py"
    assert b.resources.digest(benchmark_source)==cost["benchmark_source_sha256"]
    sources = {str(p.relative_to(b.ROOT)): b.resources.digest(p) for p in
               [root / "results.json", root / "protocol.json", root / "resource-benchmark.json",benchmark_source]}
    by_key = {(r["campaign"], r["arm"]): r for r in rows}
    task_means = {}
    for arm in ["tokens", "continuous"]:
        resource, = [r for r in cost["models"] if r["arm"] == arm]
        assert resource["complete"] and resource["checkpoint_sha256"]==protocol["checkpoint_sha256"]
        assert resource["cohort_protocol_sha256"]==result["protocol_sha256"]
        assert resource["measured_calls"] == len(resource["samples_ms"]) == 4
        assert resource["warmup_calls"]==1 and resource["batch_size"]==1
        assert resource["parameter_count"] > 0 and resource["coordinate_regressors"] == 60
        assert resource["loaded_parameter_copies"] == 1 and resource["coordinate_regressor_objects"] == 60
        assert resource["sum_loaded_parameter_elements"] == resource["parameter_count"]
        np.testing.assert_allclose(np.mean(resource["samples_ms"]),resource["latency_mean_ms"],rtol=0,atol=1e-9)
        np.testing.assert_allclose(np.std(resource["samples_ms"],ddof=1),resource["latency_sd_ms"],rtol=0,atol=1e-9)
        values = {str(s): {} for s in spar.PEOPLE}
        tasks = np.empty((9, 3, 8))
        scores = [r for r in result["scores"] if r["arm"] == arm]
        assert len(scores) == 27
        for score in scores:
            subject, seed = score["subject"], score["seed"]
            path = root / f"subject_{subject:02d}/{arm}/predictions_test_{seed}.npz"
            reference = b.ROOT / f"results/{spar.CAMPAIGN}/subject_{subject:02d}/fits/{arm}_{seed}/test_predictions.npz"
            assert b.resources.digest(path) == score["predictions_sha256"]
            with np.load(path) as z, np.load(reference) as ref:
                np.testing.assert_array_equal(z["target"], ref["target"])
                np.testing.assert_array_equal(z["tasks"], ref["tasks"])
                distance = np.linalg.norm((z["prediction"]-z["target"]).reshape(16,50,20,3),axis=-1)
                np.testing.assert_allclose(distance.mean(),score["error_mm"],rtol=0,atol=2e-6)
                for i, task in enumerate(spar.TASKS):
                    assert sum(z["tasks"] == task) == 2
                    task_error=float(distance[z["tasks"] == task].mean())
                    np.testing.assert_allclose(task_error,score["per_task_mm"][str(task)],rtol=0,atol=2e-6)
                    tasks[subject-1,spar.SEEDS.index(seed),i]=task_error
            values[str(subject)][str(seed)]=score["error_mm"]
            sources[str(path.relative_to(b.ROOT))]=score["predictions_sha256"]
            config_path=root / f"subject_{subject:02d}/{arm}/config.json"
            config=json.loads(config_path.read_text())
            assert config["source_cache_sha256"] == protocol["source_cache_sha256"][str(subject)]
            assert config["cohort_protocol_sha256"] == result["protocol_sha256"]
            assert config["components"] == 128 and config["n_estimators"] == 8
            assert config["runtime"]["checkpoint_sha256"] == protocol["checkpoint_sha256"]
            assert config["runtime"]["tabpfn"] == cost["tabpfn_version"]
            assert config["runtime"]["torch"] == cost["torch_version"]
            sources[str(config_path.relative_to(b.ROOT))]=b.resources.digest(config_path)
        means={s:float(np.mean(list(v.values()))) for s,v in values.items()}
        np.testing.assert_allclose(np.mean(list(means.values())), result["summary"]["tabpfn"][arm]["mean_mm"],rtol=0,atol=1e-12)
        row=dict(campaign=CAMPAIGN,arm=arm,representation=arm,family="TabPFN",panel="B",
            label="TabPFN 3.5 + SPAR "+("tokens" if arm=="tokens" else "features"),
            plot_label="SPAR "+("tokens" if arm=="tokens" else "features"),
            experiment_group_id="tabpfn",experiment_group="TabPFN 3.5 (Prior Labs, 2026)",
            error_mean=float(np.mean(list(means.values()))),error_participant_means=means,
            error_participant_seed_values=values,error_n_participants=9,error_n_seeds=3,
            error_evaluation="chronological_test",error_units="mm",error_precision="TabPFN 9.0.0 default auto inference precision",
            error_scope=protocol["scope"],model_parameters_million=resource["parameter_count"]/1e6,
            timing_n_calls=4,resource_campaign=CAMPAIGN,
            resource_session={k:cost[k] for k in ["gpu","cpu_threads","torch_version"]},
            model_literature=dict(citation="Prior Labs (2026), TabPFN 3.5 release",
                url="https://huggingface.co/Prior-Labs/tabpfn_3_5",
                relationship="Frozen tabular foundation model; causal SPAR summaries and training-only PCA, 60 scalar regressors with eight ensemble members each."),
            **{k:v for k,v in resource.items() if k!="arm"})
        rows.append(row)
        task_means[arm]=tasks.mean(axis=1)
    b.GROUPS.insert(-1,dict(id="tabpfn",label="TabPFN 3.5 (Prior Labs, 2026)",variants=[TOKEN,FEATURE]))
    absolute=np.abs(task_means["tokens"]-task_means["continuous"]).mean(axis=1)
    summary=dict(decoder="TabPFN",table_label="TabPFN 3.5",mean_absolute_change_mm=float(absolute.mean()),
        sd_across_participants_mm=float(absolute.std(ddof=1)),n_participants=9,
        participant_means_mm={str(s):float(v) for s,v in zip(spar.PEOPLE,absolute)})
    b.ABSOLUTE_ANALYSIS["summaries"].append(summary)
    b.REPRESENTATION_PAIRS.insert(3,("TabPFN",TOKEN,FEATURE))
    b.REPRESENTATION_TABLE_LABELS["TabPFN"]="TabPFN 3.5"
    benchmark["tabpfn_extension"]=cost
    by_key={(r["campaign"],r["arm"]):r for r in rows}
    summaries={s["decoder"]:s for s in b.ABSOLUTE_ANALYSIS["summaries"]}
    definitions=[]
    for ref in [b.L1_CNN_KEY,b.MATCHED_CNN_KEY,b.tbme.KEY]:
        for key in [TOKEN,FEATURE]:
            definitions.append(dict(kind="cnn_reference",alternative=list(key),reference=list(ref)))
    definitions += [dict(kind="tabpfn_vs_v20",alternative=list(TOKEN),reference=list(spar.TOKEN)),
                    dict(kind="tabpfn_vs_v20",alternative=list(FEATURE),reference=list(spar.FEATURE)),
                    dict(kind="tabpfn_representation",alternative=list(TOKEN),reference=list(FEATURE)),
                    dict(kind="absolute_error_change_spar",alternative=["absolute_error_change","TabPFN"],
                         reference=["absolute_error_change","SPAR"])]
    def vals(key):
        v=(summaries[key[1]]["participant_means_mm"] if key[0]=="absolute_error_change" else by_key[tuple(key)]["error_participant_means"])
        return np.array([v.get(str(s),v.get(s)) for s in spar.PEOPLE])
    def label(key):
        return b.REPRESENTATION_TABLE_LABELS[key[1]]+" absolute error change" if key[0]=="absolute_error_change" else by_key[tuple(key)]["label"]
    comparisons=copy.deepcopy(analysis["groups"][0]["comparisons"])
    assert len(comparisons)==66
    for index,definition in enumerate(definitions):
        alt,ref=definition["alternative"],definition["reference"]
        d=vals(alt)-vals(ref)
        pair=dict(contrast_kind=definition["kind"],alternative=alt,reference=ref,a=alt,b=ref,
            i=200+index,j=0,label_a=label(alt),label_b=label(ref),
            effect_alternative_minus_reference_mm=float(d.mean()),mean_difference_a_minus_b=float(d.mean()),
            participant_differences={str(s):float(v) for s,v in zip(spar.PEOPLE,d)},
            effect_ci95_pointwise_mm=b.stats.exact_location_interval(d),n_participants=9,n_seeds_per_participant=3,
            permutations=512,leave_one_participant_out_effects_mm={str(s):float(np.delete(d,s-1).mean()) for s in spar.PEOPLE})
        if definition["kind"]!="cnn_reference":
            pair.update(contrast_id=f"tabpfn_extension_{index:02d}",display_label="TabPFN" if index==9 else alt[1])
        comparisons.append(pair)
    matrix=np.array([[p["participant_differences"][str(s)] for p in comparisons] for s in spar.PEOPLE])
    assert matrix.shape==(9,76)
    observed,raw,adjusted,null=b.stats.exact_max_t(matrix)
    changes=[]
    for i,pair in enumerate(comparisons):
        if i<66:
            assert raw[i]==pair["p_raw"] and adjusted[i]>=pair["p_adjusted"]
            pair["p_adjusted_previous_66"]=pair["p_adjusted"]
            if adjusted[i]!=pair["p_adjusted"]:
                changes.append(dict(index=i,label_a=pair["label_a"],label_b=pair["label_b"],
                    alternative=pair["alternative"],reference=pair["reference"],
                    previous_p=pair["p_adjusted"],current_p=float(adjusted[i]),
                    significance_changed=bool((adjusted[i]<.05)!=(pair["p_adjusted"]<.05))))
        pair.update(t_observed=float(observed[i]),p_raw=float(raw[i]),p_adjusted=float(adjusted[i]),
                    p_maxT=float(adjusted[i]),significant=bool(adjusted[i]<.05))
    analysis["retained_66_adjusted_p_changes"]=analysis.get("previous_adjusted_p_changes",[])
    analysis["retained_66_significance_changes"]=analysis.get("previous_significance_changes",[])
    analysis["retained_66_exclusions"]=copy.deepcopy(analysis["excluded"])
    analysis["excluded"]["alternative_vs_alternative"]=(
        "Only the representation, signed-gap, absolute-change and SPAR/TabPFN decoder contrasts "
        "explicitly enumerated in the retained and extended protocols are included alongside CNN-reference tests. "
        "Other alternative-to-alternative comparisons, including the descriptive ridge/mean controls, are excluded.")
    analysis["groups"][0].update(comparisons=comparisons,family_size=76,
        label="CNN references, TabPFN versus SPAR decoders, representation gaps and absolute error changes")
    analysis.update(family_size=76,retained_66_protocol=analysis["comparison_protocol"],
        previous_adjusted_p_changes=changes,
        previous_significance_changes=[c for c in changes if c["significance_changed"]],
        comparison_protocol=dict(cohort_protocol=str((root/"protocol.json").relative_to(b.ROOT)),
            cohort_protocol_sha256=result["protocol_sha256"],retained_contrasts=66,additions=definitions,family_size=76,
            scope="TabPFN extension fixed after the Subject-1 pilot and before this all-nine run; all data remain exploratory."),
        maxT_reference_distribution=null.tolist(),significant_comparisons=sum(p["significant"] for p in comparisons),
        minimum_observed_adjusted_p=float(min(adjusted)),tabpfn_absolute_summary=summary,
        tabpfn_task_means={k:v.tolist() for k,v in task_means.items()},
        method="Two-sided exact synchronized participant sign-flip studentized max-T; 512 sign patterns, 76 contrasts. "
               "Requires independent participants and joint null sign symmetry. Previous 66 contrast vectors retained. "
               "Ten TabPFN additions declared before this cohort run; the broader adaptive research history is not corrected.")
    analysis["sources"].update(sources)
    assert len(rows)==21 and sum(r["error_n_participants"]*r["error_n_seeds"] for r in rows)==567
    return rows,formats,benchmark,analysis


def caption(supplement=False):
    if supplement:
        return ("Smaller DPose decoder baselines. Same nine participants, three seeds, chronological test endpoints and "
            "20 wrist-relative landmarks as the main figure. In the accuracy column, gray points average seeds within participant; "
            "colored symbols average participants. The dashed line is the retrained TBME CNN. Parameters exclude upstream processing and buffers. "
            "Latency symbols and whiskers show the mean and sample SD of 48 FP32, batch-one complete-pose RTX 4090 decoder calls "
            "after 12 warmups, using one fitted participant/seed; inputs are GPU-resident. "
            "Branches compare with the adapted CNN outside this view, using the main figure's exploratory 76-contrast, "
            "two-sided exact max-T family and 512 synchronized participant sign patterns. The procedure assumes independent "
            "participants and joint null sign symmetry. * p<0.05, ** p<0.01, *** p<0.001; n.s. denotes non-rejection, not equivalence.")
    return ("HD-EMG accuracy, receiver cost and quantization effects. (a) Unfiltered mean Euclidean error over 20 wrist-relative "
        "landmarks on the same personalized, same-session chronological test endpoints. In the accuracy column, gray points average three seeds within "
        "each of nine participants; colored symbols average participants. The adapted CNN uses L1 loss and weight averaging; "
        "the dashed reference is the TBME CNN retrained on this split. Matched custom decoders share their stated recipe. "
        "Qwen3 uses frozen pretrained/random backbones and a bypass control. TabPFN 3.5 uses frozen pretrained weights, "
        "training-only 128-component PCA of 1/5/25/100-frame causal SPAR summaries, and 60 scalar regressors with eight ensemble "
        "members each. Token and continuous arms use identical preparation settings, with PCA fitted separately; SPAR codebooks "
        "are frozen from participant training blocks. The learned encoder saw these recordings during unlabeled pretraining. "
        "Receiver parameters include frozen backbones; TabPFN counts one shared weight set, excluding PCA and retained training "
        "context. Latency symbols and whiskers show the mean and sample SD of batch-one complete-pose calls for one fitted "
        "participant/seed, using FP32 on the same RTX 4090: existing decoders have 48 GPU-resident calls after 12 warmups "
        "(PyTorch 2.13); TabPFN has four complete-pose calls after one warmup, including CPU summaries/PCA and API transfers "
        "(PyTorch 2.14). TabPFN retains preprocessing but recomputes transformer context for 3,200 training rows; fitting is excluded. "
        "All timings exclude upstream feature extraction/tokenization. (b) Array payload for 320 channels in decimal MB/s: uint16 tokens at "
        "50 Hz/channel, int16 raw EMG at 2,048 Hz, and dense FP32 learned/SPAR features. Headers and model/context transfer "
        "are excluded. FP32 is the plotted feature-storage format; matched/Qwen fitting used FP16-rounded learned-feature caches. "
        "(c) Participant means of absolute task-level token-feature error changes, averaging equally across eight tasks after "
        "averaging two windows and three seeds per task; lower means a smaller change, not necessarily lower error. Branches compare with the adapted "
        "CNN in (a) and proposed SPAR in (c). All stars and n.s. use one exploratory 76-contrast max-T family, retaining the "
        "previous 66 and adding ten declared TabPFN contrasts. The 512 synchronized sign patterns require independent participants "
        "and joint null sign symmetry; correction does not cover the full adaptive research history. * p<0.05, ** p<0.01, "
        "*** p<0.001; non-rejection is not equivalence.")


def export(b,root,rows,formats,benchmark,analysis,supplement=False):
    groups=b.EARLY_GROUPS if supplement else b.GROUPS
    selected=b.selected_rows(rows,groups)
    stem=b.SUPPLEMENT_STEM if supplement else b.STEM
    fig=b.make_figure(selected,formats,analysis,supplement=supplement)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for item in [*fig.texts,*[t for ax in fig.axes for t in ax.get_xticklabels()+ax.get_yticklabels()+[ax.xaxis.label]]]:
        if item.get_visible() and item.get_text():
            box=item.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,item.get_text()
    paths=[b.PDFS/f"{stem}.pdf",b.OUT/f"{stem}.svg",b.OUT/f"{stem}.png"]
    for path in paths:fig.savefig(path,dpi=400,facecolor="white")
    b.plt.close(fig)
    prose=caption(supplement)
    (b.OUT/f"{stem}.md").write_text(prose+"\n\nReproduce: `python scripts/build_dpose_combined.py`.\n")
    tex=b.latex_caption(prose).replace("p<0.05",r"$p<0.05$").replace("p<0.01",r"$p<0.01$").replace("p<0.001",r"$p<0.001$")
    (b.OUT/f"{stem}.tex").write_text("\\begin{figure*}[t]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{output/pdf/{stem}.pdf}}\n\\caption{{{tex}}}\n"
        f"\\label{{fig:{stem}}}\n\\end{{figure*}}\n")
    record=dict(complete=True,variant_count=len(selected),decoder_rows=selected,
        all_source_test_participant_seed_cells_verified=567,test_participant_seed_cells_verified=27*len(selected),
        timed_calls_verified=sum(r.get("timing_n_calls",48) for r in selected),
        latency_axis_unit="s" if b.LATENCY_SCALE==.001 else "ms",
        latency_axis_scale_from_ms=b.LATENCY_SCALE,
        figure_width_in=b.FIGURE_WIDTH_IN,figure_height_in=b.SUPPLEMENT_HEIGHT_IN if supplement else b.FIGURE_HEIGHT_IN,
        display_groups=groups,baseline_reference=b.BASELINE_REFERENCE,storage_formats=None if supplement else formats,
        statistics=analysis,significance_annotations=b.significance_brackets(selected,groups,analysis),
        paired_representation_differences=None if supplement else b.paired_representation_data(selected),
        paired_significance_annotations=None if supplement else b.paired_significance(selected,analysis),
        caption=prose,sources={**b.accuracy.SOURCES,**analysis["sources"]},builder_sha256=b.resources.digest(Path(b.__file__)),
        extension_sha256=b.resources.digest(Path(__file__)),benchmark=benchmark,
        outputs=[str(p.relative_to(b.ROOT)) for p in paths])
    spar.write(b.OUT/f"{stem}.json",record)
    flat=[dict(campaign=r["campaign"],arm=r["arm"],label=r["label"],error_mean_mm=r["error_mean"],
        participant_sd_mm=float(np.std(list(r["error_participant_means"].values()),ddof=1)),
        parameters=r["parameter_count"],latency_mean_ms=r["latency_mean_ms"],latency_sd_ms=r["latency_sd_ms"]) for r in selected]
    with (b.OUT/f"{stem}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    table=[r"\begin{tabular}{lrrr}\toprule",r"Decoder & Error (mm) & Parameters (M) & Latency (ms)\\",
           r" & Mean $\pm$ SD & & Mean $\pm$ SD\\\midrule"]
    for group in groups:
        table.append(r"\multicolumn{4}{l}{\textbf{"+group["label"]+r"}}\\")
        for row in b.selected_rows(selected,[group]):
            sd=float(np.std(list(row["error_participant_means"].values()),ddof=1))
            table.append(f"{b.plot_model_label(row)} & {row['error_mean']:.2f} $\\pm$ {sd:.2f} & "
                f"{row['model_parameters_million']:.3f} & {row['latency_mean_ms']:.1f} $\\pm$ {row['latency_sd_ms']:.1f}"+r"\\")
    table.append(r"\bottomrule\end{tabular}")
    (b.OUT/f"TABLE_{stem}.tex").write_text("\n".join(table)+"\n")


def run(b,root):
    supplement_layout={key:copy.deepcopy(getattr(b,key)) for key in
        ["METRIC_COLUMNS","METRIC_LABELS","LATENCY_SCALE","LATENCY_LIMITS","LATENCY_TICKS","ERROR_LIMITS","ERROR_TICKS"]}
    rows,formats,benchmark=b.load_data()
    analysis=b.reference_statistics(rows)
    rows,formats,benchmark,analysis=spar.extend(b,rows,formats,benchmark,analysis)
    rows,formats,benchmark,analysis=extend(b,root,rows,formats,benchmark,analysis)
    b.FIGURE_HEIGHT_IN=6.7
    b.MODEL_PANEL_RECT=(.32,.56)
    b.METRIC_COLUMNS={"error":(.285,.29),"size":(.665,.145),"latency":(.865,.12)}
    maximum=max(r["latency_mean_ms"]+r["latency_sd_ms"] for r in rows)
    if maximum>1000:
        b.LATENCY_SCALE=.001
        b.METRIC_LABELS["latency"]="Latency (s)"
        magnitude=10**int(np.floor(np.log10(maximum/1000)))
        last_tick=next(m*magnitude for m in [1,2,5,10] if m*magnitude>=maximum/1000)
        minimum=min(r["latency_mean_ms"]-r["latency_sd_ms"] for r in rows)
        assert minimum>0
        first_tick=10**int(np.floor(np.log10(minimum/1000)))
        middle_tick=10**int(round((np.log10(first_tick)+np.log10(last_tick))/2))
        # Trim the spine to enclosing ticks, retaining enough axis padding for
        # the full marker width on this narrow logarithmic panel.
        b.LATENCY_LIMITS=(first_tick/1.5,last_tick*3)
        b.LATENCY_TICKS=[first_tick,middle_tick,last_tick]
    else:
        last_tick=10**int(np.ceil(np.log10(maximum)))
        b.LATENCY_LIMITS=(.3,last_tick*3)
        b.LATENCY_TICKS=[1,100,last_tick] if last_tick>100 else [1,10,100]
    lowest=min(min(r["error_participant_means"].values()) for r in rows)
    highest=max(max(r["error_participant_means"].values()) for r in rows)
    first_error_tick=max(5,5*int(np.floor(lowest/5)))
    last_error_tick=5*int(np.ceil(highest/5))
    b.ERROR_TICKS=list(range(first_error_tick,last_error_tick+1,5))
    b.ERROR_LIMITS=(min(first_error_tick-.5,lowest-.5),last_error_tick+.5)
    max_change=max(max(s["participant_means_mm"].values()) for s in b.ABSOLUTE_ANALYSIS["summaries"])
    if max_change>b.PAIRED_LIMITS[1]:
        b.PAIRED_LIMITS=(0,int(np.ceil(max_change)))
        b.PAIRED_TICKS=list(range(b.PAIRED_LIMITS[1]+1))
    # Keep the proposed SPAR decoder as the bottom reference in panel C.
    export(b,root,rows,formats,benchmark,analysis,False)
    # The early-baseline supplement has no TabPFN rows. Preserve its approved
    # axis ranges and millisecond units while refreshing the shared statistics.
    for key,value in supplement_layout.items():setattr(b,key,value)
    export(b,root,rows,formats,benchmark,analysis,True)
    for p in [b.ROOT/"results/dpose_figure_analysis.json",b.ROOT/"results/dpose_tabpfn_figure_analysis.json",b.OUT/"dpose-figure-statistics.json"]:
        spar.write(p,analysis)
    records=[dict(alternative=p["label_a"],reference=p["label_b"],kind=p["contrast_kind"],
        effect_mm=p["effect_alternative_minus_reference_mm"],ci95_low_mm=p["effect_ci95_pointwise_mm"][0],
        ci95_high_mm=p["effect_ci95_pointwise_mm"][1],p_raw=p["p_raw"],p_adjusted_76=p["p_adjusted"],
        symbol=b.significance_symbol(p["p_adjusted"]) or "n.s.") for p in analysis["groups"][0]["comparisons"]]
    with (b.OUT/"dpose-figure-statistics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=["# DPose figure: TabPFN extension","",analysis["method"],"",
        "All 66 previous participant contrast vectors and raw p-values are retained. Ten comparisons were declared in the cohort protocol before the all-nine run.","",
        "| Alternative | Reference | Effect (mm) | Pointwise 95% interval | Raw p | Joint 76 max-T p |",
        "|---|---|---:|---|---:|---:|"]
    lines += [f"| {r['alternative']} | {r['reference']} | {r['effect_mm']:.3f} | [{r['ci95_low_mm']:.3f}, {r['ci95_high_mm']:.3f}] | {r['p_raw']:.6f} | {r['p_adjusted_76']:.6f} |" for r in records]
    for p in [b.ROOT/"output/DPOSE_TABPFN_FIGURE_ANALYSIS.md",b.OUT/"dpose-figure-statistics.md"]:p.write_text("\n".join(lines)+"\n")
    for suffix in ["json","csv","md"]:shutil.copy2(b.OUT/f"dpose-figure-statistics.{suffix}",b.OUT/f"dpose-statistical-tests.{suffix}")
    table=[r"% Requires booktabs and longtable. All 76 joint max-T comparisons.",
        r"\begin{longtable}{p{2.1in}p{2.1in}rrr}\toprule",r"Alternative & Reference & $\Delta$ (mm) & Pointwise 95\% CI & Max-T $p$\\\midrule\endhead"]
    for r in records:
        table.append(f"{r['alternative']} & {r['reference']} & {r['effect_mm']:+.3f} & $[{r['ci95_low_mm']:+.3f}, {r['ci95_high_mm']:+.3f}]$ & {r['p_adjusted_76']:.6f}"+r"\\")
    table.append(r"\bottomrule\end{longtable}")
    (b.OUT/"TABLE_dpose-statistical-tests.tex").write_text("\n".join(table)+"\n")
    (b.OUT/"dpose-statistical-tests.tex").write_text(r"\input{output/figures/TABLE_dpose-statistical-tests.tex}"+"\n")
    for name in ["build_dpose_combined.py","dpose_spar_figure_extension.py","dpose_tabpfn_figure_extension.py"]:
        shutil.copy2(b.ROOT/"scripts"/name,b.OUT/name)
    print(json.dumps(dict(main_rows=15,supplement_rows=4,statistical_family=76,all_source_cells=567),indent=2))
