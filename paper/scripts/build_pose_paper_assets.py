"""Build publication figures/tables from the completed, audited pose campaigns.

No fitting or checkpoint selection occurs here. Every displayed result is derived
from a saved report; trajectory selection uses targets only and a fixed head seed.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = paths.OUTPUT / "figures/pose_paper"
TABLES = paths.OUTPUT / "tables"
SEEDS = (260905, 260906, 260907)
COLORS = {
    "vq": "#d80032", "continuous": "#5c618e", "fsq": "#B75F96",
    "pca": "#D59000", "kmeans": "#7966A8", "codec": "#D55E00",
    "v19": "#3C4149", "raw": "#7F858E", "target": "#222222",
}
SOURCES = {}
AUDIT = {"figures": {}, "derived": {}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    path = ROOT / name
    SOURCES[name] = sha(path)
    return json.loads(path.read_text())


def npz(name):
    path = ROOT / name
    SOURCES[str(name)] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def configure():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#858B93",
        "axes.labelcolor": "#30343A", "text.color": "#24282D",
        "xtick.color": "#50565F", "ytick.color": "#50565F",
        "axes.linewidth": .65, "grid.color": "#DDE1E6",
        "grid.linewidth": .55, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=11)
    ax.text(-.11, 1.045, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="bottom")


def save(fig, name, data):
    fig.canvas.draw()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=320, bbox_inches="tight", pad_inches=.06)
    AUDIT["figures"][name] = {"data": data, "pdf_sha256": sha(OUT / f"{name}.pdf")}
    plt.close(fig)


def vector(r, labels, scope="official", subgroup="all"):
    rows = [v for v in r["rows"] if v["scope"] == scope and v["label"] in labels]
    assert len(rows) == 3 * len(labels), (labels, scope, len(rows))
    people = sorted(rows[0]["scores"][subgroup]["per_user"])
    assert all(sorted(v["scores"][subgroup]["per_user"]) == people for v in rows)
    x = np.array([[v["scores"][subgroup]["per_user"][u]["landmark_mae_palm_lengths"]
                   for u in people] for v in rows]).mean(0)
    np.testing.assert_allclose(x.mean(), np.mean([v["scores"][subgroup]["macro_user_mae"]
                                                for v in rows]), atol=1e-12)
    return people, x


def pair(r, scope, subgroup, comparison):
    found = [v for k, v in r["paired"].items()
             if k.endswith(f"/{scope}/{subgroup}/{comparison}")]
    # raw_supervised is registered under both the controls and codec families;
    # repeats are accepted only when every record is identical.
    assert found and all(v == found[0] for v in found), comparison
    result = found[0]
    average = comparison.startswith("three_encoder_mean_")
    a, b = comparison.removeprefix("three_encoder_mean_").split("_vs_")
    labels = lambda key: sorted({v["label"] for v in r["rows"]
                                if v["label"].startswith(key + "_")}) if average else [key]
    people, x = vector(r, labels(a), scope, subgroup)
    other, y = vector(r, labels(b), scope, subgroup)
    assert people == other == result["users"]
    draws = np.random.default_rng(260908).integers(0, len(x), (10000, len(x)))
    ci = np.quantile((x-y)[draws].mean(1), [.025, .975])
    np.testing.assert_allclose((x-y).mean(), result["mean_difference"], atol=1e-12)
    np.testing.assert_allclose(ci, result["difference_ci95"], atol=1e-12)
    return result


def pipeline():
    from build_tokenizer_mechanism import build
    receipt=build()
    # Frozen data/model sources are checked before regeneration. Editable layout
    # and builder hashes belong to build provenance, recorded separately below.
    SOURCES.update({rel: digest for rel, digest in receipt['source_sha256'].items()
                    if rel not in {'scripts/build_tokenizer_mechanism.py',
                                   'paper/figures/tokenizer_mechanism.tex'}})
    AUDIT['figures']['interface']={'pdf_sha256':receipt['pdf_sha256'],'data':receipt}


def transfer(r, mm_per_palm):
    from plot_meta_transfer import draw
    draw(r, mm_per_palm)


def controls(r, mm_per_palm):
    from plot_compact_controls import draw
    draw(r, mm_per_palm)


def runtime(r):
    from plot_streaming_cost import draw
    draw(r)


def meta_overview(r, mm_per_palm):
    from plot_meta_overview import draw
    draw(r, mm_per_palm)


def dpose(r, audit):
    arms = [("v19", "RaulNetV19", "v19"), ("tcn_continuous", "TCN / continuous", "continuous"),
            ("tcn_tokens", "TCN / tokens", "vq"),
            ("transformer_continuous", "Transformer / continuous", "continuous"),
            ("transformer_tokens", "Transformer / tokens", "vq")]
    people = sorted({v["subject"] for v in r["per_subject_seed"]})
    vectors = {arm: np.array([np.mean([v["mean_3d_error"] for v in r["per_subject_seed"]
                                     if v["subject"] == u and v["arm"] == arm]) for u in people])
               for arm, _, _ in arms}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.24, right=.97, bottom=.20, top=.87, wspace=.62)
    for i, (arm, _, color) in enumerate(arms):
        y = i + np.linspace(-.16, .16, len(people))
        axes[0].scatter(vectors[arm], y, color=COLORS[color], s=14, alpha=.55, linewidths=0)
        axes[0].scatter(vectors[arm].mean(), i, color=COLORS[color], edgecolor="white",
                        marker="D", s=48, zorder=3, linewidths=.7)
        np.testing.assert_allclose(vectors[arm].mean(), r["summary"][arm]["mean"], atol=1e-6)
    axes[0].set_yticks(range(5), [v[1] for v in arms]); axes[0].set_ylim(4.55, -.55)
    axes[0].set_xlabel("3D pose error\n(stored-coordinate units)")
    comparisons = [("tcn_tokens_vs_v19", "Token TCN\nvs. V19", "vq"),
                   ("tcn_tokens_vs_tcn_continuous", "Token TCN\nvs. continuous TCN", "vq"),
                   ("transformer_tokens_vs_v19", "Token transformer\nvs. V19", "raw")]
    for i, (name, _, color) in enumerate(comparisons):
        v = audit["comparisons"][name]
        a, b = name.split("_vs_")
        np.testing.assert_allclose((vectors[a]-vectors[b]).mean(), v["difference"], atol=1e-6)
        axes[1].plot(v["relative_ci95_percent"], [i, i], color=COLORS[color], lw=1.6)
        axes[1].scatter(v["relative_percent"], i, s=26, color=COLORS[color])
    axes[1].set_yticks(range(3), [v[1] for v in comparisons]); axes[1].set_ylim(2.55, -.55)
    axes[1].axvline(0, color="#5A636E", ls="--", lw=.8)
    axes[1].set_xlabel("Relative pose-error difference (%)\nPointwise 95% participant interval")
    for ax in axes:
        ax.grid(axis="x"); ax.tick_params(axis="y", length=0)
    panel(axes[0], "A", "Nine participants; three seeds")
    panel(axes[1], "B", "Paired comparisons")
    save(fig, "dpose_matched", {"per_participant": {k: v.tolist() for k, v in vectors.items()},
                                 "participants": people, "comparisons": audit["comparisons"]})


def trajectories():
    from plot_pose_trajectories import draw
    draw()


def noise(r):
    from plot_noise_robustness import draw
    draw(r)


def codec_diagnostics():
    from plot_codec_diagnostics import draw
    draw()


def bold_best(values, format_spec=".2f", *, higher=False):
    """Format a comparable group and bold all optima at displayed precision."""
    rendered = [format(value, format_spec) for value in values]
    displayed = [float(value.replace(",", "")) for value in rendered]
    optimum = (max if higher else min)(displayed)
    return [rf"\textbf{{{text}}}" if value == optimum else text
            for text, value in zip(rendered, displayed, strict=True)]


def tables(r, d, n, mm_per_palm):
    families = [("Learned tokens (1,024 codes; 10 bits)", [f"vq_{s}" for s in SEEDS], "3"),
                ("Continuous features (256D, FP32)", [f"continuous_{s}" for s in SEEDS], "3"),
                ("BioCodec, 6 codebooks (48 bits)", [f"biocodec6_{s}" for s in (260940, 260941, 260942)], "3"),
                ("FSQ (1,000 codes; 10 bits)", ["fsq1000"], "1"),
                ("Quantized PCA (4D; 10 bits)", ["pca4q10"], "1"),
                ("K-means (1,024 codes; 10 bits)", ["kmeans1024"], "--"),
                ("Engineered features (67D, FP32)", ["features67"], "--"),
                ("Pose-supervised features (256D, FP32)", ["raw_supervised"], "3 joint")]
    rows = [r"\begin{tabular}{lcrrr}\toprule", r"Representation & Encoders & Official (mm) & Unseen (mm) & Sessions (mm)\\\midrule"]
    means = {}
    for name, labels, count in families:
        values = [vector(r, labels, scope, subgroup)[1].mean()
                  for scope, subgroup in (("official", "all"), ("official", "unseen_users"), ("session", "all"))]
        means[name] = [float(v) for v in values]
    columns = [bold_best([means[name][column]*mm_per_palm for name, _, _ in families]) for column in range(3)]
    for index, (name, _, count) in enumerate(families):
        rows.append(name + " & " + count + " & " + " & ".join(column[index] for column in columns) + r"\\")
    rows += [r"\bottomrule\end{tabular}"]
    (TABLES / "completion_accuracy.tex").write_text("\n".join(rows)+"\n")
    rows = [r"\begin{tabular}{lrr}\toprule", r"Model & Mean 3D error & SD across seed macro-means\\\midrule"]
    matched = [("v19", "RaulNetV19, common recipe"), ("tcn_continuous", "TCN / continuous"),
               ("tcn_tokens", "TCN / tokens"), ("transformer_continuous", "Transformer / continuous"),
               ("transformer_tokens", "Transformer / tokens")]
    errors = bold_best([d["summary"][arm]["mean"] for arm, _ in matched] + [d["constant_baseline"]], ".3f")
    for index, (arm, label) in enumerate(matched):
        v = d["summary"][arm]
        rows.append(f"{label} & {errors[index]} & {v['seed_sd']:.3f}" + r"\\")
    rows += [f"Training-mean constant & {errors[-1]} & --" + r"\\", r"\bottomrule\end{tabular}"]
    (TABLES / "matched_accuracy.tex").write_text("\n".join(rows)+"\n")
    rows = [r"\begin{tabular}{llrrr}\toprule", r"Scope & Representation & Pose (mm) & Hold (mm) & Velocity $R^2$\\\midrule"]
    motion = []
    for scope, title in (("official", "Official"), ("session", "Held session")):
        group = []
        for name, labels, _ in [families[0], families[1], families[-1]]:
            cases = [v for v in r["rows"] if v["scope"] == scope and v["label"] in labels]
            pose = np.mean([v["scores"]["all"]["macro_user_mae"] for v in cases])
            hold = np.mean([v["hold"]["macro_user_mae"] for v in cases])
            velocity = np.mean([v["scores"]["all"]["pooled"]["landmark_velocity_r2"] for v in cases])
            short = "Raw supervised" if name.startswith("Raw") else name.replace(" (float32 storage)", "")
            group.append((short, pose*mm_per_palm, hold*mm_per_palm, velocity))
            motion.append({"scope": scope, "method": name, "pose": pose, "hold": hold,
                           "pose_mm": pose*mm_per_palm, "hold_mm": hold*mm_per_palm,
                           "velocity_r2": velocity})
        columns = [bold_best([v[column] for v in group], "+.4f" if column == 3 else ".2f", higher=column == 3)
                   for column in (1, 2, 3)]
        for index, (short, *_) in enumerate(group):
            rows.append(f"{title} & {short} & " + " & ".join(column[index] for column in columns) + r"\\")
        if scope == "official":
            rows.append(r"\midrule")
    rows.append(r"\bottomrule\end{tabular}")
    (TABLES / "motion_diagnostics.tex").write_text("\n".join(rows)+"\n")
    rows = [r"\begin{tabular}{lrrrr}\toprule", r" & \multicolumn{2}{c}{Pose error} & \multicolumn{2}{c}{Velocity $R^2$}\\",
            r"Model & Original & Filtered & Original & Filtered\\\midrule"]
    for arm, label in (("token_v20", "Token V20"), ("continuous_v20", "Continuous V20"), ("native_v19", "Native RaulNetV19")):
        vals = [next(v for v in n["summary"] if v["arm"] == arm and v["filtering"] == f and v["snr_db"] is None)
                for f in ("unfiltered", "one_euro")]
        poses = bold_best([v["mean_3d_error"] for v in vals], ".3f")
        velocities = bold_best([v["velocity_r2"] for v in vals], ".3f", higher=True)
        rows.append(label + " & " + " & ".join(poses + velocities) + r"\\")
    rows.append(r"\bottomrule\end{tabular}")
    (TABLES / "filter_diagnostics.tex").write_text("\n".join(rows)+"\n")
    rt = {device: {v["case"]["label"]: v for v in r["runtime"][device]["result"]["rows"]}
          for device in ("cpu", "cuda")}
    rows = [r"\begin{tabular}{lrrrrr}\toprule", r" & Packet rate & Setup & Tensors & CPU & V100\\",
            r"Representation & kB/s & bytes & MB & ms & ms\\\midrule"]
    costs = []
    for label, name in [("vq_260905", "Learned tokens (1,024 codes; 10 bits)"), ("continuous_260905", "Continuous features (256D, FP32)"),
                        ("fsq1000", "FSQ (1,000 codes; 10 bits)"), ("pca4q10", "Quantized PCA (4D; 10 bits)"),
                        ("kmeans1024", "K-means (1,024 codes; 10 bits)"), ("biocodec6_260940", "BioCodec, 6 codebooks (48 bits)")]:
        v = rt["cuda"][label]
        costs.append((name, v["packet_bytes_per_second"]/1000, v["setup_json_bytes"],
                      v["sum_component_tensor_bytes"]/1e6, rt["cpu"][label]["latency"]["p50_ms"],
                      v["latency"]["p50_ms"]))
    columns = [bold_best([v[column] for v in costs], ",.0f" if column == 2 else ".2f") for column in range(1, 6)]
    for index, (name, *_) in enumerate(costs):
        rows.append(name + " & " + " & ".join(column[index] for column in columns) + r"\\")
    rows.append(r"\bottomrule\end{tabular}")
    (TABLES / "runtime_storage.tex").write_text("\n".join(rows)+"\n")
    AUDIT["derived"]["meta_means"] = means
    AUDIT["derived"]["meta_means_source_unit"] = "palm lengths"
    AUDIT["derived"]["meta_means_mm"] = {k: [v*mm_per_palm for v in values] for k, values in means.items()}
    AUDIT["derived"]["motion"] = motion
    AUDIT["table_highlighting"] = {
        "ties": "All numerical optima at the displayed precision are bold; this is not a significance claim.",
        "completion_accuracy": "Minimum pose error within each cohort column.",
        "matched_accuracy": "Minimum mean pose error; seed SD is descriptive and not ranked.",
        "runtime_storage": "Minimum in each cost column.",
        "motion_diagnostics": "Within each cohort: minimum pose/hold errors and maximum velocity R2.",
        "filter_diagnostics": "Within each model and original/filtered pair: minimum pose error and maximum velocity R2.",
    }
    macro = {"MetaVQ": f"{means['Learned tokens (1,024 codes; 10 bits)'][0]*mm_per_palm:.2f}", "MetaContinuous": f"{means['Continuous (float32 storage)'][0]*mm_per_palm:.2f}",
             "SessionVQ": f"{means['Learned tokens (1,024 codes; 10 bits)'][2]*mm_per_palm:.2f}", "SessionContinuous": f"{means['Continuous (float32 storage)'][2]*mm_per_palm:.2f}",
             "MetaPalmMM": f"{mm_per_palm:.3f}",
             "MatchedToken": f"{d['summary']['tcn_tokens']['mean']:.3f}",
             "MatchedRaul": f"{d['summary']['v19']['mean']:.3f}"}
    (TABLES / "result_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macro.items()) + "\n"
    )


def main():
    import emg2pose
    configure(); OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    r = read("results/paper_completion_v1/results.json")
    audit = read("results/paper_completion_v1/local_statistics_audit.json")
    assert r["complete"] and r["independently_recomputed_cases"] == len(r["rows"]) == 114
    assert audit["results_sha256"] == SOURCES["results/paper_completion_v1/results.json"]
    conversion = read("data/reference/meta_millimetre_conversion.json")
    assert conversion["source"]["sha256"] == SOURCES["results/paper_completion_v1/results.json"]
    mm_per_palm = conversion["mm_per_palm_length"]
    assert mm_per_palm > 0 and conversion["hand_model"]["matches_hpc_model_sha256"]
    model_path = Path(emg2pose.__file__).parent / "UmeTrack/dataset/generic_hand_model.json"
    assert sha(model_path) == conversion["hand_model"]["sha256"]
    normalization = paths.REPO / "src/myoji/data/landmarks.py"
    assert sha(normalization) == conversion["validation"]["local_normalization_source_sha256"]
    AUDIT["meta_units"] = {"source": "palm lengths", "display": "mm; reference hand",
                           "mm_per_palm_length": mm_per_palm,
                           "hand_model_sha256": sha(model_path)}
    for device in ("cpu", "cuda"):
        path = f"results/paper_completion_v1/runtime_{device}.json"
        read(path)
        assert SOURCES[path] == r["runtime"][device]["sha256"]
    d = read("results/dpose_matched_v1/results.json")
    da = read("results/dpose_matched_v1/paired_participant_audit.json")
    assert d["complete"] and da["source_results_sha256"] == SOURCES["results/dpose_matched_v1/results.json"]
    n = read("results/dpose_v20_noise_followup_v2/results.json")
    na = read("results/dpose_v20_noise_followup_v2/independent_audit.json")
    assert n["complete"] and na["results_sha256"] == SOURCES["results/dpose_v20_noise_followup_v2/results.json"]
    pipeline(); meta_overview(r, mm_per_palm); dpose(d, da); trajectories(); noise(n); codec_diagnostics(); tables(r, d, n, mm_per_palm)
    AUDIT.update({"complete": True, "source_sha256": SOURCES, "builder_sha256": sha(Path(__file__)),
                  "new_training": False, "intervals": "Pointwise participant bootstrap; fixed trained encoders, not a population interval over tokenizer training."})
    for p in TABLES.glob("*.tex"):
        AUDIT.setdefault("table_sha256", {})[p.name] = sha(p)
    (paths.OUTPUT / "pose_paper_asset_audit.json").write_text(json.dumps(AUDIT, indent=2)+"\n")
    print(f"Built {len(AUDIT['figures'])} figures in PDF/PNG/SVG; checked source hashes and plotted paired statistics.")
    print(json.dumps(AUDIT["figures"]["pose_trajectories"]["data"], indent=2))


if __name__ == "__main__":
    main()
