"""Recheck paired statistics and build the completed-study paper figures."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from paper.common import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    root = ROOT / "results/paper_completion_v1"
    r = read(root / "results.json")
    assert r["complete"] and r["independently_recomputed_cases"] == len(r["rows"]) == 114
    assert r["protocol_sha256"] == sha(root / "protocol.json")
    for group, record in r["groups"].items():
        assert record["sha256"] == sha(root / f"evaluation/{group}/results.json")
    for device, record in r["runtime"].items():
        assert record["sha256"] == sha(root / f"runtime_{device}.json")
    for name, expected in r["paired"].items():
        _, scope, subgroup, comparison = name.split("/")
        average = comparison.startswith("three_encoder_mean_")
        a, b = comparison.removeprefix("three_encoder_mean_").split("_vs_")
        people = expected["users"]
        vectors = []
        for label in (a, b):
            rows = [v for v in r["rows"] if v["scope"] == scope and
                    (v["label"].startswith(label + "_") if average else v["label"] == label)]
            assert len(rows) == (9 if average else 3)
            assert all(sorted(v["scores"][subgroup]["per_user"]) == people for v in rows)
            vectors.append(np.array([[v["scores"][subgroup]["per_user"][u]["landmark_mae_palm_lengths"]
                                      for u in people] for v in rows]).mean(0))
        x, y = vectors
        draws = np.random.default_rng(260908).integers(0, len(people), (10000, len(people)))
        computed = {"mean_difference": (x-y).mean(), "difference_ci95": np.quantile((x-y)[draws].mean(1), [.025, .975]),
                    "ratio": x.mean()/y.mean(), "ratio_ci95": np.quantile(x[draws].mean(1)/y[draws].mean(1), [.025, .975])}
        for field, value in computed.items():
            np.testing.assert_allclose(value, expected[field], rtol=1e-12, atol=1e-12)
    audit = {"complete": True, "paired_comparisons_recomputed": len(r["paired"]), "results_sha256": sha(root / "results.json"),
             "builder_sha256": sha(Path(__file__)), "scope": "Independent local reaggregation of all paired participant means, differences, ratios and 10000-draw intervals from the reported per-person scores. The separate HPC collector independently verified114 prediction files."}
    (root / "local_statistics_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    output = paths.OUTPUT / "figures/completion"
    output.mkdir(parents=True, exist_ok=True)
    selected = [("vq_260905", "VQ 1024"), ("continuous_260905", "Continuous 256"), ("fsq1000", "FSQ 1000"),
                ("pca4q10", "PCA4, 10 bits"), ("kmeans1024", "Engineered K-means 1024"),
                ("biocodec6_260940", "BioCodec, 6 books")]
    runtime = {d: {v["case"]["label"]: v for v in r["runtime"][d]["result"]["rows"]} for d in ("cpu", "cuda")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True, gridspec_kw={"width_ratios": [1.35, 1, 1.15]})
    for i, (label, _) in enumerate(selected):
        pair = next((v for k, v in r["paired"].items() if k.endswith("/official/all/vq_260905_vs_" + label)), None)
        if pair is None:
            assert label == "vq_260905"
            delta, lo, hi = 0., 0., 0.
        else:
            delta, lo, hi = -pair["mean_difference"], -pair["difference_ci95"][1], -pair["difference_ci95"][0]
        axes[0].plot([lo, hi], [i, i], color="#2463A5", lw=1.7)
        axes[0].plot(delta, i, "o", color="#2463A5", markersize=5)
        packet = runtime["cuda"][label]["packet_bytes_per_second"] / 1000
        axes[1].plot(packet, i, "o", color="#2463A5")
        axes[1].annotate(f"{packet:.2f}", (packet, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
        for device, offset, color in (("cpu", -.10, "#2463A5"), ("cuda", .10, "#CF7626")):
            value = runtime[device][label]["latency"]["p50_ms"]
            axes[2].plot(value, i+offset, "o", color=color, label={"cpu": "HPC CPU", "cuda": "V100"}[device] if i == 0 else None)
    axes[0].axvline(0, color="#777777", lw=.8)
    axes[0].set_yticks(range(len(selected)), [v[1] for v in selected], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Comparator error − VQ error\n(palm lengths; pointwise 95% interval)", fontsize=9)
    axes[0].set_title("Pose accuracy: official split", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xlim(.8, 2800)
    axes[1].set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    axes[1].set_xlabel("Actual packets (kB/s)\n9-byte frame header included", fontsize=9)
    axes[1].set_title("Representation transmission", fontsize=10)
    axes[2].set_xscale("log")
    axes[2].set_xlim(2, 160)
    axes[2].set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    axes[2].axvline(20, ls=":", color="#777777", lw=1)
    axes[2].set_xlabel("Full-pipeline median (ms)\nDotted line: 20 ms output interval", fontsize=9)
    axes[2].set_title("Fixed three-second replay", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.12)
    fig.suptitle("Compact EMG features: accuracy, transmission and inference", fontsize=12)
    fig.text(.5, .015, "Original tokenizer and first BioCodec seed; three head seeds. BioCodec uses increasing-prefix replay. Model/setup bytes reported separately.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("accuracy_cost." + suffix), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    cohorts = [("official/all", "Official test: 176 people"), ("official/unseen_users", "Officially unseen: 20 people"),
               ("session/all", "Same-day held sessions: 10 scorable people")]
    for i, (key, label) in enumerate(cohorts):
        pair = r["paired"]["continuous/" + key + "/three_encoder_mean_vq_vs_continuous"]
        ax.plot(pair["difference_ci95"], [i, i], color="#2463A5", lw=1.8)
        ax.plot(pair["mean_difference"], i, "o", color="#2463A5")
    ax.set_yticks(range(3), [v[1] for v in cohorts], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", lw=.8)
    ax.set_xlabel("Token − continuous error (palm lengths); pointwise 95% participant interval", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Quantization across three independently trained encoders", fontsize=11)
    fig.text(.5, .015, "Average errors over three encoders × three heads; no prediction ensemble. Intervals conditional on these encoders.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("encoder_repetitions." + suffix), dpi=180)
    plt.close(fig)

    lines = ["# Completed Meta representation campaign", "", "9 September 2026. All 90 new downstream fits, three BioCodec adaptations, 24 reused heads, three evaluation groups and both runtime measurements completed. The reporting recovery independently checked all 114 prediction cases. All 105 paired comparisons were independently reaggregated locally.", "",
             "The original frozen protocols, fitted checkpoints and test membership are unchanged. Reporting now explicitly identifies the one session participant with zero valid scored labels. The official test has 176 scorable people, including 20 officially unseen people. The session test retains 153 windows from 11 planned people; 122 windows and 10 people have any valid labels in the scored interval (5,142 valid frames). These are same-day sessions, not documented electrode replacement.", "",
             "## Main accuracy results", "", "Error is participant-macro Euclidean landmark distance in palm-length units. Head seeds are averaged within person. Rows with three encoders also average errors across those encoders; they are not ensembles. Simple controls use only the original encoder, so their prespecified paired comparisons use that original VQ reference.", "",
             "| Representation | Encoder repetitions | Official test | Unseen people | Held sessions |", "|---|---:|---:|---:|---:|"]
    families = [("VQ", [f"vq_{s}" for s in (260905, 260906, 260907)]),
                ("Continuous, float32 storage", [f"continuous_{s}" for s in (260905, 260906, 260907)]),
                ("BioCodec, six books", [f"biocodec6_{s}" for s in (260940, 260941, 260942)]),
                ("FSQ1000", ["fsq1000"]), ("PCA4, 10 bits", ["pca4q10"]), ("Engineered K-means1024", ["kmeans1024"]),
                ("Engineered features67", ["features67"]), ("Raw jointly supervised", ["raw_supervised"])]
    for label, labels in families:
        means = []
        for scope, subgroup in (("official", "all"), ("official", "unseen_users"), ("session", "all")):
            means.append(np.mean([v["scores"][subgroup]["macro_user_mae"] for v in r["rows"] if v["scope"] == scope and v["label"] in labels]))
        repetitions = "3 joint fits" if labels == ["raw_supervised"] else str(len(labels))
        lines.append(f"| {label} | {repetitions} | {means[0]:.5f} | {means[1]:.5f} | {means[2]:.5f} |")
    lines += ["", "Across three encoders, VQ minus continuous error on the full official test is −0.000465, with a 95% participant interval of [−0.001673, +0.000735]. The unseen-person and session intervals also cross zero. This supports a small observed quantization effect in the tested systems; it does not establish equivalence or a biological error floor.", "",
              "Original-seed VQ beats the specified PCA and engineered K-means controls on the full official test, while FSQ remains competitive. Session intervals are wide and show no consistent winner. These pointwise exploratory comparisons do not support a universal VQ advantage.", "",
              "## Published-codec and movement limits", "",
              "Six-book BioCodec is close in pose error. Its averaged full-official contrast against VQ has a small pointwise interval just excluding zero; the unseen-person and session intervals cross zero. All three adaptations' unlabeled reconstruction diagnostics continue improving between 20k and 40k updates. The different pretraining budgets and still-improving codec make a broad superiority claim inappropriate. Report this as the bounded official-code adaptation that was run.", "",
              "Pose error remains the primary outcome. As a secondary derivative diagnostic, original-seed VQ has mean pooled velocity R² about −0.018 on the official test and +0.007 on the session test. These velocity scores do not overturn the position-decoding result or identify tokenization as the cause; continuous and raw references also have low velocity scores. Complete speed-stratum, amplitude and lag diagnostics remain available in the motion summary. Physical closed-loop control remains a separate, untested outcome.", "",
              "## Measured implementation cost", "",
              "At 16 channels and 50 Hz, VQ packets use 1,450 B/s versus 819,650 B/s for float32 256D continuous features, about 565× fewer transmitted bytes. A framed float32 native-EMG reference uses 128,450 B/s (about 88.6× more than the token stream). These ratios exclude amortized model/setup bytes, which are separately reported. Named model/lookup tensor components are about25.11 MB for VQ and24.06 MB for continuous in this implementation.", "",
              "Full-pipeline median latency is 4.39 ms VQ versus3.41 ms continuous on V100, and13.74 versus11.95 ms on the HPC CPU. Both stay below20 ms in the100 measured updates of this fixed replay. VQ does not improve inference latency here. Engineered K-means1024 is cheaper (2.97 ms V100,3.50 ms CPU) with a modest official-test accuracy cost. These are short software replay measurements, not long-duration or physical-device guarantees. BioCodec recomputes increasing prefixes and has different history; its latency is not a verdict on an optimized recurrent implementation.", "",
              "![Accuracy and cost](figures/completion/accuracy_cost.png)", "", "![Encoder repetitions](figures/completion/encoder_repetitions.png)", "",
              "The paper can center on a compact reusable interface with measured costs and limits. It should retain competitive simple baselines, all seeds, exploratory test reuse, and the separation among Meta transformer, DPose/V20, UCI reuse and mechanistic manifold experiments.", "",
              "[All paired results](../results/paper_completion_v1/RESULTS.md) · [Local statistics audit](../results/paper_completion_v1/local_statistics_audit.json) · [Frozen protocol](PAPER_COMPLETION_PROTOCOL.md)"]
    (paths.OUTPUT / "PAPER_COMPLETION_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    run()
