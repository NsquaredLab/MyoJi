"""Collect only complete planned comparisons; recheck metrics from predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import numpy as np
from paper.emg2pose import paper_completion_core as core
from paper.common.reporting_support import (
    recovery_provenance,
    require_scored_participants,
    scoring_support,
)


def verify_predictions(row):
    path = Path(row["predictions"])
    if pose.digest(path) != row["predictions_sha256"]:
        raise ValueError("prediction file changed")
    with np.load(path, allow_pickle=False) as a:
        p, t, v, users = a["prediction"][:, -50:], a["target"][:, -50:], a["valid"][:, -50:], a["users"]
    # Independent NumPy implementation, no training/evaluation metric helper.
    distances = np.linalg.norm((p.astype(np.float64)-t).reshape(len(p), 50, -1, 3), axis=-1)
    support = scoring_support(users, v)
    require_scored_participants(row["scores"]["all"], support)
    actual = {}
    for user in support["scored_participants"]:
        ix = users == user
        actual[user] = float(distances[ix][v[ix]].mean())
        expected = row["scores"]["all"]["per_user"][user]["landmark_mae_palm_lengths"]
        np.testing.assert_allclose(actual[user], expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.mean(list(actual.values())), row["scores"]["all"]["macro_user_mae"], atol=1e-8, rtol=1e-8)
    return actual, support


def summarize_motion(rows):
    groups = defaultdict(list)
    for row in rows:
        for group in ("all", "stage/" + row["stage"], "speed/" + str(row["motion_stratum"])):
            groups[(group, row["user"])].append(row)
    keys = ("pose_error", "velocity_mae", "pose_r2", "target_coordinate_variance", "amplitude_ratio", "lag_frames", "peak_correlation")
    by_user = {}
    for (group, user), items in groups.items():
        out = {"windows": len(items), "valid_frames": sum(r["valid_frames"] for r in items),
               "lag_eligible_windows": sum(r["lag_eligible"] for r in items),
               "lag_boundary_windows": sum(bool(r["at_boundary"]) for r in items)}
        for key in keys:
            selected = [r[key] for r in items if r[key] is not None]
            out[key] = float(np.mean(selected)) if selected else None
        by_user.setdefault(group, {})[user] = out
    return by_user


def run(args):
    p = core.verify(args.root)
    recovery = recovery_provenance(args.root, getattr(args, "recovery_manifest", None))
    protocol_hash = pose.digest(args.root / "protocol.json")
    results, unique, pairs, motion = {}, {}, {}, {}
    identities, targets, supports = {}, {}, {}
    for group in ("continuous", "controls", "codec"):
        path = args.root / f"evaluation/{group}/results.json"
        result = json.loads(path.read_text())
        if not result["complete"] or result["protocol_sha256"] != protocol_hash:
            raise ValueError("incomplete or mismatched result group")
        for scope in core.SCOPES:
            for observed, key in [(identities, "test_identities"), (targets, "target_hashes")]:
                if scope in observed and observed[scope] != result[key][scope]:
                    raise ValueError("groups evaluated different observations")
                observed[scope] = result[key][scope]
        results[group] = {"sha256": pose.digest(path), "rows": len(result["rows"])}
        pairs.update({group + "/" + key: value for key, value in result["paired"].items()})
        for row in result["rows"]:
            if row["id"] in unique:
                old = unique[row["id"]]
                if old["checkpoint_sha256"] != row["checkpoint_sha256"] or old["scores"] != row["scores"]:
                    raise ValueError("reused reference changed between groups")
                continue
            _, support = verify_predictions(row)
            if row["scope"] in supports and supports[row["scope"]] != support:
                raise ValueError("cases have different scored-label support")
            supports[row["scope"]] = support
            unique[row["id"]] = row
            filename = args.root / f"evaluation/{group}/{row['id']}_motion.json"
            m = json.loads(filename.read_text())
            if m["predictions_sha256"] != row["predictions_sha256"]:
                raise ValueError("motion diagnostic provenance differs")
            motion[row["id"]] = {"case": row["id"], "label": row["label"], "scope": row["scope"],
                                  "head_seed": row["head_seed"], "groups": summarize_motion(m["rows"])}
    if set(unique) != {r["id"] for r in p["cases"] + p["references"]}:
        raise ValueError("required case is missing or unplanned case was added")
    runtime = {}
    for device in ("cpu", "cuda"):
        path = args.root / f"runtime_{device}.json"
        value = json.loads(path.read_text())
        if not value["complete"] or value["protocol_sha256"] != protocol_hash:
            raise ValueError("required full pipeline benchmark is incomplete")
        runtime[device] = {"sha256": pose.digest(path), "result": value}
    pose.write_json(args.root / "motion_summary.json", {
        "protocol_sha256": protocol_hash, "cases": motion,
        "aggregation": "Unshifted window diagnostics averaged within each participant/stratum. Primary pose outcome remains valid-frame-weighted within participant. Invalid and constant-motion lag estimates retained as missing; boundary optima counted.",
    })
    summary = {"complete": True, "protocol_sha256": protocol_hash, "groups": results,
               "new_head_fits": len(p["cases"]), "reused_fits": len(p["references"]),
               "independently_recomputed_cases": len(unique), "rows": list(unique.values()),
               "scoring_support": supports, "reporting_recovery": recovery,
               "paired": pairs, "runtime": runtime, "motion_summary_sha256": pose.digest(args.root / "motion_summary.json"),
               "claim_limits": p["scope"]}
    pose.write_json(args.root / "results.json", summary)
    lines = ["# Meta completion results", "", "All planned fits and both pipeline benchmarks completed; metrics independently recomputed from saved predictions.", "",
             "Intervals below are pointwise paired-participant intervals, conditional on the displayed encoder. Error is normalized landmark distance. A positive difference means VQ has more error than the comparator.", "",
             "| Comparison | VQ minus comparator | 95% interval | Participants |", "|---|---:|---:|---:|"]
    for name, r in pairs.items():
        lines.append(f"| {name} | {r['mean_difference']:.5f} | [{r['difference_ci95'][0]:.5f}, {r['difference_ci95'][1]:.5f}] | {len(r['users'])} |")
    lines += ["", p["scope"], "", "Completion is not an equivalence decision."]
    for scope, support in supports.items():
        lines += ["", f"{scope}: {len(support['scored_participants'])}/{support['present_participants']} participants have scored labels; {support['windows_with_scored_labels']}/{support['windows']} windows contain at least one valid scored frame. Unscorable participant IDs: {support['unscorable_participants']}. No windows or training examples were removed."]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    figures(args.root, summary)
    print(json.dumps({"complete": True, "independently_recomputed": len(unique)}))


def figures(root, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    selected = {k: v for k, v in results["paired"].items() if "/all/" in k}
    fig, ax = plt.subplots(figsize=(10, max(4, len(selected)*.28)))
    for i, (label, value) in enumerate(selected.items()):
        lo, hi = value["difference_ci95"]
        ax.plot([lo, hi], [i, i], color="#276a93")
        ax.plot(value["mean_difference"], i, "o", color="#276a93", markersize=4)
    ax.set_yticks(range(len(selected)), list(selected), fontsize=7)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_xlabel("VQ minus comparator error (palm lengths); pointwise 95% participant interval")
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / ("paired_comparisons." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path)
    run(parser.parse_args())
