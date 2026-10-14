"""Supplementary figure: how much token history the movement loops need.

Panel A of the pose-recovery figure (participant 1, eight movements, each island
viewed face-on in a 3-D UMAP and placed at the MDS of native centroid distances)
repeated for growing history windows: the frame alone, then the frame plus the
tokens 6, 12, ... updates earlier, up to 960 ms. Each panel carries the movements'
mean routes and the mean loop roundness (4*pi*area / perimeter^2; 1 = circle).

    uv run python scripts/make_token_history_supplement.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fig5", ROOT / "scripts/make_token_pose_recovery_figure.py")
fig5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig5)

import matplotlib.pyplot as plt                                        # noqa: E402  (styled by fig5's imports)
import paths                                                           # noqa: E402
import switzer_font                                                    # noqa: E402

WINDOWS_MS = (0, 120, 240, 480, 720, 960)                              # 20 ms per update, lags every 6 updates
CACHE = fig5.RECOVERY / "history_supplement.npz"
OUT_STEM = "token-history-supplement"
NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Fist", "2-Pinch", "3-Pinch"]
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f", "#e377c2"]


def roundness(route):
    p = np.vstack([route, route[:1]])
    area = .5 * abs(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))
    return float(4 * np.pi * area / np.linalg.norm(np.diff(p, axis=0), axis=1).sum() ** 2)


def mean_route(e, m, cycle, phase, steps=20):
    bins = np.linspace(0, 1, steps + 1)
    reps = []
    for c in np.unique(cycle[m]):
        mc = m & (cycle == c)
        per = [e[mc & (phase >= a) & (phase < b)] for a, b in zip(bins[:-1], bins[1:])]
        if all(len(q) for q in per):
            reps.append([q.mean(0) for q in per])
    return np.array(reps).mean(0)


def embed_all():
    if CACHE.exists():
        z = np.load(CACHE)
        if tuple(z["windows_ms"]) == WINDOWS_MS:
            return {w: z[f"w{w}"] for w in WINDOWS_MS}
    from umap import UMAP
    from sklearn.manifold import MDS
    from myoji.spar import grid_layout
    d = np.load(fig5.RECOVERY / "pose_recovery_sub01.npz")
    ids, label = d["ids"].astype(np.int32), d["label"]
    book = np.load(paths.DATA / "figure_inputs/codebooks.npz")["book_01"]
    mask = np.tile(grid_layout().mask, (1, 6))
    out = {}
    for w in WINDOWS_MS:
        lags = tuple(range(6, w // 20 + 1, 6))
        if lags:
            fig5.HISTORY_LAGS = lags
            hist, _ = fig5.recording_history(ids, label)
            W = np.hstack([ids, hist])
        else:
            W = ids
        X = np.hstack([(book[W[:, 320 * j:320 * (j + 1)].astype(np.int64)] * mask[None]).reshape(len(W), -1)
                       for j in range(W.shape[1] // 320)]).astype(np.float32)
        X = X[:, X.var(0) > 1e-10]
        e3 = UMAP(n_components=3, **fig5.UMAP_SETTINGS).fit_transform(X)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        cen = np.array([Xn[label == c].mean(0) for c in range(8)])
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        anchor = MDS(2, dissimilarity="precomputed", random_state=42, normalized_stress="auto",
                     n_init=8).fit_transform(1 - cen @ cen.T)
        out[w] = fig5.face_on(e3, label, d["cycle"], d["phase"], anchor)
        print(f"{w} ms: {len(lags)} lags done", flush=True)
    np.savez(CACHE, windows_ms=np.array(WINDOWS_MS), **{f"w{w}": v for w, v in out.items()})
    return out


def main():
    plt.style.use(["science", "ieee", "simpetru-large"])
    plt.rcParams.update(switzer_font.rcparams())
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0})
    d = np.load(fig5.RECOVERY / "pose_recovery_sub01.npz")
    label, cycle, phase = d["label"], d["cycle"], d["phase"]
    emb = embed_all()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.3))
    fig.subplots_adjust(left=.01, right=.99, bottom=.07, top=.95, wspace=.04, hspace=.18)
    receipt = {}
    rng = np.random.default_rng(0)
    for ax, w in zip(axes.ravel(), WINDOWS_MS):
        e = emb[w]
        order = rng.permutation(len(e))
        ax.scatter(e[order, 0], e[order, 1], s=.5, c=np.array(COLOURS)[label[order]], lw=0, alpha=.5,
                   rasterized=True)
        rs = []
        for c in range(8):
            r = mean_route(e, label == c, cycle, phase)
            rs.append(roundness(r))
            ax.plot(*np.vstack([r, r[:1]]).T, color="black", lw=.8)
        lo, hi = np.percentile(e, [.5, 99.5], axis=0); pad = (hi - lo) * .05
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        title = "Single update" if w == 0 else f"{w} ms of history"
        ax.set_title(f"{title}, roundness {np.mean(rs):.2f}", fontsize=9, pad=2)
        receipt[f"{w}_ms"] = dict(lags=list(range(6, w // 20 + 1, 6)), roundness=dict(zip(NAMES, rs)),
                                  mean_roundness=float(np.mean(rs)))
    handles = [plt.Line2D([], [], marker="o", lw=0, ms=5, color=c, label=n) for n, c in zip(NAMES, COLOURS)]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8, handletextpad=.2,
               columnspacing=.9, bbox_to_anchor=(.5, 0.0))
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        x0, y1 = ax.get_position().x0, ax.get_position().y1
        fig.text(x0 + .005, y1 + .035, letter, fontsize=11, weight="bold", va="top")
    fig.savefig(fig5.PDFS / f"{OUT_STEM}.pdf")
    fig.savefig(fig5.OUT / f"{OUT_STEM}.png", dpi=300, facecolor="white")
    (fig5.OUT / f"{OUT_STEM}.json").write_text(json.dumps(dict(
        windows=receipt, settings=fig5.UMAP_SETTINGS,
        view="3-D UMAP per window, each island face-on (mean-route plane), placed at MDS of native centroid distances",
        participant=1), indent=1) + "\n")
    print(json.dumps({k: round(v["mean_roundness"], 2) for k, v in receipt.items()}))


if __name__ == "__main__":
    main()
