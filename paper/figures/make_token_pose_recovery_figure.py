"""Figure: do tokens carry the hand's pose? Two UMAP views, pose recovery, and the
decoder's accuracy-payload trade-off.

Numbers come from results/token_umap/pose_recovery/ (per-participant nearest-repetition
pose recovery, computed by pose_recovery/pose_recovery.py) and from the two SPAR
decoder campaigns (results/dpose_spar_simple_full_v1 with 1,024 codes, _v64 with 64).
The UMAP is illustration only: every quoted number is computed in the native
5,760-value frame space or by the trained decoder.

    uv run python -m paper.figures.make_token_pose_recovery_figure
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import colorcet
from paper.common import figstyle as sf
from paper.common import paths
from paper.common import switzer_font

RECOVERY = paths.RESULTS / "token_umap/pose_recovery"
UMAP = RECOVERY / "umap_sub01_tokens.npz"
# Chosen by the author from two parameter sweeps (36 settings each, PCA-50 and
# full-dimensional): cosine with few neighbours keeps the eight movements as
# separate islands and tears off the fewest frames; min_dist 0.8 gives full
# islands. Fitted through emg_manifolds.fit_umap (zero-variance columns dropped,
# RobustScaler), on the full 5,760-D dequantized token frames, no PCA.
# Fitted on the token IDs themselves: each frame is its 320 symbols and the
# distance between two frames is the share of electrodes whose token differs
# (Hamming). No codebook vectors enter, so the map shows what the symbols alone
# carry. Same neighbour count, min_dist and spread the author chose earlier.
# repulsion_strength 0.01: islands sit closer together; distances between islands carry no
# meaning in UMAP, so this only changes spacing, not within-island structure.
# Each token is replaced by its quantized codebook vector (18 values, structural
# zeros masked); cosine distance on the concatenated window. Two different codes
# that lie close in the codebook count as close, unlike Hamming on the IDs.
UMAP_SETTINGS = dict(metric="cosine", n_neighbors=50, min_dist=0.8, spread=1.0, n_epochs=500, random_state=42,
                     repulsion_strength=0.05)
# Delay embedding: each point is a frame's 320 tokens plus the tokens of the
# frames 6, 12, 18 and 24 updates earlier, i.e. the preceding 480 ms. The window
# is fixed by rule, a quarter of the median repetition (95 frames = 1.9 s), not
# chosen by appearance; loop shape is stable for windows of 0.3-1 s. History is
# taken from the continuous recording, so it may reach into the previous repetition.
HISTORY_LAGS = (6, 12, 18, 24)
if os.environ.get("FIG5_UMAP"):                       # comparison renders only, e.g. '{"n_neighbors": 200}'
    UMAP_SETTINGS.update(json.loads(os.environ["FIG5_UMAP"]))
    UMAP = UMAP.with_name(f"umap_sub01_tokens_{UMAP_SETTINGS['n_neighbors']}_{UMAP_SETTINGS['min_dist']}_{UMAP_SETTINGS['spread']}.npz")
# Display rule for the two UMAP panels, decided in the native token space with no
# reference to the picture: a frame is movement-ambiguous when fewer than half of
# its 15 nearest frames (cosine, the same metric and preprocessing as the UMAP)
# share its movement. Such frames have no movement structure for panel A to show
# and are omitted from A and B only; every number in C and D keeps every frame.
AMBIGUOUS_K, AMBIGUOUS_SHARE = 15, 0.5
CAMPAIGNS = {"1024": ROOT / "results/dpose_spar_simple_full_v1/results.json",
             "64": ROOT / "results/dpose_spar_simple_full_v64/results.json"}
OUT, PDFS = ROOT / "output/figures", ROOT / "output/pdf"
STEM = "token-pose-recovery"
# The author's hand drawing: one open hand, each finger banded with the colours of
# the movements it takes part in (thumb: thumb, 2-pinch, 3-pinch; index: index,
# 2-pinch, 3-pinch; middle: middle, 3-pinch; ring; pinky; fist bands all five).
# It is the key to panel A and sits under it, in the slot left of panel C.
HAND = ROOT / "paper/ieee/figures/assets/hand-movement-key.png"

TOKEN, FEATURE = sf.REDS[2], sf.BLUES[1]
PARTICIPANT_GREY = sf.GREYS[1]          # #939699: context marks, 3:1 on white
REFERENCE_COLOR, REFERENCE_DASHES = "#707070", (0, (4, 3))
GRID_COLOR, GRID_WIDTH = "#DEDEDE", 0.55
# Panel B colours hand closure (mm), a monotone measured quantity, so a sequential
# colour-blind-safe map is correct: light = open, dark = closed. The lightest 5 %
# of viridis is clipped so open-hand frames do not fade into the white page.
CLOSURE_CMAP_NAME = os.environ.get("FIG5_CMAP", "plasma_r")       # author's pick from an 8-map comparison; override only for sheets
_clip = {"Blues": .25, "Greys": .3, "YlGnBu": .08}.get(CLOSURE_CMAP_NAME, .05)
CLOSURE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "closure", (colorcet.cm[CLOSURE_CMAP_NAME] if CLOSURE_CMAP_NAME in colorcet.cm else plt.get_cmap(CLOSURE_CMAP_NAME))(np.linspace(_clip, 1, 256)))
# The DPose movement colours used in the earlier papers (matplotlib tab10):
# Thumb blue, Index orange, Middle green, Ring red, Little purple, Fist brown,
# three-finger pinch pink, two-finger pinch grey.
MOVEMENT_ORDER = ["Thumb", "Index", "Middle", "Ring", "Little", "Two-finger pinch", "Three-finger pinch", "Fist"]
MOVEMENT_COLOURS = {"Thumb": "#1f77b4", "Index": "#ff7f0e", "Middle": "#2ca02c", "Ring": "#d62728",
                    "Little": "#9467bd", "Fist": "#8c564b", "Three-finger pinch": "#e377c2",
                    "Two-finger pinch": "#7f7f7f"}
OKABE_ITO = [MOVEMENT_COLOURS[m] for m in MOVEMENT_ORDER]
MARKERS = ["o"] * 8
# Nominal packed payload for 320 channels at 50 updates/s.
PAYLOAD_MBIT_S = {"continuous": 320 * 50 * 18 * 32 / 1e6, "1024": 320 * 50 * 10 / 1e6, "64": 320 * 50 * 6 / 1e6}
BLE_MBIT_S = 1.4
REPS = [("continuous", "Continuous SPAR", FEATURE), ("tokens", "SPAR Tokens, 1,024 Codes", TOKEN),
        ("tokens_64", "SPAR Tokens, 64 Codes", TOKEN), ("tokens_hamming", "Token IDs, Hamming", TOKEN),
        ("pose_oracle", "Pose Space (Ceiling)", REFERENCE_COLOR)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced(err, label, cycle, phase):
    bins = np.minimum((phase * 10).astype(int), 9)
    per_movement = []
    for c in np.unique(label):
        per_cycle = []
        for cy in np.unique(cycle[label == c]):
            m = cycle == cy
            per_cycle.append(np.mean([err[m & (bins == b)].mean() for b in range(10) if (m & (bins == b)).any()]))
        per_movement.append(np.mean(per_cycle))
    return float(np.mean(per_movement))


def load_recovery():
    rows = []
    for npz in sorted(RECOVERY.glob("pose_recovery_sub*.npz")):
        d = np.load(npz)
        s = json.loads(npz.with_suffix(".json").read_text())
        label, cycle, phase = d["label"], d["cycle"], d["phase"]
        away = np.zeros(len(label), bool)
        for c in np.unique(label):
            m = label == c
            away[m] = d["from_open"][m] > 0.5 * np.percentile(d["from_open"][m], 95)
        row = dict(subject=int(s["subject"]), file=npz.name, sha256=sha(npz), frames=int(len(label)),
                   away_share=float(away.mean()), names=list(s["per_task"]))
        for rep, _, _ in REPS:
            e = d[f"err_{rep}"]
            row[rep] = balanced(e, label, cycle, phase)
            row[rep + "_away"] = float(np.median(e[away]))
        row["phase_shift"] = s["controls"]["tokens_phase_shift_median_mm"]
        row["shuffle"] = s["controls"]["tokens_global_shuffle_median_mm"]
        row["movement_mean"] = s["controls"]["movement_mean_pose_median_mm"]
        rows.append(row)
    assert rows, "no pose-recovery results found"
    return rows


def recording_history(ids_now, label):
    """For every in-cycle frame, its token IDs HISTORY_LAGS frames earlier in the
    full recording, and whether that history lies inside the recording."""
    import h5py
    from scipy.signal import butter, detrend, find_peaks, sosfiltfilt
    def cycles(sp):
        sig = sosfiltfilt(butter(4, 1.0, btype="low", fs=50, output="sos"), detrend(sp.astype(float)))
        pk, _ = find_peaks(sig, prominence=.2); tr, _ = find_peaks(-sig)
        c = np.array([(pk[i], pk[i + 1]) for i in range(len(pk) - 1)
                      if any(pk[i] < q < pk[i + 1] for q in tr)]).reshape(-1, 2)
        L = c[:, 1] - c[:, 0]; m = np.median(L)
        return c[(L >= m / 1.8) & (L <= m * 1.8)]
    hist = np.zeros((len(ids_now), 320 * len(HISTORY_LAGS)), np.int32)
    valid = np.zeros(len(ids_now), bool)
    for i, t in enumerate((2, 4, 6, 8, 10, 15, 18, 19)):
        with h5py.File(paths.RESULTS / f"token_umap/sub01_task{t}.h5") as f:
            sp, full = f["fingertip_spread_mm"][:], f["spar_ids"][:].astype(np.int32)
        inside = np.zeros(len(sp), bool)
        for a_, b_ in cycles(sp):
            inside[a_:b_] = True
        fr = np.flatnonzero(inside); rows = np.flatnonzero(label == i)
        assert len(fr) == len(rows), "frame selection differs from the recovery arrays"
        hist[rows] = np.hstack([full[np.maximum(fr - L, 0)] for L in HISTORY_LAGS])
        valid[rows] = fr >= max(HISTORY_LAGS)
    return hist, valid


def face_on(e3, label, cycle, phase, anchor, gap=.12):
    """3-D UMAP -> 2-D panel: each movement's island is projected onto its own two
    main axes (PCA of that island's points; no phase or direction information), so
    every ring faces the viewer. The plane is that of the movement's mean route
    (repetitions aligned in 5 % phase steps and averaged): a plain PCA of the
    island's points picks its longest extent, which for an elongated island is not
    the plane the loop lies in. Islands are then packed close together: start from
    their 3-D centres projected on the centres' main plane, pull toward the middle,
    and push apart any two whose 98th-percentile radii (plus a gap) overlap. Within-
    island geometry is untouched; between-island positions carry no meaning."""
    n = int(label.max()) + 1
    local, radius = {}, np.zeros(n)
    for c in range(n):
        m = label == c
        pts = e3[m] - e3[m].mean(0)
        bins = np.linspace(0, 1, 21); reps = []
        for k in np.unique(cycle[m]):
            mk = m & (cycle == k)
            per = [e3[mk & (phase >= lo_) & (phase < hi_)] for lo_, hi_ in zip(bins[:-1], bins[1:])]
            if all(len(q) for q in per):
                reps.append([q.mean(0) for q in per])
        route = np.array(reps).mean(0)
        _, _, vt = np.linalg.svd(route - route.mean(0), full_matrices=False)
        local[c] = pts @ vt[:2].T
        radius[c] = np.percentile(np.linalg.norm(local[c], axis=1), 98)
    # Island positions come from the data: `anchor` is a 2-D MDS of the cosine
    # distances between movement centroids in the native window space (fits with
    # stress ~0.01). Islands start there, scaled to their size, and are pushed
    # apart only as far as needed so that none overlap.
    g = gap * radius.mean()
    pos = anchor - anchor.mean(0)
    pos = pos * (2.2 * radius.mean() / np.median(np.linalg.norm(pos[:, None] - pos[None], axis=2)[np.triu_indices(n, 1)]))
    for _ in range(4000):
        moved = False
        for a_ in range(n):
            for b_ in range(a_ + 1, n):
                d_ = pos[a_] - pos[b_]; dist = np.linalg.norm(d_) + 1e-9; need = radius[a_] + radius[b_] + g
                if dist < need:
                    shift = (need - dist) / 2 * d_ / dist; pos[a_] += shift; pos[b_] -= shift; moved = True
        if not moved:
            break
    out = np.zeros((len(e3), 2))
    for c in range(n):
        out[label == c] = local[c] + pos[c]
    return out


def embedding():
    """Subject 1's token windows in two dimensions; refit if the cached settings differ."""
    tag = json.dumps(dict(UMAP_SETTINGS, history=HISTORY_LAGS, view="3-D UMAP, each island face-on (mean-route plane), placed at MDS of native centroid distances"), sort_keys=True)
    if UMAP.exists():
        u = np.load(UMAP)
        if "settings" in u.files and str(u["settings"]) == tag and "native_purity" in u.files:
            return u
    from umap import UMAP as _UMAP
    from sklearn.neighbors import NearestNeighbors
    d = np.load(RECOVERY / "pose_recovery_sub01.npz")
    ids = d["ids"].astype(np.int32)
    hist, valid = recording_history(ids, d["label"])
    from myoji.spar import grid_layout
    book = np.load(paths.DATA / "figure_inputs/codebooks.npz")["book_01"]
    mask = np.tile(grid_layout().mask, (1, 6))
    def quantized(window_ids):
        """(frames, 320 * k) token IDs -> concatenated codebook vectors, masked."""
        k = window_ids.shape[1] // 320
        out = np.hstack([(book[window_ids[:, 320 * j:320 * (j + 1)].astype(np.int64)] * mask[None]).reshape(len(window_ids), -1)
                         for j in range(k)]).astype(np.float32)
        return out
    W = np.hstack([ids, hist])
    X = quantized(W)
    live = X.var(0) > 1e-10
    X = X[:, live]
    e3 = _UMAP(n_components=3, **UMAP_SETTINGS).fit_transform(X)
    from sklearn.manifold import MDS
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cen = np.array([Xn[d["label"] == c].mean(0) for c in range(int(d["label"].max()) + 1)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    anchor = MDS(2, dissimilarity="precomputed", random_state=42, normalized_stress="auto", n_init=8).fit_transform(1 - cen @ cen.T)
    emb = face_on(e3, d["label"], d["cycle"], d["phase"], anchor)
    _, nn = NearestNeighbors(n_neighbors=AMBIGUOUS_K + 1, metric="cosine").fit(X).kneighbors(X)
    native_purity = (d["label"][nn[:, 1:]] == d["label"][:, None]).mean(1)
    # Control: the same window built from each repetition's frames in shuffled
    # order. If the rings came from the windowing itself, they would survive this.
    rng = np.random.default_rng(42)
    perm = np.arange(len(ids))
    for c in np.unique(d["cycle"]):
        idx = np.flatnonzero(d["cycle"] == c); perm[idx] = rng.permutation(idx)
    Xs = quantized(np.hstack([ids, hist[perm]]))[:, live]
    # The control gets exactly the same 3-D fit and face-on treatment, so the
    # rotation cannot be what makes rings.
    emb_ctrl = face_on(_UMAP(n_components=3, **UMAP_SETTINGS).fit_transform(Xs), d["label"], d["cycle"], d["phase"], anchor)
    np.savez(UMAP, embedding=emb, embedding_3d=e3, embedding_shuffled_history=emb_ctrl, label=d["label"], phase=d["phase"],
             cycle=d["cycle"], task=d["task"], from_open=d["from_open"], native_purity=native_purity,
             history_valid=valid, settings=tag)
    return np.load(UMAP)


def load_campaigns():
    out = {}
    for codes, path in CAMPAIGNS.items():
        if not path.exists():
            continue
        full = json.loads(path.read_text())
        r = full["summary"]
        paired = next(c for c in full["comparisons"] if c["reference"] == "spar_continuous")
        tok, con = np.array(r["spar_tokens"]["subject_by_seed_mm"]), np.array(r["spar_continuous"]["subject_by_seed_mm"])
        per_seed = tok.mean(0) - con.mean(0)              # the same contrast, one value per decoder seed
        out[codes] = dict(tokens=r["spar_tokens"], continuous=r["spar_continuous"], sha256=sha(path),
                          tokens_minus_continuous_mm=paired["mean_difference_mm"], p_raw=paired["p_raw"],
                          p_holm=paired["p_holm"], per_participant_mm=paired["participant_differences_mm"],
                          contrast_per_seed_mm=per_seed.tolist(), contrast_seed_sd_mm=float(per_seed.std(ddof=1)))
    assert "1024" in out, "the 1,024-code campaign results are required"
    return out


def main():
    plt.style.use(["science", "ieee", "simpetru-large"])
    plt.rcParams.update(switzer_font.rcparams())           # after the style, like Figure 5
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    rows = load_recovery()
    campaigns = load_campaigns()
    u = embedding()
    emb, label, phase = u["embedding"], u["label"], u["phase"]
    shown = (u["native_purity"] >= AMBIGUOUS_SHARE) & u["history_valid"]   # history must lie inside the recording
    ambiguous = dict(rule=f"fewer than {AMBIGUOUS_SHARE:.0%} of the {AMBIGUOUS_K} nearest frames in the native token "
                          "space share the frame's movement; omitted from panels A and B only",
                     omitted=int((~shown).sum()), share=float((~shown).mean()),
                     by_movement={n: int(((~shown) & (label == i)).sum()) for i, n in enumerate(rows[0]["names"])})
    # UMAP coordinates are arbitrary up to rotation and reflection. Rotate so the
    # direction along which the five digit islands are most ordered (Thumb ->
    # Little, by Spearman rank correlation of the island medians) runs left to
    # right. The islands' relative positions are untouched; the order the data
    # give is reported in the receipt and caption, not improved.
    from scipy.stats import spearmanr
    digit_idx = [list(rows[0]["names"]).index(n) for n in ("Thumb", "Index", "Middle", "Ring", "Little")]
    med = np.array([np.median(emb[label == c], 0) for c in digit_idx])
    angles = np.linspace(0, np.pi, 720, endpoint=False)
    rho = [spearmanr(range(5), med[:, 0] * np.cos(a) + med[:, 1] * np.sin(a))[0] for a in angles]
    theta = angles[int(np.argmax(rho))]
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # No rotation: the islands are laid out in a fixed order (see face_on); the
    # digit-axis statistic is kept in the receipt only.
    digit_axis = dict(rotation_deg=float(np.degrees(theta)), spearman=float(max(rho)),
                      left_to_right=[("Thumb", "Index", "Middle", "Ring", "Little")[i]
                                     for i in np.argsort((med @ rot.T)[:, 0])])
    names = rows[0]["names"]                       # task order in the arrays
    order = [names.index(n) for n in MOVEMENT_ORDER]
    sub1 = next(r for r in rows if r["subject"] == 1)

    # Two rows. The UMAPs are the point of the figure, so they get the top row at
    # a size where each island can carry its own name; the two dot plots are
    # short and wide underneath, which is the shape a dot plot wants.
    # 2 x 2: three views of ONE embedding (A movement, B closure, C direction) and
    # the pose-recovery chart bottom right (D).
    # FIG5_AFFINITY: render only the three UMAP panels, at the exact boxes they
    # occupy in the author's Affinity file (px on a 2148 x 1958 page at 300 dpi),
    # with no labels, colourbar, legend, letters or chart, for swapping in.
    AFF = os.environ.get("FIG5_AFFINITY")
    if AFF:
        PW, PH = 2148, 1958.4539673048805
        fig = plt.figure(figsize=(PW / 300, PH / 300))
        box = lambda x, y, w, h: [x / PW, 1 - (y + h) / PH, w / PW, h / PH]
        ax_a = fig.add_axes(box(-4, 0, 1063, 945))
        ax_b = fig.add_axes(box(1090, 0, 1063, 945))
        ax_dir = fig.add_axes(box(-4, 987, 1063, 945))
    else:
        fig = plt.figure(figsize=(7.16, 7.0))
    W, H = .44, .40
    if not AFF:
        ax_a = fig.add_axes([.035, .575, W, H])
        ax_b = fig.add_axes([.535, .575, W, H])
        ax_dir = fig.add_axes([.035, .105, W, H])
    top_strip, bot_strip = .565, .095                 # footer strips under the top and bottom UMAPs
    strip_top = top_strip
    ax_cb = fig.add_axes([.535 + .12, top_strip - .03, .20, .009])
    # Panel D spans exactly 3.35 in from its letter (x = .512) to its right edge,
    # y label included: .512 + 3.35 / 7.16 = .980 of the page width.
    D_LEFT, D_SPAN, D_YLABEL = .512, 3.35 / 7.16, .092
    ax_c = fig.add_axes([D_LEFT + D_YLABEL, .085, D_SPAN - D_YLABEL, .36])   # pose-recovery chart (lettered D)
    ax_d = fig.add_axes([0, 0, .001, .001]); ax_d.set_visible(False)
    SHOW_D = False

    # ---- A: one token UMAP, coloured by movement, each island named on itself ----
    rng = np.random.default_rng(0)
    draw = rng.permutation(len(emb))              # display order only; keeps colours interleaved
    for k, mi in enumerate(order):
        m = (label == mi) & shown
        ax_a.scatter(emb[m, 0], emb[m, 1], s=1.3, c=OKABE_ITO[k], lw=0, alpha=.55, rasterized=True)
    short = {"Two-finger pinch": "2-Pinch", "Three-finger pinch": "3-Pinch", "Little": "Pinky"}   # as in the DPose papers
    centres = np.array([[np.median(emb[(label == mi) & shown, 0]), np.median(emb[(label == mi) & shown, 1])] for mi in order])
    span = np.percentile(emb, 99.5, axis=0) - np.percentile(emb, .5, axis=0)
    # Labels start on their island; at this panel size they rarely need moving, and
    # when two would overlap they are separated along the axis with more room.
    pos = centres.copy()
    for _ in range(300):
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j:
                    continue
                dx, dy = (pos[i] - pos[j]) / span
                if abs(dx) < .22 and abs(dy) < .07:
                    if abs(dx) / .22 > abs(dy) / .07:
                        pos[i, 0] += np.sign(dx if dx else (i - j)) * (.22 - abs(dx)) * span[0] * .5
                    else:
                        pos[i, 1] += np.sign(dy if dy else (i - j)) * (.07 - abs(dy)) * span[1] * .5
    for k, mi in enumerate(order):
        name = short.get(MOVEMENT_ORDER[k], MOVEMENT_ORDER[k])
        if np.hypot(*((pos[k] - centres[k]) / span)) > .03:
            ax_a.plot([pos[k, 0], centres[k, 0]], [pos[k, 1], centres[k, 1]], color="#505050", lw=.5, zorder=4)
        if AFF:
            continue                                    # the author's hand icons stand in for the names
        ax_a.text(pos[k, 0], pos[k, 1], name, fontsize=9, ha="center", va="center", zorder=5, color="black",
                  weight="bold", path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6, foreground="white")])
    # ---- B: same coordinates, coloured by how far the hand has closed ----
    # Closure is expressed per movement, as a percentage of that movement's own
    # range: 100 % is its 95th-percentile displacement from the open-hand
    # template, the same reference the away-from-open rule uses (threshold 50 %).
    # In millimetres the pinky (10 mm of travel) would look permanently open next
    # to the fist (35 mm); per-movement scaling shows every island's gradient.
    drawn = draw[shown[draw]]
    closure_mm = u["from_open"]                       # mm, mean landmark displacement from the movement's open pose
    full_mm = np.array([np.percentile(closure_mm[label == c], 95) for c in range(len(rows[0]["names"]))])
    closure = np.minimum(100 * closure_mm / full_mm[label], 100)
    sc = ax_b.scatter(emb[drawn, 0], emb[drawn, 1], s=1.3, c=closure[drawn], cmap=CLOSURE_CMAP, vmin=0, vmax=100,
                      lw=0, alpha=.75, rasterized=True)
    bar = fig.colorbar(sc, cax=ax_cb, orientation="horizontal")
    bar.set_ticks([]); bar.outline.set_linewidth(.5)
    x0, x1 = ax_cb.get_position().x0, ax_cb.get_position().x1
    fig.text(x0, top_strip, "Hand closure", ha="left", va="top", fontsize=8, color=REFERENCE_COLOR)
    for xx, ha, t in ((x0, "left", "Open"), (x1, "right", "Closed")):
        fig.text(xx, ax_cb.get_position().y0 - .006, t, ha=ha, va="top", fontsize=8, color=REFERENCE_COLOR)

    # ---- C: what happens over time. One repetition per movement, its frames joined
    # in time order through the same map: the path out (closing) and back (opening)
    # are different routes, so each repetition draws a loop. Grey = all frames.
    direction = json.loads((paths.DATA / "figure_inputs/direction_test.json").read_text())
    dir1 = next(r for r in direction if r["subject"] == 1)
    ax_dir.scatter(emb[drawn, 0], emb[drawn, 1], s=1.0, c="#d9d9d9", lw=0, alpha=.6, zorder=1, rasterized=True)
    from matplotlib.patches import FancyArrowPatch
    from scipy.interpolate import CubicSpline
    route_bins = np.linspace(0, 1, 21)                                   # 5 % steps of the cycle

    def smooth_loop(points, sigma=1.0, n=200):
        """Closed route -> smooth closed curve: circular Gaussian smoothing over the
        bins (sigma in bins), then a periodic cubic spline through them."""
        k = len(points)
        offs = np.arange(-3, 4)
        w = np.exp(-.5 * (offs / sigma) ** 2); w /= w.sum()
        sm = sum(wi * np.roll(points, -o, axis=0) for o, wi in zip(offs, w))
        t = np.linspace(0, 1, k + 1)
        spl = CubicSpline(t, np.vstack([sm, sm[:1]]), bc_type="periodic")
        tt = np.linspace(0, 1, n)
        return spl(tt), spl
    for k, mi in enumerate(order):
        m = (label == mi) & shown
        # Keep the movement's main island only: a few frames sit in small detached
        # fragments, and a bin that mixes both would put its median in empty space.
        from sklearn.cluster import DBSCAN
        idx_m = np.flatnonzero(m)
        cl = DBSCAN(eps=.04 * float(np.mean(np.ptp(emb, 0))), min_samples=10).fit_predict(emb[idx_m])
        main = np.bincount(cl[cl >= 0]).argmax()
        m = np.zeros(len(m), bool); m[idx_m[cl == main]] = True
        # The movement's typical route: each repetition is resampled onto the same
        # 5 % phase grid (its own median position per bin), then the repetitions are
        # averaged point by point -- a mean of aligned repetitions. ROUTE_STAT=median
        # takes the point-wise median instead.
        cyc_r = u["cycle"]
        aligned = []
        for c in np.unique(cyc_r[m]):
            mc = m & (cyc_r == c)
            per = [emb[mc & (phase >= lo_) & (phase < hi_)] for lo_, hi_ in zip(route_bins[:-1], route_bins[1:])]
            if all(len(q) for q in per):
                aligned.append([np.median(q, axis=0) for q in per])
        aligned = np.array(aligned)                                       # (repetitions, 20, 2)
        pts = (np.median if os.environ.get("ROUTE_STAT") == "median" else np.mean)(aligned, axis=0)
        loop = np.vstack([pts, pts[:1]])                                  # 100 % is the same open hand as 0 %
        # Every repetition as a faint line, binned onto the same 5 % steps so each is
        # as smooth as the median; the median route drawn strong on top.
        cyc_all = u["cycle"]
        rep_bins = np.linspace(0, 1, 11)                    # 10 % steps: each repetition reads as a curve
        island = float(np.ptp(emb[m], axis=0).mean())
        drawn_reps = 0
        for c in np.unique(cyc_all[m]):
            mc = m & (cyc_all == c)
            rep = [emb[mc & (phase >= lo_) & (phase < hi_)] for lo_, hi_ in zip(rep_bins[:-1], rep_bins[1:])]
            if any(len(r) == 0 for r in rep):
                continue                                   # repetition partly outside the main island
            rp = np.array([np.median(r, axis=0) for r in rep]); rp = np.vstack([rp, rp[:1]])
            if np.linalg.norm(np.diff(rp, axis=0), axis=1).max() > .6 * island:
                continue                                   # a jump across the map, not a route
            curve, _ = smooth_loop(rp[:-1], sigma=.8)
            ax_dir.plot(curve[:, 0], curve[:, 1], color=OKABE_ITO[k], lw=.45, alpha=.2, zorder=2)
            drawn_reps += 1
        reps_drawn = locals().setdefault("reps_drawn", {}); reps_drawn[MOVEMENT_ORDER[k]] = drawn_reps
        # Light smoothing only (sigma .5 bins): heavier smoothing pulls the closing
        # and opening halves together and the loop collapses into one stroke.
        curve, spl = smooth_loop(pts, sigma=.5)
        ax_dir.plot(curve[:, 0], curve[:, 1], color=OKABE_ITO[k], lw=1.15, zorder=3, solid_capstyle="round")
        ax_dir.scatter(*spl(0.0), s=13, color=OKABE_ITO[k], ec="white", lw=.6, zorder=5)  # open hand
        for t0 in (.25, .75):                                              # one arrow closing, one opening
            ax_dir.add_patch(FancyArrowPatch(spl(t0), spl(t0 + .03), arrowstyle="-|>", mutation_scale=13,
                                             color=OKABE_ITO[k], lw=0, zorder=4))

    # View box: the central 99 % of frames, identical in all three panels.
    lo, hi = np.percentile(emb, [.5, 99.5], axis=0)
    pad = (hi - lo) * .05
    outside = int((~((emb >= lo) & (emb <= hi)).all(1)).sum())
    for ax in (ax_a, ax_b, ax_dir):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="datalim")         # rings must stay round
    fs = dict(ha="left", va="top", fontsize=8, color=REFERENCE_COLOR, linespacing=1.25)
    # The participant-1 numbers (pose from tokens, control, over-half-closed error,
    # direction) live in the caption; panel D carries the quantitative result.
    # A-C and their footer numbers are participant 1 only; D is all nine.
    # A-C show participant 1 only; stated in the caption.

    cx = ax_dir.get_position().x0 + .03
    # Legend for C: what each mark means, in neutral grey so it reads as an
    # encoding key rather than as one of the movements.
    key = "#555555"
    handles = [Line2D([], [], color=key, lw=.7, alpha=.35, label="Repetition"),
               Line2D([], [], color=key, lw=1.15, label="Mean route"),
               Line2D([], [], color=key, lw=0, marker="o", ms=5, mec="white", mew=.7, label="Open hand"),
               Line2D([], [], color=key, lw=0, marker=">", ms=5, label="Direction")]
    if not AFF:
        ax_dir.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=4, frameon=False,
                      fontsize=7.5, handlelength=1.2, handletextpad=.35, columnspacing=.8, labelcolor=REFERENCE_COLOR)


    # ---- C: paired slope chart. Each participant is one line across what the
    # receiver has: continuous features, dequantized codes at two sizes, bare
    # symbols. Flat lines are the claim; the two rules say what the level means.
    boot_rng = np.random.default_rng(42)
    def boot(d, n=10000):
        d = np.asarray(d, float); m = boot_rng.choice(d, (n, len(d))).mean(1)
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    positions = ["continuous", "tokens", "tokens_64", "tokens_hamming"]
    xlab = ["Continuous", "1,024\ncodes", "64\ncodes", "Token\nIDs"]
    xs = np.arange(len(positions))
    vals = np.array([[r[p] for p in positions] for r in rows])          # (participants, positions)
    for v in vals:
        ax_c.plot(xs, v, color="black", lw=.45, marker="o", ms=1.8, mec="none", zorder=2)
    mean = vals.mean(0)
    ax_c.plot(xs, mean, color="black", lw=1.4, zorder=3)
    ax_c.scatter([xs[0]], [mean[0]], marker="s", s=38, color=FEATURE, ec="white", lw=.6, zorder=4)
    ax_c.scatter(xs[1:], mean[1:], marker="o", s=38, color=TOKEN, ec="white", lw=.6, zorder=4)
    # Paired tests in the paper's convention (Figure 3): two-sided exact sign-flip
    # of the mean difference over all 512 participant sign patterns, Holm over the
    # three declared contrasts; stars as there, exact values in the caption.
    from paper.common.dpose_figure_statistics import exact_paired_test, holm_adjust
    tests = [exact_paired_test(vals[:, i].tolist(), vals[:, 0].tolist()) for i in range(1, len(positions))]
    holm = holm_adjust([t[1] for t in tests])
    def stars(q): return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."
    paired_c = {}
    for i, p in enumerate(positions[1:], 1):
        m, lo, hi = boot(vals[:, i] - vals[:, 0]); eff, p_raw, d = tests[i - 1]
        paired_c[p] = dict(mean=m, ci95=[lo, hi], positive=int((np.array(d) > 0).sum()), p_raw=p_raw, p_holm=holm[i - 1],
                           annotation=stars(holm[i - 1]))
        # Magnitude only: the sign-flip test stars any systematic difference however
        # small, and C's question is how large. Intervals and p are in the caption.
        ax_c.text(xs[i], 4.3, f"{m:+.2f}", ha="center", va="center", fontsize=7.5, color="black")
        # Significance bracket, Figure 3's style: each token position against the
        # continuous reference, stacked above the data, the Holm-adjusted symbol on top.
        y_bar = 11.6 + 0.75 * (i - 1)
        ax_c.plot([xs[0], xs[0], xs[i], xs[i]], [y_bar - .15, y_bar, y_bar, y_bar - .15], color="black", lw=.7,
                  zorder=2, clip_on=False)
        ax_c.text((xs[0] + xs[i]) / 2, y_bar + .05, stars(holm[i - 1]), ha="center", va="bottom", fontsize=8,
                  color="black")
    control = float(np.mean([r["phase_shift"][0] for r in rows]))
    floor = float(np.mean([r["pose_oracle"] for r in rows]))
    ctrl_test = exact_paired_test(vals[:, 1].tolist(), [r["phase_shift"][0] for r in rows])
    floor_test = exact_paired_test(vals[:, 1].tolist(), [r["pose_oracle"] for r in rows])
    for yv, text, dy, va in ((control, "No pose information", .15, "bottom"),
                             (floor, "Best achievable", -.15, "top")):
        ax_c.axhline(yv, color="black", ls=REFERENCE_DASHES, lw=.75, zorder=1)
        ax_c.text(3.4, yv + dy, text, fontsize=7.5, color="black", ha="right", va=va)
    ax_c.set_xlim(-.45, 3.45)
    ax_c.set_ylim(1.6, 13.6)
    ax_c.set_xticks(xs, xlab)
    ax_c.set_yticks([4, 6, 8, 10])                                        # the two rules carry the scale here
    ax_c.tick_params(axis="x", length=0, pad=4, labelsize=8)
    ax_c.set_ylabel("Pose Recovery Error (mm)", labelpad=4, fontsize=9.5, color="black")

    # ---- D: the decoder itself, as the paired change against continuous features,
    # with 95 % participant-bootstrap intervals and the seed-to-seed spread of the
    # same contrast as the yardstick for what a 0.1 mm change means.
    seed_sd = float(np.sqrt(np.mean([campaigns[k]["contrast_seed_sd_mm"] ** 2 for k in campaigns])))
    ax_d.axhspan(-seed_sd, seed_sd, color="#ececec", lw=0, zorder=0)
    ax_d.axhline(0, color=REFERENCE_COLOR, lw=.75, zorder=1)
    ax_d.plot(PAYLOAD_MBIT_S["continuous"], 0, marker="s", ms=6.5, color=FEATURE, mec="white", mew=.6, zorder=3)
    ax_d.annotate("Continuous SPAR", (PAYLOAD_MBIT_S["continuous"], seed_sd), xytext=(0, 5), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8.5, color=FEATURE)
    paired_d = {}
    for key, text in (("1024", "1,024 codes"), ("64", "64 codes")):
        if key not in campaigns:
            continue
        x, diffs = PAYLOAD_MBIT_S[key], np.array(campaigns[key]["per_participant_mm"])
        m, lo, hi = boot(diffs); paired_d[key] = dict(mean=m, ci95=[lo, hi])
        ax_d.scatter(x * np.exp(rng.uniform(-.1, .1, len(diffs))), diffs, s=10, color=PARTICIPANT_GREY, lw=0, zorder=2)
        ax_d.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6.5, color=TOKEN, ecolor=TOKEN,
                      elinewidth=1.1, capsize=0, mec="white", mew=.6, zorder=3)
        side = 1 if m > 0 else -1
        ax_d.annotate(text, (x, hi if side > 0 else lo), xytext=(7, 3 * side), textcoords="offset points",
                      ha="left", va="bottom" if side > 0 else "top", fontsize=8.5, color=TOKEN)
        # Significance bracket in Figure 3's style: a thin grey bar from the token
        # point across to the continuous reference, the test's symbol at its midpoint.
        y_bar = {"1024": .62, "64": .92}[key]
        symbol = "n.s." if campaigns[key]["p_holm"] >= .05 else ("***" if campaigns[key]["p_holm"] < .001 else "**" if campaigns[key]["p_holm"] < .01 else "*")
        x_ref = PAYLOAD_MBIT_S["continuous"]
        ax_d.plot([x, x, x_ref, x_ref], [y_bar - .05, y_bar, y_bar, y_bar - .05], color="#707070", lw=.7, zorder=2,
                  clip_on=False)
        ax_d.text(.42, y_bar + .02, symbol, ha="center", va="bottom", fontsize=8,   # left of the BLE rule
                  color="#707070" if symbol == "n.s." else "#303030")
    ax_d.text(22, -seed_sd - .05, "Seed-to-seed spread", fontsize=7.5, color=REFERENCE_COLOR, ha="right", va="top")
    ax_d.axvline(BLE_MBIT_S, color=REFERENCE_COLOR, ls=REFERENCE_DASHES, lw=.75, zorder=1)
    ax_d.text(BLE_MBIT_S * 1.12, 1.3, "BLE LE 2M", fontsize=8, color=REFERENCE_COLOR, ha="left", va="center")
    ax_d.set_xscale("log")
    ax_d.set_xlim(.04, 25)
    ax_d.xaxis.set_major_locator(FixedLocator([.1, 1, 10]))
    ax_d.set_xticklabels(["0.1", "1", "10"])
    ax_d.set_ylim(-1.15, 1.45)
    ax_d.set_yticks([-1, 0, 1])
    ax_d.set_xlabel("Payload (Mbit/s)", labelpad=4, fontsize=9.5)
    ax_d.set_ylabel("Decoder Error\nChange (mm)", labelpad=4, fontsize=9.5)

    for ax in (ax_c, ax_d):
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        sf.trim_spines(ax, which=("bottom",), offset=0)
        ax.tick_params(axis="y", length=0, pad=4, labelsize=9)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax_d.tick_params(axis="x", labelsize=9, length=3)
    ax_c.tick_params(axis="both", colors="black", labelcolor="black")
    ax_c.spines["bottom"].set_color("black")


    if AFF:
        # Island centres in Affinity page px (y down), for moving the hand icons.
        fig.canvas.draw()
        centres_px = {}
        for k, mi in enumerate(order):
            cx, cy = np.median(emb[(label == mi) & shown], axis=0)
            X_, Y_ = ax_a.transData.transform((cx, cy))
            centres_px[MOVEMENT_ORDER[k]] = [float(X_ / fig.dpi * 300), float(PH - Y_ / fig.dpi * 300)]
        for ax in (ax_cb, ax_c):
            ax.set_visible(False)
        for t in list(fig.texts):
            t.remove()
        fig.savefig(AFF)
        fig.savefig(Path(AFF).with_suffix(".png"), dpi=600, transparent=True)   # for in-place bitmap swaps
        Path(AFF).with_suffix(".centres.json").write_text(json.dumps(centres_px, indent=1))
        plt.close(fig); return
    for x, yy, letter in ((.012, .99, "A"), (.512, .99, "B"), (.012, .52, "C"), (.512, .52, "D")):
        fig.text(x, yy, letter, fontsize=11, weight="bold", va="top")
    assert not sf.check_font_sizes(fig)
    if os.environ.get("FIG5_OUT"):                       # comparison render: one PDF, nothing else written
        fig.text(.5, .005, f"panel B colourmap: {CLOSURE_CMAP_NAME}", ha="center", va="bottom", fontsize=8, color=REFERENCE_COLOR)
        fig.savefig(os.environ["FIG5_OUT"]); plt.close(fig); return
    fig.savefig(PDFS / f"{STEM}.pdf"); fig.savefig(OUT / f"{STEM}.svg")
    fig.savefig(OUT / f"{STEM}.png", dpi=400, facecolor="white")
    plt.close(fig)

    # ---- receipt: every number the caption can quote ----
    tok = np.array([r["tokens"] for r in rows]); con = np.array([r["continuous"] for r in rows])
    receipt = dict(
        participants=[r["subject"] for r in rows], frames=[r["frames"] for r in rows],
        rule="nearest frame in each other repetition cycle (own cycle and one neighbour excluded on each side), "
             "5 closest cycles, mean of their poses; mean 3D distance over 20 wrist-relative landmarks",
        aggregation="frame errors averaged within 10 phase bins, bins within cycles, cycles within movements, "
                    "movements within participant; participants as the replication unit",
        balanced_mm={rep: dict(mean=float(np.mean([r[rep] for r in rows])),
                               sd=float(np.std([r[rep] for r in rows], ddof=1)) if len(rows) > 1 else None,
                               per_participant=[r[rep] for r in rows]) for rep, _, _ in REPS},
        tokens_minus_continuous_mm=dict(mean=float((tok - con).mean()), per_participant=(tok - con).tolist(),
                                        positive=int(((tok - con) > 0).sum())),
        away_from_open_median_mm={rep: [r[rep + "_away"] for r in rows] for rep, _, _ in REPS},
        controls=dict(phase_shift_median_mm=[r["phase_shift"] for r in rows],
                      global_shuffle_median_mm=[r["shuffle"] for r in rows],
                      movement_mean_pose_median_mm=[r["movement_mean"] for r in rows]),
        decoder=campaigns, payload_mbit_s=PAYLOAD_MBIT_S, ble_mbit_s=BLE_MBIT_S,
        panel_c=dict(form="paired slope chart, one line per participant", positions=positions,
                     mean_mm=mean.tolist(), paired_vs_continuous=paired_c,
                     test="two-sided exact paired sign-flip of the mean difference, all 512 participant sign patterns, "
                          "Holm over the three contrasts against continuous; stars as in Figure 3",
                     rules=dict(phase_shifted_control_mm=control, neighbours_in_pose_space_mm=floor,
                                tokens_vs_control=dict(mean_mm=ctrl_test[0], p_raw=ctrl_test[1]),
                                tokens_vs_pose_space=dict(mean_mm=floor_test[0], p_raw=floor_test[1]))),
        panel_d=dict(paired_change_vs_continuous=paired_d, interval="95 % participant bootstrap, 10,000 resamples, pointwise, uncorrected",
                     seed_to_seed_band_mm=seed_sd, seed_to_seed_note="RMS over the two campaigns of the SD, across the three "
                     "decoder seeds, of the tokens-minus-continuous macro contrast"),
        umap=dict(file=UMAP.name, sha256=sha(UMAP), participant=1, frames=int(len(emb)), frames_out_of_view=outside,
                  view="central 99 % of frames per axis, 5 % padding, 16 % strip reserved below",
                  digit_axis=digit_axis, ambiguous_frames=ambiguous,
                  panel_b_colour=dict(quantity="mean displacement of the 20 landmarks from the movement's open-hand "
                                               "template (first and last 5 % of each cycle), as a percentage of that "
                                               "movement's 95th-percentile displacement; clipped at 100",
                                      full_closure_mm_per_movement=dict(zip(rows[0]["names"], full_mm.round(2).tolist())),
                                      cmap=f"{CLOSURE_CMAP_NAME}, lightest {_clip:.0%} removed"),
                  settings=dict(UMAP_SETTINGS, input="each token replaced by its quantized codebook vector; frame plus "
                                "tokens 6, 12, 18 and 24 updates earlier (480 ms); cosine distance; no scaling"),
                  space="dequantized 1,024-code token frames, 320 x 18 masked", illustration_only=True),
        sources={r["file"]: r["sha256"] for r in rows} | {str(Path(__file__).name): sha(Path(__file__))},
    )
    (OUT / f"{STEM}.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("balanced_mm", "tokens_minus_continuous_mm")}, indent=1))


if __name__ == "__main__":
    main()
