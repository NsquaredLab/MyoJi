"""Vectorized, causal 1€ filtering, matched to the authors' Python reference.

Adapted from casiez/OneEuroFilter, commit d78925584245597f2aa9c4c01a802eb0f0b77fb9.
Copyright 2019 Inria; BSD-3-Clause terms in vendor/OneEuroFilter-LICENSE.
The unchanged reference is retained in vendor/one_euro_filter.py.
"""

import numpy as np

MIN_CUTOFFS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BETAS = (0.0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1)
CANDIDATES = [{"kind": "identity"}] + [
    {"kind": "one_euro", "mincutoff": fc, "beta": beta, "dcutoff": 1.0, "freq": 50.0}
    for fc in MIN_CUTOFFS
    for beta in BETAS
]


def smooth(values, *, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=50.0):
    """Filter each coordinate/window independently along the penultimate axis.

    Inputs and outputs use stored pose units. The derivative uses the previous
    *filtered* position, exactly as in the pinned authors' implementation.
    First output equals first input, derivative state starts at zero. No shared
    state across windows or dependence on future frames.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] < 1 or not np.isfinite(values).all():
        raise ValueError("expected finite (..., time, coordinates) values")
    if (
        not np.isfinite([mincutoff, beta, dcutoff, freq]).all()
        or min(mincutoff, dcutoff, freq) <= 0
        or beta < 0
    ):
        raise ValueError(
            "positive finite cutoffs/frequency and nonnegative beta required"
        )

    def alpha(cutoff):
        return 1.0 / (1.0 + freq / (2.0 * np.pi * cutoff))

    result = np.empty_like(values)
    previous = values[..., 0, :].copy()
    derivative = np.zeros_like(previous)
    result[..., 0, :] = previous
    derivative_alpha = alpha(dcutoff)
    for t in range(1, values.shape[-2]):
        current = values[..., t, :]
        raw_derivative = (current - previous) * freq
        derivative = (
            derivative_alpha * raw_derivative + (1 - derivative_alpha) * derivative
        )
        weight = alpha(mincutoff + beta * np.abs(derivative))
        previous = weight * current + (1 - weight) * previous
        result[..., t, :] = previous
    return result


def apply(values, config):
    if config["kind"] == "identity":
        return np.asarray(values, dtype=np.float64).copy()
    assert config["kind"] == "one_euro"
    return smooth(values, **{k: v for k, v in config.items() if k != "kind"})


def metrics(prediction, target):
    """Three seeds, N windows, 100 frames, 60 coordinates; score final 50."""
    prediction = np.asarray(prediction, dtype=np.float64)[..., -50:, :]
    target = np.asarray(target, dtype=np.float64)[..., -50:, :]
    assert prediction.ndim == 4 and prediction.shape[1:] == target.shape
    assert target.shape[-2:] == (50, 60)
    residual = prediction - target
    denominator = np.square(target - target.mean((0, 1))).sum()
    pv, tv = 50 * np.diff(prediction, axis=2), 50 * np.diff(target, axis=1)
    velocity_denominator = np.square(tv - tv.mean((0, 1))).sum()
    assert denominator > 0 and velocity_denominator > 0
    return [
        {
            "mean_3d_error": float(
                np.linalg.norm(delta.reshape(-1, 20, 3), axis=-1).mean()
            ),
            "coordinate_r2": float(1 - np.square(delta).sum() / denominator),
            "velocity_r2": float(
                1 - np.square(pv[seed] - tv).sum() / velocity_denominator
            ),
            "velocity_mse": float(np.square(pv[seed] - tv).mean()),
            "velocity_energy_ratio": float(
                np.square(pv[seed]).sum() / np.square(tv).sum()
            ),
        }
        for seed, delta in enumerate(residual)
    ]


def choose(validation):
    """One global parameter set, selected exclusively from validation arrays."""
    candidates = []
    for index, config in enumerate(CANDIDATES):
        rows = []
        for subject, data in validation.items():
            scores = metrics(apply(data["prediction"], config), data["target"])
            rows.extend(
                dict(score, subject=int(subject), seed_index=seed)
                for seed, score in enumerate(scores)
            )
        candidates.append(
            {
                "index": index,
                "config": config,
                "mean_3d_error": float(np.mean([r["mean_3d_error"] for r in rows])),
                "velocity_r2": float(np.mean([r["velocity_r2"] for r in rows])),
                "per_subject_seed": rows,
            }
        )
    selected = min(candidates, key=lambda r: (r["mean_3d_error"], r["index"]))
    return {
        "selected": selected,
        "candidates": candidates,
        "criterion": "Lowest validation macro-person/seed mean 3D error; identity is eligible, earliest listed candidate wins ties. No test-based selection.",
    }


def step_responses(config):
    """Synthetic filter delay diagnostic; units/amplitude matter for beta."""
    rows = []
    for amplitude in (1.0, 10.0, 100.0):
        x = np.zeros((550, 1))
        x[50:] = amplitude
        y = apply(x, config)[:, 0]
        row = {"step_amplitude_stored_units": amplitude}
        for fraction in (0.5, 0.9):
            crossed = np.flatnonzero(y[50:] >= fraction * amplitude)
            row[f"t{int(100 * fraction)}_ms"] = (
                int(crossed[0] * 20) if len(crossed) else None
            )
        rows.append(row)
    return rows
