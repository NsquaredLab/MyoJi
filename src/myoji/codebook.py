"""SPAR tokens: train-only standardization and a Lloyd (k-means) codebook.

Descriptors are standardized with statistics pooled over training windows only
(absent boundary edges excluded), a seeded Lloyd quantizer is fitted on a uniform
training subset, and every descriptor is replaced by its nearest code index.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def fit_standardization(features, mask):
    """Pool training (windows,channels,frames,features), excluding absent edges."""
    x = np.asarray(features, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 4 or mask.shape != (x.shape[1], x.shape[3]) or not np.isfinite(x).all():
        raise ValueError("finite training features (N,C,T,D) and mask (C,D) required")
    count = mask.sum(0) * x.shape[0] * x.shape[2]
    if (count == 0).any():
        raise ValueError("each descriptor component needs at least one observed electrode")
    total = np.where(mask[None, :, None, :], x, 0).sum((0, 1, 2), dtype=np.float64)
    mean = total / count
    squared = np.where(mask[None, :, None, :], (x.astype(np.float64) - mean) ** 2, 0).sum((0, 1, 2))
    scale = np.maximum(np.sqrt(squared / count), 1e-4)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(features, mean, scale, mask):
    return (((features - mean) / scale) * np.asarray(mask)[None, :, None, :]).astype(np.float32)


@torch.inference_mode()
def nearest(vectors, codebook, batch_size=8192, return_distance=False):
    if vectors.ndim != 2 or codebook.ndim != 2 or vectors.shape[1] != codebook.shape[1] or batch_size < 1:
        raise ValueError("vectors and codebook must have matching feature dimensions")
    ids, errors = [], []
    for start in range(0, len(vectors), batch_size):
        x = vectors[start:start + batch_size]
        distance = (x.square().sum(-1, keepdim=True) + codebook.square().sum(-1)[None] - 2 * x @ codebook.T).clamp_min(0)
        error, index = distance.min(-1)
        ids.append(index)
        errors.append(error)
    if not ids:
        raise ValueError("cannot quantize an empty array")
    indices = torch.cat(ids)
    return (indices, torch.cat(errors)) if return_distance else indices


@torch.inference_mode()
def fit_codebook(training_features, *, vocabulary=1024, seed=260911, max_vectors=131072,
                 iterations=40, device="cpu", batch_size=8192):
    """Lloyd vector quantizer. Only caller-supplied training descriptors are read.

    A fixed uniform subset bounds memory/compute. Initial centres are a seeded
    sample; empty centres are replaced by the most poorly represented samples.
    Labels, validation descriptors and test descriptors never enter this function.
    """
    array = np.asarray(training_features, dtype=np.float32)
    array = array.reshape(-1, array.shape[-1])
    if not np.isfinite(array).all() or not 2 <= vocabulary <= min(len(array), max_vectors) or iterations < 1:
        raise ValueError("finite training data, sufficient vectors and positive iterations required")
    generator = np.random.default_rng(seed)
    sampled = generator.choice(len(array), size=min(max_vectors, len(array)), replace=False)
    x = torch.as_tensor(array[sampled], dtype=torch.float32, device=device)
    initial = generator.choice(len(x), size=vocabulary, replace=False)
    book = x[torch.as_tensor(initial, device=device)].clone()
    history = []
    for step in range(iterations):
        ids, errors = nearest(x, book, batch_size, return_distance=True)
        counts = torch.bincount(ids, minlength=vocabulary)
        sums = torch.zeros_like(book)
        sums.index_add_(0, ids, x)
        updated = sums / counts.clamp_min(1)[:, None]
        empty = counts == 0
        if empty.any():
            updated[empty] = x[errors.topk(int(empty.sum())).indices]
        history.append(dict(iteration=step + 1, mean_squared_distance=float(errors.mean()), used_codes=int((counts > 0).sum())))
        shift = float((updated - book).square().mean())
        book = updated
        if step >= 4 and shift < 1e-6:
            break
    _, final_errors = nearest(x, book, batch_size, return_distance=True)
    receipt = dict(kind="Lloyd vector quantizer", vocabulary=vocabulary, vector_dim=x.shape[-1], seed=seed,
                   training_vectors=len(array), fitted_vectors=len(x), iterations=len(history), history=history,
                   fitted_sample_mse=float(final_errors.mean() / x.shape[-1]), selection="Uniform training-only subset, no labels",
                   floating_point="float32; seeded sampling; CUDA index_add accumulation may vary at roundoff level")
    return book.cpu().numpy(), receipt


@torch.inference_mode()
def tokenize(features, codebook, *, device="cpu", batch_size=8192):
    x = np.asarray(features, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature vectors")
    book = torch.as_tensor(codebook, dtype=torch.float32, device=device)
    flat = x.reshape(-1, x.shape[-1])
    ids = []
    for start in range(0, len(flat), batch_size):
        vectors = torch.as_tensor(flat[start:start + batch_size], dtype=torch.float32, device=device)
        ids.append(nearest(vectors, book, batch_size).cpu().numpy().astype(np.uint16))
    if len(book) > 65536:
        raise ValueError("this compact storage format supports at most 65,536 codes")
    return np.concatenate(ids).reshape(x.shape[:-1])
