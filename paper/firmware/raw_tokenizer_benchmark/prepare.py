"""Export frozen raw-input fixtures and independent PyTorch reference outputs.

This is a numerical/engineering replay of a previously exported TRAIN segment,
not a new validation/test performance experiment. No labels or fitting are used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/raw_tokenizers_cortex_m4_v1"
SRC = ROOT / "embedded/raw_tokenizers"
from paper.common import paths
from myoji.spar import grid_layout, spar_simple_dataset_batched

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def floats(name, array):
    a = np.asarray(array, dtype=np.float32)
    return f"const float {name}[{a.size}] = {{\n" + ",\n".join(
        ",".join(float(v).hex() + "f" for v in a.ravel()[i:i+16]) for i in range(0,a.size,16)) + "\n};\n"

def raw_fixture():
    """Participant 1, task 2: the first campaign window, 2,000 samples into the 2 kHz recording.

    Its first 2,120 samples hold the first three token endpoints (2039, 2079, 2119). The saved IDs
    are the frozen tokenizer's tokens at those endpoints, an independent check of the port below.
    """
    import joblib
    from myoji.learned import Tokenizer
    from paper.hdemg.run_hdemg_pilot import resample  # the paper's causal 2,048 -> 2,000 Hz FIR
    path = paths.HDEMG / "Sub1_emg.pkl"
    assert sha(path) == "3c4e6bb5c8a6f667bc621472e9dd19ba6465a1f621ce5bbfe971a84c1a100db2", \
        "expects the paper's 8-task export of participant 1"
    raw = np.ascontiguousarray(resample(joblib.load(path)["2"])[:, 2000:4120])
    assert raw.shape == (320,2120) and np.isfinite(raw).all()
    encoded = Tokenizer(paths.LEARNED_TOKENIZER).encode(raw, sample_rate_hz=2000)
    assert encoded["end_sample_indices"][-3:] == [2039,2079,2119]
    saved = np.asarray(encoded["tokens"])[:,-3:].T.astype(np.uint16)
    return raw, saved, dict(dataset="HD-EMG", participant=1, recording=path.name, recording_sha256=sha(path), task=2,
        recording_start_sample=2000, cached_window=2, source_split="train",
        selection="First campaign window of participant 1, task 2; first three token endpoints (2039,2079,2119), chosen before running the C port",
        input_interface="320 channels of raw float32 EMG at 2000 Hz, before prefix normalization",
        upstream_resampling="Causal 2048-to-2000 Hz resampling of the whole task recording, outside the timed interface")

def torch_learned(x, state):
    p = "frontend.channel_encoder.encoder."
    out = F.conv1d(x[:,None],state[p+"0.weight"],state[p+"0.bias"],stride=40)
    out = F.conv1d(F.gelu(out),state[p+"2.weight"],state[p+"2.bias"])
    for layer in range(4):
        p = f"temporal_encoder.{layer}."
        n = F.layer_norm(out.transpose(1,2),(256,),state[p+"norm.weight"],state[p+"norm.bias"]).transpose(1,2)
        n = F.conv1d(F.pad(n,(4,0)),state[p+"depthwise.weight"],state[p+"depthwise.bias"],groups=256)
        n = F.conv1d(n,state[p+"pointwise.0.weight"],state[p+"pointwise.0.bias"])
        n = F.conv1d(F.gelu(n),state[p+"pointwise.2.weight"],state[p+"pointwise.2.bias"])
        out = out+n
    return F.layer_norm(out.transpose(1,2),(256,),state["pre_quantizer_norm.weight"],state["pre_quantizer_norm.bias"])

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    raw, saved_ids, source = raw_fixture()
    x = torch.from_numpy(raw)
    centre = x[:,:1000].sort(-1).values[:,499:500]
    scale = (1.4826*(x[:,:1000]-centre).abs().sort(-1).values[:,499:500]).clamp_min(1.e-4)
    normalized = ((x-centre)/scale).clamp(-12,12)
    raw.astype("<f4").tofile(OUT/"raw_320x2120.f32")
    ckpt = paths.LEARNED_TOKENIZER
    assert sha(ckpt) == "8bba098480d570bdbfac110771957c0cb12ce103ed7c12940858bf586612a9a7"
    checkpoint = torch.load(ckpt,map_location="cpu",weights_only=True)
    assert checkpoint["calibration_samples"] == 1000
    assert checkpoint["preprocessing"] == "prefix-sorted-median-mad-f32-clip12-v2"
    state = checkpoint["state_dict"]
    keys = [k for k in state if k.startswith("frontend.channel_encoder.") or
            k.startswith("temporal_encoder.") or k.startswith("pre_quantizer_norm.") or k == "quantizer.codebook"]
    packed, offsets, offset = [], {}, 0
    for k in keys:
        values = state[k].numpy().astype("<f4").ravel()
        offsets[k] = offset
        offset += values.size
        packed.append(values)
    np.concatenate(packed).tofile(OUT/"learned_weights.f32")
    blocks = []
    for l in range(4):
        blocks.append([offsets[f"temporal_encoder.{l}."+n] for n in (
            "norm.weight","norm.bias","depthwise.weight","depthwise.bias",
            "pointwise.0.weight","pointwise.0.bias","pointwise.2.weight","pointwise.2.bias")])
    names = dict(PATCH_W="frontend.channel_encoder.encoder.0.weight", PATCH_B="frontend.channel_encoder.encoder.0.bias",
        PROJECT_W="frontend.channel_encoder.encoder.2.weight", PROJECT_B="frontend.channel_encoder.encoder.2.bias",
        FINAL_W="pre_quantizer_norm.weight", FINAL_B="pre_quantizer_norm.bias", BOOK="quantizer.codebook")
    header = "/* Generated from the frozen release; offsets are float counts. */\n"
    header += "\n".join(f"#define {name} {offsets[k]}u" for name,k in names.items()) + "\n"
    header += "static const unsigned BLOCK_OFFSETS[4][8] = {\n"+",\n".join("{"+",".join(map(str,b))+"}" for b in blocks)+"\n};\n"
    (OUT/"weight_offsets.h").write_text(header)
    # Participant 1's SPAR codebook, standardization and mask, from MYOJI_ASSETS.
    data_path = paths.ASSETS/"spar/codebooks-1024.npz"
    bundle = np.load(data_path)
    data = {k: bundle[f"subject_01/{k}"] for k in ("codebook", "feature_mean", "feature_scale", "mask")}
    layout = grid_layout()
    assert np.array_equal(data["mask"],np.tile(layout.mask,(1,6)))
    constants = '#include "common.h"\n'
    for name,a in [("spar_codebook",data["codebook"]),("spar_mean",data["feature_mean"]),("spar_scale",data["feature_scale"]),
                   ("hann128",torch.hann_window(128,periodic=False).numpy()),
                   ("fft_cos",np.cos(-2*np.pi*np.arange(64)/128)),("fft_sin",np.sin(-2*np.pi*np.arange(64)/128))]:
        constants += floats(name,a)
    constants += "const uint16_t spar_members[320][4] = {\n"+",\n".join("{"+",".join(map(str,r))+"}" for r in layout.members)+"\n};\n"
    constants += "const uint8_t spar_mask[320][3] = {\n"+",\n".join("{"+",".join(map(str,r.astype(int)))+"}" for r in layout.mask)+"\n};\n"
    (OUT/"constants.c").write_text(constants)
    # All 17 causal patch dependencies are present at endpoint 2039. Earlier
    # zero padding is flushed by four four-tap-history residual blocks.
    continuous = torch.cat([torch_learned(normalized[c:c+16,1080:],state)[:,-3:] for c in range(0,320,16)])
    learned = continuous.permute(1,0,2).contiguous().numpy()
    book = state["quantizer.codebook"]
    flattened = continuous.reshape(-1,256)
    distance = flattened.square().sum(-1,keepdim=True)-2*flattened@book.T+book.square().sum(-1)[None]
    ids = distance.argmin(-1).reshape(320,3).T.numpy().astype("uint16")
    assert np.array_equal(ids,saved_ids), "Reference port must match all 960 previously saved IDs"
    endpoints = np.array([2039,2079,2119])
    windows = normalized.numpy()[:,endpoints[:,None]-np.arange(319,-1,-1)].transpose(1,2,0).copy()
    spar_raw = spar_simple_dataset_batched(windows,channelwise=True)
    spar = ((spar_raw-data["feature_mean"])/data["feature_scale"])*data["mask"][None]
    s = torch.from_numpy(spar.reshape(-1,18))
    b = torch.from_numpy(data["codebook"])
    spar_distance = s.square().sum(-1,keepdim=True)+b.square().sum(-1)[None]-2*s@b.T
    spar_ids = spar_distance.argmin(-1).reshape(3,320).numpy().astype("uint16")
    np.savez(OUT/"reference.npz",centre=centre.numpy(),scale=scale.numpy(),normalized=normalized.numpy(),
             learned_features=learned,learned_ids=ids,spar_features=spar,spar_ids=spar_ids,
             spar_book=data["codebook"],learned_book=book.numpy())
    record = dict(schema=1,scope="Frozen raw EMG to token IDs, both routes; no decoder, fitting, labels, or BLE",
        source=source,raw_fixture_sha256=sha(OUT/"raw_320x2120.f32"), endpoints=endpoints.tolist(),
        learned_checkpoint_sha256=sha(ckpt),learned_constants_bytes=offset*4,
        learned_encoder_parameters=sum(state[k].numel() for k in keys if k != "quantizer.codebook"),
        learned_codebook_bytes=book.numel()*4, spar_checkpoint_source_sha256=sha(data_path),
        spar_codebook_bytes=int(data["codebook"].nbytes),learned_saved_tokens_checked=960,
        learned_saved_token_agreement=1.,weight_offsets=offsets,
        calibration="First 1000 samples/channel: lower median and lower MAD * 1.4826, scale floor 1e-4, clip +/-12",
        reference_sha256=sha(OUT/"reference.npz"),torch_version=torch.__version__,cpu_threads=4,
        reference_source_sha256={p:sha(paths.REPO/p) for p in ["src/myoji/spar.py",
            "src/myoji/learned/encoder.py","src/myoji/preprocessing.py",
            "paper/firmware/raw_tokenizer_benchmark/prepare.py"]})
    write(OUT/"fixture_receipt.json",record)
    print(json.dumps({k:v for k,v in record.items() if k not in ("weight_offsets","source","reference_source_sha256")}),flush=True)

if __name__ == "__main__": main()
