"""Frozen language-model backbones for a bounded DPose adaptation pilot."""

from contextlib import nullcontext
import hashlib

import torch
from torch import nn
from myoji._vendor.channel_tokenization import GeometryAwareChannelPool


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ARMS = ("tokens_pretrained", "tokens_random", "tokens_bypass", "continuous_pretrained")


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class EMGToLanguagePose(nn.Module):
    """Learn frame embeddings and pose regression; keep the backbone frozen.

    EMG IDs are looked up in the EMG codebook outside this class. They are
    never interpreted as text-vocabulary IDs. Backpropagation through the
    frozen backbone must remain enabled to train the input adapter.
    """

    def __init__(self, backbone, hidden_size, *, seed, width=64, slots=4):
        super().__init__()
        self.backbone = backbone
        if backbone is not None:
            backbone.requires_grad_(False)
            backbone.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Linear(256, width)
            self.pool = GeometryAwareChannelPool(
                embedding_dim=width,
                geometry_dim=14,
                num_slots=slots,
                num_heads=4,
                query_residual=False,
            )
            self.bridge = nn.Sequential(
                nn.LayerNorm(slots * width),
                nn.Linear(slots * width, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.pose = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 60))
        self.input_scale = 0.02

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def adapter_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def forward(self, features, descriptors):
        if features.ndim != 4 or features.shape[-1] != 256:
            raise ValueError("Expected batch, channels, time, 256 EMG features")
        x = self.project(features)
        spatial = self.pool.spatialize(x, descriptors)
        pooled = self.pool(spatial, descriptors, spatialized=True)
        batch, slots, steps, width = pooled.shape
        frames = pooled.permute(0, 2, 1, 3).reshape(batch, steps, slots * width)
        frames = self.bridge(frames) * self.input_scale
        if self.backbone is not None:
            dtype = next(self.backbone.parameters()).dtype
            frames = self.backbone(
                inputs_embeds=frames.to(dtype), use_cache=False, return_dict=True
            ).last_hidden_state
        return self.pose(frames.float())


def build_model(model_dir, arm, seed, device="cpu", dtype=torch.float32):
    from transformers import AutoConfig, AutoModel

    if arm not in ARMS:
        raise ValueError(arm)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    config.use_cache = False
    config._attn_implementation = "sdpa"
    loading = {}
    if arm.endswith("pretrained"):
        backbone, loading = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        unexpected = [k for k in loading["unexpected_keys"] if k != "lm_head.weight"]
        if loading["missing_keys"] or loading["mismatched_keys"] or unexpected or loading["error_msgs"]:
            raise RuntimeError(f"Incomplete pretrained backbone: {loading}")
    elif arm == "tokens_random":
        config.vocab_size = 1  # Unused text embedding; all inputs use inputs_embeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100000)
            backbone = AutoModel.from_config(config, trust_remote_code=False).to(dtype)
    else:
        backbone = None
    if backbone is not None:
        # Retain one unused row instead of storing the 151k-word embedding table.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            backbone.set_input_embeddings(nn.Embedding(1, config.hidden_size).to(dtype))
        backbone.config.vocab_size = 1
    model = EMGToLanguagePose(backbone, config.hidden_size, seed=seed).to(device)
    return model, loading


def autocast_for(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
