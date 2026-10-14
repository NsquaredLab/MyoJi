from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from myoji.learned import (
    TokenizerConfig,
    UniversalEMGTokenizerModule,
    atomic_torch_save,
    build_default_tokenizer,
    last_checkpoint_path,
    load_universal_checkpoint,
    restore_training_state,
    training_state,
)


def test_training_checkpoint_round_trip(tmp_path) -> None:
    config = TokenizerConfig(embedding_dim=16, temporal_layers=1, codebook_size=32)
    module = UniversalEMGTokenizerModule(
        build_default_tokenizer(config), spectral_bins=8
    )
    optimizer = module.configure_optimizers()
    source = SimpleNamespace(
        name="test",
        montage=SimpleNamespace(channels=4),
        source_rate_hz=2_000,
        seen_recordings={"one"},
        available_recordings=2,
    )
    rng = np.random.default_rng(7)
    progress = {"phase": "quantized", "continuous_step": 4, "quantized_step": 3}
    history = [{"loss": 1.0}]
    path = tmp_path / "run.last.pt"
    reference = next(module.parameters()).detach().clone()

    atomic_torch_save(
        training_state(
            module,
            optimizer,
            config,
            [source],
            progress,
            history,
            rng,
            torch.device("cpu"),
        ),
        path,
    )
    with torch.no_grad():
        next(module.parameters()).zero_()
    source.seen_recordings.clear()

    restored_progress, restored_history = restore_training_state(
        path,
        module,
        optimizer,
        config,
        [source],
        rng,
        torch.device("cpu"),
    )

    torch.testing.assert_close(next(module.parameters()), reference)
    assert restored_progress == progress
    assert restored_history == history
    assert source.seen_recordings == {"one"}
    assert last_checkpoint_path(tmp_path / "run.pt").name == "run.last.pt"


def small_config() -> TokenizerConfig:
    return TokenizerConfig(
        embedding_dim=16,
        temporal_layers=1,
        temporal_kernel_size=3,
        codebook_size=32,
        muscle_slots=2,
        channel_attention_heads=2,
    )


def test_load_universal_checkpoint_requires_completion_and_restores_weights(tmp_path) -> None:
    config = small_config()
    module = UniversalEMGTokenizerModule(build_default_tokenizer(config), spectral_bins=8)
    path = tmp_path / "universal.pt"
    torch.save(
        {"state_dict": module.state_dict(), "config": asdict(config), "progress": {"phase": "quantized"}},
        path,
    )
    with pytest.raises(ValueError):
        load_universal_checkpoint(path, torch.device("cpu"))

    torch.save(
        {"state_dict": module.state_dict(), "config": asdict(config), "progress": {"phase": "complete"}},
        path,
    )
    loaded, checkpoint = load_universal_checkpoint(path, torch.device("cpu"))
    assert not loaded.training
    assert loaded.spectral_bins == 8
    assert checkpoint["progress"]["phase"] == "complete"
    for name, value in module.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
