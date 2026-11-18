"""Tests for Hot SSD (256GB) <-> Cold HDD (1TB) Model Vault tiering."""

from pathlib import Path

from nglai.config import NGLAIConfig
from nglai.engine.catalog import MODEL_CATALOG
from nglai.engine.vault import ModelVault


def test_hot_cold_vault_promotion_and_eviction(tmp_path: Path) -> None:
    hot = tmp_path / "ssd_256gb"
    cold = tmp_path / "hdd_1tb"
    cfg = NGLAIConfig(
        base_dir=str(tmp_path),
        hot_ssd_dir=str(hot),
        cold_hdd_dir=str(cold),
    )
    vault = ModelVault(cfg)

    m1 = MODEL_CATALOG["qwen3.6-35b-a3b-ud-iq3"]
    m2 = MODEL_CATALOG["qwen3-coder-30b-a3b-ud-iq3"]

    # Simulate two models archived in Cold 1TB HDD
    (cold / m1.filename).write_bytes(b"FAKE_GGUF_WEIGHTS_1")
    (cold / m2.filename).write_bytes(b"FAKE_GGUF_WEIGHTS_2")

    # Activate m1 -> promoted to Hot SSD
    active_m1 = vault.resolve_active_path(m1.id, promote_to_hot=True)
    assert active_m1 == hot / m1.filename
    assert (hot / m1.filename).exists()

    # Switch to m2 -> m1 should be automatically evicted from Hot SSD while staying safe on Cold HDD!
    active_m2 = vault.resolve_active_path(m2.id, promote_to_hot=True)
    assert active_m2 == hot / m2.filename
    assert (hot / m2.filename).exists()
    assert not (hot / m1.filename).exists()
    assert (cold / m1.filename).exists()
