"""Tests for the architecture adapter."""

import pytest
import torch.nn as nn

from tests.mocks.architecture_adapter import (
    MockArchitectureAdapter,
    mock_adapter,
    mock_model_adapter,
)
from tests.mocks.models import MockGemma3Model
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)


def test_get_remote_component_with_mock(
    mock_adapter: MockArchitectureAdapter, mock_model_adapter: nn.Module
):
    """Test get_remote_component with the mock adapter."""
    # Test direct mapping
    ln_final = mock_adapter.get_component(mock_model_adapter, "ln_final")
    assert isinstance(ln_final, nn.LayerNorm)

    # Test block mapping
    block = mock_adapter.get_component(mock_model_adapter, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = mock_adapter.get_component(mock_model_adapter, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = mock_adapter.get_component(mock_model_adapter, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = mock_adapter.get_component(mock_model_adapter, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


class DummyHFConfig:
    def __init__(self):
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 128
        # Add any other attributes needed by the adapter here


@pytest.fixture
def cfg():
    return DummyHFConfig()


@pytest.fixture
def adapter(cfg) -> Gemma3ArchitectureAdapter:
    """Create a Gemma3 adapter."""
    return Gemma3ArchitectureAdapter(cfg)


@pytest.fixture
def model() -> MockGemma3Model:
    """Create a mock Gemma 3 model."""
    return MockGemma3Model()


def test_translate_transformer_lens_path(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation from TransformerLens to Remote paths."""
    # Test direct mapping
    assert adapter.translate_transformer_lens_path("embed") == "model.embed_tokens"
    assert adapter.translate_transformer_lens_path("ln_final") == "model.norm"
    assert adapter.translate_transformer_lens_path("unembed") == "lm_head"

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks") == "model.layers"
    assert adapter.translate_transformer_lens_path("blocks.0") == "model.layers.0"
    assert adapter.translate_transformer_lens_path("blocks.1") == "model.layers.1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1") == "model.layers.0.input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1_post")
        == "model.layers.0.post_attention_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2")
        == "model.layers.0.pre_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2_post")
        == "model.layers.0.post_feedforward_layernorm"
    )
    assert adapter.translate_transformer_lens_path("blocks.0.attn") == "model.layers.0.self_attn"
    assert adapter.translate_transformer_lens_path("blocks.0.mlp") == "model.layers.0.mlp"

    # Test deeper subcomponent paths
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.q")
        == "model.layers.0.self_attn.q_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.k")
        == "model.layers.0.self_attn.k_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.v")
        == "model.layers.0.self_attn.v_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.o")
        == "model.layers.0.self_attn.o_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.gate")
        == "model.layers.0.mlp.gate_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.in") == "model.layers.0.mlp.up_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.out")
        == "model.layers.0.mlp.down_proj"
    )


def test_translate_transformer_lens_path_last_component(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation with last_component_only=True."""
    # Test direct mapping
    assert (
        adapter.translate_transformer_lens_path("embed", last_component_only=True) == "embed_tokens"
    )
    assert adapter.translate_transformer_lens_path("ln_final", last_component_only=True) == "norm"
    assert adapter.translate_transformer_lens_path("unembed", last_component_only=True) == "lm_head"

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks", last_component_only=True) == "layers"
    assert adapter.translate_transformer_lens_path("blocks.0", last_component_only=True) == "0"
    assert adapter.translate_transformer_lens_path("blocks.1", last_component_only=True) == "1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1", last_component_only=True)
        == "input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1_post", last_component_only=True)
        == "post_attention_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2", last_component_only=True)
        == "pre_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2_post", last_component_only=True)
        == "post_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn", last_component_only=True)
        == "self_attn"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp", last_component_only=True) == "mlp"
    )

    # Test deeper subcomponent paths with last_component_only
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.q", last_component_only=True)
        == "q_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.k", last_component_only=True)
        == "k_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.v", last_component_only=True)
        == "v_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.o", last_component_only=True)
        == "o_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.gate", last_component_only=True)
        == "gate_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.in", last_component_only=True)
        == "up_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.out", last_component_only=True)
        == "down_proj"
    )


def test_component_mapping_structure(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test that the component mapping has the expected structure."""
    mapping = adapter.get_component_mapping()

    # Test that we have the expected top-level components
    assert "embed" in mapping
    assert "blocks" in mapping
    assert "ln_final" in mapping
    assert "unembed" in mapping

    # Test that components are bridge instances
    from transformer_lens.model_bridge.generalized_components import (
        AttentionBridge,
        BlockBridge,
        EmbeddingBridge,
        LinearBridge,
        MLPBridge,
        NormalizationBridge,
        UnembeddingBridge,
    )

    assert isinstance(mapping["embed"], EmbeddingBridge)
    assert isinstance(mapping["blocks"], BlockBridge)
    assert isinstance(mapping["ln_final"], NormalizationBridge)
    assert isinstance(mapping["unembed"], UnembeddingBridge)

    # Test that blocks has submodules
    blocks_bridge = mapping["blocks"]
    assert hasattr(blocks_bridge, "submodules")
    assert "ln1" in blocks_bridge.submodules
    assert "ln2" in blocks_bridge.submodules
    assert "attn" in blocks_bridge.submodules
    assert "mlp" in blocks_bridge.submodules

    # Test that the submodules are the expected types
    assert isinstance(blocks_bridge.submodules["ln1"], NormalizationBridge)
    assert isinstance(blocks_bridge.submodules["ln2"], NormalizationBridge)
    assert isinstance(blocks_bridge.submodules["attn"], AttentionBridge)
    assert isinstance(blocks_bridge.submodules["mlp"], MLPBridge)

    # Test that attention has submodules
    attn_bridge = blocks_bridge.submodules["attn"]
    assert hasattr(attn_bridge, "submodules")
    assert "q" in attn_bridge.submodules
    assert "k" in attn_bridge.submodules
    assert "v" in attn_bridge.submodules
    assert "o" in attn_bridge.submodules
    assert isinstance(attn_bridge.submodules["q"], LinearBridge)
    assert isinstance(attn_bridge.submodules["k"], LinearBridge)
    assert isinstance(attn_bridge.submodules["v"], LinearBridge)
    assert isinstance(attn_bridge.submodules["o"], LinearBridge)

    # Test that MLP has submodules
    mlp_bridge = blocks_bridge.submodules["mlp"]
    assert hasattr(mlp_bridge, "submodules")
    assert "gate" in mlp_bridge.submodules
    assert "in" in mlp_bridge.submodules
    assert "out" in mlp_bridge.submodules
    assert isinstance(mlp_bridge.submodules["gate"], LinearBridge)
    assert isinstance(mlp_bridge.submodules["in"], LinearBridge)
    assert isinstance(mlp_bridge.submodules["out"], LinearBridge)


def test_get_component(adapter: Gemma3ArchitectureAdapter, model: MockGemma3Model) -> None:
    """Test getting components from the model."""
    # Test direct mapping
    assert isinstance(adapter.get_component(model, "embed"), nn.Embedding)

    # Test block mapping
    block = adapter.get_component(model, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = adapter.get_component(model, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = adapter.get_component(model, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = adapter.get_component(model, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


def test_invalid_paths(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test handling of invalid paths."""
    with pytest.raises(ValueError, match="Component not_found not found in component mapping"):
        adapter.translate_transformer_lens_path("not_found")

    with pytest.raises(ValueError, match="Expected item index, got invalid"):
        adapter.translate_transformer_lens_path("blocks.invalid")

    with pytest.raises(ValueError, match="Component not_found not found in blocks components"):
        adapter.translate_transformer_lens_path("blocks.0.not_found")


def test_get_component_invalid_paths(
    adapter: Gemma3ArchitectureAdapter, model: MockGemma3Model
) -> None:
    """Test handling of invalid paths in get_component."""
    with pytest.raises(ValueError, match="Component not_found not found in component mapping"):
        adapter.get_component(model, "not_found")

    with pytest.raises(ValueError, match="Expected item index, got invalid"):
        adapter.get_component(model, "blocks.invalid")
