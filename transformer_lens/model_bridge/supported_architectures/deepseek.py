"""DeepSeek architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    MoEBridge,
    UnembeddingBridge,
)


class DeepseekArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for DeepSeek models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the DeepSeek architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                # Attention weights
                "blocks.{i}.attn.W_Q": "model.layers.{i}.self_attn.q_proj.weight",
                "blocks.{i}.attn.W_K": "model.layers.{i}.self_attn.k_proj.weight",
                "blocks.{i}.attn.W_V": "model.layers.{i}.self_attn.v_proj.weight",
                "blocks.{i}.attn.W_O": "model.layers.{i}.self_attn.o_proj.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                # MLP weights for dense layers
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight",
                # MoE weights
                "blocks.{i}.moe.gate.w": "model.layers.{i}.mlp.gate.weight",
                "blocks.{i}.moe.experts.W_gate.{j}": "model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                "blocks.{i}.moe.experts.W_in.{j}": "model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                "blocks.{i}.moe.experts.W_out.{j}": "model.layers.{i}.mlp.experts.{j}.down_proj.weight",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "blocks": (
                "model.layers",
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "ln2": ("post_attention_layernorm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                    "moe": ("mlp", MoEBridge),
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        } 