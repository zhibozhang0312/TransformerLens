"""OPT architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


class OptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the OPT architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.decoder.embed_tokens.weight",
                "pos_embed.W_pos": "model.decoder.embed_positions.weight",
                "blocks.{i}.ln1.w": "model.decoder.layers.{i}.self_attn_layer_norm.weight",
                "blocks.{i}.ln1.b": "model.decoder.layers.{i}.self_attn_layer_norm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.decoder.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.decoder.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.decoder.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.decoder.layers.{i}.self_attn.out_proj.weight",
                    RearrangeWeightConversion("m (n h) -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.b_Q": "model.decoder.layers.{i}.self_attn.q_proj.bias",
                "blocks.{i}.attn.b_K": "model.decoder.layers.{i}.self_attn.k_proj.bias",
                "blocks.{i}.attn.b_V": "model.decoder.layers.{i}.self_attn.v_proj.bias",
                "blocks.{i}.attn.b_O": "model.decoder.layers.{i}.self_attn.out_proj.bias",
                "blocks.{i}.ln2.w": "model.decoder.layers.{i}.final_layer_norm.weight",
                "blocks.{i}.ln2.b": "model.decoder.layers.{i}.final_layer_norm.bias",
                "blocks.{i}.mlp.W_in": "model.decoder.layers.{i}.fc1.weight",
                "blocks.{i}.mlp.b_in": "model.decoder.layers.{i}.fc1.bias",
                "blocks.{i}.mlp.W_out": "model.decoder.layers.{i}.fc2.weight",
                "blocks.{i}.mlp.b_out": "model.decoder.layers.{i}.fc2.bias",
                "ln_final.w": "model.decoder.final_layer_norm.weight",
                "ln_final.b": "model.decoder.final_layer_norm.bias",
                "unembed.W_U": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "pos_embed": EmbeddingBridge(name="model.decoder.embed_positions"),
            "blocks": BlockBridge(
                name="model.decoder.layers",
                submodules={
                    "ln1": LayerNormBridge(name="self_attn_layer_norm"),
                    "attn": AttentionBridge(name="self_attn"),
                    "ln2": LayerNormBridge(name="final_layer_norm"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": LayerNormBridge(name="model.decoder.final_layer_norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
