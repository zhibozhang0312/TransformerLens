from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class NanogptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NanoGPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the NanoGPT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "pos_embed.pos": "transformer.wpe.weight",
                "embed.e": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion("d_model (3 n_head d_head) -> 3 n_head d_head d_model"),
                ),
                "blocks.{i}.attn.k": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion("d_model (3 n_head d_head) -> 3 n_head d_head d_model"),
                ),
                "blocks.{i}.attn.v": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeHookConversion("d_model (3 n_head d_head) -> 3 n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeHookConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.o": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeHookConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "unembed.u": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),  # Word token embeddings
            "pos_embed": EmbeddingBridge(name="transformer.wpe"),  # Positional embeddings
            "blocks": BlockBridge(
                name="transformer.h",  # Base path for blocks
                submodules={
                    "ln1": NormalizationBridge(name="ln_1"),  # Pre-attention layer norm
                    "ln2": NormalizationBridge(name="ln_2"),  # Pre-MLP layer norm
                    "attn": AttentionBridge(name="attn", config=self.cfg),  # Full attention module
                    "mlp": MLPBridge(name="mlp"),  # Full MLP module
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f"),  # Final layer norm
            "unembed": UnembeddingBridge(name="lm_head"),  # Language model head
        }

    def convert_weights(self, remote_module: Any) -> dict[str, torch.Tensor]:
        # Nanogpt models saved after torch.compile() have this unwanted prefix
        # This is a simple way to remove it
        unwanted_prefix = "_orig_mod."
        state_dict: dict[str, torch.Tensor] = (
            remote_module.state_dict() if hasattr(remote_module, "state_dict") else remote_module
        )
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        return super().convert_weights(remote_module)
