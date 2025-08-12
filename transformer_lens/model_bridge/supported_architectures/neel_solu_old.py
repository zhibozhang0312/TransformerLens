"""Neel Solu Old architecture adapter."""

from typing import Any

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


class NeelSoluOldArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neel's SOLU models (old style)."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Neel SOLU old-style architecture adapter.

        Args:
            cfg: The configuration object.
        """
        self.default_config: dict[str, Any] = {}
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "pos_embed.pos": "wpe.weight",
                "embed.e": "wte.weight",
                "blocks.{i}.ln1.w": "blocks.{i}.ln1.w",
                "blocks.{i}.ln1.b": "blocks.{i}.ln1.b",
                "blocks.{i}.ln2.w": "blocks.{i}.ln2.w",
                "blocks.{i}.ln2.b": "blocks.{i}.ln2.b",
                "blocks.{i}.attn.q": (
                    "blocks.{i}.attn.W_Q",
                    RearrangeHookConversion("d_model n_head d_head -> n_head d_model d_head"),
                ),
                "blocks.{i}.attn.k": (
                    "blocks.{i}.attn.W_K",
                    RearrangeHookConversion("d_model n_head d_head -> n_head d_model d_head"),
                ),
                "blocks.{i}.attn.v": (
                    "blocks.{i}.attn.W_V",
                    RearrangeHookConversion("d_model n_head d_head -> n_head d_model d_head"),
                ),
                "blocks.{i}.attn.o": (
                    "blocks.{i}.attn.W_O",
                    RearrangeHookConversion("n_head d_head d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": "blocks.{i}.attn.b_Q",
                "blocks.{i}.attn.b_K": "blocks.{i}.attn.b_K",
                "blocks.{i}.attn.b_V": "blocks.{i}.attn.b_V",
                "blocks.{i}.attn.b_O": "blocks.{i}.attn.b_O",
                "blocks.{i}.mlp.in": "blocks.{i}.mlp.W_in",
                "blocks.{i}.mlp.b_in": "blocks.{i}.mlp.b_in",
                "blocks.{i}.mlp.out": "blocks.{i}.mlp.W_out",
                "blocks.{i}.mlp.b_out": "blocks.{i}.mlp.b_out",
                "ln_final.w": "ln_f.w",
                "ln_final.b": "ln_f.b",
                "unembed.u": "unembed.W_U",
                "unembed.b_U": "unembed.b_U",
            }
        )
        self.component_mapping = {
            "embed": EmbeddingBridge(name="wte"),
            "pos_embed": EmbeddingBridge(name="wpe"),
            "blocks": BlockBridge(
                name="blocks",
                submodules={
                    "ln1": NormalizationBridge(name="ln1"),
                    "attn": AttentionBridge(name="attn", config=self.cfg),
                    "ln2": NormalizationBridge(name="ln2"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": NormalizationBridge(name="ln_f"),
            "unembed": UnembeddingBridge(name="unembed"),
        }


def convert_neel_solu_old_weights(state_dict: dict, cfg: Any):
    """
    Converts the weights of my old SoLU models to the HookedTransformer format.
    Takes as input a state dict, *not* a model object.

    There are a bunch of dumb bugs in the original code, sorry!

    Models 1L, 2L, 4L and 6L have left facing weights (ie, weights have shape
    [dim_out, dim_in]) while HookedTransformer does right facing (ie [dim_in,
    dim_out]).

    8L has *just* a left facing W_pos, the rest right facing.

    And some models were trained with
    """
    # Early models have left facing W_pos
    reverse_pos = cfg.n_layers <= 8

    # Models prior to 8L have left facing everything (8L has JUST left facing W_pos - sorry! Stupid bug)
    reverse_weights = cfg.n_layers <= 6

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm", "ln")
        if k.startswith("ln."):
            k = k.replace("ln.", "ln_final.")
        new_state_dict[k] = v

    if reverse_pos:
        new_state_dict["pos_embed.W_pos"] = new_state_dict["pos_embed.W_pos"].T
    if reverse_weights:
        for k, v in new_state_dict.items():
            if "W_" in k and "W_pos" not in k:
                new_state_dict[k] = v.transpose(-2, -1)
    return new_state_dict
