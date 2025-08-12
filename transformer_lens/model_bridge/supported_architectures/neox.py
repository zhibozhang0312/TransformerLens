"""NeoX architecture adapter."""

from typing import Any

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
    SplitHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.chain_hook_conversion import (
    ChainHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class NeoxArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NeoX models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the NeoX architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "gpt_neox.embed_in.weight",
                "blocks.{i}.ln1.w": "gpt_neox.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "gpt_neox.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "gpt_neox.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "gpt_neox.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(0, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.num_attention_heads,
                                d_head=self.cfg.hidden_size // self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.k": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(1, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.num_attention_heads,
                                d_head=self.cfg.hidden_size // self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainHookConversion(
                        [
                            SplitHookConversion(2, 3),
                            RearrangeHookConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.cfg.num_attention_heads,
                                d_head=self.cfg.hidden_size // self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(0, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(1, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainHookConversion(
                        [
                            SplitHookConversion(2, 3),
                            RearrangeHookConversion(
                                "(head d_head) -> head d_head",
                                head=self.cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "gpt_neox.layers.{i}.attention.dense.weight",
                    RearrangeHookConversion("d_model (head d_head) -> head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "gpt_neox.layers.{i}.attention.dense.bias",
                "blocks.{i}.mlp.in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                "ln_final.w": "gpt_neox.final_layer_norm.weight",
                "ln_final.b": "gpt_neox.final_layer_norm.bias",
                "unembed.u": "embed_out.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="gpt_neox.embed_in"),
            "rotary_emb": EmbeddingBridge(name="gpt_neox.rotary_emb"),
            "blocks": BlockBridge(
                name="gpt_neox.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm"),
                    "ln2": NormalizationBridge(name="post_attention_layernorm"),
                    "attn": JointQKVAttentionBridge(
                        name="attention",
                        model_config=self.cfg,
                        submodules={
                            "W_QKV": LinearBridge(
                                name="query_key_value",
                            ),
                            "W_O": LinearBridge(name="dense"),
                        },
                        qkv_config={"split_qkv_matrix": self.split_qkv_matrix},
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "W_in": LinearBridge(name="dense_h_to_4h"),
                            "W_out": LinearBridge(name="dense_4h_to_h"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="gpt_neox.final_layer_norm"),
            "unembed": UnembeddingBridge(name="embed_out"),
        }

    def split_qkv_matrix(
        self, original_attention_component: Any
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split the QKV matrix into separate linear transformations.
        Args:
            attention_component: The original attention layer component
        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations
        """

        # Keep mypy happy
        assert original_attention_component is not None
        assert original_attention_component.query_key_value is not None

        qkv_weights = original_attention_component.query_key_value.weight

        # Keep mypy happy
        assert isinstance(qkv_weights, torch.Tensor)

        # Original qkv_weights shape: [d_model * 3 * d_model]
        # Split into three equal parts along dimension 1 to get Q, K, V weights
        W_Q, W_K, W_V = torch.tensor_split(qkv_weights, 3, dim=1)

        qkv_bias = original_attention_component.query_key_value.bias

        # Keep mypy happy
        assert isinstance(qkv_bias, torch.Tensor)

        # Original qkv_bias shape: [n_heads * 3 * d_head]
        # Reshape to [3, n_heads * d_head] to split by Q, K, V
        qkv_bias = qkv_bias.reshape(3, self.cfg.n_heads * self.cfg.d_head)
        b_Q, b_K, b_V = qkv_bias[0, :], qkv_bias[1, :], qkv_bias[2, :]

        # Create nn.Linear modules
        # After tensor_split, W_Q, W_K, W_V shapes are [d_model, d_model] ([in_features, out_features])
        # nn.Linear expects weight shape [out_features, in_features]
        # So we need to transpose the weights
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=True)
        W_Q_transformation.weight = torch.nn.Parameter(W_Q.T)
        W_Q_transformation.bias = torch.nn.Parameter(b_Q)

        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=True)
        W_K_transformation.weight = torch.nn.Parameter(W_K.T)
        W_K_transformation.bias = torch.nn.Parameter(b_K)

        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=True)
        W_V_transformation.weight = torch.nn.Parameter(W_V.T)
        W_V_transformation.bias = torch.nn.Parameter(b_V)

        return W_Q_transformation, W_K_transformation, W_V_transformation
