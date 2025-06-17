"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge import ArchitectureAdapterFactory
from transformer_lens.model_bridge.bridge import TransformerBridge


def boot(
    model_name: str,
    model_config: dict | None = None,
    tokenizer_config: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        model_config: Additional configuration parameters to override the default config.
        tokenizer_config: The config dict to use for tokenizer loading. If None, will use default settings.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.

    Returns:
        The bridge to the loaded model.
    """
    hf_config = AutoConfig.from_pretrained(model_name, **(model_config or {}))
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(hf_config)
    default_config = adapter.default_cfg
    merged_config = {**default_config, **(model_config or {})}

    # Load the model from HuggingFace using the original config
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=hf_config,
        torch_dtype=dtype,
        **merged_config,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_config or {}))

    return TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    )
