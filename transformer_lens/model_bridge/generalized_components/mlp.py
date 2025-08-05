"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""


from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MLPBridge(GeneralizedComponent):
    """Bridge component for MLP layers.

    This component wraps an MLP layer from a remote model and provides a consistent interface
    for accessing its weights and performing MLP operations.
    """

    hook_aliases = {
        "hook_pre": "in.hook_out",
        "hook_post": "out.hook_in",
    }

    property_aliases = {
        "W_gate": "gate.weight",
        "b_gate": "gate.bias",
        "W_in": "in.weight",
        "b_in": "in.bias",
        "W_out": "out.weight",
        "b_out": "out.bias",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the MLP bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for MLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
        """
        super().__init__(name, config, submodules=submodules)

        # No extra hooks; use only hook_in and hook_out

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the MLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = args[0]
        hidden_states = self.hook_in(hidden_states)
        new_args = (hidden_states,) + args[1:]
        output = self.original_component(*new_args, **kwargs)
        output = self.hook_out(output)

        return output
