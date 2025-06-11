"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any

import torch.nn as nn

from transformer_lens.HookedTransformer import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component wraps attention layers from different architectures and provides
    a standardized interface for hook registration and execution.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any,
    ):
        """Initialize the attention bridge.

        Args:
            original_component: The original attention component to wrap
            name: The name of this component
            architecture_adapter: Architecture adapter for component-specific operations
        """
        super().__init__(original_component, name, architecture_adapter)

        # Add all the hooks from the old attention components
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # Value vectors
        self.hook_z = HookPoint()  # Attention output
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

        # Optional hooks based on positional embedding type
        self.hook_attn_input = HookPoint()  # [batch, pos, d_model] (for shortformer)
        self.hook_rot_k = HookPoint()  # [batch, pos, head_index, d_head] (for rotary)
        self.hook_rot_q = HookPoint()  # [batch, pos, head_index, d_head] (for rotary)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the attention bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output from the original component
        """
        return self.original_component(*args, **kwargs)
