"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""

from typing import Any

import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component wraps a transformer block from a remote model and provides a consistent interface
    for accessing its weights and performing block operations.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any | None = None,
    ):
        """Initialize the block bridge.

        Args:
            original_component: The original block component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the block bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            The output from the original component
        """
        return self.original_component(*args, **kwargs)
