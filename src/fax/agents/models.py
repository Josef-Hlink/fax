"""
State space models for SSBM agents using JAX.
"""

# import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Optional, Callable
import flax.linen as nn
# import distrax


class StateSpaceBlock(nn.Module):
    """
    State Space Model (SSM) block.

    This implements a simplified version of a state space model layer
    that can be used as a building block for sequence models.
    """

    hidden_dim: int
    state_dim: int

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, input_dim = x.shape

        # Projection to higher dimension
        x_proj = nn.Dense(self.hidden_dim)(x)

        # Initialize state
        state = jnp.zeros((batch_size, self.state_dim))

        # Process sequence
        outputs = []
        for t in range(seq_len):
            # Get input at current timestep
            x_t = x_proj[:, t, :]

            # State update
            state_proj = nn.Dense(self.state_dim)(x_t)
            state = nn.tanh(state + state_proj)

            # Output projection
            output = nn.Dense(input_dim)(state)
            outputs.append(output)

        # Stack outputs
        return jnp.stack(outputs, axis=1)
