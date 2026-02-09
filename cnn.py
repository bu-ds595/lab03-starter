"""
Lab 3: Galaxy Morphology Classification with CNNs
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

N_CLASSES = 4


class CNN(nn.Module):
    """CNN for classifying galaxy morphology.

    Classes:
        0: smooth and round
        1: smooth and cigar-shaped
        2: edge-on disk
        3: unbarred spiral

    Your network should:
        - Accept input of shape (batch, 64, 64, 3)
        - Return logits of shape (batch, 4) — no softmax!

    You are free to choose any architecture. A good starting point:
    one or more Conv -> ReLU -> Pool blocks, then flatten and a Dense head.

    Useful Flax operations:
        nn.Conv(features=16, kernel_size=(3, 3))                       # conv layer
        nn.relu(x)                                                      # activation
        nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))            # pooling
        nn.BatchNorm(use_running_average=True)(x)                       # batch normalization
        x.reshape((x.shape[0], -1))                                    # flatten to (batch, features)
        nn.Dense(features=64)                                           # fully connected layer
    """

    @nn.compact
    def __call__(self, x):
        # TODO: Implement your CNN.

        ...


def train_step(params, opt_state, X_batch, y_batch, model, optimizer):
    """Single training step: forward pass, cross-entropy loss, backward pass, update.

    Args:
        params: Model parameters (pytree from model.init)
        opt_state: Optimizer state
        X_batch: Input batch, shape (B, 64, 64, 3), float32 in [0, 1]
        y_batch: Integer class labels, shape (B,), values in {0, 1, 2, 3}
        model: Flax CNN module instance
        optimizer: optax optimizer

    Returns:
        new_params: Updated parameters
        new_opt_state: Updated optimizer state
        loss: Scalar loss value
    """
    # TODO: Implement the training step.
    # Hint: use jax.value_and_grad, optax.softmax_cross_entropy_with_integer_labels

    ...


# ---- Do not edit below this line ---- #


def save_model(params, path="model_params.pkl"):
    """Save model parameters to a file.

    Call this after training to save your model for grading.

    Args:
        params: Model parameters (pytree from model.init / training)
        path: File path to save to
    """
    import pickle
    params_np = jax.tree.map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(params_np, f)


def load_model(path="model_params.pkl"):
    """Load model parameters from a file.

    Args:
        path: File path to load from

    Returns:
        params: Model parameters as JAX arrays
    """
    import pickle
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree.map(jnp.asarray, params_np)
