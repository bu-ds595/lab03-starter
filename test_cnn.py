"""
Autograding tests for Lab 3: Galaxy Morphology Classification with CNNs.

Grading (4 points total):
  test_loss_decreases        (1 pt) — train_step reduces loss
  test_accuracy_70           (1 pt) — saved model accuracy > 70%
  test_accuracy_80           (2 pt) — saved model accuracy > 80% (bonus)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from cnn import CNN, train_step, load_model


def _load_data():
    data = np.load("galaxy_data.npz")
    return {
        k: jnp.array(data[k].astype(np.float32) / 255.0) if k.startswith("X") else jnp.array(data[k])
        for k in data.files
    }


def test_loss_decreases():
    """Train for 20 steps on a small batch; loss should decrease. (1 pt)"""
    data = _load_data()
    X_batch, y_batch = data["X_train"][:64], data["y_train"][:64]

    model = CNN()
    params = model.init(jax.random.PRNGKey(0), X_batch)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    logits = model.apply(params, X_batch)
    initial_loss = float(optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean())

    for _ in range(20):
        params, opt_state, loss = train_step(
            params, opt_state, X_batch, y_batch, model, optimizer
        )

    final_loss = float(loss)
    assert np.isfinite(initial_loss), "Initial loss is not finite"
    assert np.isfinite(final_loss), "Final loss is not finite"
    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_accuracy_70():
    """Load saved model, accuracy > 70% on test set. (1 pt)"""
    data = _load_data()

    model = CNN()
    params = load_model()
    logits = model.apply(params, data["X_test"])
    accuracy = float((logits.argmax(-1) == data["y_test"]).mean())

    assert accuracy > 0.70, f"Test accuracy: {accuracy:.2%} (need > 70%)"


def test_accuracy_80():
    """Load saved model, accuracy > 80% on test set. (2 pts, bonus)"""
    data = _load_data()

    model = CNN()
    params = load_model()
    logits = model.apply(params, data["X_test"])
    accuracy = float((logits.argmax(-1) == data["y_test"]).mean())

    assert accuracy > 0.80, f"Test accuracy: {accuracy:.2%} (need > 80%)"
