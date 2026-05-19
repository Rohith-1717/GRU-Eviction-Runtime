import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from .model import GRUModel


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, learning_rate: float = 1e-3, hidden_size: int = 16):
    model = GRUModel(hidden_size=hidden_size)
    params = model.init(rng, jnp.zeros((1, 5)))
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(params, batch, labels):
    predictions = GRUModel().apply(params, batch)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(predictions, labels))


def train_epoch(state, features, labels, batch_size: int = 64):
    data_size = features.shape[0]
    perms = np.random.permutation(data_size)
    for start in range(0, data_size, batch_size):
        idx = perms[start:start + batch_size]
        batch = features[idx]
        label_batch = labels[idx]

        def grad_fn(params):
            loss = loss_fn(params, batch, label_batch)
            return loss

        grads = jax.grad(grad_fn)(state.params)
        state = state.apply_gradients(grads=grads)
    return state


def load_dataset(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    data = np.load(path, allow_pickle=True)
    features = np.stack([data['features'][i] for i in range(data['features'].shape[0])], axis=-1).T
    labels = data['labels']
    return jnp.array(features), jnp.array(labels)


def save_params(params, output_path: str):
    state_dict = serialization.to_state_dict(params)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, state_dict, allow_pickle=True)


def train_from_file(data_path: str, output_path: str, epochs: int = 10):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng)
    features, labels = load_dataset(data_path)

    for epoch in range(epochs):
        start = time.time()
        state = train_epoch(state, features, labels)
        duration = time.time() - start
        print(f"Epoch {epoch + 1}/{epochs} finished in {duration:.2f}s")

    save_params(state.params, output_path)
    return state.params
