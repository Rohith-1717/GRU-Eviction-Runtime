import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import serialization
from flax.training import train_state

from collect.preprocess import load_trace, preprocess, make_training_samples
from .model import GRUEvictionModel


DEFAULT_SEQ_LEN = 32
DEFAULT_ACCESS_FEATURES = 5
DEFAULT_PAGE_FEATURES = 4
DEFAULT_HIDDEN_SIZE = 64


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, learning_rate=1e-3, hidden_size=DEFAULT_HIDDEN_SIZE):
    model = GRUEvictionModel(hidden_size=hidden_size)

    variables = model.init(
        rng,
        jnp.zeros((1, DEFAULT_SEQ_LEN, DEFAULT_ACCESS_FEATURES), dtype=jnp.float32),
        jnp.zeros((1, DEFAULT_PAGE_FEATURES), dtype=jnp.float32)
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate)
    )


def loss_fn(apply_fn, params, sequences, page_features, labels):
    logits = apply_fn({"params": params}, sequences, page_features)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

def train_epoch(state, sequences, page_features, labels, batch_size=256, rng=None):
    rng = rng or np.random.default_rng()
    permutation = rng.permutation(len(labels))
    losses = []
    for start in range(0, len(labels), batch_size):
        indices = permutation[start:start + batch_size]
        sequence_batch = jnp.asarray(sequences[indices], dtype=jnp.float32)
        page_batch = jnp.asarray(page_features[indices], dtype=jnp.float32)
        label_batch = jnp.asarray(labels[indices], dtype=jnp.float32)
        def batch_loss(params):
            return loss_fn(state.apply_fn, params, sequence_batch, page_batch, label_batch)
        loss, grads = jax.value_and_grad(batch_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        losses.append(float(loss))
    return state, float(np.mean(losses))


def load_dataset(path, seq_len=DEFAULT_SEQ_LEN, stride=1):
    trace = load_trace(path)
    processed = preprocess(trace)
    sequences, page_features, labels = make_training_samples(
        processed["access_features"], processed["page_features"], processed["labels"],
        sequence_length=seq_len,
        stride=stride
    )

    return (
        sequences.astype(np.float32),
        page_features.astype(np.float32),
        labels.astype(np.float32)
    )


def validate_shapes(sequences, page_features, labels, hidden_size=DEFAULT_HIDDEN_SIZE):
    assert sequences.ndim == 3
    assert page_features.ndim == 2
    assert labels.ndim == 1
    assert len(sequences) == len(page_features)
    assert len(sequences) == len(labels)
    model = GRUEvictionModel(hidden_size=hidden_size)
    variables = model.init(
        jax.random.PRNGKey(0),jnp.asarray(sequences[:1]),
        jnp.asarray(page_features[:1])
    )
    logits = model.apply(
        variables,
        jnp.asarray(sequences[:4]),
        jnp.asarray(page_features[:4])
    )

    print(f"Sequences: {sequences.shape}")
    print(f"Page Features: {page_features.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Output: {logits.shape}")


def save_params(params, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        path,
        serialization.to_state_dict(params),
        allow_pickle=True
    )


def train_from_file(
    data_path,
    output_path,
    epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    seq_len=DEFAULT_SEQ_LEN,
    stride=1
):
    sequences, page_features, labels = load_dataset(
        data_path,
        seq_len=seq_len,
        stride=stride)
    validate_shapes(
        sequences,
        page_features,
        labels,
        hidden_size=hidden_size
    )
    state = create_train_state(
        jax.random.PRNGKey(0),
        learning_rate=learning_rate,
        hidden_size=hidden_size
    )
    np_rng = np.random.default_rng(0)

    for epoch in range(epochs):
        start = time.time()
        state, mean_loss = train_epoch(
            state,
            sequences,
            page_features,
            labels,
            batch_size=batch_size,
            rng=np_rng
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss={mean_loss:.6f} "
            f"Time={time.time() - start:.2f}s"
        )

    save_params(state.params, output_path)

    return state.params
