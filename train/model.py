from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class GRUCell(nn.Module):
    hidden_size: int
    @nn.compact
    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray) -> Any:
        features = jnp.concatenate([x, carry], axis=-1)
        gates = nn.Dense(2 * self.hidden_size, name="gates")(features)
        reset_gate, update_gate = jnp.split(gates,2, axis=-1)
        reset_gate = nn.sigmoid(reset_gate)
        update_gate = nn.sigmoid(update_gate)
        candidate_input = jnp.concatenate([x, reset_gate * carry],axis=-1)
        candidate = nn.tanh(
            nn.Dense(
                self.hidden_size,
                name="candidate"
            )(candidate_input)
        )
        new_carry = ((1.0 - update_gate) * candidate
            + update_gate * carry)

        return new_carry, new_carry


class WorkloadEncoder(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, access_sequence: jnp.ndarray):
        batch_size = access_sequence.shape[0]
        carry = jnp.zeros(
            (batch_size, self.hidden_size))
        cell = GRUCell(self.hidden_size,name="gru_cell")
        seq_len = access_sequence.shape[1]
        for timestep in range(seq_len):
            carry, _ = cell(carry,access_sequence[:, timestep, :])
        return carry


class CandidateScorer(nn.Module):
    @nn.compact
    def __call__(self, workload_state: jnp.ndarray,page_features: jnp.ndarray):
        x = jnp.concatenate([workload_state, page_features],axis=-1)
        x = nn.Dense(64,name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(32,name="fc2")(x)
        x = nn.relu(x)
        logits = nn.Dense(1,name="output")(x)
        return logits.squeeze(-1)


class GRUEvictionModel(nn.Module):
    hidden_size: int = 64
    @nn.compact
    def __call__(self,access_sequence: jnp.ndarray,page_features: jnp.ndarray):
        workload_state = WorkloadEncoder( hidden_size=self.hidden_size,name="encoder")(access_sequence)
        logits = CandidateScorer(name="scorer")(workload_state,page_features)
        return logits
