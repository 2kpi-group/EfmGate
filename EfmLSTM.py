from typing import Union
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
import numpy as np
  

class EfmLSTM(nn.Module):
    units: int
    signature_depth: int = 2
    signature_input_size: int = 5
    return_sequences: bool = False
    return_state: bool = False

    @nn.compact
    def __call__(self, inputs, initial_state=None):
        B, T, D = inputs.shape

        # ----------------- Projection pour signature -----------------
        x_proj = nn.Dense(self.signature_input_size, use_bias=False, name="sig_projection")(inputs)

        # ----------------- Signature computation -----------------
        # t_grid : forme (1, T, 1), broadcastable
        t_grid = jnp.linspace(0.0, 1.0, T)[None, :, None]
        path = jnp.concatenate([t_grid.repeat(B, axis=0), x_proj], axis=-1)

        # Calcul signature par batch
        # ta.path_to_fm_signature attend un chemin 2D
        def _path_signature(p):
            return ta.path_to_fm_signature(
                path=p,
                trunc=self.signature_depth,
                t_grid=jnp.linspace(0.0, 1.0, p.shape[0]),
                lam=1.0
            ).array

        signatures = jax.vmap(_path_signature)(path)  # (B, sig_dim)
        signatures = self._normalize_signature_by_time(signatures)  # normalisation par temps

        sig_dim = signatures.shape[-1]

        # ----------------- Définition des poids LSTM -----------------
        input_kernel = self.param("input_kernel", nn.initializers.glorot_uniform(), (D, self.units * 3))
        recurrent_kernel = self.param("recurrent_kernel", nn.initializers.glorot_uniform(), (self.units, self.units * 3))
        forget_kernel = self.param("forget_kernel", nn.initializers.glorot_uniform(), (sig_dim, self.units))
        bias = self.param("bias", nn.initializers.zeros, (self.units * 4,))

        b_i, b_f, b_c, b_o = jnp.split(bias, 4, axis=-1)

        # ----------------- Initial state -----------------
        if initial_state is None:
            h0 = jnp.zeros((B, self.units))
            c0 = jnp.zeros((B, self.units))
        else:
            h0, c0 = initial_state

        # ----------------- Step function -----------------
        def step(carry, x_t):
            h_prev, c_prev = carry
            x_in = x_t  # (B, 3*U)
            sig = signatures[:, 0]  # Signature globale pour forget gate

            gates = jnp.dot(h_prev, recurrent_kernel) + x_in
            i_t, c_t, o_t = jnp.split(gates, 3, axis=-1)

            f_t = jnp.dot(sig, forget_kernel) + b_f

            i_t = jax.nn.sigmoid(i_t + b_i)
            f_t = jax.nn.sigmoid(f_t)
            c_t = jnp.tanh(c_t + b_c)
            o_t = jax.nn.sigmoid(o_t + b_o)

            c_new = f_t * c_prev + i_t * c_t
            h_new = o_t * jnp.tanh(c_new)

            return (h_new, c_new), h_new

        # ----------------- Precompute input transformations -----------------
        all_x_transformed = jnp.einsum("bti,ij->btj", inputs, input_kernel)  # (B, T, 3U)

        # ----------------- Scan sur le temps -----------------
        (final_h, final_c), h_seq = lax.scan(
            step,
            (h0, c0),
            all_x_transformed.transpose(1, 0, 2)  # (T, B, 3U)
        )

        if self.return_sequences:
            outputs = h_seq.transpose(1, 0, 2)  # (B, T, U)
        else:
            outputs = final_h  # (B, U)

        if self.return_state:
            return outputs, final_h, final_c
        return outputs

    # ----------------- Normalisation signature -----------------
    def _normalize_signature_by_time(self, signatures):
        seq_length = signatures.shape[1] if signatures.ndim == 3 else 1
        time_indices = jnp.arange(1, seq_length + 1, dtype=signatures.dtype)
        time_indices = time_indices[None, :, None]
        if seq_length > 1:
            return jnp.concatenate([jnp.zeros_like(signatures[:, :1, :]), signatures / time_indices], axis=1)
        else:
            return signatures / time_indices


import jax

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 4, 2))  # batch=2, time=4, features=2

model = EfmLSTM(units=8, return_sequences=True)
params = model.init(key, x)
y = model.apply(params, x)

print(y.shape)  # (2, 4, 8)

import optax
import jax
import jax.numpy as jnp

# --- Hyperparamètres ---
learning_rate = 0.01
num_steps = 100

# --- Optimiseur ---
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Fonction de perte simple (MSE)
def loss_fn(params, x, y_true):
    y_pred = model.apply(params, x)
    return jnp.mean((y_pred - y_true) ** 2)

# --- Gradient + update ---
grad_fn = jax.value_and_grad(loss_fn)

# --- Boucle d'entraînement ---
for step in range(num_steps):
    loss_val, grads = grad_fn(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    print(f"Step {step+1:02d}, Loss: {loss_val:.6f}")
