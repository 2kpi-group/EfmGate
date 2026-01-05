
# Imports & Seeds
from data import generate_ou_signal, split_data, to_jax, r2_mean_std
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
import optax



# EFM-LSTM avec signature

class EfmLSTM(nn.Module):
    units: int
    signature_depth: int = 2
    signature_input_size: int = 5
    return_sequences: bool = True

    @nn.compact
    def __call__(self, inputs, initial_state=None):
        B, T, D = inputs.shape

        # Projection pour signature
        x_proj = nn.Dense(self.signature_input_size, use_bias=False, name="sig_projection")(inputs)
        t_grid = jnp.linspace(0.0, 1.0, T)[None, :, None]
        t_grid = jnp.repeat(t_grid, B, axis=0)
        path = jnp.concatenate([t_grid, x_proj], axis=-1)

        # signature
        def _path_signature(p_2d):
            return ta.path_to_fm_signature(
                path=p_2d,
                trunc=self.signature_depth,
                t_grid=jnp.linspace(0.0, 1.0, p_2d.shape[0]),
                lam=1.0
            ).array

        signatures = jax.vmap(_path_signature)(path)
        if signatures.ndim == 3:
            signatures = signatures[:, -1, :]  # signature globale
        sig_dim = signatures.shape[-1]

        # Paramètres LSTM
        input_kernel = self.param("input_kernel", nn.initializers.glorot_uniform(), (D, self.units*3))
        recurrent_kernel = self.param("recurrent_kernel", nn.initializers.glorot_uniform(), (self.units, self.units*3))
        forget_kernel = self.param("forget_kernel", nn.initializers.glorot_uniform(), (sig_dim, self.units))
        bias = self.param("bias", nn.initializers.zeros, (self.units*4,))
        b_i, b_f, b_c, b_o = jnp.split(bias, 4, axis=-1)

        if initial_state is None:
            h0 = jnp.zeros((B, self.units))
            c0 = jnp.zeros((B, self.units))
        else:
            h0, c0 = initial_state

        all_x_transformed = jnp.einsum("btd,du->btu", inputs, input_kernel)

        def step(carry, x_t):
            h_prev, c_prev = carry
            gates = jnp.dot(h_prev, recurrent_kernel) + x_t
            i_t, c_t, o_t = jnp.split(gates, 3, axis=-1)
            f_t = jnp.dot(signatures, forget_kernel) + b_f
            i_t = jax.nn.sigmoid(i_t + b_i)
            f_t = jax.nn.sigmoid(f_t)
            c_t = jnp.tanh(c_t + b_c)
            o_t = jax.nn.sigmoid(o_t + b_o)
            c_new = f_t*c_prev + i_t*c_t
            h_new = o_t*jnp.tanh(c_new)
            return (h_new, c_new), h_new

        (_, _), h_seq = lax.scan(step, (h0, c0), all_x_transformed.transpose(1,0,2))
        return h_seq.transpose(1,0,2) if self.return_sequences else h_seq[:,-1,:]

