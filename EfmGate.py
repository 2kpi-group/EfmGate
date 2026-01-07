
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

# 
# EFM-LSTM avec signature
# 

class EfmLSTM(nn.Module):
    units: int = 8
    signature_depth: int = 2
    signature_input_size: int = 3
    return_sequences: bool = True

    @nn.compact
    def __call__(self, inputs):
        B, T, D = inputs.shape

        # ---- Projection pour signature
        x_proj = nn.Dense(self.signature_input_size, use_bias=False)(inputs)  # (B,T,D') OUI
        t_grid = jnp.linspace(0., 1., T)[None, :, None]                        # (1,T,1) OUI
        t_grid = jnp.repeat(t_grid, B, axis=0)                                 # (B,T,1) OUI
        path = jnp.concatenate([t_grid, x_proj], axis=-1)                      # (B,T,D'+1) OUI

        # ---- Calcul des signatures hors JIT
        sig_list = []
        for b in range(B):
            sig = ta.path_to_fm_signature(
                path=path[b], # path[b] est array de 2D car il selectionne le b+1 batch de path
                trunc=self.signature_depth, # Troncature
                t_grid=jnp.linspace(0.,1.,T), # Temps T
                lam=1.0 # Parametre lambda
            ).array  # (sig_dim, T) : C'est à dire la path est sous la forme colonne =T et logne = sig_dim
            sig_list.append(sig) # Il ajoute les signatire
        signatures = jnp.stack(sig_list, axis=0)        # (B, sig_dim, T)
        signatures = jnp.transpose(signatures, (0,2,1)) # (B, T, sig_dim)
        sig_dim = signatures.shape[-1] # 

        # ---- LSTM parameters
        input_kernel = self.param("input_kernel", nn.initializers.glorot_uniform(), (D, 3*self.units))
        recurrent_kernel = self.param("recurrent_kernel", nn.initializers.glorot_uniform(), (self.units, 3*self.units))
        forget_kernel = self.param("forget_kernel", nn.initializers.glorot_uniform(), (sig_dim, self.units))
        bias = self.param("bias", nn.initializers.zeros, (4*self.units,))
        b_i, b_f, b_c, b_o = jnp.split(bias, 4, axis=-1)

        h0 = jnp.zeros((B, self.units))
        c0 = jnp.zeros((B, self.units))
        x_proj_lstm = jnp.einsum("btd,du->btu", inputs, input_kernel)

        # ---- Scan
        def step(carry, xs):
            x_t, sig_t = xs
            h_prev, c_prev = carry
            gates = jnp.dot(h_prev, recurrent_kernel) + x_t
            i_t, c_hat, o_t = jnp.split(gates, 3, axis=-1)
            f_t = jnp.dot(sig_t, forget_kernel) + b_f
            i_t = jax.nn.sigmoid(i_t + b_i)
            f_t = jax.nn.sigmoid(f_t)
            c_hat = jnp.tanh(c_hat + b_c)
            o_t = jax.nn.sigmoid(o_t + b_o)
            c_new = f_t*c_prev + i_t*c_hat
            h_new = o_t*jnp.tanh(c_new)
            return (h_new, c_new), h_new

        scan_inputs = (
            x_proj_lstm.transpose(1,0,2),   # (T,B,3*units)
            signatures.transpose(1,0,2)     # (T,B,sig_dim)
        )
        (_, _), h_seq = lax.scan(step, (h0, c0), scan_inputs)
        h_seq = h_seq.transpose(1,0,2)  # (B,T,units)
        return h_seq if self.return_sequences else h_seq[:,-1]

