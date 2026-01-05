# Imports & Seeds
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
import optax

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
jax_key = jax.random.PRNGKey(SEED)


# Générer OU signal

def generate_ou_signal(T=500, seed=0, kappa=15, theta=0., nu=1.5):
    rng = np.random.default_rng(seed)
    
    dt = 1.0 / T          # 🔴 CORRECTION CLÉ
    x = np.zeros(T)
    x[0] = 0.0

    for t in range(1, T):
        x[t] = (
            x[t-1]
            + kappa * (theta - x[t-1]) * dt
            + nu * np.sqrt(dt) * rng.standard_normal()
        )

    return np.linspace(0, 1, T), x

