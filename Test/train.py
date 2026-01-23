# Imports & Seeds

from efmgate import  build_dataset, split_data, r2_score

import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
import optax





(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(paths, targets)

X_train, y_train = jnp.array(X_train), jnp.array(y_train)
X_val, y_val     = jnp.array(X_val), jnp.array(y_val)
X_test, y_test   = jnp.array(X_test), jnp.array(y_test)


# Définition du modèle

class EfmLSTMPredictor(nn.Module):
    units: int = 16
    out_size: int = 1
    signature_depth: int = 3
    signature_input_size: int = 5

    @nn.compact
    def __call__(self, x):
        # Remplacez EfmLSTM par un simple Dense pour l'exemple
        h = EfmLSTM(self.units, self.signature_depth, self.signature_input_size)(x)
        return nn.Dense(self.out_size)(h)


# Fonction d'entraînement

def train_model(x_train, y_train, x_val, y_val, epochs=10, batch_size=32, lr=0.001):
    model = EfmLSTMPredictor()
    key = jax.random.PRNGKey(0)
    params = model.init(key, x_train[:1])  # init avec une batch pour définir les shapes

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Fonction de loss MSE
    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)

    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    for epoch in range(epochs):
        # Shuffle des données
        perm = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[perm], y_train[perm]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            loss, grads = grad_fn(params, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Validation
        val_loss = loss_fn(params, x_val, y_val)
        print(f"Epoch {epoch+1}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

    return model, params


# Exemple d'utilisation

# Suppose que tes données sont déjà en jnp.array
# X_train, y_train, X_val, y_val, X_test, y_test

model, params = train_model(X_train, y_train, X_val, y_val, epochs=5, batch_size=8, lr=0.001)

# Prédiction sur test
y_pred = model.apply(params, X_test)
print("Exemple de prédictions :", y_pred[:5])

