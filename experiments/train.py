# Imports & Seeds

from efmgate.models.efm_gate import EfmLSTM
from efmgate.data.dataset import generate_ou_signal, split_data, r2_score

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






class EfmLSTMPredictor(nn.Module):
    units: int = 16
    out_size: int = 1
    signature_depth: int = 3
    signature_input_size: int = 5

    @nn.compact
    def __call__(self, x):
        h = EfmLSTM(self.units, self.signature_depth, self.signature_input_size)(x)
        h = EfmLSTM(self.units, self.signature_depth, self.signature_input_size)(h)
        return nn.Dense(self.out_size)(h)

# ============================================
# Training
# ============================================
def train_model(
    x_train, y_train, x_val, y_val, 
    epochs=100, lr=0.01, batch_size=100, 
    patience_es=10, min_delta=1e-5,
    lr_reduce_patience=5, lr_factor=0.25, min_lr=2.5e-5
):
    model = EfmLSTMPredictor()
    params = model.init(jax.random.PRNGKey(0), x_train)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y)**2)

    grad_fn = jax.value_and_grad(loss_fn)


    best_val_loss = float("inf")
    epochs_no_improve_es = 0
    epochs_no_improve_lr = 0
    current_lr = lr


    num_batches = int(np.ceil(x_train.shape[0] / batch_size))

    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        x_train_shuff = x_train[perm]
        y_train_shuff = y_train[perm]
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train_shuff[start:end]
            y_batch = y_train_shuff[start:end]
            loss, grads = grad_fn(params, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            val_loss = float(loss_fn(params, x_val, y_val))
            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                best_params = params
                epochs_no_improve_es = 0
                epochs_no_improve_lr = 0
            else:
                epochs_no_improve_es += 1
                epochs_no_improve_lr += 1
                
            if epochs_no_improve_lr >= lr_reduce_patience:
                current_lr = max(current_lr * lr_factor, min_lr)
                optimizer = optax.adam(current_lr)
                opt_state = optimizer.init(params)
                epochs_no_improve_lr = 0
                    
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1} | Val loss: {val_loss:.5f} | LR: {current_lr:.6f}")
                 
            if epochs_no_improve_es >= patience_es:
                print(f"Early stopping triggered at epoch {epoch+1}")
                params = best_params
                break


# ============================================
# Entraînement
# ============================================

# --- Initialisation des résultats ---
n_aheads = [1]  # ou [1,6] si tu veux plusieurs horizons
results_r2 = {n: [] for n in n_aheads}
results_rmse = {n: [] for n in n_aheads}

for n_ahead in n_aheads:
    for run in range(5):
        # Générer des données pour ce run
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(paths, targets)
        X_train, y_train = jnp.array(X_train), jnp.array(y_train)
        X_val, y_val = jnp.array(X_val), jnp.array(y_val)
        X_test, y_test = jnp.array(X_test), jnp.array(y_test)

        # Entraîner le modèle
        model, params = train_model(X_train, y_train, X_val, y_val)

        # Prédictions
        y_pred = model.apply(params, X_test)

        # Calcul R² et RMSE
        r2 = r2_score(y_test, y_pred)
        rmse = jnp.sqrt(jnp.mean((y_pred - y_test)**2))

        # Stocker les résultats
        results_r2[n_ahead].append(float(r2))
        results_rmse[n_ahead].append(float(rmse))

        print(f"Run {run+1}, n_ahead={n_ahead} | R²: {r2:.4f}, RMSE: {rmse:.4f}")

