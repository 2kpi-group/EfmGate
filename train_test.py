# Imports & Seeds

from EfmGate import EfmLSTM
from data import generate_ou_signal, split_data, to_jax, r2_mean_std
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
import optax


class EfmLSTMPredictor(nn.Module):
    units: int = 16
    out_size: int = 1
    signature_depth: int = 2
    signature_input_size: int = 5

    @nn.compact
    def __call__(self, x):
        h1 = EfmLSTM(self.units, self.signature_depth, self.signature_input_size, return_sequences=True)(x)
        h2 = EfmLSTM(self.units, self.signature_depth, self.signature_input_size, return_sequences=True)(h1)
        y_pred = nn.Dense(self.out_size)(h2)
        return y_pred


# Entraînement

def train_model(
    x_train, y_train,
    x_val, y_val,
    epochs=1000,
    lr_init=0.01,
    patience_es=10,
    min_delta=1e-5,
    patience_lr=5,
    lr_factor=0.25,
    min_lr=2.5e-5
):
    model = EfmLSTMPredictor()
    key = jax.random.PRNGKey(0)
    params = model.init(key, x_train)

    lr = lr_init
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y)**2)

    grad_fn = jax.value_and_grad(loss_fn)

    best_val = np.inf
    best_params = params
    patience_es_cnt = 0
    patience_lr_cnt = 0

    for epoch in range(epochs):
        train_loss, grads = grad_fn(params, x_train, y_train)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        val_loss = float(loss_fn(params, x_val, y_val))

        #  EARLY STOPPING 
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_params = params
            patience_es_cnt = 0
            patience_lr_cnt = 0
        else:
            patience_es_cnt += 1
            patience_lr_cnt += 1

        # LR REDUCTION
        if patience_lr_cnt >= patience_lr and lr > min_lr:
            lr = max(lr * lr_factor, min_lr)
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(params)
            patience_lr_cnt = 0
            print(f"🔽 LR reduced to {lr:.6f}")

        #  STOP 
        if patience_es_cnt >= patience_es:
            print(f" Early stopping at epoch {epoch}")
            break

        if (epoch+1) % 50 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {lr:.6f}"
            )

    return model, best_params


# SECTION 1 : 1-step ahead

print("\n=== 1-step ahead ===")
def build_dataset_1step(n_paths=50, T=500):
    signals = []
    for i in range(n_paths):
        _, s = generate_ou_signal(T=T, seed=100+i)
        signals.append(s)
    signals = np.stack(signals)
    X = signals[:, :-1]
    y = signals[:, 1:]
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(2)
    return X, y

X1, y1 = build_dataset_1step()
(X1_train, y1_train), (X1_val, y1_val), (X1_test, y1_test) = split_data(X1, y1)

x1_train, y1_train = to_jax(X1_train), to_jax(y1_train)
x1_val, y1_val     = to_jax(X1_val), to_jax(y1_val)
x1_test, y1_test   = to_jax(X1_test), to_jax(y1_test)

model1, params1 = train_model(x1_train, y1_train, x1_val, y1_val)
y1_pred = model1.apply(params1, x1_test)
r2_mean1, r2_std1 = r2_mean_std(y1_test, y1_pred)
print("R² 1-step → Mean:", r2_mean1, "Std:", r2_std1)

# SECTION 2 : 9-step ahead

print("\n=== 9-step ahead ===")
def build_dataset_9step(n_paths=50, T=500, step=9):
    signals = []
    for i in range(n_paths):
        _, s = generate_ou_signal(T=T, seed=100+i)
        signals.append(s)
    signals = np.stack(signals)
    X = signals[:, :-step]
    y = signals[:, step:]
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(2)
    return X, y

X9, y9 = build_dataset_9step()
(X9_train, y9_train), (X9_val, y9_val), (X9_test, y9_test) = split_data(X9, y9)

x9_train, y9_train = to_jax(X9_train), to_jax(y9_train)
x9_val, y9_val     = to_jax(X9_val), to_jax(y9_val)
x9_test, y9_test   = to_jax(X9_test), to_jax(y9_test)

model9, params9 = train_model(x9_train, y9_train, x9_val, y9_val)
y9_pred = model9.apply(params9, x9_test)
r2_mean9, r2_std9 = r2_mean_std(y9_test, y9_pred)
print("R² 9-step → Mean:", r2_mean9, "Std:", r2_std9)
