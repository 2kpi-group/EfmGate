
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

# Split train/val/test

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    N = X.shape[0]
    n_train = int(N*train_ratio)
    n_val = int(N*val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Convert Torch -> JAX

def to_jax(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return jnp.asarray(a)


# R² mean & std

def r2_mean_std(y_true, y_pred):
    r2s = []
    B = y_true.shape[0]
    for b in range(B):
        ss_res = jnp.sum((y_true[b]-y_pred[b])**2)
        ss_tot = jnp.sum((y_true[b]-jnp.mean(y_true[b]))**2)
        r2 = 1 - ss_res/ss_tot
        r2s.append(float(r2))
    r2s = np.array(r2s)
    return r2s.mean(), r2s.std()


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
