import logging
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import os
import matplotlib.pyplot as plt
from data_generator import gen_data, Runge_Kutta_Method, ODE_fxn
import flax.linen as nn
logging.getLogger('jax').setLevel(logging.WARNING)

plot_dir = "Ouput_Plots"

class MLP(nn.Module):
    """Multilayer Perceptron model."""
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.tanh(x)
        return x

class TrainState(train_state.TrainState):
    metrics: dict


def create_train_state(model, init_key, learning_rate,momentum, input_shape):
    params = model.init(init_key, jnp.ones(input_shape))['params']
    tx = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def mse_loss(params, apply_fn, batch):
    x, y_true = batch
    y_pred = apply_fn({'params': params}, x)
    if y_true.ndim == 1 and y_pred.ndim == 2:
        y_true = jnp.expand_dims(y_true, axis=-1)  
        y_true = jnp.tile(y_true, (1, y_pred.shape[1]))
    return jnp.mean((y_pred - y_true) ** 2)

@jax.jit
def compute_metrics(state, batch):
    loss = mse_loss(state.params, state.apply_fn, batch)
    return {'mse': loss}

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        return mse_loss(params, state.apply_fn, batch)
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(state, batch)
    return state, metrics


@jax.jit
def val_step(state, batch):
    return compute_metrics(state, batch)


def train_model(state, train_data, val_data, epochs):
    metrics_history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        for batch in train_data:
            state, train_metrics = train_step(state, batch)
        for batch in val_data:
            val_metrics = val_step(state, batch)
        if epoch % 2000 == 0:
            metrics_history['train_loss'].append(train_metrics['mse'])
            metrics_history['val_loss'].append(val_metrics['mse'])
            print(f"Epoch {epoch}, Train MSE: {train_metrics['mse']}, Val MSE: {val_metrics['mse']}")
    return state, metrics_history

def plot_loss_curve(metrics_history, plot_dir, filename):
    epochs = range(len(metrics_history['train_loss']))
    plt.plot(epochs, metrics_history['train_loss'], label="Train Loss")
    plt.plot(epochs, metrics_history['val_loss'], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curves saved to {plot_path}")


@hydra.main(version_base=None, config_path="config", config_name="config_train")
def main(cfg: DictConfig):
    y0 = jnp.array(cfg.data.y0)
    t0 = cfg.data.t0
    t_n = cfg.data.t_n
    h = cfg.data.h
    b = cfg.data.b
    m = cfg.data.m
    l = cfg.data.l
    g = cfg.data.g

    t, y = Runge_Kutta_Method(ODE_fxn, y0, t0, t_n, h, b, m, l, g)
    t_train, y_train, t_test, y_test = gen_data(t, y)

    t_train = t_train[:, None]
    t_test = t_test[:, None]

    train_data = [(t_train, y_train)]
    val_data = [(t_test, y_test)]

    key = jax.random.PRNGKey(0)
    model = MLP(cfg.model.features)
    input_shape = (t_train.shape[1],)
    
    state = create_train_state(model, key, cfg.optimizer.learning_rate,cfg.optimizer.momentum, input_shape)

    state, metrics_history = train_model(state, train_data, val_data, cfg.train.epochs)

    filename = 'MSE_loss_curve.png'
    plot_loss_curve(metrics_history, plot_dir, filename)

    print(" ")
    test_metrics = val_step(state, (t_test, y_test))
    print(f"Test MSE: {test_metrics['mse']}")

if __name__ == "__main__":
    main()
