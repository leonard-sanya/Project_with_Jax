import logging  
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad
import matplotlib.pyplot as plt
import os
import optax
from train import MLP, create_train_state
from data_generator import Runge_Kutta_Method, ODE_fxn
import hydra
from omegaconf import DictConfig
logging.getLogger('jax').setLevel(logging.WARNING)

def ode_loss(params, apply_fn, batch, b, m, l, g):
    t = batch
    t_resized = t.reshape(-1, 1)
    mlp_pred = apply_fn({'params': params}, t_resized).squeeze()
    
    d_mlp_dt = vmap(lambda time: grad(lambda t: apply_fn({'params': params}, t.reshape(-1, 1)).squeeze())(time))(t)
    d2_mlp_dt2 = vmap(lambda time: grad(lambda t: grad(lambda t: apply_fn({'params': params}, t.reshape(-1, 1)).squeeze())(time))(t))(t)
    ode_residual = d2_mlp_dt2 + (b / m) * d_mlp_dt + (g / l) * jnp.sin(mlp_pred)

    initial_condition_angle = (mlp_pred[0] - 2 * jnp.pi / 3) ** 2
    initial_condition_velocity = d_mlp_dt[0] ** 2
    total_loss = jnp.mean(ode_residual**2) + initial_condition_angle + initial_condition_velocity

    return total_loss

@jit
def ode_train_step(state, batch,b, m, l, g):
    loss_fn = lambda params: ode_loss(params, state.apply_fn, batch, b, m, l, g)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss

def train_ode_model(key, batch, model, epochs, learning_rate,momentum,b, m, l, g):
    input_shape = (1, 1)
    init_key, _ = jax.random.split(key)
    state = create_train_state(model, init_key,learning_rate,momentum, input_shape)
    losses = []

    for epoch in range(epochs):
        state, loss = ode_train_step(state, batch,b, m, l, g)
        losses.append(loss)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return state, losses

@hydra.main(version_base=None, config_path="config", config_name="config_ode_train")
def main(cfg: DictConfig):
    y0 = jnp.array(cfg.data.y0)
    t0 = cfg.data.t0
    t_n = cfg.data.t_n
    h = cfg.data.h
    b = cfg.data.b
    m = cfg.data.m
    l = cfg.data.l
    g = cfg.data.g

    t, _ = Runge_Kutta_Method(ODE_fxn, y0, t0, t_n, h, b, m, l, g)
  
    key = jax.random.PRNGKey(0)
    model = MLP(cfg.model.features)
    learning_rate = cfg.optimizer.learning_rate
    momentum = cfg.optimizer.momentum
    epochs = cfg.train.epochs

    state, ode_metrics_history = train_ode_model(key, t, model, epochs, learning_rate,momentum,b, m, l, g)

    filename = 'ode_train_loss_curve.png'
    plt.plot(ode_metrics_history)
    plt.title('ODE Training Loss over Epochs')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plot_dir = "loss_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curves saved to {plot_path}")

if __name__ == "__main__":
    main()
