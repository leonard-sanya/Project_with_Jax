import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import hydra
import logging
from omegaconf import DictConfig
import time
from jax import lax,jit
import os

logging.getLogger('jax').setLevel(logging.WARNING)
plot_dir = "Ouput_Plots"
os.makedirs(plot_dir, exist_ok=True)

def ODE_fxn(t, y, b, m, l, g):
    '''
    ODE_fxn takes in y and returns F(y)
    '''
    return jnp.array([y[1], -b/m * y[1] -g/l * jnp.sin(y[0])])


def Eulers_Method(F, y0, t0, t_n, h, b, m, l, g):
    '''
    f: The ODE function (ODE_fxn).
    y0: Our initial y condition.
    t0: Starting step.
    t_n: Ending step.
    h: Step size. 
    '''
    t_steps = jnp.arange(t0, t_n, h)
    y_values = [y0]
    for t in t_steps[:-1]:
        y_values.append(y_values[-1] + h * F(t,y_values[-1], b, m, l, g))
    return t_steps, jnp.array(y_values)



def Scan_Eulers_Method(ODE_fxn,y0, t0, t_n, h, b, m, l, g):
    t_vals = jnp.arange(t0, t_n, h)
    init = (y0, t0) 

    def euler_step(carry, t):
        y, t = carry
        dy = ODE_fxn(t, y, b, m, l, g)
        y_next = y + h * dy  
        return (y_next, t + h), y_next

    carry, y_values = lax.scan(euler_step, init, t_vals)
    return carry, y_values 
   

@jit
def Scan_Eulers_Method_Jit(y0, t_vals, h, b, m, l, g):
    # Ensure y0 is a JAX array
    y0 = jnp.array(y0)
    init = (y0, jnp.array(t_vals[0]))  # Ensure t0 is part of t_vals
    
    def euler_step(carry, t):
        y, t = carry
        dy = ODE_fxn(t, y, b, m, l, g)  # JAX operations here
        y_next = y + h * dy
        return (y_next, t + h), y_next
    
    carry, y_vals = lax.scan(euler_step, init, t_vals[:-1])
    return t_vals, jnp.vstack([y0, y_vals]) 



def Runge_Kutta_Method(F, y0, t0, t_n, h, b, m, l, g):
    t_steps = jnp.arange(t0, t_n, h)
    y_values = [y0]

    for t in t_steps[:-1]:
        y_n = y_values[-1]
        k1 = F(t,y_n, b, m, l, g)
        k2 = F(t + (h/2) , y_n + h * (k1/2), b, m, l, g)
        k3 = F(t + (h/2) , y_n + h * (k2/2), b, m, l, g)
        k4 = F(t + h , y_n + h * k3, b, m, l, g)
        
        y_next = y_n + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y_values.append(y_next)
    return t_steps, jnp.array(y_values)

def Scan_Runge_Kutta_Method(F, y0, t0, t_n, h, b, m, l, g):
    t_steps = jnp.arange(t0, t_n, h)
    init = (y0, t0)  

    def rk4_step(carry, t):
        y_n, t = carry
        
        k1 = F(t, y_n, b, m, l, g)
        k2 = F(t + (h / 2), y_n + h * (k1 / 2), b, m, l, g)
        k3 = F(t + (h / 2), y_n + h * (k2 / 2), b, m, l, g)
        k4 = F(t + h, y_n + h * k3, b, m, l, g)
        
        y_next = y_n + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return (y_next, t + h), y_next 

    carry, y_vals = lax.scan(rk4_step, init, t_steps[:-1])
    return t_steps, jnp.vstack([y0, y_vals]) 

def Scan_Runge_Kutta_Method_Jit(F, y0, t_vals, h, b, m, l, g):
    y0 = jnp.array(y0)
    init = (y0, jnp.array(t_vals[0]))  

    def rk4_step(carry, t):
        y_n, t = carry
        
        k1 = F(t, y_n, b, m, l, g)
        k2 = F(t + (h / 2), y_n + h * (k1 / 2), b, m, l, g)
        k3 = F(t + (h / 2), y_n + h * (k2 / 2), b, m, l, g)
        k4 = F(t + h, y_n + h * k3, b, m, l, g)
        
        y_next = y_n + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return (y_next, t + h), y_next

    carry, y_vals = lax.scan(rk4_step, init, t_vals[:-1])
    return t_vals, jnp.vstack([y0, y_vals])


def plot_solution(t_values, y_values,filename,method_name="Method"):
    plt.plot(t_values, y_values[:, 0], label='Theta (angle)')
    plt.plot(t_values, y_values[:, 1], label='Angular Velocity')
    plt.title(f"Solution using {method_name}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curves saved to {plot_path}")

def gen_data(t, y):
    """Generate test and train data from the solution of the numerical method."""
    t_sliced, y_sliced = (
        t[jnp.arange(t.size, step=200)],
        y[jnp.arange(t.size, step=200)],
    )
    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]

    return t_train, y_train, t_test, y_test


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

    filename_euler = "Euler_Method.png"
    t_euler, y_euler = Eulers_Method(ODE_fxn, y0, t0, t_n, h, b, m, l, g)
    plot_solution(t_euler, y_euler,filename_euler,"Euler_Method")
    
    filename_rk4 = "Runge_Kutta_Method.png"
    t_rk, y_rk = Runge_Kutta_Method(ODE_fxn, y0, t0, t_n, h, b, m, l, g)
    plot_solution(t_rk, y_rk,filename_rk4,"Runge-Kutta Method")
   

    t_vals = jnp.arange(t0, t_n, h)
    print("  ")
    print("============ Euler_Methods Time Comparison ===========")
    start = time.time()
    euler_for_loop = Eulers_Method(ODE_fxn,y0, t0, t_n, h, b, m, l, g)
    end = time.time()
    print(f"Euler's method using loop took: {end - start} seconds")

    start = time.time()
    euler_scan_nojit = Scan_Eulers_Method(ODE_fxn,y0, t0, t_n, h, b, m, l, g)
    end = time.time()
    print(f"Scan Euler's method without Jit took: {end - start} seconds")

    start = time.time()
    euler_scan_jit = Scan_Eulers_Method_Jit(y0, t_vals, h, b, m, l, g)
    end = time.time()
    print(f"Scan Euler's method with Jit took: {end - start} seconds")

   

    print("  ")
    print("====== Runge_Kutta_Methods Time Comparison =======")
    start = time.time()
    rk_for_loop = Runge_Kutta_Method(ODE_fxn,y0, t0, t_n, h, b, m, l, g)
    end = time.time()
    print(f"Runge-Kutta method using for loop took: {end - start} seconds")
    
    start = time.time()
    rk_scan_nojit = Scan_Runge_Kutta_Method(ODE_fxn,y0, t0, t_n, h, b, m, l, g)
    end = time.time()
    print(f"Scan Runge-Kutta method without Jit took: {end - start} seconds")

    start = time.time()
    rk_scan_jit= Scan_Runge_Kutta_Method_Jit(ODE_fxn, y0, t_vals, h, b, m, l, g)
    end = time.time()
    print(f"Scan Runge-Kutta method with Jit took: {end - start} seconds")


if __name__ == "__main__":
    main()
