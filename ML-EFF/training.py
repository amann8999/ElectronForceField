from scipy.io import loadmat
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import time
%matplotlib inline
from ipywidgets import interact, fixed

import psEeFF
from data_utils import *
from lossFunction import lossFunction
from relaxMolecule import relaxMolecule
from mol2feature import mol2feature
from interpolation_utils import *
from vibrations import vibrations
from showResultsEFF import *

# Get Data
molSet = 5  # 0: 18 atoms, 1: 12 mols, 2: 22 mols, 3: 36 mols, 4: 66 mols, 5: 112 mols, 6: 556 mols
lr = 1e-2  # Initial Learning Rate

expMatData = loadmat('expNIST.mat', simplify_cells=True)['molExp']
expJaxData = mat2jax(expMatData, molSet)
molExp, eExp, wExp = data2mol(expJaxData)

effMatData = loadmat('effNIST.mat', simplify_cells=True)['molEFF']
effJaxData = mat2jax(effMatData, molSet)
molEFF, eEFF, wEFF = data2mol(effJaxData)

# Initialize model and optimizer
key = jax.random.PRNGKey(0)
interp_fns = get_interp_fns()
model = psEeFF.EnergyModel(key=jax.random.PRNGKey(0), interp_ae=interp_fns[0], interp_ee_same=interp_fns[1],
                           interp_ee_opp=interp_fns[2], interp_e=interp_fns[3])
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def update(model, opt_state, molEFF, molExp, eExp, wExp):
    loss, grads = eqx.filter_value_and_grad(lossFunction)(model, molEFF, molExp, eExp, wExp)
    grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, 0.), grads)  # Avoid NaNs in gradients
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Training loop with adaptive learning rate
losses = []
best_loss = float("inf")
stagnant_steps = 0
patience = 10  # Number of steps to wait before decreasing learning rate

for step in range(100001):
    start = time.time()
    model, opt_state, loss = update(model, opt_state, molEFF, molExp, eExp, wExp)
    
    # Check for improvement
    if loss < best_loss:
        best_loss = loss
        best_model = model
        stagnant_steps = 0  # Reset counter
    else:
        stagnant_steps += 1
    losses.append(best_loss)

    # Adjust learning rate if no improvement
    if stagnant_steps >= patience:
        lr /= 10  # Change learning rate
        optimizer = optax.adam(learning_rate=lr)  # Create new optimizer
        model = best_model
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))  # Reset optimizer state
        stagnant_steps = 0  # Reset counter
        print(f"Step {step}: Learning rate reduced to {lr:.2e}")

    if step % 1 == 0:
        print(f"Step {step}, Train Loss: {loss:.6f}, Best Loss: {best_loss:.6f}, Time: {time.time() - start:.4f} sec")
    if lr < 1e-8:
        break


# Save Best Model
eqx.tree_serialise_leaves('ae.eqx', best_model.ae)
eqx.tree_serialise_leaves('ees.eqx', best_model.ee_same)
eqx.tree_serialise_leaves('eeo.eqx', best_model.ee_opp)
eqx.tree_serialise_leaves('e.eqx', best_model.e)
eqx.tree_serialise_leaves('a.eqx', best_model.a)
