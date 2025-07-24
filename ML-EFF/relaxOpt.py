import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxopt


def relaxBatch(mol_batch, model, diff_keys=['ra','rb','wb']):
    # === Step 1: Separate static and differentiable keys ===
    diff = {k: mol_batch[k] for k in diff_keys}
    static = {k: mol_batch[k] for k in mol_batch if k not in diff_keys}

    # === Step 2: Flatten differentiable batch ===
    flat_diff, unravel_fn = ravel_pytree(diff)
    
    # === Step 3: Define total energy function ===
    def total_energy(flat_inputs):
        diff_unraveled = unravel_fn(flat_inputs)
        full_batch = {**diff_unraveled, **static}
        energies = jax.vmap(model)(full_batch)
        return jnp.sum(energies)  

    def safe_value_and_grad(f):
        def f_vg(x):
            val, grad = jax.value_and_grad(f)(x)
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return val, grad
        return f_vg

    # === Step 4: Optimize ===
    # LBFGS with custom gradient 
    opt = jaxopt.LBFGS(
        fun=safe_value_and_grad(total_energy),
        value_and_grad=True,
        maxiter=20,
        unroll=True
    )
    result = opt.run(flat_diff)

    # === Step 5: Unpack final parameters ===
    flat_final = result.params
    diff_final = unravel_fn(flat_final)

    # Recombine with static mol data
    full_batch_final = {**static, **diff_final}
    e_final = jax.vmap(model)(full_batch_final)

    return full_batch_final, e_final


def relaxSingle(mol, model, diff_keys=['ra','rb','wb']):

    diff_mol = {k: mol[k] for k in diff_keys}
    static_mol = {k: mol[k] for k in mol if k not in diff_keys}
    flat_diff, unravel_fn = ravel_pytree(diff_mol)

    def wrapped_model(flat_inputs):
        mol_diff = unravel_fn(flat_inputs)
        full_mol = {**mol_diff, **static_mol}
        return model(full_mol)

    opt = jaxopt.LBFGS(fun=wrapped_model, maxiter = 100)
    result = opt.run(flat_diff)
    flat_diff_final = result.params
    mol_diff_final = unravel_fn(flat_diff_final)
    full_mol_final = {**mol_diff_final, **static_mol}
    e_final = model(full_mol_final)

    return full_mol_final, e_final
