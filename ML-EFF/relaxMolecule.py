import jax
import jax.numpy as jnp
import equinox as eqx


def relaxMolecule(mol, model, lr=1e-1, max_steps=20, constraints=jnp.array([]).astype(int)):
   
    diff_keys=['ra','rb','wb']

    def step_fn(carry, _):
        mol, E_prev, stop = carry

        # Compute gradient
        dEdx = eqx.filter_grad(model)(mol)
        dEdx = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), dEdx)
        dEdx['ra'] = jax.lax.cond(
            jnp.size(constraints) > 0,
            lambda ra: ra.at[:,constraints].set(0.),
            lambda ra: ra,
            dEdx['ra']
        )

        # Update only differentiable keys, skip update if stopped
        mol_updated = {k: jnp.where(stop, mol[k], mol[k] - lr * dEdx[k]) if k in diff_keys else mol[k] for k in mol}

        # Compute new energy, skip update if stopped
        E_new = jnp.where(stop, E_prev, model(mol_updated))

        dE = E_new - E_prev
        reject = dE > 0.0

        # Reject update if energy increased
        mol_updated = jax.tree.map(lambda new, old: jnp.where(reject, old, new), mol_updated, mol)
        E_new = jnp.where(reject, E_prev, E_new)

        stop_new = stop | reject

        return (mol_updated, E_new, stop_new), None

    # Initialize scan
    E0 = model(mol)
    init = (mol, E0, False)
    (final_carry, _) = jax.lax.scan(step_fn, init, xs=None, length=max_steps)

    mol_final, E_final, converged = final_carry

    return mol_final, E_final
