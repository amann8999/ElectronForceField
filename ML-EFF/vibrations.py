import jax
import jax.numpy as jnp
from jax import hessian
from jax.flatten_util import ravel_pytree


atomic_masses = jnp.array([1e-8, 1.0080, 4.0030, 6.9410, 9.0120, 10.8110, 12.0110, 14.0070, 15.9990,
                           18.9980, 20.1800, 22.9900, 24.3050, 26.9820, 28.0860, 30.9740, 32.0660,
                           35.4530, 39.9480])

amu = 1822.8884  # conversion factor to atomic units


def flat_hessian(mol, model, diff_keys=('ra', 'rb', 'wb')):
    diff_mol = {k: mol[k] for k in diff_keys}
    static_mol = {k: mol[k] for k in mol if k not in diff_keys}
    flat_diff, unravel_fn = ravel_pytree(diff_mol)

    def wrapped_model(flat_inputs):
        mol_diff = unravel_fn(flat_inputs)
        full_mol = {**mol_diff, **static_mol}
        return model(full_mol)

    H = hessian(wrapped_model)(flat_diff)
    return H, unravel_fn


def vibrations(mol, model):
    H, _ = flat_hessian(mol, model)

    H = jnp.nan_to_num(H)
    H = 0.5 * (H + H.T)

    na = mol['za'].shape[0]
    nb = mol['sb'].shape[0]

    ma = atomic_masses[mol['za']] * amu
    mb = jnp.ones(nb)
    mw = 3.0 * jnp.ones(nb)

    mass_vec = jnp.concatenate([ma] * 3 + [mb] * 3 + [mw])

    M = jnp.sqrt(mass_vec[:, None] * mass_vec[None, :])

    w2, v = jnp.linalg.eigh(H/M)

    iva = jnp.argsort(-jnp.sum(v[:3*na] ** 2, axis=0))[:3*na]
    ivb = jnp.argsort(jnp.sum(v[:3*na] ** 2, axis=0))[:4*nb]

    va = v[:, iva]
    vb = v[:, ivb]

    wa2 = w2[iva]
    wb2 = w2[ivb]

    iwa = jnp.argsort(wa2)
    iwb = jnp.argsort(wb2)

    wa2 = wa2[iwa]
    wb2 = wb2[iwb]

    va = va[:, iwa] / jnp.sqrt(mass_vec)[:,None]
    vb = vb[:, iwb] / jnp.sqrt(mass_vec)[:,None]

    va /= jnp.linalg.norm(va, axis=0)[None,:]
    vb /= jnp.linalg.norm(vb, axis=0)[None,:]

    wa = jnp.sign(wa2) * jnp.sqrt(jnp.abs(wa2))
    wb = jnp.sign(wb2) * jnp.sqrt(jnp.abs(wb2))

    return wa, wb, va, vb
