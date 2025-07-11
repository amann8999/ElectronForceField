import jax
import jax.numpy as jnp
from relaxMolecule import relaxMolecule
from mol2feature import mol2feature
from vibrations import vibrations


def lossFunction(model, molEFF, molExp, eExp, wExp):
    # Energy and final relaxed structure
    relMolEFF, eEFF = jax.vmap(relaxMolecule, in_axes=(0,None))(molEFF, model)
    wEFF = jax.vmap(vibrations, in_axes=(0, None))(relMolEFF, model)[0]

    # Energies
    energy_loss = jnp.mean(jnp.square(eEFF - eExp))

    # Geometry loss (only using aa for now)
    featExp = jax.vmap(mol2feature)(molExp)
    featEFF = jax.vmap(mol2feature)(relMolEFF)

    def safe_mse(feff, fexp):
        delta = feff[..., -1] - fexp[..., -1]
        mask = fexp[..., -1] != 0
        return jnp.sum(jnp.square(delta) * mask) / jnp.maximum(jnp.sum(mask), 1)

    geometry_loss = safe_mse(featEFF['aa'], featExp['aa'])

    # Vibrational loss
    mask = (wExp != 0) & (jnp.abs(wExp - wEFF[:, ::-1][:, :wExp.shape[1]]) < 0.1)
    vib_sq_error = jnp.square(jnp.sort(wExp,axis=1) - jnp.sort(wEFF[:, ::-1][:, :wExp.shape[1]],axis=1)) * mask
    vib_loss = jnp.sum(vib_sq_error) / jnp.maximum(jnp.sum(mask), 1)

    # Final combined loss (rescaled)
    loss = jnp.sqrt(energy_loss) / 1e-1 + jnp.sqrt(geometry_loss) / 1e-1 + jnp.sqrt(vib_loss) / 1.5e-3
    
#    jax.debug.print("{x}, {y}, {z}", x=jnp.sqrt(energy_loss), y=jnp.sqrt(geometry_loss), z=jnp.sqrt(vib_loss))
    
    return loss
