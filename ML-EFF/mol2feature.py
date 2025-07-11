import jax
import jax.numpy as jnp


def mol2feature(mol):
    max_na = jnp.size(mol['za']); max_nb = jnp.size(mol['sb'])
    na = jnp.count_nonzero(mol['za']); nb = jnp.count_nonzero(mol['sb'])
    
    feature_dict = dict()
    
    def get_aa_features(pair):
        i, j = pair
        rij = mol['ra'][:, i] - mol['ra'][:, j] + 1e-8
        dij = jnp.linalg.norm(rij)
        z1, z2 = mol['za'][i], mol['za'][j]
        return jnp.where((z1 != 0) & (z2 != 0), jnp.array([z1, z2, dij]), jnp.zeros(3))

    def get_ae_features(pair):
        i, j = pair
        j1 = mol['sb'][j]
        z = mol['za'][i]
        w = mol['wb'][j] + 1e-8
        rij = mol['ra'][:, i] - mol['rb'][:, j] + 1e-8
        dij = jnp.linalg.norm(rij)
        feat = jnp.array([z, dij/w, w])
        return jnp.where((z != 0) & (j1 != 0), feat, jnp.zeros_like(feat))

    def get_ee_same_features(pair):
        i, j = pair
        i1 = mol['sb'][i]
        j1 = mol['sb'][j]
        rij = mol['rb'][:, i] - mol['rb'][:, j] + 1e-8
        dij = jnp.linalg.norm(rij)
        wij = jnp.sqrt(mol['wb'][i]**2 + mol['wb'][j]**2) + 1e-8
        wmax = jnp.maximum(mol['wb'][i], mol['wb'][j]) + 1e-8
        wmin = jnp.minimum(mol['wb'][i], mol['wb'][j]) + 1e-8
        feat = jnp.array([dij/wij, wij, wmax/wmin])
        return jnp.where((i1 - j1 == 0) & (i1 != 0) & (j1 != 0), feat, jnp.zeros_like(feat))

    def get_ee_opp_features(pair):
        i, j = pair
        i1 = mol['sb'][i]
        j1 = mol['sb'][j]
        rij = mol['rb'][:, i] - mol['rb'][:, j] + 1e-8
        dij = jnp.linalg.norm(rij)
        wij = jnp.sqrt(mol['wb'][i]**2 + mol['wb'][j]**2) + 1e-8
        wmax = jnp.maximum(mol['wb'][i], mol['wb'][j]) + 1e-8
        wmin = jnp.minimum(mol['wb'][i], mol['wb'][j]) + 1e-8
        feat = jnp.array([dij/wij, wij, wmax/wmin])
        return jnp.where((i1 + j1 == 0) & (i1 != 0) & (j1 != 0), feat, jnp.zeros_like(feat))

    def get_a_features(i):
        z = mol['za'][i]
        feat = jnp.zeros(18).at[z-1].add(1)
        return jnp.where(z != 0, feat, jnp.zeros_like(feat))

    def get_e_features(i):
        w = mol['wb'][i]
        return jnp.where(w != 0, jnp.array([w]), jnp.zeros(1))
    
    ia = jnp.arange(max_na); ib = jnp.arange(max_nb)
    
    feature_dict['a'] = jax.vmap(get_a_features)(ia) if max_na > 0 else jnp.zeros((0, 18))
    
    feature_dict['e'] = jax.vmap(get_e_features)(ib) if max_nb > 0 else jnp.zeros((0, 1))
    
    i, j = jnp.triu_indices(max_na, k=1)  # k=1 excludes the diagonal
    pairs = jnp.stack([i, j], axis=-1)
    feature_dict['aa'] = jax.vmap(get_aa_features, in_axes=0)(pairs) if max_na > 1 else jnp.zeros((0, 3))
        
    i, j = jnp.meshgrid(ia, ib, indexing='ij')
    i, j = jnp.reshape(i, (-1,)), jnp.reshape(j, (-1,))
    pairs = jnp.stack([i, j], axis=-1)
    feature_dict['ae'] = jax.vmap(get_ae_features)(pairs) if (max_na > 0) and (max_nb > 0) else jnp.zeros((0, 3))
        
    i, j = jnp.triu_indices(max_nb, k=1)  # k=1 excludes the diagonal
    pairs = jnp.stack([i, j], axis=-1)
    feature_dict['ee_same'] = jax.vmap(get_ee_same_features)(pairs) if max_nb > 1 else jnp.zeros((0, 3))
    feature_dict['ee_opp'] = jax.vmap(get_ee_opp_features)(pairs) if max_nb > 1 else jnp.zeros((0, 3))

    return feature_dict
