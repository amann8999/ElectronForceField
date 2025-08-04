import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.special import erf
from mol2feature import mol2feature


def YuOfYs(ys):
    # Maps raw model outputs to physical energy units using a scaled sinh
    # Ensures output has the correct scale and symmetry
    ecut = 1e-2
    yu = ecut*jnp.sinh(ys)
    return yu


def get_zval(x):
    # Maps atomic number Z to effective core+nucleus charge used in electrostatics
    # This can be thought of as a pseudopotential-inspired lookup table
    zval = jnp.array([0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
    return zval[jnp.int8(x)]


def Exact_AA(x):
    # Analytic atom-atom electrostatic interaction
    # x = [Z1, Z2, r], where r is the atomic separation
    mask = x[2] > 1e-8
    r = x[2]
    out = get_zval(x[0]) * get_zval(x[1]) / r
    return jnp.where(mask, out, 0.0)


def Exact_AE(x):
    # Analytic atom-electron electrostatic interaction
    # x = [Z, r, w], where w is the electron width
    mask = x[2] > 1e-8
    z = get_zval(x[0])
    rw = x[1]
    w = x[2]
    out = -z / (rw*w) * erf(rw / jnp.sqrt(2))
    return jnp.where(mask, out, 0.0)


def Exact_EE(x):
    # Analytic electron-electron electrostatic interaction
    # x = [r, w_ij, w_i/w_j], where wij = sqrt(w_i^2+w_j^2) and w_i > w_j
    mask = x[2] > 1e-8
    rw = x[0]
    wij = x[1]
    out = 1 / (rw*wij) * erf(rw / jnp.sqrt(2))
    return jnp.where(mask, out, 0.0)


class AA(eqx.Module):
    # Atom-atom neural network using atomic number embedding
    embed: eqx.nn.Embedding
    net: eqx.nn.MLP
    
    def __init__(self, key, embed_dim=4, hidden_dim=10, depth=3, Z_max=18):
        # Create embedding for each atomic number (Z ∈ [1, Z_max])
        # Input: [embedding(Z1), embedding(Z2), r] → energy
        k1, k2 = jax.random.split(key)
        self.embed = eqx.nn.Embedding(Z_max, embed_dim, key=k1)
        self.net = eqx.nn.MLP(2 * embed_dim + 1, 1, hidden_dim, depth, key=k2, activation=jax.nn.gelu)
        
    def __call__(self, x):
        Z1, Z2 = x[0], x[1]
        zvec1 = self.embed((Z1 - 1).astype(int))
        zvec2 = self.embed((Z2 - 1).astype(int))
        features = jnp.concatenate([zvec1, zvec2, x[2:3]])
        raw = self.net(features)
        return jnp.where(x[0]>1e-8, YuOfYs(raw), 0.0)


class AE(eqx.Module):
    # Atom-electron neural network using atomic number embedding
    embed: eqx.nn.Embedding
    net: eqx.nn.MLP

    def __init__(self, key, embed_dim=4, hidden_dim=10, depth=3, Z_max=18):
        # Input: [embedding(Z), r, log(w)] → energy
        k1, k2 = jax.random.split(key)
        self.embed = eqx.nn.Embedding(Z_max, embed_dim, key=k1)
        self.net = eqx.nn.MLP(embed_dim + 2, 1, hidden_dim, depth, key=k2, activation=jax.nn.gelu)

    def __call__(self, x):
        Z = x[0]
        zvec = self.embed((Z - 1).astype(int))
        features = jnp.concatenate([zvec, x[1:2], jnp.log(x[2:3])])
        raw = self.net(features)
        return jnp.where(x[2] > 1e-8, YuOfYs(raw), 0.0)


class EE(eqx.Module):
    # Electron-electron neural network
    net: eqx.nn.MLP

    def __init__(self, key, hidden_dim=10, embed_dim=0, depth=3):
        # Input: [r, log(wij), log(wi/wj)] → energy
        self.net = eqx.nn.MLP(3, 1, hidden_dim, depth, key=key, activation=jax.nn.gelu)

    def __call__(self, x):
        features = jnp.concatenate([x[:1], jnp.log(x[1:])])
        raw = self.net(features)
        return jnp.where(x[2] > 1e-8, YuOfYs(raw), 0.0)


class E(eqx.Module):
    # Single-electron self-energy neural network
    net: eqx.nn.MLP

    def __init__(self, key, in_dim=1, hidden_dim=10, out_dim=1, depth=2):
        # Input: [w] → output = energy * w^2
        self.net = eqx.nn.MLP(in_dim, out_dim, hidden_dim, depth, key=key, activation=jax.nn.gelu)

    def __call__(self, x):
        mask = x[0] > 1e-8
        w = x[0]
        out = self.net(x) / w ** 2
        return jnp.where(mask, YuOfYs(out), 0.0)


class A(eqx.Module):
    # Atomic self-energy linear model: w⋅x
    w: jnp.ndarray

    def __init__(self):
        self.w = jnp.array([-0.102, -0.374, -7.271, -13.711, -22.315, -33.092, -45.951, -60.977, -78.934,
                                   -97.564, -162.078, -199.307, -240.428, -285.887, -335.441, -388.912, -446.201, -507.607])

    def __call__(self, x):
        out = jnp.dot(self.w, x)
        return out


class EnergyModel(eqx.Module):
    # Main energy model combining all neural and analytic components
    aa: AA
    ae: AE
    ee_same: EE
    ee_opp: EE
    e: E
    a: A
    interp_ae_fn: callable
    interp_ee_same_fn: callable
    interp_ee_opp_fn: callable
    interp_e_fn: callable

    def __init__(self, key, interp_ae, interp_ee_same, interp_ee_opp, interp_e):
        keys = jax.random.split(key, 5)

        # Initialize all neural networks with randomized parameters
        self.aa = AA(keys[0])
        self.ae = AE(keys[1])
        self.ee_same = EE(keys[2])
        self.ee_opp = EE(keys[3])
        self.e = E(keys[4])
        self.a = A()

        # Initialize all neural networks with loaded parameters
#        self.aa = eqx.tree_deserialise_leaves('aa.eqx',AA(keys[0]))
#        self.ae = eqx.tree_deserialise_leaves('ae.eqx',AE(keys[1]))
#        self.ee_same = eqx.tree_deserialise_leaves('ees.eqx',EE(keys[2]))
#        self.ee_opp = eqx.tree_deserialise_leaves('eeo.eqx',EE(keys[3]))
#        self.e = eqx.tree_deserialise_leaves('e.eqx',E(keys[4]))
#        self.a = eqx.tree_deserialise_leaves('a.eqx',A())

        # External interpolation functions (e.g. splines)
        self.interp_ae_fn = interp_ae
        self.interp_ee_same_fn = interp_ee_same
        self.interp_ee_opp_fn = interp_ee_opp
        self.interp_e_fn = interp_e

    def __call__(self, mol_dict):
        # Compute total energy for a given molecule dictionary
        E = 0.0
        feature_dict = mol2feature(mol_dict)

        # Atom-atom components
        if 'aa' in feature_dict:
#            E += jnp.sum(jax.vmap(self.aa)(feature_dict['aa']))
            E += jnp.sum(jax.vmap(Exact_AA)(feature_dict['aa']))

        # Atom-electron components
        if 'ae' in feature_dict:
#            E += jnp.sum(jax.vmap(self.ae)(feature_dict['ae']))
            E += jnp.sum(jax.vmap(Exact_AE)(feature_dict['ae']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ae_fn)(feature_dict['ae'])))

        # Electron-electron (same spin) components
        if 'ee_same' in feature_dict:
#            E += jnp.sum(jax.vmap(self.ee_same)(feature_dict['ee_same']))
            E += jnp.sum(jax.vmap(Exact_EE)(feature_dict['ee_same']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ee_same_fn)(feature_dict['ee_same'])))

        # Electron-electron (opposite spin) components
        if 'ee_opp' in feature_dict:
#            E += jnp.sum(jax.vmap(self.ee_opp)(feature_dict['ee_opp']))
            E += jnp.sum(jax.vmap(Exact_EE)(feature_dict['ee_opp']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ee_opp_fn)(feature_dict['ee_opp'])))

        # Single-electron components
        if 'e' in feature_dict:
#            E += jnp.sum(jax.vmap(self.e)(feature_dict['e']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_e_fn)(feature_dict['e'])))

        # Single atom component
#        if 'a' in feature_dict:
#            E += jnp.sum(jax.vmap(self.a)(feature_dict['a']))

        return E
