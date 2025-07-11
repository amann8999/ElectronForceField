import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.special import erf
from mol2feature import mol2feature


def YuOfYs(ys):
    ecut = 1e-2
    yu = ecut*jnp.sinh(ys)
    return yu


def get_zval(x):
    zval = jnp.array([0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
    return zval[jnp.int8(x)]


def Exact_AA(x):
    mask = x[2] > 1e-8
    r = x[2] + 1e-8
    out = get_zval(x[0]) * get_zval(x[1]) / r
    return jnp.where(mask, out, 0.0)


def Exact_AE(x):
    mask = x[2] > 1e-8
    z = get_zval(x[0])
    rw = x[1] + 1e-8
    w = x[2] + 1e-8
    out = -z / (rw*w) * erf(rw / jnp.sqrt(2))
    return jnp.where(mask, out, 0.0)


def Exact_EE(x):
    mask = x[2] > 1e-8
    rw = x[0] + 1e-8
    wij = x[1] + 1e-8
    out = 1 / (rw*wij) * erf(rw / jnp.sqrt(2))
    return jnp.where(mask, out, 0.0)


class AE(eqx.Module):
    embed: eqx.nn.Embedding
    net: eqx.nn.MLP

    def __init__(self, key, embed_dim=4, hidden_dim=10, depth=3, Z_max=18):
        k1, k2 = jax.random.split(key)
        self.embed = eqx.nn.Embedding(Z_max, embed_dim, key=k1)
        self.net = eqx.nn.MLP(embed_dim + 2, 1, hidden_dim, depth, key=k2, activation=jax.nn.gelu)

    def __call__(self, x):
        Z = x[0]
        zvec = self.embed(jnp.int32(Z) - 1)
        features = jnp.concatenate([zvec, x[1:2], jnp.log(x[2:3])])
        raw = self.net(features)
        return jnp.where(x[2] > 1e-8, YuOfYs(raw), 0.0)


class EE(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, key, hidden_dim=10, embed_dim=0, depth=3):
        self.net = eqx.nn.MLP(3, 1, hidden_dim, depth, key=key, activation=jax.nn.gelu)

    def __call__(self, x):
        features = jnp.concatenate([x[:1], jnp.log(x[1:])])
        raw = self.net(features)
        return jnp.where(x[2] > 1e-8, YuOfYs(raw), 0.0)


class E(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, key, in_dim=1, hidden_dim=10, out_dim=1, depth=2):
        self.net = eqx.nn.MLP(in_dim, out_dim, hidden_dim, depth, key=key, activation=jax.nn.gelu)

    def __call__(self, x):
        mask = x[0] > 1e-8
        w = x[0] + 1e-8
        out = self.net(x) / w ** 2
        return jnp.where(mask, YuOfYs(out), 0.0)


class A(eqx.Module):
    w: jnp.ndarray
    baseline: jnp.ndarray

    def __init__(self, key):
        self.w = jax.random.normal(key, (18,)) * 0.1
        self.baseline = jnp.array([-0.102, -0.374, -7.271, -13.711, -22.315, -33.092, -45.951, -60.977, -78.934,
                                   -97.564, -162.078, -199.307, -240.428, -285.887, -335.441, -388.912, -446.201, -507.607])

    def __call__(self, x):
        out = jnp.dot(self.w, x) + jnp.dot(self.baseline, x)
        return out


class EnergyModel(eqx.Module):
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
        keys = jax.random.split(key, 22)
        self.ae = AE(keys[0])
        self.ee_same = EE(keys[1])
        self.ee_opp = EE(keys[2])
        self.e = E(keys[3])
        self.a = eqx.tree_deserialise_leaves('a.eqx',A(keys[4]))
        self.interp_ae_fn = interp_ae
        self.interp_ee_same_fn = interp_ee_same
        self.interp_ee_opp_fn = interp_ee_opp
        self.interp_e_fn = interp_e

    def __call__(self, mol_dict):
        E = 0.0
        feature_dict = mol2feature(mol_dict)

        if 'aa' in feature_dict:
            E += jnp.sum(jax.vmap(Exact_AA)(feature_dict['aa']))

        if 'ae' in feature_dict:
            E += jnp.sum(jax.vmap(self.ae)(feature_dict['ae']))
            E += jnp.sum(jax.vmap(Exact_AE)(feature_dict['ae']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ae_fn)(feature_dict['ae'])))

        if 'ee_same' in feature_dict:
            E += jnp.sum(jax.vmap(self.ee_same)(feature_dict['ee_same']))
            E += jnp.sum(jax.vmap(Exact_EE)(feature_dict['ee_same']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ee_same_fn)(feature_dict['ee_same'])))

        if 'ee_opp' in feature_dict:
            E += jnp.sum(jax.vmap(self.ee_opp)(feature_dict['ee_opp']))
            E += jnp.sum(jax.vmap(Exact_EE)(feature_dict['ee_opp']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_ee_opp_fn)(feature_dict['ee_opp'])))

        if 'e' in feature_dict:
            E += jnp.sum(jax.vmap(self.e)(feature_dict['e']))
            E += jnp.sum(YuOfYs(jax.vmap(self.interp_e_fn)(feature_dict['e'])))

        if 'a' in feature_dict:
            E += jnp.sum(jax.vmap(self.a)(feature_dict['a']))

        return E
