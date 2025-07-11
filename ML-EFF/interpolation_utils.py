import jax
import jax.numpy as jnp
from scipy.io import loadmat


def YsOfYu(yu):
    ecut = 1e-2
    ys = jnp.asinh(yu/ecut)
    return ys


def catmull_rom_weights(t):
    t2 = t * t
    t3 = t2 * t
    return jnp.array([
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    ])


def cubic_interp_1d(x, x_vals, y_vals):
    dx = x_vals[1] - x_vals[0]
    N = x_vals.shape[0]

    x = jnp.clip(x, x_vals[1], x_vals[-2])
    i = jnp.clip(jnp.floor((x - x_vals[0]) / dx).astype(int), 1, N - 3)
    t = (x - (x_vals[0] + i * dx)) / dx

    w = catmull_rom_weights(t)
    patch = jax.lax.dynamic_slice(y_vals, (i - 1,), (4,))  # shape (4,)
    return jnp.dot(w, patch)


def bicubic_interp_2d(x, y, x_vals, y_vals, grid):
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    Nx, Ny = grid.shape

    x = jnp.clip(x, x_vals[1], x_vals[-2])
    y = jnp.clip(y, y_vals[1], y_vals[-2])

    i = jnp.clip(jnp.floor((x - x_vals[0]) / dx).astype(int), 1, Nx - 3)
    j = jnp.clip(jnp.floor((y - y_vals[0]) / dy).astype(int), 1, Ny - 3)

    tx = (x - (x_vals[0] + i * dx)) / dx
    ty = (y - (y_vals[0] + j * dy)) / dy

    wx = catmull_rom_weights(tx)
    wy = catmull_rom_weights(ty)

    patch = jax.lax.dynamic_slice(grid, (i - 1, j - 1), (4, 4))  # shape (4, 4)
    return jnp.dot(wx, jnp.dot(patch, wy))


def tricubic_interp_3d(x, y, z, x_vals, y_vals, z_vals, grid):
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    dz = z_vals[1] - z_vals[0]
    Nx, Ny, Nz = grid.shape

    x = jnp.clip(x, x_vals[1], x_vals[-2])
    y = jnp.clip(y, y_vals[1], y_vals[-2])
    z = jnp.clip(z, z_vals[1], z_vals[-2])

    i = jnp.clip(jnp.floor((x - x_vals[0]) / dx).astype(int), 1, Nx - 3)
    j = jnp.clip(jnp.floor((y - y_vals[0]) / dy).astype(int), 1, Ny - 3)
    k = jnp.clip(jnp.floor((z - z_vals[0]) / dz).astype(int), 1, Nz - 3)

    tx = (x - (x_vals[0] + i * dx)) / dx
    ty = (y - (y_vals[0] + j * dy)) / dy
    tz = (z - (z_vals[0] + k * dz)) / dz

    wx = catmull_rom_weights(tx)
    wy = catmull_rom_weights(ty)
    wz = catmull_rom_weights(tz)

    patch = jax.lax.dynamic_slice(grid, (i - 1, j - 1, k - 1), (4, 4, 4))  # shape (4, 4, 4)
    return jnp.tensordot(wx, jnp.tensordot(wy, jnp.tensordot(wz, patch, axes=[0,2]), axes=[0,1]), axes=[0,0])


def make_ae_interp_fn(r_vals, w_vals, ae_grid):
    def fn(x):
        Z, r, w = jnp.int8(x[0]), x[1], x[2]
        val = bicubic_interp_2d(r, jnp.log(w), r_vals, w_vals, ae_grid[Z-1,:,:])
        return jnp.where(w > 1e-5, val, 0.0)
    return fn


def make_ee_same_interp_fn(r_vals, w1_vals, w2_vals, ee_same_grid):
    def fn(x):
        r, w1, w2 = x[0], x[1], x[2]
        val = tricubic_interp_3d(r, jnp.log(w1), jnp.log(w2), r_vals, w1_vals, w2_vals, ee_same_grid)
        return jnp.where(w2 > 1e-5, val, 0.0)
    return fn


def make_ee_opp_interp_fn(r_vals, w1_vals, w2_vals, ee_opp_grid):
    def fn(x):
        r, w1, w2 = x[0], x[1], x[2]
        val = tricubic_interp_3d(r, jnp.log(w1), jnp.log(w2), r_vals, w1_vals, w2_vals, ee_opp_grid)
        return jnp.where(w2 > 1e-5, val, 0.0)
    return fn


def make_e_interp_fn(w_vals, e_grid):
    def fn(x):
        w = x[0]
        val = cubic_interp_1d(jnp.log(w), w_vals, e_grid)
        return jnp.where(w > 1e-5, val, 0.0)
    return fn


def get_interp_fns():
    Ek = jnp.array(loadmat('ek.mat', simplify_cells=True)['ek'])
    Ecv = jnp.array(loadmat('ecv.mat', simplify_cells=True)['ecv'])
    Euu = jnp.array(loadmat('euu.mat', simplify_cells=True)['euu'])
    Eud = jnp.array(loadmat('eud.mat', simplify_cells=True)['eud'])

    Ecv = jnp.concatenate([Ecv[:,1:2,:], Ecv], axis=1)
    Euu = jnp.concatenate([Euu[1:2,:,:], Euu], axis=0)
    Eud = jnp.concatenate([Eud[1:2,:,:], Eud], axis=0)

    Ek = YsOfYu(Ek)
    Ecv = YsOfYu(Ecv)
    Euu = YsOfYu(Euu)
    Eud = YsOfYu(Eud)

    rw = jnp.arange(-0.1,5.1,0.1)
    lw = jnp.linspace(jnp.log(0.1),jnp.log(10),100)
    lwiwj = jnp.linspace(jnp.log(0.5), jnp.log(5), 50)
    w = jnp.exp(lw)

    #rw = jnp.arange(-0.1,10.1,0.1)
    #wij = jnp.arange(0.1,10.1,0.1)
    #wiwj = jnp.arange(0.5,5.1,0.1)

    interp_ae_fn = make_ae_interp_fn(rw, lw, Ecv)
    interp_ee_same_fn = make_ee_same_interp_fn(rw, lw, lwiwj, Euu)
    interp_ee_opp_fn = make_ee_opp_interp_fn(rw, lw, lwiwj, Eud)
    interp_e_fn = make_e_interp_fn(lw, Ek)

    return interp_ae_fn, interp_ee_same_fn, interp_ee_opp_fn, interp_e_fn
