"""Finite-element model of galvanic corrosion under an electrolyte layer.

Geometry (2D cross-section): a rectangle of electrolyte, width W and
thickness h. The bottom edge is metal: zinc coating for x < 0 and bare steel
for x >= 0 (a scratch or cut edge in galvanized steel). Other edges insulate.

Physics:
    electrolyte:   div(sigma * grad(phi)) = 0
    metal surface: current density i(E) leaves the metal, E = phi_metal - phi,
                   with the metal at phi_metal = 0 (zinc and steel are connected)
    zinc:  anodic Tafel     i =  i0_zn * 10**((E - E_zn) / b_zn)
    steel: cathodic Tafel   i = -i0_st * 10**(-(E - E_st) / b_st)   (oxygen reduction)

Solved with linear triangular elements and a damped Newton method.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

LN10 = np.log(10.0)


@dataclass
class Params:
    """Illustrative values in the range reported for zinc and steel in ~3.5% NaCl.

    They are not fitted to a specific experiment; change them for your system.
    Potentials in V vs SCE, current densities in A/m^2, conductivity in S/m.
    """
    sigma: float = 5.0          # 3.5% NaCl
    E_zn: float = -1.03         # zinc corrosion potential
    i0_zn: float = 0.1          # zinc exchange-like current density at E_zn
    b_zn: float = 0.04          # zinc anodic Tafel slope, V/decade
    E_st: float = -0.60         # steel (oxygen reduction) reference potential
    i0_st: float = 0.1
    b_st: float = 0.12          # cathodic Tafel slope, V/decade


def i_zinc(E, p):
    return p.i0_zn * 10 ** ((E - p.E_zn) / p.b_zn)


def i_steel(E, p):
    return -p.i0_st * 10 ** (-(E - p.E_st) / p.b_st)


@dataclass
class Solution:
    x: np.ndarray               # bottom-node positions (m)
    E: np.ndarray               # electrode potential along the surface (V vs SCE)
    i: np.ndarray               # current density along the surface (A/m^2), + = metal dissolving
    phi: np.ndarray             # electrolyte potential at every node
    nodes: np.ndarray
    iterations: int
    residual: float
    width: float
    thickness: float
    info: dict = field(default_factory=dict)


def mesh(width, thickness, nx, ny):
    """Structured mesh of linear triangles, refined toward the zinc/steel junction."""
    s = np.linspace(-1, 1, nx)
    xs = 0.5 * width * np.sign(s) * np.abs(s) ** 1.5           # finer near x = 0
    ys = thickness * np.linspace(0, 1, ny) ** 1.3              # finer near the metal
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    tri = []
    for j in range(ny - 1):
        for k in range(nx - 1):
            a = j * nx + k
            b, c, d = a + 1, a + nx, a + nx + 1
            tri += [(a, b, d), (a, d, c)]
    return nodes, np.array(tri), np.arange(nx)                 # bottom nodes are the first row


def stiffness(nodes, tri, sigma):
    p = nodes[tri]                                             # (nt, 3, 2)
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    # gradients of the three linear shape functions
    b = np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 1] - p[:, 0, 1], p[:, 0, 1] - p[:, 1, 1]], 1)
    c = np.stack([p[:, 2, 0] - p[:, 1, 0], p[:, 0, 0] - p[:, 2, 0], p[:, 1, 0] - p[:, 0, 0]], 1)
    ke = sigma * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area[:, None, None])
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, 3).ravel()
    n = len(nodes)
    return coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def mixed_potential(p, width):
    """Hand calculation: the potential where total zinc and steel currents balance (uniform potential)."""
    half = width / 2
    return brentq(lambda E: i_zinc(E, p) * half + i_steel(E, p) * half, -1.5, 0.0)


def solve(thickness, width=0.06, p=Params(), nx=301, ny=25, tol=1e-9, max_iter=100):
    nodes, tri, bottom = mesh(width, thickness, nx, ny)
    K = stiffness(nodes, tri, p.sigma)
    xb = nodes[bottom, 0]
    # lumped boundary weights (length of surface each bottom node represents)
    w = np.zeros_like(xb)
    seg = np.diff(xb)
    w[:-1] += seg / 2
    w[1:] += seg / 2
    zinc = xb < 0

    def surface(E):
        i = np.where(zinc, i_zinc(E, p), i_steel(E, p))
        di = np.where(zinc, LN10 / p.b_zn, LN10 / p.b_st) * np.abs(i)    # dI/dE > 0 for both
        return i, di

    def residual(phi):
        i, di = surface(-phi[bottom])
        F = np.zeros(len(nodes))
        F[bottom] = i * w
        R = K @ phi - F
        return R, di, np.linalg.norm(R) / max(np.linalg.norm(F), 1e-30)

    phi = np.full(len(nodes), -mixed_potential(p, width))      # start from the uniform-potential answer
    R, di, res = residual(phi)
    converged_by = None
    for it in range(1, max_iter + 1):
        if res < tol:
            converged_by = "residual"
            break
        J = (K + diags(np.bincount(bottom, weights=di * w, minlength=len(nodes)))).tocsc()
        step = spsolve(J, -R)
        step *= min(1.0, 0.2 / max(np.abs(step).max(), 1e-30))    # limit to 200 mV per iteration
        # Backtracking line search: halve the step until the residual drops.
        for _ in range(30):
            R_new, di_new, res_new = residual(phi + step)
            if res_new < res:
                break
            step *= 0.5
        phi += step
        R, di, res = R_new, di_new, res_new
        if np.abs(step).max() < 1e-9:      # potential no longer changing: round-off floor reached
            converged_by = "step size"
            break
    E = -phi[bottom]
    i, _ = surface(E)
    return Solution(x=xb, E=E, i=i, phi=phi, nodes=nodes, iterations=it, residual=float(res),
                    width=width, thickness=thickness, info={"weights": w, "converged_by": converged_by})


def protection_distance(sol, p=Params(), shift=0.05):
    """Length of steel (from the junction) held at least `shift` volts below the steel's own potential."""
    steel = sol.x >= 0
    protected = sol.E[steel] <= p.E_st - shift
    xs = sol.x[steel]
    if not protected.any():
        return 0.0
    if protected.all():
        return float(xs.max())
    return float(xs[np.argmin(protected)])
