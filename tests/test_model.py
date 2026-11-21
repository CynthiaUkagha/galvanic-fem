import numpy as np
import pytest

from galvanicfem import Params, mixed_potential, protection_distance, solve


def _total_anodic(sol):
    w = sol.info["weights"]
    return (np.clip(sol.i, 0, None) * w).sum()


@pytest.mark.parametrize("h", [1e-5, 1e-4, 1e-3])
def test_charge_balance(h):
    sol = solve(h)
    w = sol.info["weights"]
    assert sol.info["converged_by"] is not None
    assert abs((sol.i * w).sum()) / _total_anodic(sol) < 1e-6


def test_matches_mixed_potential_hand_calculation_when_electrolyte_is_very_conductive():
    p = Params(sigma=1e6)
    sol = solve(1e-3, p=p)
    E_mix = mixed_potential(p, sol.width)
    assert np.max(np.abs(sol.E - E_mix)) < 1e-3          # within 1 mV everywhere


def test_mesh_convergence():
    coarse = solve(1e-4, nx=151, ny=13)
    fine = solve(1e-4, nx=301, ny=25)
    finer = solve(1e-4, nx=601, ny=49)
    i1, i2, i3 = (_total_anodic(s) for s in (coarse, fine, finer))
    assert abs(i3 - i2) / i3 < 0.01                       # default mesh within 1% of a 2x finer one
    assert abs(i3 - i2) < abs(i2 - i1)                    # and the error shrinks with refinement
    assert abs(protection_distance(finer) - protection_distance(fine)) < 0.5e-3


def test_thinner_film_protects_less():
    d = [protection_distance(solve(h)) for h in (1e-5, 3e-5, 1e-4, 3e-4)]
    assert all(a <= b for a, b in zip(d, d[1:]))
    assert d[0] < 0.5 * d[-1]


def test_zinc_dissolves_and_steel_is_cathodic():
    sol = solve(1e-4)
    assert (sol.i[sol.x < 0] > 0).all() and (sol.i[sol.x >= 0] < 0).all()
