import numpy as np
import pytest

from pyequate import linear_se


@pytest.fixture
def two_forms():
    a = np.random.default_rng(0).normal(20, 4, 3000)
    b = np.random.default_rng(1).normal(22, 6, 3000)
    return a, b


def test_linear_se_matches_delta_method(two_forms):
    a, b = two_forms
    x = np.array([10.0, 20.0, 30.0])
    sx, sy = a.std(ddof=1), b.std(ddof=1)
    nx, ny = len(a), len(b)
    z = (x - a.mean()) / sx
    expected = np.sqrt((sy / sx) ** 2 * sx ** 2 / nx + sy ** 2 / ny
                       + z ** 2 * sy ** 2 / (2 * (nx - 1))
                       + z ** 2 * sy ** 2 / (2 * (ny - 1)))
    got = linear_se(a, b, x).SEE.to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_linear_se_matches_kolen_brennan_normal_formula(two_forms):
    #K&B (2004): Var = sigma_Y^2/2 * (1/N_X + 1/N_Y) * (2 + z^2)
    a, b = two_forms
    z = (20 - a.mean()) / a.std(ddof=1)
    kb = np.sqrt(b.var(ddof=1) / 2 * (1 / len(a) + 1 / len(b)) * (2 + z ** 2))
    assert linear_se(a, b, [20]).SEE[0] == pytest.approx(kb, rel=1e-3)
