import numpy as np
import pytest

from pyequate import irtOS, irtTS, load_dataset
from pyequate.irt.irt_helper import equipercentile_irt


@pytest.fixture
def params():
    px = load_dataset("itemparx").rename(columns={"g": "c"})
    py = load_dataset("itempary").rename(columns={"g": "c"})
    return px, py


@pytest.mark.parametrize("pmf", [[0.5, 0.5], [0.2, 0.3, 0.5], [0.1, 0.1, 0.3, 0.5]])
def test_equipercentile_irt_identity(pmf):
    np.testing.assert_allclose(equipercentile_irt(pmf, pmf), np.arange(len(pmf)), atol=1e-12)


def test_irtOS_identity_same_form(params):
    px, _ = params
    out = irtOS(px, px, theta_points=40, model="3pl")
    np.testing.assert_allclose(out["eyx"], out["Scale"], atol=1e-8)
