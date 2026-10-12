import numpy as np
import pandas as pd
import pytest

from pyequate import fe, joint_distribution, load_dataset


@pytest.fixture
def kb_joint():
    kx, ky = load_dataset("KBneatx"), load_dataset("KBneaty")
    gx = joint_distribution(kx, "total", "anchor", range(37), range(13))
    gy = joint_distribution(ky, "total", "anchor", range(37), range(13))
    return gx, gy


def _synthetic(seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 5, 500)
    t = np.clip(v + rng.integers(0, 6, 500), 0, 9)
    return pd.DataFrame({"t": t, "v": v})


def test_lowest_score_is_not_nan():
    gx = joint_distribution(_synthetic(3), "t", "v", range(10), range(5))
    gy = joint_distribution(_synthetic(4), "t", "v", range(10), range(5))
    out = fe(gx, gy, 0, 9, 0.5)
    assert out["e_yx"].notna().all()
    assert np.isfinite(out["e_yx"].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("w1", [0.0, 0.5, 1.0])
def test_identity_when_populations_identical(w1):
    g = joint_distribution(_synthetic(3), "t", "v", range(10), range(5))
    out = fe(g, g.copy(), 0, 9, w1)
    np.testing.assert_allclose(out["e_yx"], np.arange(10), atol=1e-10)


def test_interior_values_unchanged(kb_joint):
    #Values produced by the pre-fix implementation, which was correct away from the lowest bin
    out = fe(*kb_joint, 0, 36, 1.0)
    np.testing.assert_allclose(out["e_yx"].iloc[10:13], [10.8203, 11.9125, 13.2203], atol=1e-4)


def test_score_range_mismatch_raises(kb_joint):
    with pytest.raises(ValueError, match="score_min"):
        fe(*kb_joint, 0, 40, 0.5)
