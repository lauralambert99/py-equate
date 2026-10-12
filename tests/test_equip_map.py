import numpy as np
import pandas as pd

from pyequate import equipercen, load_dataset
from pyequate.helper_fxns.equip_map import equip_map


def _freq(s, lo, hi):
    return s.value_counts().reindex(range(lo, hi + 1), fill_value=0).sort_index()


def test_identity_when_distributions_equal():
    f = pd.Series([2.0, 3.0, 5.0], index=[0, 1, 2])
    np.testing.assert_allclose(equip_map(f, f), [0.0, 1.0, 2.0])


def test_matches_equipercen_on_same_scale():
    kx, ky = load_dataset("KBneatx"), load_dataset("KBneaty")
    got = equip_map(_freq(kx.total, 0, 36), _freq(ky.total, 0, 36))
    np.testing.assert_allclose(got, equipercen(kx.total, ky.total, 0, 36).equated)


def test_different_scales_and_zero_frequency_ends():
    from_f = pd.Series([0, 4, 6, 0], index=[0, 1, 2, 3])
    to_f = pd.Series([0, 1, 1, 1, 1, 0], index=[10, 11, 12, 13, 14, 15])
    out = equip_map(from_f, to_f)
    assert np.all(np.isfinite(out))
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 9.5 and out.max() <= 15.5
