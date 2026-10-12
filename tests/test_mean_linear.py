import numpy as np
import pytest

from pyequate import linear, load_dataset


@pytest.fixture
def xy():
    kx, ky = load_dataset("KBneatx"), load_dataset("KBneaty")
    return kx.total, ky.total


def test_zscore_option_removed(xy):
    with pytest.raises(ValueError, match="Use 'mean' or 'linear'"):
        linear(*xy, 0, 36, type="zscore")


@pytest.mark.parametrize("kind", ["mean", "linear"])
def test_rescale_is_linear_transform_of_raw_result(xy, kind):
    raw = linear(*xy, 0, 36, type=kind)
    res = linear(*xy, 0, 36, type=kind, rescale=True)
    np.testing.assert_allclose(res["ex"], (raw["ex"] - 0) / 36 * 100)
    np.testing.assert_array_equal(res["Score"], raw["Score"])
