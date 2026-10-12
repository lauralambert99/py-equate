import numpy as np
import pandas as pd
import pytest

from pyequate import joint_distribution, load_dataset


@pytest.fixture
def kbx():
    return load_dataset("KBneatx")


def test_probabilities_match_brute_force_counts(kbx):
    g = joint_distribution(kbx, "total", "anchor", range(37), range(13))
    counts = kbx.groupby(["total", "anchor"]).size()
    for (x, v), n in counts.items():
        assert g.at[x, v] == pytest.approx(n / len(kbx))
    assert g["Marginal"].sum() == pytest.approx(1.0)
    assert list(g.columns[-2:]) == ["Marginal", "Cumulative"]


def test_out_of_range_score_raises(kbx):
    #KBneatx totals run 2..36, so range(5, 37) excludes real scores
    with pytest.raises(ValueError, match="outside form_range"):
        joint_distribution(kbx, "total", "anchor", range(5, 37), range(13))


def test_float_columns_match_int_columns(kbx):
    as_float = kbx.astype({"total": float, "anchor": float})
    g_int = joint_distribution(kbx, "total", "anchor", range(37), range(13))
    g_flt = joint_distribution(as_float, "total", "anchor", range(37), range(13))
    np.testing.assert_allclose(g_flt.to_numpy(), g_int.to_numpy())


def test_data_derived_range(kbx):
    g = joint_distribution(kbx, "total", "anchor")
    assert g.index.min() == 2 and g.index.max() == 36
    assert not g.isna().any().any()
