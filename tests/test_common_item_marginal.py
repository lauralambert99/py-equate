import numpy as np
import pytest

from pyequate import common_item_marginal, joint_distribution, load_dataset


@pytest.fixture
def g():
    return joint_distribution(load_dataset("KBneatx"), "total", "anchor", range(37), range(13))


def test_exclude_zero_keeps_all_columns(g):
    probs = g.iloc[:, :-2]
    out = common_item_marginal(probs, exclude_last_n=0)
    row = out.loc["Marginal_Common"]
    assert row.notna().all()
    np.testing.assert_allclose(row.to_numpy(dtype=float), probs.sum(axis=0).to_numpy())


def test_default_excludes_marginal_and_cumulative(g):
    row = common_item_marginal(g).loc["Marginal_Common"]
    assert row[["Marginal", "Cumulative"]].isna().all()
    assert row.iloc[:-2].notna().all()
    assert row.iloc[:-2].astype(float).sum() == pytest.approx(1.0)
