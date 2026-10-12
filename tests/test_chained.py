import numpy as np
import pytest

from pyequate import chained, equipercen, load_dataset


@pytest.fixture
def kb():
    kx, ky = load_dataset("KBneatx"), load_dataset("KBneaty")
    return kx.total, ky.total, kx.anchor, ky.anchor


def test_eq_runs_when_anchor_range_differs(kb):
    out = chained(*kb, 0, 36, 0, 12, type="eq")
    assert len(out) == 37
    e = out["e_yx"].to_numpy()
    assert np.all(np.isfinite(e))
    assert np.all(np.diff(e) >= -1e-12)
    assert e.min() >= -0.5 and e.max() <= 36.5


def test_eq_reduces_to_v_to_y_step_when_pop1_anchor_is_x(kb):
    #With V = X in pop 1, the X->V step is the identity wherever f(x) > 0, so the
    #chain must equal direct equipercentile equating of pop-2 anchor onto Y.
    #This pins the V->Y step, which used the wrong percentile before the fix.
    x, y, _, vy = kb
    w = vy * 3  #pop-2 anchor stretched onto the 0..36 scale
    out = chained(x, y, x, w, 0, 36, 0, 36, type="eq")
    direct = equipercen(w, y, 0, 36)
    has_mass = x.value_counts().reindex(range(37), fill_value=0).to_numpy() > 0
    np.testing.assert_allclose(out["e_yx"].to_numpy()[has_mass],
                               direct["equated"].to_numpy()[has_mass], atol=1e-10)


def test_eq_accepts_lists(kb):
    x, y, vx, vy = kb
    a = chained(list(x), list(y), list(vx), list(vy), 0, 36, 0, 12, type="eq")
    b = chained(x, y, vx, vy, 0, 36, 0, 12, type="eq")
    np.testing.assert_allclose(a["e_yx"], b["e_yx"])


def test_eq_float_scores(kb):
    x, y, vx, vy = (s.astype(float) for s in kb)
    out = chained(x, y, vx, vy, 0, 36, 0, 12, type="eq")
    np.testing.assert_allclose(out["e_yx"], chained(*kb, 0, 36, 0, 12, type="eq")["e_yx"])


def _prank_knots(freq):
    #K&B Eq. 2.14: percentile rank is piecewise linear with knots at score - 0.5
    pts = freq.index.to_numpy(dtype=float)
    knots = np.append(pts - 0.5, pts[-1] + 0.5)
    vals = np.append(0.0, (freq / freq.sum()).cumsum().to_numpy())
    return knots, vals


def test_eq_matches_continuous_kolen_brennan_chain(kb):
    #e_Y(x) = G2^-1( P2( e_V1(x) ) ), every step via the continuous percentile-rank function
    x, y, vx, vy = kb
    fr = lambda s, hi: s.value_counts().reindex(range(hi + 1), fill_value=0).sort_index()
    kx_, px_ = _prank_knots(fr(x, 36))
    kv1, pv1 = _prank_knots(fr(vx, 12))
    kv2, pv2 = _prank_knots(fr(vy, 12))
    ky_, py_ = _prank_knots(fr(y, 36))
    pts = np.arange(37.0)
    e_v = np.interp(np.interp(pts, kx_, px_), pv1, kv1)
    ref = np.interp(np.interp(e_v, kv2, pv2), py_, ky_)
    out = chained(*kb, 0, 36, 0, 12, type="eq")["e_yx"].to_numpy()
    interior = (fr(x, 36).to_numpy() > 0) & (pts >= 3) & (pts <= 35)
    np.testing.assert_allclose(out[interior], ref[interior], atol=1e-8)


def test_eq_strictly_increasing_where_x_has_mass(kb):
    out = chained(*kb, 0, 36, 0, 12, type="eq")["e_yx"].to_numpy()
    has_mass = kb[0].value_counts().reindex(range(37), fill_value=0).to_numpy() > 0
    assert np.all(np.diff(out[has_mass]) > 0)


def test_eq_out_of_range_score_raises(kb):
    #KBneatx totals run 2..36
    with pytest.raises(ValueError, match="outside"):
        chained(*kb, 5, 36, 0, 12, type="eq")
