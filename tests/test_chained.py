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
