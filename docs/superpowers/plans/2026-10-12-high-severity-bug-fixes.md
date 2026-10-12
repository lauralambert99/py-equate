# High-Severity Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 11 confirmed high-severity bugs from the 2026-10-12 package review, in order of severity, each guarded by a pytest test that fails before the fix and passes after.

**Architecture:** One task per bug (or per tightly-coupled bug group), ordered from "silently wrong numbers" to "loud crash" to "edge case". A small shared helper, `helper_fxns/equip_map.py`, replaces the duplicated (and buggy) Y\*-lookup code in `fe` and `chained`. A new `tests/` directory (pytest) is created; there is no existing test suite.

**Tech Stack:** Python ≥3.9, numpy, pandas, scipy, statsmodels, pytest (new, dev-only).

**Spec:** The review report in this conversation ("Review: pyequate 0.1.2.3", Bugs section, confirmed-by-running items).

## Global Constraints

- `requires-python = ">=3.9"` — no syntax newer than 3.9 without `from __future__ import annotations`.
- Do not change public function signatures or return column names (other API inconsistencies are out of scope).
- House style: relative intra-package imports (`from ..x.y import z`), `#Comment` with no space, NumPy-style docstrings in newer modules.
- Kolen & Brennan (2004) convention: X = new form, Y = old form; equating produces e_Y(x).
- Work on branch `fix/high-severity-bugs`, never on `main`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions requiring approval (defaults used below)

1. **`joint_distribution` with scores outside a user-supplied range** — APPROVED default: **raise `ValueError`** naming the offending scores. (Alternative: silently expand the table with zero-filled rows/columns, which keeps today's "expand" intent but then misaligns with `score_min..score_max` in `fe`/`bh`.)
2. **`linear(type="zscore")`** — APPROVED: **remove the option**; `type="zscore"` now raises `ValueError` (z-score equating is linear equating, so `type="linear"` covers it).
3. **`linear(rescale=True)`** — APPROVED default: rescale x, y **and the score points** with the *possible* range `score_min..score_max` (same transform for both forms), so `ex` is reported on the 0–100 scale; the `Score` column stays as raw score points. (Today's code uses each form's observed min/max, which changes the equating relationship itself.)

## Review Focus

1. Score columns read as float (e.g., CSV with NaNs) — `joint_distribution` and `chained` must give the same results as int columns. → test in Task 2.
2. Score points with zero frequency at the bottom or top of either distribution — `equip_map`, `fe`, `chained` must return finite numbers, never NaN/inf. → tests in Tasks 3 and 4.
3. Interior equated scores from `fe` must not change (the old code was right there). → regression test in Task 3.
4. Plain Python lists passed to `chained` (docstring says "Array") — must work. → test in Task 4.
5. A fresh `import pyequate` must still succeed after adding `helper_fxns/equip_map.py` (helper_fxns ↔ random_groups import cycle risk). → import test in Task 3.

## Out of scope (lower severity; separate plan)

`equi_se` loop indentation (O(n²) but correct), `bh` score-range alignment guard, 2PL `c` inconsistency, unused imports, `load_dataset` `Unnamed: 0` column, column-name/docstring/old-vs-new-form consistency fixes.

---

### Task 0: Branch and test infrastructure

**Files:**
- Modify: `pyproject.toml` (append)
- Create: `tests/test_import.py`

- [ ] **Step 1: Create branch and install pytest**

```bash
git checkout -b fix/high-severity-bugs
python -m pip install pytest
```

- [ ] **Step 2: Add pytest config so tests import from `src/` without installing**

Append to `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["pytest"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Write a smoke test**

`tests/test_import.py`:

```python
import importlib
import sys


def test_fresh_import_succeeds():
    for name in [m for m in sys.modules if m.startswith("pyequate")]:
        del sys.modules[name]
    pq = importlib.import_module("pyequate")
    for name in pq.__all__:
        assert hasattr(pq, name), name
```

- [ ] **Step 4: Run** `python -m pytest -q` — Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_import.py
git commit -m "test: add pytest setup and import smoke test"
```

---

### Task 1: `linear_se` — wrong derivative w.r.t. μX (silently wrong SEs)

**Files:**
- Modify: `src/pyequate/se_equating/linear_se.py:52`
- Test: `tests/test_linear_se.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_linear_se.py -v` — Expected: both FAIL (got ≈0.119 vs ≈0.154 at x=20).

- [ ] **Step 3: Fix** — in `linear_se.py` line 52 replace

```python
        d_mu_x = -sd_x / sd_y
```
with
```python
        d_mu_x = -sd_y / sd_x
```

- [ ] **Step 4: Run** `python -m pytest tests/test_linear_se.py -v` — Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/se_equating/linear_se.py tests/test_linear_se.py
git commit -m "fix(linear_se): correct derivative of l(x) with respect to mu_X"
```

---

### Task 2: `joint_distribution` — out-of-range score turns whole table into NaN

**Files:**
- Modify: `src/pyequate/freq_tab/joint_distribution.py:32-57` (replace loop with `pd.crosstab`)
- Test: `tests/test_joint_distribution.py`

**Interfaces:** Produces the same DataFrame shape as before: rows = form scores, columns = common scores then `'Marginal'`, `'Cumulative'`; no index/column names.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_joint_distribution.py -v` — Expected: `test_out_of_range_score_raises` FAILS (no error raised; table is NaN). Others may pass — they guard the rewrite.

- [ ] **Step 3: Replace the body of `joint_distribution`** (keep the docstring; add a `Raises:` line):

```python
    form_scores = df[form_col]
    common_scores = df[common_col]

    #Determine score ranges from data if not provided
    if form_range is None:
        form_range = range(int(form_scores.min()), int(form_scores.max()) + 1)

    if common_range is None:
        common_range = range(int(common_scores.min()), int(common_scores.max()) + 1)

    #Refuse scores outside the requested ranges rather than silently reshaping the table
    out_x = sorted(set(form_scores.dropna()) - set(form_range))
    out_v = sorted(set(common_scores.dropna()) - set(common_range))
    if out_x:
        raise ValueError(f"Scores in '{form_col}' outside form_range: {out_x}")
    if out_v:
        raise ValueError(f"Scores in '{common_col}' outside common_range: {out_v}")

    #Joint frequency table, including score points nobody obtained
    joint = pd.crosstab(form_scores, common_scores).reindex(
        index=list(form_range), columns=list(common_range), fill_value=0)
    joint.index.name = None
    joint.columns.name = None

    #Normalize to probabilities
    total = joint.to_numpy().sum()
    joint_prob = joint / total

    #Marginal and cumulative distributions
    joint_prob['Marginal'] = joint_prob.sum(axis = 1)
    joint_prob['Cumulative'] = joint_prob['Marginal'].cumsum()

    return joint_prob
```

Add to the docstring after `Returns:`:

```
    Raises:
        ValueError: if any score falls outside form_range or common_range
```

Remove the now-unused `import numpy as np` only if nothing else uses it (nothing does).

- [ ] **Step 4: Run** `python -m pytest tests/ -v` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/freq_tab/joint_distribution.py tests/test_joint_distribution.py
git commit -m "fix(joint_distribution): raise on out-of-range scores instead of producing NaN table"
```

---

### Task 3: Shared `equip_map` helper + `fe` lowest-bin NaN and missing clamp

**Files:**
- Create: `src/pyequate/helper_fxns/equip_map.py`
- Modify: `src/pyequate/neat_methods/fe.py:96-140` (replace Y\* lookup with `equip_map`)
- Test: `tests/test_equip_map.py`, `tests/test_fe.py`

**Interfaces:**
- Produces: `equip_map(from_freq: pd.Series, to_freq: pd.Series) -> np.ndarray`. Both Series are frequencies or probabilities indexed by ascending score points (integers). Returns one equated score per entry of `from_freq`, on `to_freq`'s score scale. Not added to any `__all__` (internal helper). Task 4 consumes it.

- [ ] **Step 1: Write the failing tests**

`tests/test_equip_map.py`:

```python
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
```

`tests/test_fe.py`:

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_equip_map.py tests/test_fe.py -v` — Expected: `test_equip_map.py` fails with `ModuleNotFoundError`; `test_lowest_score_is_not_nan`, `test_identity_when_populations_identical` and `test_score_range_mismatch_raises` FAIL; `test_interior_values_unchanged` passes (regression guard).

- [ ] **Step 3: Create `src/pyequate/helper_fxns/equip_map.py`**

```python
# -*- coding: utf-8 -*-
"""
Shared equipercentile mapping used by fe() and chained().

@author: laycocla
"""

import numpy as np


def equip_map(from_freq, to_freq):
    """
    Equipercentile map from one score scale to another (Kolen & Brennan, 2004, Eq. 2.17).

    Parameters
    ----------
    from_freq : pd.Series
        Frequencies (or probabilities) of the form being equated, indexed by
        ascending score points.
    to_freq : pd.Series
        Frequencies (or probabilities) of the target form, indexed by
        ascending score points. May use a different score range than `from_freq`.

    Returns
    -------
    np.ndarray
        Equated score on the `to_freq` scale for each score point of `from_freq`.
    """
    f = from_freq / from_freq.sum()
    g = to_freq / to_freq.sum()

    #Percentile ranks (as proportions) of the from-scores, and target cumulative distribution
    P = (f.cumsum().shift(1, fill_value=0) + f / 2).to_numpy(dtype=float)
    G = g.cumsum().to_numpy(dtype=float)
    to_scores = to_freq.index.to_numpy(dtype=float)

    out = np.empty(len(P))
    for i, p in enumerate(P):
        #Smallest target score whose cumulative proportion reaches p (Y*_U)
        k = min(int(np.searchsorted(G, p, side="left")), len(G) - 1)
        Ghi = G[k]
        Glo = G[k - 1] if k > 0 else 0.0
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            out[i] = to_scores[k]
        else:
            #Linear interpolation within the target bin, with 0.5 continuity correction
            out[i] = (to_scores[k] - 0.5) + (p - Glo) / (Ghi - Glo)
    return out
```

- [ ] **Step 4: Rewrite the tail of `fe` (lines 96–140)**

Add the import at the top of `fe.py` next to the other package imports:

```python
from ..helper_fxns.equip_map import equip_map
```

Replace everything from `  #Cumulative synthetic distributions` to the end of the function with:

```python
    if len(fsx) != len(scores) or len(gsy) != len(scores):
        raise ValueError("Joint distributions must cover exactly score_min..score_max; "
                         "pass form_range=range(score_min, score_max + 1) to joint_distribution().")

  #Equipercentile equating of the synthetic distributions
    fsx = pd.Series(fsx.to_numpy(dtype=float), index=scores)
    gsy = pd.Series(gsy.to_numpy(dtype=float), index=scores)
    e_yx = equip_map(fsx, gsy)

    return pd.DataFrame({'Score': scores, 'e_yx': e_yx})
```

- [ ] **Step 5: Run** `python -m pytest tests/ -v` — Expected: all pass, including `test_import.py` (checks no import cycle).

- [ ] **Step 6: Commit**

```bash
git add src/pyequate/helper_fxns/equip_map.py src/pyequate/neat_methods/fe.py tests/test_equip_map.py tests/test_fe.py
git commit -m "fix(fe): no NaN at lowest score; share equipercentile mapping via equip_map"
```

---

### Task 4: `chained(type="eq")` — crash, wrong percentile in V→Y step, wrong lag bound

**Files:**
- Modify: `src/pyequate/random_groups/chained.py:57-167` (replace the whole `elif type == "eq":` branch)
- Test: `tests/test_chained.py`

**Interfaces:**
- Consumes: `equip_map(from_freq, to_freq) -> np.ndarray` from Task 3.
- Produces: unchanged return shape — DataFrame with `'Score'`, `'e_yx'`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_chained.py -v` — Expected: FAIL with `ValueError: All arrays must be of the same length` (or wrong values for the whole-test case).

- [ ] **Step 3: Replace the `elif type == "eq":` branch**

Add next to the existing imports in `chained.py`:

```python
from ..helper_fxns.equip_map import equip_map
```

Replace lines 57–167 (the entire `elif type == "eq":` block, up to but not including `else:`) with:

```python
    elif type == "eq":
        x, y = pd.Series(x), pd.Series(y)
        common_x, common_y = pd.Series(common_x), pd.Series(common_y)

        score_pts = range(score_min, score_max + 1)
        anchor_pts = range(anchor_min, anchor_max + 1)

        #Frequency tables, keeping score points nobody obtained
        f_x = x.value_counts().reindex(score_pts, fill_value=0).sort_index()
        h1_v = common_x.value_counts().reindex(anchor_pts, fill_value=0).sort_index()
        h2_v = common_y.value_counts().reindex(anchor_pts, fill_value=0).sort_index()
        g_y = y.value_counts().reindex(score_pts, fill_value=0).sort_index()

        #Form X to common items in Population 1: e_V1(x)
        e_vx = equip_map(f_x, h1_v)

        #Common items to Form Y in Population 2: e_Y2(v)
        e_yv = equip_map(h2_v, g_y)

        #Chain them: e_Y(x) = e_Y2(e_V1(x)); e_V1(x) is non-integer, so interpolate e_Y2
        e_yx = np.interp(e_vx, np.asarray(anchor_pts, dtype=float), e_yv)

        return pd.DataFrame({'Score': scores,
                             'e_yx': e_yx})

```

- [ ] **Step 4: Run** `python -m pytest tests/ -v` — Expected: all pass. Spot-check: `chained(*kb, 0, 36, 0, 12, type="eq")` at x = 10, 20, 30 ≈ 10.59, 20.66, 30.99 (prototype values).

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/random_groups/chained.py tests/test_chained.py
git commit -m "fix(chained): rewrite equipercentile chain; fixes length crash, wrong V->Y percentile, lag bound"
```

---

### Task 5: `equipercentile_irt` — 0.5 too high at the lowest score

**Files:**
- Modify: `src/pyequate/irt/irt_helper.py:221`
- Test: `tests/test_irt.py` (created here; Task 6 appends to it)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_irt.py -v` — Expected: FAIL; first element is 0.5 instead of 0.

- [ ] **Step 3: Fix** — in `irt_helper.py` line 221 replace

```python
                eyx[i] = px / Gy[0] if Gy[0] > 0 else 0.0
```
with
```python
                eyx[i] = -0.5 + px / Gy[0] if Gy[0] > 0 else 0.0
```

- [ ] **Step 4: Run** `python -m pytest tests/test_irt.py -v` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/irt/irt_helper.py tests/test_irt.py
git commit -m "fix(irt): add missing -0.5 continuity term in lowest bin of equipercentile_irt"
```

---

### Task 6: `irtTS` — default score range uses Form Y's length

**Files:**
- Modify: `src/pyequate/irt/irtTS.py:56`
- Test: `tests/test_irt.py` (append)

- [ ] **Step 1: Append the failing test to `tests/test_irt.py`**

```python
def test_irtTS_default_range_uses_form_x(params):
    px, py = params
    out = irtTS(px, py.iloc[:30], model="3pl")  #X has 36 items, Y has 30
    assert len(out) == 37
    np.testing.assert_array_equal(out["X"], np.arange(37))
```

- [ ] **Step 2: Run** `python -m pytest tests/test_irt.py::test_irtTS_default_range_uses_form_x -v` — Expected: FAIL (31 rows).

- [ ] **Step 3: Fix** — in `irtTS.py` line 56 replace

```python
        score_max = len(formY_params)
```
with
```python
        score_max = len(formX_params)
```

- [ ] **Step 4: Run** `python -m pytest tests/ -v` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/irt/irtTS.py tests/test_irt.py
git commit -m "fix(irtTS): default score range spans Form X number-correct scores"
```

---

### Task 7: `linear` — remove broken `type="zscore"`; `rescale=True` mixes scales

**Files:**
- Modify: `src/pyequate/random_groups/mean_linear.py:12-81` (docstring + body)
- Test: `tests/test_mean_linear.py`

Implements Decisions 2 and 3. Note: removing `zscore` is a breaking API change for users of the PyPI release — mention it in release notes.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_mean_linear.py -v` — Expected: all FAIL.

- [ ] **Step 3: Replace the body of `linear`** (from the `#TODO: potential errors` comment through `return eq`):

```python
    #This should be the only score consideration, right?
    if group == "Single":
        #Make sure scores are paired for single group design
        if len(x) != len(y):
            raise ValueError("X and Y must have the same length for single-group equating (paired scores).")

    #Define scores
    scores = np.arange(score_min, score_max + 1)
    eq_points = scores.astype(float)

    if rescale:
        #Same transform for both forms and the score points, based on the possible score range
        span = score_max - score_min
        x = (np.asarray(x, dtype=float) - score_min) / span * 100
        y = (np.asarray(y, dtype=float) - score_min) / span * 100
        eq_points = (eq_points - score_min) / span * 100

    #Compute the means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if type == "mean":
        #Mean equating: align the means of x and y
        intercept = mean_y - mean_x
        slope = 1  #No change in slope with mean equating, just intercepts
    elif type == "linear":
        #Linear equating: align both the means and standard deviations
        sd_x = np.std(x)  #Standard deviation of x
        sd_y = np.std(y)  #Standard deviation of y
        slope = sd_y / sd_x  #Ratio of standard deviations
        intercept = mean_y - slope * mean_x
    else:
        raise ValueError("Invalid 'type'. Use 'mean' or 'linear'.")

    ex = slope * eq_points + intercept

    eq =  pd.DataFrame({'Score': scores,
                        'ex': ex})
    #Return equated scores as a dataframe
    return eq
```

Update the docstring entries:

```
    type : str, optional
        Type of equating. "mean" (mean equating) or "linear" (linear equating).
        Default is "linear". (z-score equating, setting z_X(x) = z_Y(y), is linear equating.)
    rescale : bool, optional
        If True, x, y and the score points are rescaled to 0-100 using the possible
        score range (score_min..score_max), and 'ex' is reported on that 0-100 scale.
        'Score' stays on the raw scale. Default is False.
```

Remove the unused `import itertools`.

- [ ] **Step 4: Run** `python -m pytest tests/ -v` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/random_groups/mean_linear.py tests/test_mean_linear.py
git commit -m "fix(linear): remove broken zscore option; rescale applies one transform to forms and score points"
```

---

### Task 8: `common_item_marginal(exclude_last_n=0)` blanks every column

**Files:**
- Modify: `src/pyequate/freq_tab/common_item_marginal.py:31`
- Test: `tests/test_common_item_marginal.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `python -m pytest tests/test_common_item_marginal.py -v` — Expected: `test_exclude_zero_keeps_all_columns` FAILS.

- [ ] **Step 3: Fix** — replace lines 30–32:

```python
    # Fill the excluded columns with NaNs to maintain shape
    for col in joint_probs.columns[-exclude_last_n:]:
        marginal_row[col] = pd.NA
```
with
```python
    # Fill the excluded columns with NaNs to maintain shape
    excluded_cols = joint_probs.columns[-exclude_last_n:] if exclude_last_n > 0 else []
    for col in excluded_cols:
        marginal_row[col] = pd.NA
```

- [ ] **Step 4: Run** `python -m pytest tests/ -v` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyequate/freq_tab/common_item_marginal.py tests/test_common_item_marginal.py
git commit -m "fix(common_item_marginal): exclude_last_n=0 no longer blanks every column"
```

---

### Task 9: Final verification

- [ ] **Step 1:** `python -m pytest -v` — Expected: all tests pass, 0 failures.
- [ ] **Step 2:** Re-run the review probe script (`scratchpad/probe.py`) and confirm each previously failing line now reports the corrected behavior.
- [ ] **Step 3:** `git log --oneline main..HEAD` — Expected: 9 commits, one per task.
- [ ] **Step 4:** Use superpowers:finishing-a-development-branch to decide merge / PR.
