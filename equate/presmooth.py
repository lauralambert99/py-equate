# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025

@author: laycocla
"""

import numpy as np
import pandas as pd
import statsmodels.tools as smt
import statsmodels.api as sm
import warnings

from typing import Literal
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from scipy.optimize import minimize
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Single Poisson GLM fit function
# ---------------------------------------------------------------------------
def _fit_glm(design: pd.DataFrame, counts: np.ndarray) -> GLMResultsWrapper:
    """
    Fit a Poisson log-linear model using IRLS.

    `design` should NOT include an intercept column

    """
    X = sm.add_constant(design, has_constant = "add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            counts,
            X,
            family = sm.families.Poisson(),
        )
        result = model.fit(maxiter = 100, tol = 1e-8, disp = False)
    return result

# ---------------------------------------------------------------------------
# Orthogonal polynomial basis matrix
# ---------------------------------------------------------------------------
#Need this to prevent multicollinearity in higher-order terms
#Mimicing poly() from R

def _ortho_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute an orthogonal polynomial basis for `x` up to `degree`, using
    the Gram-Schmidt process.

    Parameters
    ----------
    x : 1-D array of predictor values (e.g. integer scores).
    degree : maximum polynomial degree.

    Returns
    -------
    Z : ndarray, shape (len(x), degree + 1)
        Column 0 is the intercept (all ones); columns 1..degree are
        orthogonal polynomial contrasts.
    """
    n = len(x)
    Z = np.zeros((n, degree + 1))
    Z[:, 0] = 1.0  #Intercept

    if degree == 0:
        return Z

    #Gram-Schmidt orthogonalisation bit
    Z[:, 1] = x - x.mean()

    for k in range(2, degree + 1):
        z_prev  = Z[:, k - 1]
        z_prev2 = Z[:, k - 2]

        v = x * z_prev
        alpha = np.dot(v, z_prev) / np.dot(z_prev, z_prev)
        beta  = np.dot(v, z_prev2) / np.dot(z_prev2, z_prev2)

        Z[:, k] = v - alpha * z_prev - beta * z_prev2

    #Normalise columns 1+ to unit norm
    for k in range(1, degree + 1):
        nrm = np.linalg.norm(Z[:, k])
        if nrm > 0:
            Z[:, k] /= nrm

    return Z

# ---------------------------------------------------------------------------
# Design matrix build (i.e., score function)
# ---------------------------------------------------------------------------
#Build matrix to pass to GLM fit
def _build_scorefun(
    scores: np.ndarray,
    degrees: list[int],
    stepup: bool,
    compare: bool,) -> tuple[pd.DataFrame, list[int] | None, list[str] | None]:
    """
    Build the design matrix from orthogonal polynomials.

    For a univariate design `degrees` is a list with one element, (e.g.
    `[4]`)

    Returns
    -------
    scorefun : DataFrame  — columns are named "s1", "s1^2", …
    models   : list[int] | None
    mnames   : list[str] | None
    """
    degree = int(degrees[0])  #Univariate: only the first element matters
    Z = _ortho_poly(scores, degree)  #Shape (n, degree+1); col 0 = intercept

    col_names = [f"s^{k}" for k in range(1, degree + 1)]
    scorefun = pd.DataFrame(Z[:, 1:], columns = col_names)

    if stepup or compare:
        #Models[i] indicates which step model column i belongs to
        models = list(range(1, degree + 1))   #Need +1 to include final degree b/c indexing difference
        mnames = [f"Degree {d}" for d in range(1, degree + 1)]
    else:
        models = None
        mnames = None

    return scorefun, models, mnames

# ---------------------------------------------------------------------------
# ANOVA table
# ---------------------------------------------------------------------------
#Report out some fit stats (Deviance, AIC, etc.) to allow model evaluation

def _aovtab(
    models: dict[str, GLMResultsWrapper],
    counts: np.ndarray,) -> pd.DataFrame:
    """
    Build a deviance fit table comparing nested Poisson models.

    Columns: Model, Df, Deviance, Resid.Df, Resid.Dev, Pr(>Chi), AIC, BIC
    """
    n = len(counts)
    rows = []
    prev_dev  = None
    prev_df   = None
    #model_names = list(models.keys())

    for name, res in models.items(): #res = GLMResultsWrapper object
        df_resid = int(res.df_resid)
        deviance = float(res.deviance)
        aic      = float(res.aic)
        n_params = int(res.df_model) + 1         #+1 for intercept
        bic      = deviance + np.log(n) * n_params #np.log = ln

        if prev_dev is not None:  #For not-the-first-model
            delta_df  = prev_df  - df_resid
            delta_dev = prev_dev - deviance
            p_val     = 1.0 - chi2.cdf(delta_dev, delta_df) if delta_df > 0 else np.nan
        else:
            delta_df  = np.nan
            delta_dev = np.nan
            p_val     = np.nan

        rows.append({
            "Model"    : name,
            "Df"       : delta_df, #Gain in df from previous model
            "Deviance" : delta_dev, #Deviance decrease from previous model
            "Resid.Df" : df_resid, #Remaining df
            "Resid.Dev": deviance, #Remaining deviance
            "Pr(>Chi)" : p_val, #p-value of increase in order
            "AIC"      : aic,
            "BIC"      : bic,
        })
        prev_dev = deviance
        prev_df  = df_resid

    return pd.DataFrame(rows).set_index("Model")

# ---------------------------------------------------------------------------
# Model selection option
# ---------------------------------------------------------------------------

def _glmselect(
    atab: pd.DataFrame,
    choosemethod: Literal["chi", "aic", "bic"] = "aic",
    chip: float = 0.05,) -> str:
    """
    Select the best model from deviance table.

    Parameters
    ----------
    atab         : output of _aovtab().
    choosemethod : "chi" uses the chi-squared p-value threshold `chip`;
                   "aic" / "bic" pick the model minimising AIC / BIC.
    chip         : significance threshold for chi-squared selection.

    Returns
    -------
    name of the selected model (index label in atab).
    """
    if choosemethod == "chi":
        #Last model whose improvement is significant at level chip
        sig = atab.index[atab["Pr(>Chi)"].fillna(1) < chip].tolist()
        return sig[-1] if sig else atab.index[0]
    elif choosemethod == "aic":
        return atab["AIC"].idxmin() #Smallest AIC
    elif choosemethod == "bic":
        return atab["BIC"].idxmin() #Smallest BIC
    else:
        raise ValueError(f"Unknown choosemethod: {choosemethod!r}")
        
# ---------------------------------------------------------------------------
# Overall function
# ---------------------------------------------------------------------------
def presmooth(
        freq: "array-like", 
        score_min: int, 
        score_max: int, 
        degrees: list[int] | None = None,
        scorefun: pd.DataFrame | None = None,
        compare: bool = False,
        choose: bool = False,
        choosemethod: Litera["chi", "aic", "bic"] = "aic",
        chip: float = 0.05,
        verbose: bool = False) -> pd.DataFrame | pd.Series | dict:
    """
    Perform log-linear Poisson presmoothing for a single form using polynomial
    orders.

    Parameters
    ----------
    freq: Observed score frequencies (length = score_max - score_min + 1,
                 or a Series/dict indexed by score).
    score_min: Minimum possible score.
    score_max: Maximum possible score.
    degrees: Max polynomial degree.  Pass a list with one integer for the
                 univariate case.  Defaults to `[4]`.
    scorefun: Optional pre-built design matrix (DataFrame, one row per score,
                 no intercept column).  If supplied, `degrees` is ignored.
    stepup: If True, fit models of degree 1, 2, …, [degrees] and return all
                 fitted frequencies as a DataFrame (one column per model).
                 Defaults to True when `compare` or `choose` is True.
    compare: If True, return the ANOVA/AIC/BIC comparison table instead
                 of fitted frequencies.
    choose: If True, run `compare` and additionally select the best model;
                 return the smoothed frequencies for that model only.
    choosemethod: Criterion for `choose`: "chi", "aic", or "bic".
    chip: p-value threshold used when `choosemethod="chi"`.
    verbose: If True, return the raw statsmodels GLM result object(s).

    Returns
    -------
    Default (stepup = False, compare = False, choose = False, verbose = False):
        pd.Series — smoothed frequencies, indexed by score.

    stepup = True (and not compare/choose/verbose):
        pd.DataFrame — one column per model ("Degree 1", …, "Degree N"),
        indexed by score.

    compare = True (and choose = False):
        pd.DataFrame — ANOVA/AIC/BIC table.

    choose = True:
        dict with keys:
            "fitted" → pd.Series of smoothed frequencies for chosen model,
            "anova" → ANOVA table,
            "model" → name of chosen model.

    verbose = True:
        Single GLMResultsWrapper or dict of them (stepup=True).
    """
    if degrees is None:
        degrees = [4]

    if choose:
        compare = True
        
    if stepup is None:
        stepup = compare
        
    # ------------------------------------------------------------------
    # First: Build score vector and observed counts
    # ------------------------------------------------------------------
    score_range = np.arange(score_min, score_max + 1, dtype = float)
    n_scores = len(score_range)

    freq_series = pd.Series(freq)
    if not freq_series.index.equals(pd.RangeIndex(n_scores)):
        #If freq is indexed by score values, reindex 
        #Otherwise assume it is ordered from score_min to score_max.
        freq_series = freq_series.reindex(score_range, fill_value = 0) #Don't eliminate scores with zero counts
    counts = freq_series.values.astype(float)
    
    # ------------------------------------------------------------------
    # Second: Build design matrix
    # ------------------------------------------------------------------
    if scorefun is not None:
        sf_df = scorefun.copy().reset_index(drop = True)
        if len(sf_df) != n_scores:
            raise ValueError(
                "'scorefun' must contain the same number of rows as 'freq'"
            )
        if stepup or compare:
            #Treat each column as one additional model term
            models_idx = list(range(1, len(sf_df.columns) + 1))
            mnames = [f"Term {i}" for i in models_idx]
        else:
            models_idx = None
            mnames = None
    else:
        sf_df, models_idx, mnames = _build_scorefun(
            score_range, degrees, stepup, compare
        )

    if (stepup or compare) and sf_df.shape[1] < 2:
        raise ValueError(
            f"Cannot run multiple models with only {sf_df.shape[1]} model term(s). "
            "Increase `degrees` or provide a wider `scorefun`."
        )
    
    # ------------------------------------------------------------------
    # Third: Fit model(s)
    # ------------------------------------------------------------------
    if stepup or compare:
        #One nested model per unique step value
        unique_steps = sorted(set(models_idx))
        col_names = list(sf_df.columns)

        glm_results: dict[str, sm.GLMResultsWrapper] = {}
        for step in unique_steps:
            # Include all columns assigned to steps <= current step
            keep_cols = [c for c, m in zip(col_names, models_idx) if m <= step]
            glm_results[mnames[step - 1]] = _fit_glm(sf_df[keep_cols], counts)
    else:
        glm_results = {"model": _fit_glm(sf_df, counts)}
    
    # ------------------------------------------------------------------
    # Fourth: Return requested things
    # ------------------------------------------------------------------
    scores_int = score_range.astype(int)

    #If verbose: return raw GLM object(s)
    if verbose:
        if stepup or compare:
            return glm_results
        return glm_results["model"]

    #If compare without choose: return ANOVA table
    if compare and not choose:
        return _aovtab(glm_results, counts)

    #If choose: run model selection, return smoothed freq + metadata
    if choose:
        atab = _aovtab(glm_results, counts)
        best = _glmselect(atab, choosemethod, chip)
        fitted = pd.Series(
            glm_results[best].fittedvalues,
            index=scores_int,
            name="smoothed",
        )
        return {"fitted": fitted, "anova": atab, "model": best}

    #If stepup without compare/choose: return all fitted values as DataFrame
    if stepup:
        out = pd.DataFrame(
            {name: res.fittedvalues for name, res in glm_results.items()},
            index=scores_int,
        )
        out.index.name = "score"
        return out

    #Default: return smoothed frequencies as a Series
    return pd.Series(
        glm_results["model"].fittedvalues,
        index=scores_int,
        name="smoothed",
    )
