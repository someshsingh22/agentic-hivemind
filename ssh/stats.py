"""Run each hypothesis' frozen regression test on a CSV whose feature columns are populated."""

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

SIGN = {"+": 1, "-": -1}


def fit(formula, term, df, group):
    """OLS coefficient of `term`; cluster-robust SE by `group` (HC1 if no group)."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    cov = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(df.loc[X.index, group])[0]}} if group else {"cov_type": "HC1"}
    result = sm.OLS(y, X).fit(**cov)
    return {"coef": result.params.get(term, np.nan), "p": result.pvalues.get(term, np.nan), "n": len(y)}


def run(plan, df, alpha=0.05):
    group = plan["env"]["group"]
    results = pd.DataFrame(
        {**h, **(fit(h["formula"], h["term"], df, group) if h["term"] else {"coef": np.nan, "p": np.nan, "n": 0})}
        for h in plan["hypotheses"]
    )
    tested = results.p.notna()
    results["p_holm"] = np.nan
    results.loc[tested, "p_holm"] = multipletests(results.p[tested], method="holm")[1]
    agrees = (results.sign == "two_sided") | (np.sign(results.coef) == results.sign.map(SIGN))
    results["ssh"] = (results.p_holm <= alpha) & agrees
    return results
