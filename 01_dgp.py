"""
01_dgp.py -- Data Generating Process for the Direct Mail ROI demo.

Simulates a universe of 100,000 households for a high-value direct mail
program (value per sale $1,000, cost per piece $1.50):

  - Household features: income, net worth, modeled cruise-ship likelihood,
    prior direct-mail response frequency.
  - A true (hidden) response propensity driven by those features.
  - 90% of households are mailed; 10% are held out (not mailed) at random.
    Holdouts still buy at a low baseline rate, so they let us measure lift.
  - Mail multiplies the odds of a sale ~6x (calibrated so the top decile
    responds at ~0.7% mailed and the overall mailed rate is ~0.27%).

Output: output/households.csv with one row per household.
"""

import numpy as np
import pandas as pd

SEED = 130
N_HH = 100_000
HOLDOUT_PCT = 0.10
MAIL_ODDS_MULT = 6.0          # mail multiplies odds of sale ~6x
TARGET_MAILED_RATE = 0.00266  # overall mailed sales rate (~239 / 90,000)


rng = np.random.default_rng(SEED)


def zscore(x):
    return (x - x.mean()) / x.std()


def generate_households(n, rng):
    """Household features. A latent 'affluence' factor ties income,
    net worth, and cruise likelihood together, as in real prospect files."""
    affluence = rng.normal(0, 1, n)

    ln_income = 11.05 + 0.55 * affluence + rng.normal(0, 0.35, n)
    income = np.round(np.exp(ln_income), -2)  # median ~$63K

    ln_networth = 12.4 + 0.85 * affluence + rng.normal(0, 0.70, n)
    net_worth = np.round(np.exp(ln_networth), -3)

    # Modeled cruise-ship likelihood: 1-99 score, partly driven by affluence
    cruise = np.clip(np.round(45 + 14 * affluence + rng.normal(0, 18, n)), 1, 99)

    # Prior direct-mail response frequency: count of past responses (0-9),
    # driven by its own latent responsiveness plus a touch of affluence
    responsiveness = 0.3 * affluence + rng.normal(0, 1, n)
    lam = np.exp(-0.6 + 0.55 * responsiveness)
    prior_dm = np.minimum(rng.poisson(lam), 9)

    return pd.DataFrame({
        "hh_id": np.arange(1, n + 1),
        "income": income,
        "net_worth": net_worth,
        "cruise_likelihood": cruise.astype(int),
        "prior_dm_resp_freq": prior_dm,
    })


def true_logit(df, intercept):
    """Hidden truth: how features drive the baseline (unmailed) sale logit.
    Coefficients sized so, when ranked into deciles, the top decile responds
    ~0.7% mailed and the top 6 deciles clear the mail-cost breakeven --
    matching the shape of DM_ROI_Example.xlsx."""
    return (
        intercept
        + 0.16 * zscore(np.log(df["income"]))
        + 0.19 * zscore(np.log(df["net_worth"]))
        + 0.23 * zscore(df["cruise_likelihood"])
        + 0.30 * zscore(df["prior_dm_resp_freq"])
    )


def calibrate_intercept(df, rng):
    """Solve for the intercept that hits the target overall mailed rate."""
    from scipy.optimize import brentq

    def gap(b0):
        p_mail = 1 / (1 + np.exp(-(true_logit(df, b0) + np.log(MAIL_ODDS_MULT))))
        return p_mail.mean() - TARGET_MAILED_RATE

    return brentq(gap, -15, -3)


def main():
    df = generate_households(N_HH, rng)

    b0 = calibrate_intercept(df, rng)
    logit_base = true_logit(df, b0)
    p_base = 1 / (1 + np.exp(-logit_base))                          # not mailed
    p_mail = 1 / (1 + np.exp(-(logit_base + np.log(MAIL_ODDS_MULT))))  # mailed

    # Random 90/10 split: mailed vs holdout (holdout is modeled, not mailed)
    df["mailed"] = (rng.random(N_HH) >= HOLDOUT_PCT).astype(int)

    # Realize sales: mailed HHs sell at p_mail, holdouts at baseline p_base
    p = np.where(df["mailed"] == 1, p_mail, p_base)
    df["sold"] = (rng.random(N_HH) < p).astype(int)

    df.to_csv("output/households.csv", index=False)

    m = df["mailed"] == 1
    print(f"Households: {len(df):,}  (mailed {m.sum():,}, holdout {(~m).sum():,})")
    print(f"Mailed sales rate:  {df.loc[m, 'sold'].mean():.4%}  "
          f"({df.loc[m, 'sold'].sum()} sales)")
    print(f"Holdout sales rate: {df.loc[~m, 'sold'].mean():.4%}  "
          f"({df.loc[~m, 'sold'].sum()} sales)")
    print(f"Calibrated intercept: {b0:.3f}")


if __name__ == "__main__":
    main()
