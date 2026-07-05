"""
02_model_deciles.py -- Train the targeting model and build the decile lift table.

Mirrors real-world practice:
  1. Train logistic regression on MAILED households only (sold ~ features).
  2. Score ALL households (mailed + holdout) and rank into deciles
     (decile 1 = best 10% by model score).
  3. Build a decile lift table: mailed rate, holdout rate, estimated
     baseline, and incremental sales per decile.

Per-decile holdout counts are small (~1,000 HHs, a handful of sales), so
the per-decile baseline rate is estimated by scaling each decile's mailed
rate by the GLOBAL lift ratio (overall mailed rate / overall holdout rate)
-- the same 5-6x relationship built into DM_ROI_Example.xlsx. Raw holdout
counts are kept in the table for honesty.

Input:  output/households.csv   Output: output/decile_lift.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = ["log_income", "log_net_worth", "cruise_likelihood", "prior_dm_resp_freq"]


def main():
    df = pd.read_csv("output/households.csv")
    df["log_income"] = np.log(df["income"])
    df["log_net_worth"] = np.log(df["net_worth"])

    # --- 1. Train on mailed households only ---------------------------------
    train = df[df["mailed"] == 1]
    X, y = train[FEATURES], train["sold"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print("Model AUC (mailed HHs):", round(auc, 3))
    print("Coefficients:")
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:22s} {c:+.4f}")

    # --- 2. Score everyone, rank into deciles (1 = best) --------------------
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    df["decile"] = 10 - pd.qcut(df["score"], 10, labels=False)  # 1..10
    df.to_csv("output/households_scored.csv", index=False)

    # --- 3. Decile lift table ------------------------------------------------
    g = df.groupby("decile").apply(
        lambda d: pd.Series({
            "n_hh": len(d),
            "n_mailed": (d["mailed"] == 1).sum(),
            "n_holdout": (d["mailed"] == 0).sum(),
            "mailed_sales": d.loc[d["mailed"] == 1, "sold"].sum(),
            "holdout_sales": d.loc[d["mailed"] == 0, "sold"].sum(),
        }), include_groups=False,
    ).reset_index()

    g["mailed_rate"] = g["mailed_sales"] / g["n_mailed"]
    g["holdout_rate_raw"] = g["holdout_sales"] / g["n_holdout"]

    # Global lift ratio smooths the tiny per-decile holdout counts
    overall_mailed_rate = g["mailed_sales"].sum() / g["n_mailed"].sum()
    overall_holdout_rate = g["holdout_sales"].sum() / g["n_holdout"].sum()
    lift_ratio = overall_mailed_rate / overall_holdout_rate
    print(f"\nGlobal lift ratio (mailed vs holdout): {lift_ratio:.1f}x")

    g["baseline_rate_est"] = g["mailed_rate"] / lift_ratio
    g["incremental_sales"] = (
        g["mailed_sales"] - g["n_mailed"] * g["baseline_rate_est"]
    ).round(1)

    g.to_csv("output/decile_lift.csv", index=False)

    print("\nDecile lift table:")
    print(g[["decile", "n_mailed", "mailed_sales", "mailed_rate",
             "n_holdout", "holdout_sales", "incremental_sales"]]
          .to_string(index=False,
                     formatters={"mailed_rate": "{:.4%}".format}))


if __name__ == "__main__":
    main()
