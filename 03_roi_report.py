"""
03_roi_report.py -- No-Model vs With-Model ROI comparison.

Two scenarios, same universe, same economics ($1.50/piece, $1,000/sale):

  NO MODEL:   mail every non-holdout household (all 10 deciles).
  WITH MODEL: mail only the deciles whose incremental response clears
              the mail-cost breakeven (top 6) -- fewer pieces, and every
              piece goes to a household worth mailing.

Incremental sales = mailed sales minus the sales that would have happened
anyway (estimated from the holdout group's baseline rate).

Input:  output/decile_lift.csv
Output: output/roi_by_decile.csv, output/roi_comparison.csv,
        output/summary.json, output/chart_*.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COST_PER_PIECE = 1.50
VALUE_PER_SALE = 1000.0

NAVY, TEAL, RED, GRAY = "#1f3a5f", "#2a9d8f", "#c1443c", "#9aa5b1"


def scenario(g, label):
    pieces = g["n_mailed"].sum()
    sales = g["mailed_sales"].sum()
    incr = g["incremental_sales"].sum()
    cost = pieces * COST_PER_PIECE
    value = incr * VALUE_PER_SALE
    net = value - cost
    return {
        "scenario": label,
        "deciles_mailed": len(g),
        "pieces_mailed": int(pieces),
        "mailed_sales": int(sales),
        "incremental_sales": round(float(incr), 1),
        "mail_cost": round(cost, 0),
        "incremental_value": round(value, 0),
        "net": round(net, 0),
        "roi": round(net / cost, 2),
    }


def main():
    g = pd.read_csv("output/decile_lift.csv")

    # Per-decile economics
    g["mail_cost"] = g["n_mailed"] * COST_PER_PIECE
    g["incr_value"] = g["incremental_sales"] * VALUE_PER_SALE
    g["net"] = g["incr_value"] - g["mail_cost"]
    g["roi"] = g["net"] / g["mail_cost"]
    g.to_csv("output/roi_by_decile.csv", index=False)

    # A decile is worth mailing if its incremental value covers its mail cost
    profitable = g[g["roi"] > 0]
    cutoff = int(profitable["decile"].max())
    print(f"Profitable deciles: 1-{cutoff}\n")

    no_model = scenario(g, "No model (mail everyone)")
    with_model = scenario(g[g["decile"] <= cutoff], f"With model (mail deciles 1-{cutoff})")
    comp = pd.DataFrame([no_model, with_model])
    comp.to_csv("output/roi_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # Numbers the web page needs
    summary = {
        "cost_per_piece": COST_PER_PIECE,
        "value_per_sale": VALUE_PER_SALE,
        "cutoff_decile": cutoff,
        "deciles": {
            "decile": g["decile"].tolist(),
            "n_mailed": g["n_mailed"].astype(int).tolist(),
            "mailed_rate_pct": (g["mailed_rate"] * 100).round(4).tolist(),
            "incremental_sales": g["incremental_sales"].tolist(),
            "net": g["net"].round(0).tolist(),
            "roi": g["roi"].round(2).tolist(),
        },
        "no_model": no_model,
        "with_model": with_model,
    }
    with open("output/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Charts --------------------------------------------------------------
    dec = g["decile"]

    # 1. Response rate by decile with breakeven line
    breakeven_rate = COST_PER_PIECE / VALUE_PER_SALE / (
        g["incremental_sales"].sum() / g["mailed_sales"].sum())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [TEAL if d <= cutoff else GRAY for d in dec]
    ax.bar(dec, g["mailed_rate"] * 100, color=colors)
    ax.axhline(breakeven_rate * 100, color=RED, ls="--", lw=1.5,
               label=f"Breakeven ({breakeven_rate:.2%} response)")
    ax.set_xlabel("Model decile (1 = best)")
    ax.set_ylabel("Mailed response rate (%)")
    ax.set_title("Response rate by model decile")
    ax.set_xticks(dec)
    ax.legend()
    fig.tight_layout()
    fig.savefig("output/chart_response_by_decile.png", dpi=150)

    # 2. Net $ by decile
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [TEAL if n > 0 else RED for n in g["net"]]
    ax.bar(dec, g["net"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Model decile (1 = best)")
    ax.set_ylabel("Net $ (incremental value - mail cost)")
    ax.set_title("Each decile's contribution to profit")
    ax.set_xticks(dec)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    fig.tight_layout()
    fig.savefig("output/chart_net_by_decile.png", dpi=150)

    # 3. Scenario comparison
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    labels = ["Mail everyone", f"Mail deciles 1-{cutoff}"]
    for ax, key, title, fmt in [
        (axes[0], "pieces_mailed", "Pieces mailed", "{:,.0f}"),
        (axes[1], "net", "Net profit", "${:,.0f}"),
        (axes[2], "roi", "ROI", "{:.0%}"),
    ]:
        vals = [no_model[key], with_model[key]]
        bars = ax.bar(labels, vals, color=[GRAY, TEAL])
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.15)
    fig.suptitle("Same file, same economics -- fewer, smarter pieces", y=1.02)
    fig.tight_layout()
    fig.savefig("output/chart_scenarios.png", dpi=150, bbox_inches="tight")

    print("\nCharts and CSVs written to output/")


if __name__ == "__main__":
    main()
