# Direct Mail Targeting ROI

A seeded, self-contained simulation showing what a direct-mail targeting model is worth in dollars: score 100,000 households, rank them into deciles, and mail only where incremental sales cover the postage.

**Result: 40% fewer pieces, $22K more net profit, ROI up from 45% to 102% — same file, same economics.**

Full write-up: [mcbelli.github.io/DM_ROI/dm_same-budget-double-the-roi](https://mcbelli.github.io/DM_ROI/dm_same-budget-double-the-roi)

## Run it

```bash
pip install -r requirements.txt
python 01_dgp.py             # generate 100K households, 90/10 mail/holdout, sales
python 02_model_deciles.py   # fit logistic regression, build decile lift table
python 03_roi_report.py      # decile economics, scenario comparison, charts
```

Everything is seeded (`SEED = 130`), so all numbers and figures reproduce exactly. The two large generated files (`output/households.csv`, `output/households_scored.csv`) are gitignored — the scripts rebuild them in seconds.

## The design

- **Economics:** $1,000 value per sale, $1.50 per mail piece → breakeven at ~0.18% incremental response.
- **Holdout:** a random 10% receives no mail, pinning down the ~5x lift mail provides and the sales that would have happened anyway.
- **Honest information set:** the true response propensity is hidden in the DGP; the model trains only on mailed households' outcomes, as a real analyst would.

| Scenario | Pieces | Incremental sales | Net | ROI |
| --- | ---: | ---: | ---: | ---: |
| No model — mail everyone | 89,806 | 195 | $60,191 | 45% |
| With model — mail deciles 1–6 | 53,841 | 163 | $82,338 | 102% |
