# Claims Multilabel Signals

One score and one threshold can hide which review reasons have enough evidence to act on.

This independent portfolio experiment by Vedant Thaker runs locally on Python 3.11. All inputs are original fictional or synthetic fixtures. It requires no cloud account, paid API, credentials, external weights or employer data.

## The decision

I transferred a multi-label learning topic to original fictional operational signals. Four actual logistic heads predict co-occurring review reasons from eight decision-time fields. Train, validation and test families are disjoint. Each label gets a validation-selected threshold only after a minimum-support check. The 0.5 baseline and selected policy are compared on the same admitted-label population; the rare withheld label stays visible as a diagnostic.

## Reproduce and inspect

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/ci
```

The default `python run.py` refreshes committed `results/`; `--output` writes a fresh run to an ignored directory. The seeded generator in `generate.py` writes the original input fixtures to `data/`. [Measured results](results/report.md), [raw metrics](results/metrics.json), and [source/data/result digests](results/manifest.json) make the experiment inspectable. Tests verify known answers and failure boundaries, not just successful execution.

## Measured comparison and next decision

The actual CPU experiment fits four logistic heads on **2,000 training records**, selects thresholds on **600 validation records**, and tests **800 records from 200 disjoint families**. Three labels meet the support gate; the rare label is withheld. On those same three labels, macro F1 is **0.6075 with selected thresholds versus 0.5830 at 0.5**; macro AP is **0.6475**. The gain is not universal: document-gap F1 declines from **0.6766 to 0.6601**.

The declared changed-relation slice has lower macro AP than the reference slice (**0.6078 versus 0.7194**). The next decision is an authorized independent label set with review costs and capacity; validation F1 alone cannot define a production action policy.

## What the result can support

All 3,400 records are synthetic numeric fixtures, not real insurance, lending, medical or customer records. A changed feature-label relation in half of test is a declared stress slice. Thresholds optimize validation F1, not business utility or fairness. There are no approval/eligibility actions. Portable JSON coefficients are actual fitted weights; no unknown pickle or downloaded model is used.

The next decision follows from the recorded comparison, including weak or negative outcomes. No synthetic result is presented as an employer deployment or a real-world business improvement.

See [data card](docs/data-card.md) and [reference credits](docs/provenance.md). Original code and fixtures are MIT licensed.
