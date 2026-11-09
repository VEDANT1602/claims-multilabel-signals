# Review reasons are not one binary score

The actual CPU run fits four independent logistic heads on 2,000 records from 500 families, selects thresholds on 600 records from 150 separate families, then evaluates 800 records from 200 unseen families. Decisions are supported for 3 of 4 labels; the rare label is withheld when validation support is below 25 positives or 25 negatives. It remains a separately labeled probability diagnostic, not an admitted decision.

On the same supported-label population, a fixed 0.5 threshold has macro F1 0.5830; validation-selected thresholds have macro F1 0.6075, micro F1 0.6189, and macro AP 0.6475. The paired family-bootstrap interval for macro F1 difference is [0.0029866496326184304, 0.04531820556794989]; fitted models and thresholds are held fixed.

My implementation separates each label's evidence and admission rule, commits portable learned coefficients and per-label raw predictions, and reports the generator's changed-relation slice. Reference/shifted supported macro AP is 0.7194/0.6078; the shift is deliberately synthetic and not a real drift estimate. A label with weak support should enter review rather than disappear inside a favorable micro-average.

The aggregate improvement is not universal: document_gap F1 falls from 0.6766 to 0.6601. That tradeoff remains in the per-label table.

The source topic was multi-label learning; this project contains no medical/drug data or real claims. The equation is simple, reasons are synthetic, and random-family separation does not establish deployment generalization. The next step is an independently curated and authorized label set with costs and review capacity, not a production eligibility rule.
