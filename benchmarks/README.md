# Benchmarks

This directory is the source of truth for evaluation datasets and benchmark reports.

Do not mix these files with train/retrain data under `data/`. Benchmark samples must stay frozen so active and candidate models can be compared on the same evidence.

## Layout

- `cleanliness/pilot_benchmark.csv`: real public-image ground-truth set for `PASS/PENDING/FAIL`.
- `cleanliness/case_studies.csv`: real selected cases for thesis/demo discussion.
- `ppe/pilot_benchmark.csv`: real public-image PPE ground-truth set.
- `reports/`: created only after a real model run. Empty report placeholders are intentionally not committed.

There is no committed golden mask manifest yet because the project does not currently contain a real frozen set of reviewed polygon/mask labels. Add that file only after the masks exist.

## Rules

- Do not train on samples listed here.
- Do not overwrite historical reports; create a new report per model/date.
- Use rectangle-only annotations as weak labels, not golden masks.
- Do not invent model outputs. Leave `predicted_*`, score, latency, and visualization fields empty until a real inference run fills them.
- If images are large/private, keep them out of git and store only stable blob URLs or object keys in the CSV.
