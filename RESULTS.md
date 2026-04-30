# Results Coverage

The checked-in result set contains current aggregate JSON rows only. Per-query rankings are packaged separately as a release artifact.

## Included Aggregate Rows

- Total current rows: 704
- Dense rows: 487
- Late-interaction rows: 116
- Hybrid rows: 88
- Sparse rows: 13

## Datasets

BEIR:

- `nfcorpus`
- `scifact`
- `fiqa`
- `arguana`
- `scidocs`
- `trec-covid`

BRIGHT:

- `biology`
- `economics`
- `psychology`
- `robotics`
- `stackoverflow`
- `leetcode`

LIMIT:

- `limit`

## Status Rows

Most rows are completed benchmark rows. The remaining status rows are intentional and should be shown as constraints or skips, not silent missing data:

- `backend_oom_full8192_a100`: 12 rows
- `component_latency_unavailable`: 28 rows
- `not_applicable_small_corpus`: 13 rows
- `skipped_high_dim_opq_compute`: 10 rows

## Validation

Run:

```bash
python scripts/validate_v1_coverage.py --results-dir results --latency-sample-size 200
```

Expected summary:

```text
Missing cells: 0
Cells with missing/stale latency: 0
```

The validator treats documented OOM and skip statuses as intentional coverage.
