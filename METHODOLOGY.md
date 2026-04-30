# Methodology

Retrieval Pareto evaluates deployed retrieval systems, not just embedding checkpoints. A row is defined by:

```text
benchmark x dataset x model x retrieval_system
```

This separates model quality from backend choices such as exact search, approximate search, compression, and late-interaction serving.

## Datasets

The public result set covers:

- BEIR: `nfcorpus`, `scifact`, `fiqa`, `arguana`, `scidocs`, `trec-covid`
- BRIGHT: `biology`, `economics`, `psychology`, `robotics`, `stackoverflow`, `leetcode`
- LIMIT: `limit`

Only evaluation queries with qrels are scored.

## Models

Dense models:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `Alibaba-NLP/gte-large-en-v1.5`
- `intfloat/e5-large-v2`
- `Qwen/Qwen3-Embedding-8B`
- `nvidia/NV-Embed-v2`

Late-interaction models:

- `mixedbread-ai/mxbai-edge-colbert-v0-32m`
- `lightonai/colbertv2.0`
- `lightonai/GTE-ModernColBERT-v1`
- `lightonai/Reason-ModernColBERT`

## Retrieval Systems

Dense systems:

- Flat exact dense search
- HNSWlib graph search
- FAISS OPQ-IVF-PQ
- FAISS IVF-RaBitQ
- ScaNN anisotropic hashing
- Binary Hamming first stage with dense rerank

Late-interaction systems:

- Exact MaxSim on small corpora
- FastPlaid approximate search
- MUVERA first stage with MaxSim rerank

Sparse and hybrid systems:

- BM25
- Reciprocal-rank fusion with `k=60`

## Quality Metrics

For every completed row:

- `ndcg_at_10`
- `recall_at_100`
- `mrr_at_10`
- `map_at_100`

Per-query rankings are written as `*.rankings.json` sidecars during evaluation. They are distributed as a release artifact rather than committed to git.

## Latency Protocol

Latency is split into offline indexing and online serving.

Offline indexing records corpus encoding time, index preparation time, docs/sec, precision, max sequence length, and batch size. Corpus encoding uses the largest stable batch available for the model/hardware when possible.

Online serving latency is measured after warmup on a fixed query sample:

- `latency_sample_seed = 13`
- target sample size `200`, or all queries if fewer than 200 exist
- p50 and p99 latency
- batch-1 latency as the primary fair comparison

Rows record query encoding latency and retrieval latency separately. End-to-end query latency is:

```text
query_encode_ms_p50 + retrieval_ms_p50_topk100
```

Fixed serving batches record OOM or unavailable status rather than silently falling back.

## Storage

Rows record representation/index sizes where available:

- FP16 representation size
- INT8 estimate
- PQ/compressed backend size where applicable
- binary-rerank index size
- HNSW index file size
- MUVERA index size

The quality-vs-storage plot should use the backend-specific serving index size when present.

## Hardware And Cost

The reference latency hardware is an A100 SXM 40GB Lambda instance. Cost fields convert measured p50 latency into an A100 reference cost using the hourly price stored in each JSON row. This is a derived convenience field, not a separate economic benchmark.

## Important Caveats

HNSW rows use a high-recall setting (`M=32`, `ef_search=128`). On BEIR-scale corpora, high-recall HNSW can be slower than exact flat search because graph traversal overhead dominates. This is why HNSW should be read as one backend point, not as a fully tuned ANN curve.

FastPlaid full8192 rows marked `backend_oom_full8192_a100` are explicit deployment constraints on the A100 latency host. They are not missing results.

OPQ-IVF-PQ is skipped for corpora below 10k documents because small corpora under-train PQ codebooks and produce misleading results.

Qwen3-Embedding-8B full-dimension OPQ-IVF-PQ is skipped because construction cost dominated the benchmark; lower Matryoshka dimensions are included where useful.

The website's benchmark-level `Complete configs` view only averages configurations that cover every dataset in the selected benchmark. Partial-scope configurations remain visible on the individual dataset tabs.
