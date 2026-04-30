# Retrieval Pareto

Production-oriented cost/quality benchmark for modern dense, late-interaction, sparse, and hybrid retrievers.

Evaluates deployed retriever configurations across quality, latency, and storage: model plus search backend plus compression choices.

## What Is Included

- 13 datasets: 6 BEIR tasks, 6 BRIGHT tasks, and LIMIT.
- Dense models: BGE small/large, GTE large, E5 large, Qwen3-Embedding-8B, NV-Embed-v2.
- Late-interaction models: mxbai-edge-colbert, ColBERTv2, GTE-ModernColBERT, Reason-ModernColBERT.
- Retrieval systems: flat exact, HNSW, OPQ-IVF-PQ, RaBitQ, ScaNN, binary rerank, FastPlaid, MUVERA, BM25, and RRF hybrids.
- Metrics: nDCG@10, recall@100, MRR@10, MAP@100, p50/p99 query latency, storage, and A100 reference cost.

The checked-in aggregate result files live under:

```text
results/<benchmark>/<dataset>/<retrieval_system>/<model>.json
```

The interactive website lives in `site/`. It includes Pareto charts, benchmark filters, a measured-systems table, result detail panels, and a methodology page. Its compact data file is:

```text
site/data/results.json
```

Per-query ranking sidecars are not committed because they are large. Download them from the [`v1-data` release](https://github.com/SoumilRathi/retrieval-pareto/releases/tag/v1-data) as `retrieval-pareto-rankings-json.tar.gz`.

## Quickstart

```bash
make install
make eval BENCHMARK=beir DATASET=scifact MODEL=BAAI/bge-small-en-v1.5 SYSTEM=dense-flat
make site-data
```

ScaNN is Linux-only in this setup. Install it with `pip install '.[scann]'` on a Linux host before reproducing `dense-scann` rows.

Serve the interactive website locally:

```bash
python scripts/serve_site.py
```

Validate result coverage:

```bash
python scripts/validate_v1_coverage.py --results-dir results --latency-sample-size 200
```

## Retrieval Systems

Dense:

- `dense-flat`: exact dense search over normalized embeddings.
- `dense-hnsw`: HNSWlib graph search with recorded `M`, `ef_construction`, and `ef_search`.
- `dense-opq-ivfpq`: FAISS OPQ + IVF + PQ.
- `dense-rabitq`: FAISS IVF + RaBitQ.
- `dense-scann`: ScaNN anisotropic hashing.
- `dense-binary-rerank`: sign-bit Hamming candidate generation with dense rerank.

Late interaction:

- `li-exact`: exact MaxSim on small corpora.
- `li-fastplaid`: PyLate/FastPlaid approximate late-interaction search.
- `li-muvera`: MUVERA fixed-dimensional encoding with MaxSim rerank.

Sparse and hybrid:

- `sparse-bm25`
- `hybrid-rrf-bm25-dense`
- `hybrid-rrf-bm25-li`
- `hybrid-rrf-bm25-dense-li`
- `hybrid-rrf-dense-li`

## Reading The Results

Useful comparisons:

- Quality vs latency: what quality can be bought at a serving-latency budget?
- Quality vs storage: what quality can be bought at an index-size budget?
- Dense vs late interaction on LIMIT and BRIGHT: where single-vector dense retrieval loses separation signal.

Several rows are intentional skip/constraint rows:

- `backend_oom_full8192_a100`: FastPlaid full8192 did not fit on the A100 latency host for that model/dataset pair.
- `not_applicable_small_corpus`: OPQ-IVF-PQ skipped on small corpora to avoid under-trained PQ codebooks.
- `skipped_high_dim_opq_compute`: Qwen3-Embedding-8B full-dimension OPQ was skipped because construction cost dominated; lower Matryoshka dimensions are included.
- `component_latency_unavailable`: hybrid quality exists, but one component latency was unavailable.

HNSW is reported as a high-recall CPU graph configuration (`M=32`, `ef_search=128`), not as a latency-tuned sweep. On several BEIR-scale corpora, exact flat search is faster because contiguous matrix multiplication beats graph traversal overhead. That is an empirical result, not a rendering bug.

The website's `Complete configs` view only averages configurations that cover every dataset in the selected benchmark. Partial-scope rows such as `li-exact` are still available on individual dataset tabs.

## Hardware And Cost

Latency rows use an A100 reference host. Cost is a derived conversion from measured p50 latency using the recorded A100 hourly price; it is included for intuition, not as an independent measurement. The headline plots should remain quality-vs-latency and quality-vs-storage.

See [METHODOLOGY.md](METHODOLOGY.md) for the full protocol and [RESULTS.md](RESULTS.md) for coverage details.
