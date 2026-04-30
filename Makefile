.PHONY: install eval plot site-data smoke clean

MODEL ?= BAAI/bge-small-en-v1.5
DATASET ?= nfcorpus
BENCHMARK ?= beir
FAMILY ?=
SYSTEM ?=
LIMIT ?=
BATCH_SIZE ?=
CORPUS_ENCODE_BATCH ?=
LATENCY_BATCH_SIZES ?= 1
MAX_FEASIBLE_BATCH_CANDIDATES ?= 1,2,4,8,16,32,64,128
LATENCY_SAMPLES ?= 100
LATENCY_SAMPLE_SIZE ?= 200
LATENCY_SAMPLE_SEED ?= 13
LATENCY_WARMUP ?= 10
CACHE_DIR ?= data/cache
INDEX_DIR ?= indexes
LI_SEARCH ?= exact
HNSW_M ?= 32
HNSW_EF_CONSTRUCTION ?= 200
HNSW_EF_SEARCH ?= 128
FAISS_NLIST ?=
FAISS_PQ_M ?= 32
BINARY_CANDIDATES ?= 1000
TRUNCATE_DIM ?=
MAX_SEQ_LENGTH ?=
PRECISION ?= auto
GPU_HOURLY_USD ?= 1.99
GPU_PRICE_LABEL ?= Lambda A100 SXM 40GB
GPU_PRICING_SOURCE ?= https://lambda.ai/pricing
PLAID_NBITS ?= 8
PLAID_N_IVF_PROBE ?= 32
PLAID_N_FULL_SCORES ?= 8192
PLAID_KMEANS_NITERS ?= 4
MUVERA_K_SIM ?= 5
MUVERA_DIM_PROJ ?= 16
MUVERA_R_REPS ?= 20
MUVERA_CANDIDATES ?= 1000
NO_CACHE ?=
CACHE_ONLY ?=
SKIP_LATENCY ?=
SKIP_EXISTING ?=
PYTHON ?= $(shell command -v python3.12 >/dev/null 2>&1 && echo python3.12 || echo python3)
RUNNER := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo ".venv/bin/python")

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		$(PYTHON) -m venv .venv; \
		.venv/bin/python -m pip install -U pip; \
		.venv/bin/python -m pip install -e .; \
	fi

eval:
	$(RUNNER) -m retrieval_pareto.eval --model "$(MODEL)" --dataset "$(DATASET)" --benchmark "$(BENCHMARK)" --cache-dir "$(CACHE_DIR)" --index-dir "$(INDEX_DIR)" --li-search "$(LI_SEARCH)" --latency-batch-sizes "$(LATENCY_BATCH_SIZES)" --max-feasible-batch-candidates "$(MAX_FEASIBLE_BATCH_CANDIDATES)" --latency-samples "$(LATENCY_SAMPLES)" --latency-sample-size "$(LATENCY_SAMPLE_SIZE)" --latency-sample-seed "$(LATENCY_SAMPLE_SEED)" --latency-warmup "$(LATENCY_WARMUP)" --precision "$(PRECISION)" --gpu-hourly-usd "$(GPU_HOURLY_USD)" --gpu-price-label "$(GPU_PRICE_LABEL)" --gpu-pricing-source "$(GPU_PRICING_SOURCE)" --hnsw-m "$(HNSW_M)" --hnsw-ef-construction "$(HNSW_EF_CONSTRUCTION)" --hnsw-ef-search "$(HNSW_EF_SEARCH)" --faiss-pq-m "$(FAISS_PQ_M)" --binary-candidates "$(BINARY_CANDIDATES)" --plaid-nbits "$(PLAID_NBITS)" --plaid-n-ivf-probe "$(PLAID_N_IVF_PROBE)" --plaid-n-full-scores "$(PLAID_N_FULL_SCORES)" --plaid-kmeans-niters "$(PLAID_KMEANS_NITERS)" --muvera-k-sim "$(MUVERA_K_SIM)" --muvera-dim-proj "$(MUVERA_DIM_PROJ)" --muvera-r-reps "$(MUVERA_R_REPS)" --muvera-candidates "$(MUVERA_CANDIDATES)" $(if $(FAISS_NLIST),--faiss-nlist "$(FAISS_NLIST)",) $(if $(TRUNCATE_DIM),--truncate-dim "$(TRUNCATE_DIM)",) $(if $(MAX_SEQ_LENGTH),--max-seq-length "$(MAX_SEQ_LENGTH)",) $(if $(CORPUS_ENCODE_BATCH),--corpus-encode-batch "$(CORPUS_ENCODE_BATCH)",) $(if $(SYSTEM),--system "$(SYSTEM)",) $(if $(FAMILY),--family "$(FAMILY)",) $(if $(BATCH_SIZE),--batch-size "$(BATCH_SIZE)",) $(if $(LIMIT),--limit "$(LIMIT)",) $(if $(NO_CACHE),--no-cache,) $(if $(CACHE_ONLY),--cache-only,) $(if $(SKIP_LATENCY),--skip-latency,) $(if $(SKIP_EXISTING),--skip-existing,)

plot:
	$(RUNNER) -m retrieval_pareto.plots.generate_plots

site-data:
	$(PYTHON) scripts/export_site_data.py --results-dir results --output site/data/results.json

smoke:
	python3 -m compileall src scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
