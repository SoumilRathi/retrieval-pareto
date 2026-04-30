from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_pareto.adapters import DenseRetriever, LateInteractionRetriever
from retrieval_pareto.adapters.dense_adapter import DenseIndex, DenseQueryRepr, _build_faiss_index
from retrieval_pareto.benchmarks import load_beir_dataset, load_bright_dataset, load_limit_dataset
from retrieval_pareto.cache import cache_info, cache_path, load_cached, save_cached
from retrieval_pareto.metrics.cost import cost_summary
from retrieval_pareto.metrics.latency import summarize_ms, timed_call
from retrieval_pareto.metrics.quality import compute_quality_metrics
from retrieval_pareto.metrics.storage import storage_summary
from retrieval_pareto.model_registry import infer_family, sanitize_model_id

SYSTEMS = {
    "dense-flat": {
        "family": "dense",
        "backend": "numpy-flat",
        "search_mode": "flat",
        "exact": True,
        "compression": "fp16",
    },
    "dense-hnsw": {
        "family": "dense",
        "backend": "hnswlib",
        "search_mode": "hnsw",
        "exact": False,
        "compression": "fp16+hnsw-graph",
    },
    "dense-opq-ivfpq": {
        "family": "dense",
        "backend": "faiss-opq-ivf-pq",
        "search_mode": "opq_ivf_pq",
        "exact": False,
        "compression": "opq-ivf-pq",
    },
    "dense-rabitq": {
        "family": "dense",
        "backend": "faiss-ivf-rabitq",
        "search_mode": "rabitq",
        "exact": False,
        "compression": "rabitq",
    },
    "dense-scann": {
        "family": "dense",
        "backend": "scann-ah",
        "search_mode": "scann",
        "exact": False,
        "compression": "scann-ah",
    },
    "dense-binary-rerank": {
        "family": "dense",
        "backend": "binary-hamming-fp32-rerank",
        "search_mode": "binary_rerank",
        "exact": False,
        "compression": "binary-codes+fp16-rerank-store",
    },
    "li-exact": {
        "family": "late_interaction",
        "backend": "pylate-maxsim",
        "search_mode": "exact",
        "exact": True,
        "compression": "fp16-token-vectors",
    },
    "li-fastplaid": {
        "family": "late_interaction",
        "backend": "pylate-fastplaid",
        "search_mode": "plaid",
        "exact": False,
        "compression": "plaid-pq",
    },
    "li-muvera": {
        "family": "late_interaction",
        "backend": "fastembed-muvera-fde-maxsim-rerank",
        "search_mode": "muvera",
        "exact": False,
        "compression": "muvera-fde+fp16-token-rerank-store",
    },
}

OPQ_IVFPQ_MIN_DOCS = 10_000
HIGH_DIM_OPQ_SKIP_MODEL = "Qwen/Qwen3-Embedding-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one retrieval benchmark evaluation.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--dataset", default="nfcorpus", help="Dataset short name.")
    parser.add_argument("--benchmark", default="beir", choices=["beir", "bright", "limit"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--system", choices=sorted(SYSTEMS), help="Retrieval system to evaluate.")
    parser.add_argument("--family", choices=["dense", "late_interaction", "hybrid"])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--limit", type=int, help="Optional query limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, help="Backward-compatible alias for corpus encode batch size.")
    parser.add_argument(
        "--corpus-encode-batch",
        default=None,
        help="Corpus encoding batch size or 'auto-max-feasible'.",
    )
    parser.add_argument("--latency-batch-sizes", default="1,32,max")
    parser.add_argument("--max-feasible-batch-candidates", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--latency-sample-size", type=int, default=200)
    parser.add_argument("--latency-sample-seed", type=int, default=13)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--li-search", default="exact", choices=["exact", "plaid", "muvera"])
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument("--faiss-nlist", type=int)
    parser.add_argument("--faiss-pq-m", type=int, default=32)
    parser.add_argument("--binary-candidates", type=int, default=1000)
    parser.add_argument("--truncate-dim", type=int, help="Optional dense embedding truncation for MRL models.")
    parser.add_argument("--max-seq-length", type=int, help="Optional encoder max sequence length.")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--gpu-hourly-usd", type=float, default=1.99)
    parser.add_argument("--gpu-pricing-source", default="https://lambda.ai/pricing")
    parser.add_argument("--gpu-price-label", default="Lambda A100 SXM 40GB")
    parser.add_argument("--plaid-nbits", type=int, default=8)
    parser.add_argument("--plaid-n-ivf-probe", type=int, default=32)
    parser.add_argument("--plaid-n-full-scores", type=int, default=8192)
    parser.add_argument("--plaid-kmeans-niters", type=int, default=4)
    parser.add_argument("--plaid-search-batch-size", type=int, default=262144)
    parser.add_argument("--muvera-k-sim", type=int, default=5)
    parser.add_argument("--muvera-dim-proj", type=int, default=16)
    parser.add_argument("--muvera-r-reps", type=int, default=20)
    parser.add_argument("--muvera-candidates", type=int, default=1000)
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache reads/writes.")
    parser.add_argument("--cache-only", action="store_true", help="Require cached embeddings; do not load models.")
    parser.add_argument("--skip-latency", action="store_true", help="Compute quality/storage only.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if a current result JSON already exists.")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_system = resolve_retrieval_system(args)
    family = retrieval_system["family"]
    dataset = load_dataset(args)
    output_path = result_path(
        Path(args.output_dir),
        dataset.benchmark,
        dataset.name,
        retrieval_system["name"],
        args.model,
        limit=args.limit,
        result_suffix=result_suffix(args, retrieval_system),
    )
    if args.skip_existing and current_result_exists(output_path, require_latency=not args.skip_latency):
        print(f"Skipping existing current result: {output_path}")
        return
    if should_skip_opq_ivfpq_small_corpus(retrieval_system, dataset):
        write_not_applicable_small_corpus(args, retrieval_system, dataset, output_path)
        return
    if should_skip_high_dim_opq_compute(args, retrieval_system):
        write_skipped_high_dim_opq_compute(args, retrieval_system, dataset, output_path)
        return

    retriever = build_retriever_from_args(args, retrieval_system, dataset)
    corpus_batch = resolve_corpus_encode_batch(args, retriever, dataset)
    retriever.batch_size = corpus_batch["batch_size"]

    print(f"Loaded {dataset.hf_id}: {len(dataset.documents)} docs, {len(dataset.queries)} eval queries")
    print(
        "Corpus encode batch: "
        f"{corpus_batch['batch_size']} ({corpus_batch['mode']})"
    )
    print(f"Preparing corpus with {args.model} ({args.system or retrieval_system['name']})")
    index, corpus_encode_ms, corpus_load_ms, corpus_cache = load_or_encode_corpus(
        args, family, retriever, dataset
    )
    index, index_prepare_ms = timed_call(lambda: prepare_index(retriever, index))

    print("Preparing queries")
    query_repr, query_encode_total_ms, query_load_ms, query_cache = load_or_encode_queries(
        args, family, retriever, dataset
    )

    print(f"Searching top-{args.k}")
    rankings, retrieval_total_ms = timed_call(lambda: retriever.search(query_repr, index, args.k))

    quality = compute_quality_metrics(dataset.queries, rankings, dataset.qrels)
    latency = (
        skipped_latency_summary(args)
        if args.skip_latency
        else measure_latency(retriever, dataset.queries, query_repr, index, args.k, args)
    )
    latency["corpus_encode_ms_total"] = corpus_encode_ms
    latency["corpus_load_ms_total"] = corpus_load_ms
    latency["corpus_encode_batch_size"] = corpus_batch["batch_size"]
    latency["corpus_encode_batch_mode"] = corpus_batch["mode"]
    latency["corpus_encode_batch_candidates"] = corpus_batch["candidates"]
    latency["index_prepare_ms_total"] = index_prepare_ms
    latency["query_encode_ms_total"] = query_encode_total_ms
    latency["query_load_ms_total"] = query_load_ms
    latency["retrieval_ms_total"] = retrieval_total_ms
    latency["e2e_query_ms_p50"] = (
        None
        if args.skip_latency
        else latency["query_encode_ms_p50"] + latency["retrieval_ms_p50_topk100"]
    )

    index_size_fp16 = retriever.index_size_bytes(index, "fp16")
    index_size_pq = retriever.index_size_bytes(index, "pq") if retrieval_system["name"] == "li-fastplaid" else None
    index_size_hnsw = retriever.index_size_bytes(index, "hnsw") if retrieval_system["name"] == "dense-hnsw" else None
    index_size_backend = (
        retriever.index_size_bytes(index, "backend")
        if retrieval_system["name"] in {"dense-opq-ivfpq", "dense-rabitq", "dense-scann"}
        else None
    )
    index_size_binary_rerank = (
        retriever.index_size_bytes(index, "binary_rerank")
        if retrieval_system["name"] == "dense-binary-rerank"
        else None
    )
    index_size_muvera = (
        retriever.index_size_bytes(index, "muvera")
        if retrieval_system["name"] == "li-muvera"
        else None
    )
    storage = storage_summary(index_size_fp16, len(dataset.documents), index_size_pq=index_size_pq)
    if index_size_hnsw is not None:
        storage["index_bytes_hnsw"] = index_size_hnsw
    if index_size_backend is not None:
        storage["index_bytes_backend"] = index_size_backend
    if index_size_binary_rerank is not None:
        storage["index_bytes_binary_rerank"] = index_size_binary_rerank
    if index_size_muvera is not None:
        storage["index_bytes_muvera"] = index_size_muvera
    docs_indexed_per_second = (
        len(dataset.documents) / (corpus_encode_ms / 1000.0) if corpus_encode_ms > 0 else None
    )
    latency["docs_indexed_per_second"] = docs_indexed_per_second
    latency["max_seq_length"] = args.max_seq_length
    latency["precision"] = args.precision

    rankings_path = output_path.with_suffix(".rankings.json")
    result = {
        "schema_version": "0.3",
        "model": {
            "id": args.model,
            "family": family,
            "revision": None,
            "params_m": None,
            "license": None,
            "search_mode": retrieval_system["search_mode"],
            "truncate_dim": args.truncate_dim,
            "max_seq_length": args.max_seq_length,
            "precision": args.precision,
        },
        "retrieval_system": retrieval_system,
        "dataset": {
            "benchmark": dataset.benchmark,
            "id": dataset.hf_id,
            "name": dataset.name,
            "split": dataset.split,
            "query_limit": args.limit,
            "corpus_rows": len(dataset.documents),
            "query_rows": len(dataset.queries),
            "qrels_rows": dataset.qrels_rows,
        },
        "quality": quality,
        "latency": latency,
        "storage": storage,
        "cost": cost_summary(
            latency["e2e_query_ms_p50"],
            docs_indexed_per_second,
            hourly_rate_usd=args.gpu_hourly_usd,
            hardware_label=args.gpu_price_label,
            pricing_source=args.gpu_pricing_source,
        ),
        "benchmark_protocol": {
            "serving_latency_batch_sizes": args.latency_batch_sizes,
            "max_feasible_batch_candidates": args.max_feasible_batch_candidates,
            "latency_samples": args.latency_samples,
            "latency_sample_size": args.latency_sample_size,
            "latency_sample_seed": args.latency_sample_seed,
            "latency_warmup": args.latency_warmup,
            "latency_skipped": args.skip_latency,
            "cache_only": args.cache_only,
            "corpus_encode_batch": args.corpus_encode_batch or args.batch_size,
            "precision": args.precision,
            "max_seq_length": args.max_seq_length,
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "gpu_price_label": args.gpu_price_label,
            "gpu_pricing_source": args.gpu_pricing_source,
            "index_residency": "warm in process memory/page cache before timed retrieval",
            "note": (
                "Corpus indexing throughput and online serving latency are measured separately. "
                "Fixed serving batches report OOM rather than silently falling back. "
                "Batch>1 per-query latency is total batch time divided by batch size."
                if not args.skip_latency
                else "Latency was intentionally skipped for this parallel quality/storage pass."
            ),
        },
        "system": system_summary(),
        "index": index_metadata(retriever, index),
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware_label": hardware_label(),
            "git_sha": git_sha(),
            "rankings_path": str(rankings_path),
            "cache": {
                "corpus": corpus_cache.__dict__,
                "queries": query_cache.__dict__,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rankings_sidecar(rankings_path, output_path, result, dataset, rankings, args.k)
    print(f"Wrote {output_path}")
    print(json.dumps({"quality": quality, "latency": latency, "storage": storage}, indent=2))


def load_dataset(args: argparse.Namespace):
    if args.benchmark == "beir":
        return load_beir_dataset(args.dataset, split=args.split, limit=args.limit)
    if args.benchmark == "bright":
        return load_bright_dataset(args.dataset, limit=args.limit)
    if args.benchmark == "limit":
        return load_limit_dataset(split=args.split, limit=args.limit)
    raise ValueError(f"Unsupported benchmark: {args.benchmark}")


def write_rankings_sidecar(
    rankings_path: Path,
    output_path: Path,
    result: dict[str, Any],
    dataset,
    rankings,
    k: int,
) -> None:
    payload = {
        "schema_version": "0.1",
        "result_schema_version": result["schema_version"],
        "result_path": str(output_path),
        "model": result["model"],
        "retrieval_system": result["retrieval_system"],
        "dataset": result["dataset"],
        "k": k,
        "queries": [
            {
                "query_id": query.query_id,
                "qrels": dataset.qrels.get(query.query_id, {}),
                "hits": [
                    {
                        "rank": rank,
                        "doc_id": hit.doc_id,
                        "score": hit.score,
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
            for query, hits in zip(dataset.queries, rankings)
        ],
    }
    rankings_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")


def should_skip_opq_ivfpq_small_corpus(retrieval_system: dict[str, Any], dataset) -> bool:
    return retrieval_system["name"] == "dense-opq-ivfpq" and len(dataset.documents) < OPQ_IVFPQ_MIN_DOCS


def should_skip_high_dim_opq_compute(args: argparse.Namespace, retrieval_system: dict[str, Any]) -> bool:
    return (
        retrieval_system["name"] == "dense-opq-ivfpq"
        and args.model == HIGH_DIM_OPQ_SKIP_MODEL
        and args.truncate_dim is None
    )


def write_not_applicable_small_corpus(args, retrieval_system: dict[str, Any], dataset, output_path: Path) -> None:
    reason = (
        "OPQ-IVF-PQ is skipped for corpora with fewer than "
        f"{OPQ_IVFPQ_MIN_DOCS} documents to avoid under-trained PQ codebooks."
    )
    result = {
        "schema_version": "0.3",
        "status": "not_applicable_small_corpus",
        "skip_reason": reason,
        "model": {
            "id": args.model,
            "family": retrieval_system["family"],
            "revision": None,
            "params_m": None,
            "license": None,
            "search_mode": retrieval_system["search_mode"],
            "truncate_dim": args.truncate_dim,
            "max_seq_length": args.max_seq_length,
            "precision": args.precision,
        },
        "retrieval_system": retrieval_system,
        "dataset": {
            "benchmark": dataset.benchmark,
            "id": dataset.hf_id,
            "name": dataset.name,
            "split": dataset.split,
            "query_limit": args.limit,
            "corpus_rows": len(dataset.documents),
            "query_rows": len(dataset.queries),
            "qrels_rows": dataset.qrels_rows,
        },
        "quality": {
            "ndcg_at_10": None,
            "recall_at_100": None,
            "mrr_at_10": None,
            "map_at_100": None,
        },
        "latency": {
            "e2e_query_ms_p50": None,
            "skip_reason": reason,
        },
        "storage": {
            "index_bytes_fp16": None,
            "index_bytes_int8": None,
            "index_bytes_pq": None,
            "index_bytes_per_doc": None,
        },
        "cost": {
            "cost_per_million_queries_usd": None,
            "cost_per_million_docs_indexed_usd": None,
            "hardware_hourly_usd": args.gpu_hourly_usd,
            "hardware_label": args.gpu_price_label,
            "pricing_source": args.gpu_pricing_source,
        },
        "benchmark_protocol": {
            "serving_latency_batch_sizes": args.latency_batch_sizes,
            "max_feasible_batch_candidates": args.max_feasible_batch_candidates,
            "latency_samples": args.latency_samples,
            "latency_sample_size": args.latency_sample_size,
            "latency_sample_seed": args.latency_sample_seed,
            "latency_warmup": args.latency_warmup,
            "latency_skipped": args.skip_latency,
            "cache_only": args.cache_only,
            "corpus_encode_batch": args.corpus_encode_batch or args.batch_size,
            "precision": args.precision,
            "max_seq_length": args.max_seq_length,
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "gpu_price_label": args.gpu_price_label,
            "gpu_pricing_source": args.gpu_pricing_source,
            "index_residency": None,
            "note": reason,
        },
        "system": system_summary(),
        "index": {
            "search_mode": retrieval_system["search_mode"],
            "backend_index_status": "not_applicable_small_corpus",
            "backend_index_spec": None,
            "min_docs_required": OPQ_IVFPQ_MIN_DOCS,
        },
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware_label": hardware_label(),
            "git_sha": git_sha(),
            "cache": {
                "corpus": cache_info(False, False, None).__dict__,
                "queries": cache_info(False, False, None).__dict__,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps({"status": result["status"], "skip_reason": reason}, indent=2))


def write_skipped_high_dim_opq_compute(args, retrieval_system: dict[str, Any], dataset, output_path: Path) -> None:
    reason = (
        "Qwen3-Embedding-8B full-dimension OPQ-IVF-PQ is skipped because OPQ construction "
        "is compute-dominated at 4096 dimensions; Qwen8 flat, HNSW, RaBitQ, and "
        "binary-rerank remain in scope, and Qwen8 OPQ is still evaluated for lower "
        "Matryoshka dimensions."
    )
    result = {
        "schema_version": "0.3",
        "status": "skipped_high_dim_opq_compute",
        "skip_reason": reason,
        "model": {
            "id": args.model,
            "family": retrieval_system["family"],
            "revision": None,
            "params_m": None,
            "license": None,
            "search_mode": retrieval_system["search_mode"],
            "truncate_dim": args.truncate_dim,
            "max_seq_length": args.max_seq_length,
            "precision": args.precision,
        },
        "retrieval_system": retrieval_system,
        "dataset": {
            "benchmark": dataset.benchmark,
            "id": dataset.hf_id,
            "name": dataset.name,
            "split": dataset.split,
            "query_limit": args.limit,
            "corpus_rows": len(dataset.documents),
            "query_rows": len(dataset.queries),
            "qrels_rows": dataset.qrels_rows,
        },
        "quality": {
            "ndcg_at_10": None,
            "recall_at_100": None,
            "mrr_at_10": None,
            "map_at_100": None,
        },
        "latency": {
            "e2e_query_ms_p50": None,
            "skip_reason": reason,
        },
        "storage": {
            "index_bytes_fp16": None,
            "index_bytes_int8": None,
            "index_bytes_pq": None,
            "index_bytes_per_doc": None,
        },
        "cost": {
            "cost_per_million_queries_usd": None,
            "cost_per_million_docs_indexed_usd": None,
            "hardware_hourly_usd": args.gpu_hourly_usd,
            "hardware_label": args.gpu_price_label,
            "pricing_source": args.gpu_pricing_source,
        },
        "benchmark_protocol": {
            "serving_latency_batch_sizes": args.latency_batch_sizes,
            "max_feasible_batch_candidates": args.max_feasible_batch_candidates,
            "latency_samples": args.latency_samples,
            "latency_sample_size": args.latency_sample_size,
            "latency_sample_seed": args.latency_sample_seed,
            "latency_warmup": args.latency_warmup,
            "latency_skipped": True,
            "cache_only": args.cache_only,
            "corpus_encode_batch": args.corpus_encode_batch or args.batch_size,
            "precision": args.precision,
            "max_seq_length": args.max_seq_length,
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "gpu_price_label": args.gpu_price_label,
            "gpu_pricing_source": args.gpu_pricing_source,
            "index_residency": None,
            "note": reason,
        },
        "system": system_summary(),
        "index": {
            "search_mode": retrieval_system["search_mode"],
            "backend_index_status": "skipped_high_dim_opq_compute",
            "backend_index_spec": None,
            "skipped_model": HIGH_DIM_OPQ_SKIP_MODEL,
        },
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware_label": hardware_label(),
            "git_sha": git_sha(),
            "cache": {
                "corpus": cache_info(False, False, None).__dict__,
                "queries": cache_info(False, False, None).__dict__,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps({"status": result["status"], "skip_reason": reason}, indent=2))


def resolve_retrieval_system(args: argparse.Namespace) -> dict[str, Any]:
    if args.system is None:
        family = infer_family(args.model, args.family)
        if family == "dense":
            args.system = "dense-flat"
        elif family == "late_interaction":
            args.system = "li-fastplaid" if args.li_search == "plaid" else "li-exact"
        else:
            args.system = "hybrid-rrf"
    if args.system not in SYSTEMS:
        raise ValueError(f"Unsupported retrieval system: {args.system}")

    retrieval_system = dict(SYSTEMS[args.system])
    model_family = infer_family(args.model, args.family)
    if model_family != retrieval_system["family"]:
        raise ValueError(
            f"System {args.system} expects {retrieval_system['family']} model, "
            f"but {args.model} was inferred as {model_family}."
        )

    retrieval_system["name"] = args.system
    retrieval_system["params"] = retrieval_system_params(args, args.system)
    return retrieval_system


def retrieval_system_params(args: argparse.Namespace, system_name: str) -> dict[str, Any]:
    if system_name == "dense-hnsw":
        return {
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "ef_search": args.hnsw_ef_search,
        }
    if system_name in {"dense-opq-ivfpq", "dense-rabitq"}:
        return {
            "nlist": args.faiss_nlist or "auto",
            "pq_m": args.faiss_pq_m if system_name == "dense-opq-ivfpq" else None,
        }
    if system_name in {"dense-scann", "dense-binary-rerank"}:
        return {
            "candidates": args.binary_candidates,
        }
    if system_name == "li-fastplaid":
        return {
            "nbits": args.plaid_nbits,
            "n_ivf_probe": args.plaid_n_ivf_probe,
            "n_full_scores": args.plaid_n_full_scores,
            "kmeans_niters": args.plaid_kmeans_niters,
            "use_fast": True,
        }
    if system_name == "li-muvera":
        return {
            "k_sim": args.muvera_k_sim,
            "dim_proj": args.muvera_dim_proj,
            "r_reps": args.muvera_r_reps,
            "candidates": args.muvera_candidates,
        }
    return {}


def build_retriever_from_args(args: argparse.Namespace, retrieval_system: dict[str, Any], dataset):
    family = retrieval_system["family"]
    if family == "dense":
        return DenseRetriever(
            model_id=args.model,
            batch_size=initial_corpus_encode_batch(args, default=64),
            search_mode=retrieval_system["search_mode"],
            index_folder=str(Path(args.index_dir) / dataset.benchmark / dataset.name),
            index_name=dense_index_name(args),
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            faiss_nlist=args.faiss_nlist,
            faiss_pq_m=args.faiss_pq_m,
            binary_candidates=args.binary_candidates,
            truncate_dim=args.truncate_dim,
            max_seq_length=args.max_seq_length,
            precision=args.precision,
            load_model=not args.cache_only,
        )
    if family == "late_interaction":
        return LateInteractionRetriever(
            model_id=args.model,
            batch_size=initial_corpus_encode_batch(args, default=16),
            search_mode=retrieval_system["search_mode"],
            index_folder=str(Path(args.index_dir) / dataset.benchmark / dataset.name),
            index_name=li_index_name(args),
            plaid_nbits=args.plaid_nbits,
            plaid_n_ivf_probe=args.plaid_n_ivf_probe,
            plaid_n_full_scores=args.plaid_n_full_scores,
            plaid_kmeans_niters=args.plaid_kmeans_niters,
            plaid_search_batch_size=args.plaid_search_batch_size,
            muvera_k_sim=args.muvera_k_sim,
            muvera_dim_proj=args.muvera_dim_proj,
            muvera_r_reps=args.muvera_r_reps,
            muvera_candidates=args.muvera_candidates,
            load_model=not args.cache_only,
        )
    raise NotImplementedError(
        "Hybrid retrieval is implemented by scripts/run_hybrid_supplement.py, "
        "not the single-system evaluator."
    )


def initial_corpus_encode_batch(args: argparse.Namespace, default: int) -> int:
    if args.corpus_encode_batch and args.corpus_encode_batch != "auto-max-feasible":
        return int(args.corpus_encode_batch)
    return args.batch_size or default


def resolve_corpus_encode_batch(args: argparse.Namespace, retriever, dataset) -> dict[str, Any]:
    requested = args.corpus_encode_batch
    candidates = parse_int_list(args.max_feasible_batch_candidates)
    if requested == "auto-max-feasible":
        batch_size = find_max_feasible_corpus_batch(retriever, dataset.documents, candidates)
        return {
            "mode": "auto-max-feasible",
            "batch_size": batch_size,
            "candidates": candidates,
        }
    batch_size = initial_corpus_encode_batch(args, default=retriever.batch_size)
    return {
        "mode": "fixed" if requested or args.batch_size else "default",
        "batch_size": batch_size,
        "candidates": candidates,
    }


def find_max_feasible_corpus_batch(retriever, documents, candidates: list[int]) -> int:
    if not documents:
        return 1
    original_batch_size = retriever.batch_size
    max_fit = 1
    try:
        for candidate in candidates:
            retriever.batch_size = candidate
            batch_documents = [documents[i % len(documents)] for i in range(candidate)]
            try:
                retriever.encode_corpus(batch_documents)
                max_fit = candidate
            except Exception as exc:
                if is_oom_error(exc):
                    clear_accelerator_cache()
                    break
                raise
    finally:
        retriever.batch_size = original_batch_size
    return max_fit


def prepare_index(retriever, index):
    if hasattr(retriever, "prepare_index"):
        return retriever.prepare_index(index)
    return index


def index_metadata(retriever, index) -> dict[str, Any]:
    if hasattr(retriever, "index_metadata"):
        return retriever.index_metadata(index)
    return {"search_mode": "flat"}


def dense_index_name(args: argparse.Namespace) -> str:
    base = sanitize_model_id(args.model)
    dim = f".dim{args.truncate_dim}" if args.truncate_dim else ""
    seq = f".seq{args.max_seq_length}" if args.max_seq_length else ""
    if args.system == "dense-hnsw":
        return (
            f"{base}{dim}{seq}.hnsw"
            f".m{args.hnsw_m}"
            f".efc{args.hnsw_ef_construction}"
            f".ef{args.hnsw_ef_search}"
        )
    if args.system == "dense-opq-ivfpq":
        return f"{base}{dim}{seq}.opqivfpq.nlist{args.faiss_nlist or 'auto'}.m{args.faiss_pq_m}"
    if args.system == "dense-rabitq":
        return f"{base}{dim}{seq}.rabitq.nlist{args.faiss_nlist or 'auto'}"
    if args.system == "dense-scann":
        return f"{base}{dim}{seq}.scann.candidates{args.binary_candidates}"
    if args.system == "dense-binary-rerank":
        return f"{base}{dim}{seq}.binary-rerank.candidates{args.binary_candidates}"
    return f"{base}{dim}{seq}.flat"


def li_index_name(args: argparse.Namespace) -> str:
    base = sanitize_model_id(args.model)
    if args.system != "li-fastplaid":
        if args.system == "li-muvera":
            return (
                f"{base}.muvera"
                f".ksim{args.muvera_k_sim}"
                f".proj{args.muvera_dim_proj}"
                f".reps{args.muvera_r_reps}"
                f".cand{args.muvera_candidates}"
            )
        return base
    return (
        f"{base}.fastplaid"
        f".nbits{args.plaid_nbits}"
        f".probe{args.plaid_n_ivf_probe}"
        f".full{args.plaid_n_full_scores}"
        f".kmeans{args.plaid_kmeans_niters}"
    )


def result_suffix(args: argparse.Namespace, retrieval_system: dict[str, Any]) -> str:
    if retrieval_system["family"] == "dense":
        base = sanitize_model_id(args.model)
        return dense_index_name(args).removeprefix(base)
    if retrieval_system["name"] not in {"li-fastplaid", "li-muvera"}:
        return ""
    base = sanitize_model_id(args.model)
    return li_index_name(args).removeprefix(base)


def load_or_encode_corpus(args, family: str, retriever, dataset):
    path = cache_path(
        Path(args.cache_dir),
        kind="corpus",
        family=family,
        model_id=args.model,
        dataset_id=dataset.hf_id,
        dataset_name=dataset.name,
        split="corpus",
        representation_key=representation_key(args, family),
    )
    if not args.no_cache and path.exists():
        print(f"Loading cached corpus embeddings: {path}")
        value, load_ms = timed_call(lambda: load_cached(path, family, "corpus"))
        return value, 0.0, load_ms, cache_info(True, True, path)
    if not args.no_cache:
        fallback = load_truncated_dense_cache_from_full(args, family, dataset, "corpus", path)
        if fallback is not None:
            return fallback

    print("Encoding corpus")
    value, encode_ms = timed_call(lambda: retriever.encode_corpus(dataset.documents))
    if not args.no_cache:
        print(f"Saving corpus embeddings: {path}")
        save_cached(path, family, "corpus", value)
    return value, encode_ms, 0.0, cache_info(not args.no_cache, False, path if not args.no_cache else None)


def load_or_encode_queries(args, family: str, retriever, dataset):
    path = cache_path(
        Path(args.cache_dir),
        kind="queries",
        family=family,
        model_id=args.model,
        dataset_id=dataset.hf_id,
        dataset_name=dataset.name,
        split=dataset.split,
        query_limit=args.limit,
        representation_key=representation_key(args, family),
    )
    if not args.no_cache and path.exists():
        print(f"Loading cached query embeddings: {path}")
        value, load_ms = timed_call(lambda: load_cached(path, family, "queries"))
        return value, 0.0, load_ms, cache_info(True, True, path)
    if not args.no_cache:
        fallback = load_truncated_dense_cache_from_full(args, family, dataset, "queries", path)
        if fallback is not None:
            return fallback

    print("Encoding queries")
    value, encode_ms = timed_call(lambda: retriever.encode_queries(dataset.queries))
    if not args.no_cache:
        print(f"Saving query embeddings: {path}")
        save_cached(path, family, "queries", value)
    return value, encode_ms, 0.0, cache_info(not args.no_cache, False, path if not args.no_cache else None)


def load_truncated_dense_cache_from_full(args, family: str, dataset, kind: str, target_path: Path):
    if family != "dense" or args.truncate_dim is None:
        return None
    full_path = cache_path(
        Path(args.cache_dir),
        kind=kind,
        family=family,
        model_id=args.model,
        dataset_id=dataset.hf_id,
        dataset_name=dataset.name,
        split="corpus" if kind == "corpus" else dataset.split,
        query_limit=None if kind == "corpus" else args.limit,
        representation_key=dense_full_representation_key(args),
    )
    if not full_path.exists():
        return None

    def load_truncate_and_save():
        full_value = load_cached(full_path, family, kind)
        truncated = truncate_dense_cache_value(full_value, args.truncate_dim, kind)
        save_cached(target_path, family, kind, truncated)
        return truncated

    print(f"Deriving truncated {kind} embeddings from full-dim cache: {full_path}")
    value, load_ms = timed_call(load_truncate_and_save)
    return value, 0.0, load_ms, cache_info(True, True, target_path)


def dense_full_representation_key(args: argparse.Namespace) -> str | None:
    if args.max_seq_length:
        return f"max_seq_length={args.max_seq_length}"
    return None


def truncate_dense_cache_value(value, truncate_dim: int, kind: str):
    embeddings = value.embeddings
    if embeddings.shape[1] < truncate_dim:
        raise ValueError(
            f"Cannot truncate {kind} embeddings with dim {embeddings.shape[1]} to {truncate_dim}."
        )
    truncated = np.ascontiguousarray(embeddings[:, :truncate_dim].astype("float32"))
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    nonzero = norms.squeeze(axis=1) > 0
    truncated[nonzero] = truncated[nonzero] / norms[nonzero]
    if kind == "corpus":
        return DenseIndex(
            doc_ids=value.doc_ids,
            embeddings=truncated,
            faiss_index=_build_faiss_index(truncated),
        )
    if kind == "queries":
        return DenseQueryRepr(query_ids=value.query_ids, embeddings=truncated)
    raise ValueError(f"Unsupported dense cache kind: {kind}")


def measure_latency(retriever, queries, query_repr, index, k: int, args: argparse.Namespace) -> dict[str, Any]:
    latency_queries, latency_query_repr, latency_query_ids = fixed_latency_sample(
        queries,
        query_repr,
        sample_size=args.latency_sample_size,
        seed=args.latency_sample_seed,
    )
    batch_tokens = [token.strip() for token in args.latency_batch_sizes.split(",") if token.strip()]
    fixed_batches = [int(token) for token in batch_tokens if token != "max"]
    max_candidates = parse_int_list(args.max_feasible_batch_candidates)
    latency: dict[str, Any] = {
        "latency_query_ids": latency_query_ids,
        "latency_sample_size": len(latency_query_ids),
        "latency_sample_seed": args.latency_sample_seed,
    }

    measured: dict[int, dict[str, Any]] = {}
    for batch_size in fixed_batches:
        result = measure_serving_batch(
            retriever,
            latency_queries,
            latency_query_repr,
            index,
            k,
            batch_size=batch_size,
            samples=args.latency_samples,
            warmup=args.latency_warmup,
        )
        measured[batch_size] = result
        latency.update(flatten_serving_batch_result(batch_size, result))

    if "max" in batch_tokens:
        max_batch = find_max_feasible_query_batch(retriever, latency_queries, max_candidates)
        result = measured.get(max_batch)
        if result is None:
            result = measure_serving_batch(
                retriever,
                latency_queries,
                latency_query_repr,
                index,
                k,
                batch_size=max_batch,
                samples=args.latency_samples,
                warmup=args.latency_warmup,
            )
        latency.update(flatten_serving_batch_result("max", result))
        latency["serving_max_batch_size"] = max_batch
        latency["serving_max_batch_candidates"] = max_candidates

    batch1 = measured.get(1) or measure_serving_batch(
        retriever,
        latency_queries,
        latency_query_repr,
        index,
        k,
        batch_size=1,
        samples=args.latency_samples,
        warmup=args.latency_warmup,
    )
    batch32 = measured.get(32)
    latency["query_encode_ms_p50"] = batch1.get("query_encode_ms_p50")
    latency["query_encode_ms_p99"] = batch1.get("query_encode_ms_p99")
    latency["query_encode_ms_p50_batch32"] = (
        batch32.get("query_encode_ms_per_query_p50")
        if batch32 and batch32.get("status") == "ok"
        else None
    )
    latency["retrieval_ms_p50_topk100"] = batch1.get("retrieval_ms_p50_topk100")
    latency["retrieval_ms_p99_topk100"] = batch1.get("retrieval_ms_p99_topk100")
    return latency


def skipped_latency_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": "parallel_quality_storage_pass",
        "latency_query_ids": [],
        "latency_sample_size": 0,
        "latency_sample_seed": args.latency_sample_seed,
        "query_encode_ms_p50": None,
        "query_encode_ms_p99": None,
        "query_encode_ms_p50_batch32": None,
        "retrieval_ms_p50_topk100": None,
        "retrieval_ms_p99_topk100": None,
    }


def fixed_latency_sample(queries, query_repr, *, sample_size: int, seed: int):
    if not queries:
        return [], query_repr, []
    sample_count = max(1, min(sample_size, len(queries)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(queries)), sample_count))
    sampled_queries = [queries[i] for i in indices]
    sampled_query_repr = query_repr_from_indices(query_repr, indices)
    sampled_query_ids = [query.query_id for query in sampled_queries]
    return sampled_queries, sampled_query_repr, sampled_query_ids


def query_repr_from_indices(query_repr: Any, indices: list[int]):
    cls = type(query_repr)
    query_ids = [query_repr.query_ids[i] for i in indices]
    if hasattr(query_repr.embeddings, "shape"):
        import numpy as np

        embeddings = query_repr.embeddings[np.asarray(indices, dtype=np.int64)]
    else:
        embeddings = [query_repr.embeddings[i] for i in indices]
    return cls(query_ids=query_ids, embeddings=embeddings)


def measure_serving_batch(
    retriever,
    queries,
    query_repr,
    index,
    k: int,
    *,
    batch_size: int,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    if not queries:
        return {"status": "no_queries", "batch_size": batch_size}

    query_encode_samples: list[float] = []
    tokenize_samples: list[float] = []
    forward_samples: list[float] = []
    retrieval_samples: list[float] = []
    sample_count = max(1, min(samples, len(queries)))
    warmup_count = max(0, min(warmup, sample_count))

    for i in range(warmup_count):
        ok = try_encode_query_batch(retriever, repeat_query_batch(queries, batch_size, i), batch_size)
        if ok != "ok":
            return {"status": ok, "batch_size": batch_size}

    for i in range(sample_count):
        batch = repeat_query_batch(queries, batch_size, i)
        components, ms, status = timed_encode_query_batch(retriever, batch, batch_size)
        if status != "ok":
            return {"status": status, "batch_size": batch_size}
        query_encode_samples.append(ms)
        if components and components.get("tokenize_ms") is not None:
            tokenize_samples.append(float(components["tokenize_ms"]))
        if components and components.get("forward_ms") is not None:
            forward_samples.append(float(components["forward_ms"]))

    for i in range(sample_count):
        batch_query_repr = batch_query_repr_from_cache(query_repr, batch_size, i * batch_size)
        _, ms, status = timed_search_batch(retriever, batch_query_repr, index, k)
        if status != "ok":
            return {"status": status, "batch_size": batch_size}
        retrieval_samples.append(ms)

    query_summary = summarize_ms(query_encode_samples)
    tokenize_summary = summarize_ms(tokenize_samples) if tokenize_samples else None
    forward_summary = summarize_ms(forward_samples) if forward_samples else None
    retrieval_summary = summarize_ms(retrieval_samples)
    return {
        "status": "ok",
        "batch_size": batch_size,
        "tokenize_ms_p50": tokenize_summary["p50"] if tokenize_summary else None,
        "tokenize_ms_p99": tokenize_summary["p99"] if tokenize_summary else None,
        "tokenize_ms_per_query_p50": tokenize_summary["p50"] / batch_size
        if tokenize_summary
        else None,
        "tokenize_ms_per_query_p99": tokenize_summary["p99"] / batch_size
        if tokenize_summary
        else None,
        "query_forward_ms_p50": forward_summary["p50"] if forward_summary else None,
        "query_forward_ms_p99": forward_summary["p99"] if forward_summary else None,
        "query_forward_ms_per_query_p50": forward_summary["p50"] / batch_size
        if forward_summary
        else None,
        "query_forward_ms_per_query_p99": forward_summary["p99"] / batch_size
        if forward_summary
        else None,
        "query_encode_ms_p50": query_summary["p50"],
        "query_encode_ms_p99": query_summary["p99"],
        "query_encode_ms_per_query_p50": query_summary["p50"] / batch_size,
        "query_encode_ms_per_query_p99": query_summary["p99"] / batch_size,
        "retrieval_ms_p50_topk100": retrieval_summary["p50"],
        "retrieval_ms_p99_topk100": retrieval_summary["p99"],
        "retrieval_ms_per_query_p50_topk100": retrieval_summary["p50"] / batch_size,
        "retrieval_ms_per_query_p99_topk100": retrieval_summary["p99"] / batch_size,
        "e2e_ms_p50": query_summary["p50"] + retrieval_summary["p50"],
        "e2e_ms_p99": query_summary["p99"] + retrieval_summary["p99"],
        "e2e_ms_per_query_p50": (query_summary["p50"] + retrieval_summary["p50"]) / batch_size,
        "e2e_ms_per_query_p99": (query_summary["p99"] + retrieval_summary["p99"]) / batch_size,
    }


def flatten_serving_batch_result(label: int | str, result: dict[str, Any]) -> dict[str, Any]:
    prefix = f"serving_batch_{label}"
    return {f"{prefix}_{key}": value for key, value in result.items()}


def find_max_feasible_query_batch(retriever, queries, candidates: list[int]) -> int:
    if not queries:
        return 1
    max_fit = 1
    for candidate in candidates:
        status = try_encode_query_batch(retriever, repeat_query_batch(queries, candidate, 0), candidate)
        if status == "ok":
            max_fit = candidate
            continue
        if status == "oom":
            break
        raise RuntimeError(f"Unexpected status while probing max feasible batch: {status}")
    return max_fit


def try_encode_query_batch(retriever, queries, batch_size: int) -> str:
    _, _, status = timed_encode_query_batch(retriever, queries, batch_size)
    return status


def timed_encode_query_batch(retriever, queries, batch_size: int):
    original_batch_size = retriever.batch_size
    retriever.batch_size = batch_size
    try:
        if hasattr(retriever, "measure_query_encode_components"):
            components = retriever.measure_query_encode_components(queries)
            return components, float(components["total_ms"]), "ok"
        _, ms = timed_call(lambda: retriever.encode_queries(queries))
        return {"tokenize_ms": None, "forward_ms": None, "total_ms": ms}, ms, "ok"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if is_oom_error(exc):
            clear_accelerator_cache()
            return None, 0.0, "oom"
        raise
    finally:
        retriever.batch_size = original_batch_size


def timed_search_batch(retriever, query_repr, index, k: int):
    try:
        value, ms = timed_call(lambda: retriever.search(query_repr, index, k))
        return value, ms, "ok"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if is_oom_error(exc):
            clear_accelerator_cache()
            return None, 0.0, "oom"
        raise


def _slice_query_repr(query_repr: Any, index: int):
    cls = type(query_repr)
    embeddings = query_repr.embeddings[index : index + 1]
    query_ids = query_repr.query_ids[index : index + 1]
    return cls(query_ids=query_ids, embeddings=embeddings)


def repeat_query_batch(queries, batch_size: int, offset: int):
    return [queries[(offset + i) % len(queries)] for i in range(batch_size)]


def batch_query_repr_from_cache(query_repr: Any, batch_size: int, offset: int):
    cls = type(query_repr)
    total = len(query_repr.query_ids)
    indices = [(offset + i) % total for i in range(batch_size)]
    query_ids = [query_repr.query_ids[i] for i in indices]
    if hasattr(query_repr.embeddings, "shape"):
        import numpy as np

        embeddings = query_repr.embeddings[np.asarray(indices, dtype=np.int64)]
    else:
        embeddings = [query_repr.embeddings[i] for i in indices]
    return cls(query_ids=query_ids, embeddings=embeddings)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        return [1]
    return sorted(set(parsed))


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    return "outofmemory" in exc_name or "out of memory" in message or "cuda oom" in message


def clear_accelerator_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def result_path(
    output_dir: Path,
    benchmark: str,
    dataset: str,
    retrieval_system: str,
    model_id: str,
    limit: int | None,
    result_suffix: str,
) -> Path:
    if limit is not None:
        return (
            output_dir
            / "smoke"
            / benchmark
            / dataset
            / retrieval_system
            / f"{sanitize_model_id(model_id)}{result_suffix}.limit{limit}.json"
        )
    return output_dir / benchmark / dataset / retrieval_system / f"{sanitize_model_id(model_id)}{result_suffix}.json"


def current_result_exists(path: Path, *, require_latency: bool = True) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except Exception:
        return False
    if not isinstance(result, dict) or result.get("schema_version") != "0.3":
        return False
    if not require_latency:
        return True
    latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
    return latency.get("status") != "skipped" and latency.get("e2e_query_ms_p50") is not None


def representation_key(args: argparse.Namespace, family: str) -> str | None:
    if family == "dense" and args.truncate_dim:
        parts = [f"truncate_dim={args.truncate_dim}"]
        if args.max_seq_length:
            parts.append(f"max_seq_length={args.max_seq_length}")
        return ",".join(parts)
    if family == "dense" and args.max_seq_length:
        return f"max_seq_length={args.max_seq_length}"
    return None


def system_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "peak_index_ram_gb": None,
        "peak_serving_ram_gb": None,
    }
    try:
        import psutil

        summary["ram_total_gb"] = psutil.virtual_memory().total / 1e9
    except Exception:
        summary["ram_total_gb"] = None
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        summary["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        pass
    return summary


def hardware_label() -> str:
    summary = system_summary()
    if summary.get("cuda_available"):
        return "cuda"
    if summary.get("mps_available"):
        return "local_mps"
    return "cpu"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
