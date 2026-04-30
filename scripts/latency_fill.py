#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval_pareto.benchmarks import load_beir_dataset, load_bright_dataset, load_limit_dataset
from retrieval_pareto.cache import cache_info
from retrieval_pareto.eval import (
    build_retriever_from_args,
    dense_full_representation_key,
    hardware_label,
    index_metadata,
    load_or_encode_corpus,
    load_or_encode_queries,
    measure_latency,
    prepare_index,
    representation_key,
    resolve_retrieval_system,
    system_summary,
)
from retrieval_pareto.metrics.cost import cost_summary
from retrieval_pareto.metrics.latency import timed_call


INTENTIONAL_NO_LATENCY = {
    "not_applicable_small_corpus",
    "skipped_high_dim_opq_compute",
    "latency_oom",
}
DIRECT_FAMILIES = {"dense", "late_interaction"}
MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill or re-measure isolated A100 latency for existing result JSONs."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--manifest", help="CSV manifest from --write-manifest.")
    parser.add_argument("--result", action="append", help="Specific result JSON to fill.")
    parser.add_argument("--write-manifest", help="Write a latency-fill manifest CSV and exit.")
    parser.add_argument("--include-stale", action="store_true", help="Include rows with non-200 samples.")
    parser.add_argument("--latency-batch-sizes", default="1")
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--latency-sample-size", type=int, default=200)
    parser.add_argument("--latency-sample-seed", type=int, default=13)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--max-feasible-batch-candidates", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--gpu-hourly-usd", type=float, default=1.99)
    parser.add_argument("--gpu-price-label", default="Lambda A100 SXM 40GB")
    parser.add_argument("--gpu-pricing-source", default="https://lambda.ai/pricing")
    parser.add_argument("--allow-encode", action="store_true", help="Allow embedding encoding on cache miss.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_manifest:
        rows = discover_rows(args)
        write_manifest(Path(args.write_manifest), rows)
        print(f"Wrote {args.write_manifest} with {len(rows)} rows")
        return

    targets = load_targets(args)
    print(f"Latency-fill targets: {len(targets)}")
    failures: list[tuple[str, str]] = []
    for path in targets:
        target = Path(path)
        try:
            result = json.loads(target.read_text(encoding="utf-8"))
            if not should_fill_latency(result, include_stale=args.include_stale):
                print(f"SKIP already fresh {target}", flush=True)
                continue
            fill_latency_for_result(target, args)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            print(f"FAILED {target}: {exc!r}", flush=True)
            failures.append((str(target), repr(exc)))
    if failures:
        failure_path = Path(args.results_dir).parent / "run_status" / "latency_fill_failures.csv"
        write_failures(failure_path, failures)
        raise SystemExit(f"Latency fill completed with {len(failures)} failure(s); see {failure_path}")


def load_targets(args: argparse.Namespace) -> list[str]:
    if args.result:
        return args.result
    if args.manifest:
        with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
            return [row["path"] for row in csv.DictReader(handle)]
    return [str(path) for path in discover_rows(args)]


def discover_rows(args: argparse.Namespace) -> list[Path]:
    rows: list[Path] = []
    for path in sorted(Path(args.results_dir).rglob("*.json")):
        if path.name.endswith(".rankings.json"):
            continue
        if "smoke" in path.parts or "sweeps" in path.parts:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not should_fill_latency(result, include_stale=args.include_stale):
            continue
        rows.append(path)
    return rows


def should_fill_latency(result: dict[str, Any], *, include_stale: bool) -> bool:
    if result.get("schema_version") != "0.3":
        return False
    status = result.get("status")
    if status in INTENTIONAL_NO_LATENCY:
        return False
    retrieval_system = result.get("retrieval_system") or {}
    if retrieval_system.get("family") not in DIRECT_FAMILIES:
        return False
    latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
    if latency.get("e2e_query_ms_p50") is None:
        return True
    if include_stale:
        query_rows = (result.get("dataset") or {}).get("query_rows")
        expected = min(200, int(query_rows)) if query_rows else 200
        return latency.get("latency_sample_size") != expected
    return False


def write_manifest(path: Path, rows: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": row.as_posix()})


def write_failures(path: Path, failures: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "error"])
        writer.writeheader()
        for result_path, error in failures:
            writer.writerow({"path": result_path, "error": error})


def fill_latency_for_result(path: Path, cli_args: argparse.Namespace) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    dataset_info = result["dataset"]
    model_info = result["model"]
    retrieval_system = result["retrieval_system"]
    args = args_from_result(result, cli_args)
    dataset = load_dataset_from_result(dataset_info)
    resolved = resolve_retrieval_system(args)
    if resolved["name"] != retrieval_system["name"]:
        raise RuntimeError(f"System mismatch for {path}: {resolved['name']} != {retrieval_system['name']}")

    if cli_args.dry_run:
        print(f"DRY RUN {path}")
        return

    retriever = build_retriever_for_latency(args, resolved, dataset)
    previous_batch = (result.get("latency") or {}).get("corpus_encode_batch_size") or args.batch_size
    if previous_batch:
        retriever.batch_size = int(previous_batch)

    print(f"\n==> latency fill {dataset.benchmark}/{dataset.name} {resolved['name']} {model_info['id']}")
    index, corpus_encode_ms, corpus_load_ms, corpus_cache = load_or_encode_corpus(
        args, resolved["family"], retriever, dataset
    )
    if not cli_args.allow_encode and corpus_encode_ms > 0:
        raise RuntimeError(f"Corpus cache miss caused encoding for {path}; rerun with --allow-encode if intended.")

    index, index_prepare_ms = timed_call(lambda: prepare_index(retriever, index))
    query_repr, query_encode_total_ms, query_load_ms, query_cache = load_or_encode_queries(
        args, resolved["family"], retriever, dataset
    )
    if not cli_args.allow_encode and query_encode_total_ms > 0:
        raise RuntimeError(f"Query cache miss caused encoding for {path}; rerun with --allow-encode if intended.")

    latency = measure_latency(retriever, dataset.queries, query_repr, index, args.k, args)
    latency["corpus_encode_ms_total"] = corpus_encode_ms
    latency["corpus_load_ms_total"] = corpus_load_ms
    latency["corpus_encode_batch_size"] = previous_batch
    latency["corpus_encode_batch_mode"] = "latency_fill_existing_cache"
    latency["corpus_encode_batch_candidates"] = [
        int(item.strip()) for item in args.max_feasible_batch_candidates.split(",") if item.strip()
    ]
    latency["index_prepare_ms_total"] = index_prepare_ms
    latency["query_encode_ms_total"] = query_encode_total_ms
    latency["query_load_ms_total"] = query_load_ms
    latency["retrieval_ms_total"] = None
    query_p50 = latency.get("query_encode_ms_p50")
    retrieval_p50 = latency.get("retrieval_ms_p50_topk100")
    latency["e2e_query_ms_p50"] = (
        float(query_p50) + float(retrieval_p50)
        if query_p50 is not None and retrieval_p50 is not None
        else None
    )
    latency["docs_indexed_per_second"] = None
    latency["max_seq_length"] = args.max_seq_length
    latency["precision"] = args.precision

    result["latency"] = latency
    if latency["e2e_query_ms_p50"] is None and latency.get("serving_batch_1_status") == "oom":
        result["status"] = "latency_oom"
    elif result.get("status") in {None, "", "quality_only", "latency_skipped"} and latency["e2e_query_ms_p50"] is not None:
        result["status"] = "completed"
    result["cost"] = cost_summary(
        latency["e2e_query_ms_p50"],
        None,
        hourly_rate_usd=args.gpu_hourly_usd,
        hardware_label=args.gpu_price_label,
        pricing_source=args.gpu_pricing_source,
    )
    protocol = result.setdefault("benchmark_protocol", {})
    protocol.update(
        {
            "serving_latency_batch_sizes": args.latency_batch_sizes,
            "latency_samples": args.latency_samples,
            "latency_sample_size": args.latency_sample_size,
            "latency_sample_seed": args.latency_sample_seed,
            "latency_warmup": args.latency_warmup,
            "latency_skipped": False,
            "latency_fill": True,
            "note": "Latency was filled in an isolated serial A100 pass from cached embeddings/indexes; quality/rankings were preserved.",
        }
    )
    result["system"] = system_summary()
    result["index"] = index_metadata(retriever, index)
    run = result.setdefault("run", {})
    run["latency_refilled_at"] = datetime.now(timezone.utc).isoformat()
    run["latency_refill_hardware_label"] = hardware_label()
    run["cache"] = {
        "corpus": corpus_cache.__dict__,
        "queries": query_cache.__dict__,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote latency {latency['e2e_query_ms_p50']} ms -> {path}")


def args_from_result(result: dict[str, Any], cli_args: argparse.Namespace) -> SimpleNamespace:
    model = result["model"]
    dataset = result["dataset"]
    retrieval_system = result["retrieval_system"]
    params = retrieval_system.get("params") or {}
    latency = result.get("latency") or {}
    protocol = result.get("benchmark_protocol") or {}
    return SimpleNamespace(
        model=model["id"],
        dataset=dataset["name"],
        benchmark=dataset["benchmark"],
        split=dataset.get("split") or "test",
        system=retrieval_system["name"],
        family=retrieval_system["family"],
        k=100,
        limit=dataset.get("query_limit"),
        batch_size=None,
        corpus_encode_batch=str(latency.get("corpus_encode_batch_size") or protocol.get("corpus_encode_batch") or ""),
        latency_batch_sizes=cli_args.latency_batch_sizes,
        max_feasible_batch_candidates=cli_args.max_feasible_batch_candidates,
        latency_samples=cli_args.latency_samples,
        latency_sample_size=cli_args.latency_sample_size,
        latency_sample_seed=cli_args.latency_sample_seed,
        latency_warmup=cli_args.latency_warmup,
        cache_dir=cli_args.cache_dir,
        index_dir=cli_args.index_dir,
        li_search="exact",
        hnsw_m=params.get("m", 32),
        hnsw_ef_construction=params.get("ef_construction", 200),
        hnsw_ef_search=params.get("ef_search", 128),
        faiss_nlist=None if params.get("nlist") == "auto" else params.get("nlist"),
        faiss_pq_m=params.get("pq_m") or 32,
        binary_candidates=params.get("candidates", 1000),
        truncate_dim=model.get("truncate_dim"),
        max_seq_length=model.get("max_seq_length"),
        precision=model.get("precision") or "auto",
        gpu_hourly_usd=cli_args.gpu_hourly_usd,
        gpu_pricing_source=cli_args.gpu_pricing_source,
        gpu_price_label=cli_args.gpu_price_label,
        plaid_nbits=params.get("nbits", 8),
        plaid_n_ivf_probe=params.get("n_ivf_probe", 32),
        plaid_n_full_scores=params.get("n_full_scores", 8192),
        plaid_kmeans_niters=params.get("kmeans_niters", 4),
        plaid_search_batch_size=params.get("search_batch_size", 262144),
        muvera_k_sim=params.get("k_sim", 5),
        muvera_dim_proj=params.get("dim_proj", 16),
        muvera_r_reps=params.get("r_reps", 20),
        muvera_candidates=params.get("candidates", 1000),
        no_cache=False,
        cache_only=False,
        skip_latency=False,
        skip_existing=False,
        output_dir=str(Path(result.get("run", {}).get("result_path", "results")).parent),
    )


def build_retriever_for_latency(args: SimpleNamespace, resolved: dict[str, Any], dataset):
    cache_key = (
        resolved["family"],
        args.model,
        args.precision,
        args.max_seq_length,
    )
    cached_model = MODEL_CACHE.get(cache_key)
    original_cache_only = args.cache_only
    if cached_model is not None:
        args.cache_only = True
    try:
        retriever = build_retriever_from_args(args, resolved, dataset)
    finally:
        args.cache_only = original_cache_only
    if cached_model is not None:
        retriever.model = cached_model
    elif hasattr(retriever, "model"):
        MODEL_CACHE[cache_key] = retriever.model
    return retriever


def load_dataset_from_result(dataset: dict[str, Any]):
    benchmark = dataset["benchmark"]
    name = dataset["name"]
    split = dataset.get("split") or "test"
    limit = dataset.get("query_limit")
    if benchmark == "beir":
        return load_beir_dataset(name, split=split, limit=limit)
    if benchmark == "bright":
        return load_bright_dataset(name, limit=limit)
    if benchmark == "limit":
        return load_limit_dataset(split=split, limit=limit)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


if __name__ == "__main__":
    main()
