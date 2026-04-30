#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import pickle
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval_pareto.benchmarks import load_beir_dataset, load_bright_dataset, load_limit_dataset
from retrieval_pareto.benchmarks.beir_loader import (
    DATASET_IDS,
    RetrievalDataset,
    _row_id,
    _row_text,
)
from retrieval_pareto.eval import (
    build_retriever_from_args,
    load_or_encode_corpus,
    load_or_encode_queries,
    prepare_index,
    resolve_retrieval_system,
    result_suffix,
    write_rankings_sidecar,
)
from retrieval_pareto.metrics.cost import cost_summary
from retrieval_pareto.metrics.quality import compute_quality_metrics
from retrieval_pareto.metrics.storage import storage_summary
from retrieval_pareto.model_registry import sanitize_model_id
from retrieval_pareto.types import Hit


DEFAULT_DATASETS = [
    "beir:nfcorpus",
    "beir:scifact",
    "beir:fiqa",
    "beir:arguana",
    "beir:scidocs",
    "beir:trec-covid",
    "bright:biology",
    "bright:economics",
    "bright:psychology",
    "bright:robotics",
    "bright:stackoverflow",
    "bright:leetcode",
    "limit:limit",
]

HYBRID_SYSTEMS = {
    "sparse-bm25": ("sparse",),
    "hybrid-rrf-bm25-dense": ("sparse", "dense"),
    "hybrid-rrf-bm25-li": ("sparse", "li"),
    "hybrid-rrf-bm25-dense-li": ("sparse", "dense", "li"),
    "hybrid-rrf-dense-li": ("dense", "li"),
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Component:
    name: str
    result: dict[str, Any]
    rankings: dict[str, list[Hit]]
    storage_bytes: int | float | None
    e2e_ms_p50: float | None


@dataclass
class Bm25Index:
    doc_ids: list[str]
    doc_len: list[int]
    avgdl: float
    postings: dict[str, list[tuple[int, int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 and RRF hybrid supplement from cached results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--systems", nargs="*", default=sorted(HYBRID_SYSTEMS))
    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dense-system", default="dense-hnsw")
    parser.add_argument("--li-model", default="mixedbread-ai/mxbai-edge-colbert-v0-32m")
    parser.add_argument("--li-system", default="li-fastplaid")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--latency-sample-size", type=int, default=200)
    parser.add_argument("--latency-sample-seed", type=int, default=13)
    parser.add_argument("--gpu-hourly-usd", type=float, default=1.99)
    parser.add_argument("--gpu-price-label", default="Lambda A100 SXM 40GB")
    parser.add_argument("--gpu-pricing-source", default="https://lambda.ai/pricing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[dict[str, str]] = []
    written: list[str] = []
    for dataset_spec in args.datasets:
        benchmark, dataset_name = dataset_spec.split(":", 1)
        print(f"\n==> hybrid dataset {benchmark}/{dataset_name}", flush=True)
        try:
            dataset = load_dataset(benchmark, dataset_name)
            bm25 = build_or_load_bm25(args, dataset)
            bm25_component = run_bm25(args, dataset, bm25)
            components = {"sparse": bm25_component}
            for system_name in args.systems:
                if system_name not in HYBRID_SYSTEMS:
                    raise ValueError(f"Unknown hybrid system: {system_name}")
                required = HYBRID_SYSTEMS[system_name]
                if "dense" in required and "dense" not in components:
                    components["dense"] = ensure_component(
                        args, dataset, args.dense_model, args.dense_system, "dense"
                    )
                if "li" in required and "li" not in components:
                    components["li"] = ensure_component(
                        args, dataset, args.li_model, args.li_system, "li"
                    )
                output = write_system_result(args, dataset, system_name, components)
                written.append(str(output))
        except Exception as exc:
            print(f"FAILED {dataset_spec}: {exc}", flush=True)
            failures.append({"dataset": dataset_spec, "error": repr(exc)})

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written": written,
        "failures": failures,
    }
    manifest_path = Path(args.results_dir) / "hybrid_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {manifest_path}")
    if failures:
        raise SystemExit(f"Hybrid supplement completed with {len(failures)} dataset failure(s).")


def load_dataset(benchmark: str, name: str):
    if benchmark == "beir":
        try:
            return load_beir_dataset(name)
        except ValueError as exc:
            if "multiple" not in str(exc).lower() and "configuration" not in str(exc).lower():
                raise
            return load_beir_dataset_offline(name)
    if benchmark == "bright":
        return load_bright_dataset(name)
    if benchmark == "limit":
        try:
            return load_limit_dataset()
        except ValueError as exc:
            if "multiple" not in str(exc).lower() and "configuration" not in str(exc).lower():
                raise
            return load_limit_dataset_offline()
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_beir_dataset_offline(name: str) -> RetrievalDataset:
    from datasets import load_dataset

    hf_id = DATASET_IDS[name]
    corpus_rows = load_dataset(hf_id, "corpus", split="corpus")
    query_rows = load_dataset(hf_id, "queries", split="queries")
    qrel_rows = load_dataset(hf_id, "default", split="test")

    qrels: dict[str, dict[str, float]] = {}
    qrels_rows = 0
    for row in qrel_rows:
        query_id = _row_id(row, "query-id", "query_id", "qid")
        doc_id = _row_id(row, "corpus-id", "corpus_id", "doc_id")
        score = float(row.get("score", row.get("relevance", 1.0)))
        if score <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = score
        qrels_rows += 1

    query_by_id = {
        _row_id(row, "_id", "id", "query-id", "query_id"): {
            "query_id": _row_id(row, "_id", "id", "query-id", "query_id"),
            "text": _row_text(row),
        }
        for row in query_rows
    }
    from retrieval_pareto.types import Document, Query

    queries = [Query(**query_by_id[qid]) for qid in qrels if qid in query_by_id]
    queries.sort(key=lambda query: query.query_id)
    documents = [
        Document(
            doc_id=_row_id(row, "_id", "id", "corpus-id", "corpus_id"),
            text=_row_text(row),
            title=str(row.get("title") or ""),
        )
        for row in corpus_rows
    ]
    return RetrievalDataset(
        benchmark="beir",
        name=name,
        hf_id=hf_id,
        split="test",
        documents=documents,
        queries=queries,
        qrels=qrels,
        qrels_rows=qrels_rows,
    )


def load_limit_dataset_offline() -> RetrievalDataset:
    from datasets import load_dataset
    from retrieval_pareto.benchmarks.beir_loader import LIMIT_DATASET_ID
    from retrieval_pareto.types import Document, Query

    corpus_rows = load_dataset(LIMIT_DATASET_ID, "corpus", split="corpus")
    query_rows = load_dataset(LIMIT_DATASET_ID, "queries", split="queries")
    qrel_rows = load_dataset(LIMIT_DATASET_ID, "default", split="test")

    qrels: dict[str, dict[str, float]] = {}
    qrels_rows = 0
    for row in qrel_rows:
        query_id = _row_id(row, "query-id", "query_id", "qid")
        doc_id = _row_id(row, "corpus-id", "corpus_id", "doc_id")
        score = float(row.get("score", row.get("relevance", 1.0)))
        if score <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = score
        qrels_rows += 1

    query_by_id = {
        _row_id(row, "_id", "id", "query-id", "query_id"): Query(
            query_id=_row_id(row, "_id", "id", "query-id", "query_id"),
            text=_row_text(row),
        )
        for row in query_rows
    }
    queries = [query_by_id[qid] for qid in qrels if qid in query_by_id]
    queries.sort(key=lambda query: query.query_id)
    documents = [
        Document(
            doc_id=_row_id(row, "_id", "id", "corpus-id", "corpus_id"),
            text=_row_text(row),
            title=str(row.get("title") or ""),
        )
        for row in corpus_rows
    ]
    return RetrievalDataset(
        benchmark="limit",
        name="limit",
        hf_id=LIMIT_DATASET_ID,
        split="test",
        documents=documents,
        queries=queries,
        qrels=qrels,
        qrels_rows=qrels_rows,
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_or_load_bm25(args: argparse.Namespace, dataset) -> Bm25Index:
    cache_path = Path(args.cache_dir) / "bm25" / dataset.benchmark / f"{dataset.name}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    print(f"Building BM25 index for {dataset.benchmark}/{dataset.name}", flush=True)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_ids: list[str] = []
    doc_len: list[int] = []
    for doc_idx, doc in enumerate(dataset.documents):
        doc_ids.append(doc.doc_id)
        terms = tokenize(f"{doc.title}\n{doc.text}")
        doc_len.append(len(terms))
        for term, tf in Counter(terms).items():
            postings[term].append((doc_idx, tf))

    index = Bm25Index(
        doc_ids=doc_ids,
        doc_len=doc_len,
        avgdl=(sum(doc_len) / len(doc_len)) if doc_len else 0.0,
        postings=dict(postings),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def run_bm25(args: argparse.Namespace, dataset, index: Bm25Index) -> Component:
    output_path = result_output_path(args, dataset, "sparse-bm25", "bm25-py")
    rankings_path = output_path.with_suffix(".rankings.json")
    if args.skip_existing and output_path.exists() and rankings_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        return Component("sparse", result, load_rankings(rankings_path), pick_storage(result), pick_latency(result))

    print(f"Searching BM25 for {dataset.benchmark}/{dataset.name}", flush=True)
    per_query_ms: list[float] = []
    query_rankings: list[list[Hit]] = []
    for query in dataset.queries:
        start = time.perf_counter()
        query_rankings.append(search_bm25(index, query.text, args.k))
        per_query_ms.append((time.perf_counter() - start) * 1000.0)

    quality = compute_quality_metrics(dataset.queries, query_rankings, dataset.qrels)
    storage_bytes = bm25_storage_bytes(args, dataset)
    latency = {
        "query_encode_ms_p50": 0.0,
        "query_encode_ms_p99": 0.0,
        "retrieval_ms_p50_topk100": percentile(per_query_ms, 50),
        "retrieval_ms_p99_topk100": percentile(per_query_ms, 99),
        "e2e_query_ms_p50": percentile(per_query_ms, 50),
        "e2e_query_ms_p99": percentile(per_query_ms, 99),
        "latency_sample_size": len(per_query_ms),
        "latency_sample_seed": None,
        "latency_query_ids": [query.query_id for query in dataset.queries],
        "status": "completed",
    }
    result = base_result(
        args,
        dataset,
        model_id="bm25-py",
        family="sparse",
        system_name="sparse-bm25",
        backend="python-bm25",
        compression="inverted-index",
        components=["sparse"],
        quality=quality,
        latency=latency,
        storage={
            **storage_summary(storage_bytes, len(dataset.documents)),
            "index_bytes_backend": storage_bytes,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rankings_sidecar(rankings_path, output_path, result, dataset, query_rankings, args.k)
    print(f"Wrote {output_path}", flush=True)
    return Component("sparse", result, as_query_map(dataset, query_rankings), storage_bytes, latency["e2e_query_ms_p50"])


def search_bm25(index: Bm25Index, query_text: str, k: int) -> list[Hit]:
    k1 = 0.9
    b = 0.4
    scores: dict[int, float] = defaultdict(float)
    n_docs = len(index.doc_ids)
    for term in set(tokenize(query_text)):
        postings = index.postings.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for doc_idx, tf in postings:
            denom = tf + k1 * (1.0 - b + b * index.doc_len[doc_idx] / max(index.avgdl, 1e-9))
            scores[doc_idx] += idf * (tf * (k1 + 1.0) / denom)
    top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
    return [Hit(doc_id=index.doc_ids[doc_idx], score=float(score)) for doc_idx, score in top]


def ensure_component(
    args: argparse.Namespace,
    dataset,
    model_id: str,
    system_name: str,
    component_name: str,
) -> Component:
    output_path = find_result_path(Path(args.results_dir), dataset, system_name, model_id)
    if output_path is None:
        raise FileNotFoundError(f"No aggregate result found for {dataset.benchmark}/{dataset.name} {system_name} {model_id}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    rankings_path = output_path.with_suffix(".rankings.json")
    if not rankings_path.exists():
        print(f"Generating missing rankings sidecar for {output_path}", flush=True)
        rankings = generate_component_rankings(args, dataset, model_id, system_name)
        write_rankings_sidecar(rankings_path, output_path, result, dataset, rankings, args.k)
    return Component(
        component_name,
        result,
        load_rankings(rankings_path),
        pick_storage(result),
        pick_latency(result),
    )


def generate_component_rankings(args: argparse.Namespace, dataset, model_id: str, system_name: str) -> list[list[Hit]]:
    eval_args = argparse.Namespace(
        model=model_id,
        dataset=dataset.name,
        benchmark=dataset.benchmark,
        split="test",
        system=system_name,
        family=None,
        k=args.k,
        limit=None,
        batch_size=None,
        corpus_encode_batch="128",
        latency_batch_sizes="1",
        max_feasible_batch_candidates="1,2,4,8,16,32,64,128",
        latency_samples=0,
        latency_sample_size=args.latency_sample_size,
        latency_sample_seed=args.latency_sample_seed,
        latency_warmup=0,
        cache_dir=args.cache_dir,
        index_dir=args.index_dir,
        li_search="plaid",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=128,
        faiss_nlist=None,
        faiss_pq_m=32,
        binary_candidates=1000,
        truncate_dim=None,
        max_seq_length=args.max_seq_length,
        precision=args.precision,
        gpu_hourly_usd=args.gpu_hourly_usd,
        gpu_pricing_source=args.gpu_pricing_source,
        gpu_price_label=args.gpu_price_label,
        plaid_nbits=8,
        plaid_n_ivf_probe=32,
        plaid_n_full_scores=8192,
        plaid_kmeans_niters=4,
        muvera_k_sim=5,
        muvera_dim_proj=16,
        muvera_r_reps=20,
        muvera_candidates=1000,
        no_cache=False,
        cache_only=False,
        skip_latency=True,
        skip_existing=False,
        output_dir=args.results_dir,
    )
    retrieval_system = resolve_retrieval_system(eval_args)
    retriever = build_retriever_from_args(eval_args, retrieval_system, dataset)
    index, _, _, _ = load_or_encode_corpus(eval_args, retrieval_system["family"], retriever, dataset)
    index = prepare_index(retriever, index)
    query_repr, _, _, _ = load_or_encode_queries(eval_args, retrieval_system["family"], retriever, dataset)
    return retriever.search(query_repr, index, args.k)


def write_system_result(
    args: argparse.Namespace,
    dataset,
    system_name: str,
    components: dict[str, Component],
) -> Path:
    output_path = result_output_path(args, dataset, system_name, model_id_for_system(system_name, components))
    rankings_path = output_path.with_suffix(".rankings.json")
    if args.skip_existing and output_path.exists() and rankings_path.exists():
        print(f"Skipping existing hybrid result: {output_path}", flush=True)
        return output_path

    component_names = HYBRID_SYSTEMS[system_name]
    if system_name == "sparse-bm25":
        return Path(components["sparse"].result["run"]["result_path"]) if "result_path" in components["sparse"].result.get("run", {}) else output_path

    fused_rankings = fuse_rrf(dataset, [components[name] for name in component_names], args.rrf_k, args.k)
    quality = compute_quality_metrics(dataset.queries, fused_rankings, dataset.qrels)
    storage_bytes = sum_storage(components[name] for name in component_names)
    latency = hybrid_latency([components[name] for name in component_names])
    result = base_result(
        args,
        dataset,
        model_id=model_id_for_system(system_name, components),
        family="hybrid",
        system_name=system_name,
        backend="rrf-postprocess",
        compression=f"rrf-k{args.rrf_k}",
        components=list(component_names),
        quality=quality,
        latency=latency,
        storage={
            **storage_summary(storage_bytes or 0, len(dataset.documents)),
            "index_bytes_backend": storage_bytes,
            "component_index_bytes": {name: components[name].storage_bytes for name in component_names},
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rankings_sidecar(rankings_path, output_path, result, dataset, fused_rankings, args.k)
    print(f"Wrote {output_path}", flush=True)
    return output_path


def fuse_rrf(dataset, components: list[Component], rrf_k: int, k: int) -> list[list[Hit]]:
    rankings: list[list[Hit]] = []
    for query in dataset.queries:
        scores: dict[str, float] = defaultdict(float)
        for component in components:
            for rank, hit in enumerate(component.rankings.get(query.query_id, []), start=1):
                scores[hit.doc_id] += 1.0 / (rrf_k + rank)
        top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        rankings.append([Hit(doc_id=doc_id, score=float(score)) for doc_id, score in top])
    return rankings


def base_result(
    args: argparse.Namespace,
    dataset,
    *,
    model_id: str,
    family: str,
    system_name: str,
    backend: str,
    compression: str,
    components: list[str],
    quality: dict[str, float],
    latency: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "0.3",
        "model": {
            "id": model_id,
            "family": family,
            "revision": None,
            "params_m": None,
            "license": None,
            "search_mode": system_name,
            "truncate_dim": None,
            "max_seq_length": None,
            "precision": "component-dependent",
        },
        "retrieval_system": {
            "name": system_name,
            "family": family,
            "backend": backend,
            "search_mode": system_name,
            "exact": False,
            "compression": compression,
            "params": {
                "rrf_k": args.rrf_k if family == "hybrid" else None,
                "components": components,
                "dense_model": args.dense_model if "dense" in components else None,
                "dense_system": args.dense_system if "dense" in components else None,
                "li_model": args.li_model if "li" in components else None,
                "li_system": args.li_system if "li" in components else None,
            },
        },
        "dataset": {
            "benchmark": dataset.benchmark,
            "id": dataset.hf_id,
            "name": dataset.name,
            "split": dataset.split,
            "query_limit": None,
            "corpus_rows": len(dataset.documents),
            "query_rows": len(dataset.queries),
            "qrels_rows": dataset.qrels_rows,
        },
        "quality": quality,
        "latency": latency,
        "storage": storage,
        "cost": cost_summary(
            latency.get("e2e_query_ms_p50"),
            None,
            hourly_rate_usd=args.gpu_hourly_usd,
            hardware_label=args.gpu_price_label,
            pricing_source=args.gpu_pricing_source,
        ),
        "benchmark_protocol": {
            "fusion": f"rrf-k{args.rrf_k}" if family == "hybrid" else None,
            "latency_note": "Hybrid latency is a conservative sequential sum of available component p50 latencies; quality is computed from fused top-100 rankings.",
            "component_rankings": "existing sidecars or generated from cached embeddings without overwriting aggregate component JSON",
        },
        "system": {},
        "index": {
            "search_mode": system_name,
            "backend_index_status": "postprocess",
        },
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware_label": "postprocess",
            "git_sha": None,
        },
    }


def as_query_map(dataset, rankings: list[list[Hit]]) -> dict[str, list[Hit]]:
    return {query.query_id: hits for query, hits in zip(dataset.queries, rankings)}


def load_rankings(path: Path) -> dict[str, list[Hit]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["query_id"]: [Hit(doc_id=hit["doc_id"], score=float(hit["score"])) for hit in row["hits"]]
        for row in payload["queries"]
    }


def find_result_path(results_dir: Path, dataset, system_name: str, model_id: str) -> Path | None:
    stem = sanitize_model_id(model_id)
    system_dir = results_dir / dataset.benchmark / dataset.name / system_name
    candidates = [
        path
        for path in system_dir.glob(f"{stem}*.json")
        if not path.name.endswith(".rankings.json")
    ]
    completed = []
    for path in candidates:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if result.get("status") in {None, "completed", "quality_only", "latency_skipped"}:
            completed.append(path)
    if not completed:
        return None
    return sorted(completed, key=lambda path: (".seq512" not in path.name, len(path.name), path.name))[0]


def result_output_path(args: argparse.Namespace, dataset, system_name: str, model_id: str) -> Path:
    return (
        Path(args.results_dir)
        / dataset.benchmark
        / dataset.name
        / system_name
        / f"{sanitize_model_id(model_id)}.json"
    )


def model_id_for_system(system_name: str, components: dict[str, Component]) -> str:
    names = HYBRID_SYSTEMS[system_name]
    if names == ("sparse",):
        return "bm25-py"
    return "rrf:" + "+".join(components[name].result["model"]["id"] for name in names)


def pick_storage(result: dict[str, Any]) -> int | float | None:
    storage = result.get("storage") or {}
    for key in (
        "index_bytes_backend",
        "index_bytes_binary_rerank",
        "index_bytes_muvera",
        "index_bytes_pq",
        "index_bytes_hnsw",
        "index_bytes_fp16",
    ):
        value = storage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def pick_latency(result: dict[str, Any]) -> float | None:
    value = (result.get("latency") or {}).get("e2e_query_ms_p50")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def sum_storage(components) -> int | float | None:
    values = [component.storage_bytes for component in components]
    if any(value is None for value in values):
        return None
    return sum(values)  # type: ignore[arg-type]


def hybrid_latency(components: list[Component]) -> dict[str, Any]:
    values = [component.e2e_ms_p50 for component in components]
    available = [value for value in values if value is not None]
    e2e = sum(available) if len(available) == len(values) else None
    return {
        "query_encode_ms_p50": None,
        "query_encode_ms_p99": None,
        "retrieval_ms_p50_topk100": e2e,
        "retrieval_ms_p99_topk100": None,
        "e2e_query_ms_p50": e2e,
        "e2e_query_ms_p99": None,
        "latency_sample_size": None,
        "latency_query_ids": [],
        "status": "derived" if e2e is not None else "quality_only",
        "component_e2e_ms_p50": {component.name: component.e2e_ms_p50 for component in components},
    }


def bm25_storage_bytes(args: argparse.Namespace, dataset) -> int:
    path = Path(args.cache_dir) / "bm25" / dataset.benchmark / f"{dataset.name}.pkl"
    return path.stat().st_size if path.exists() else 0


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    if pct == 50:
        return float(statistics.median(values))
    values = sorted(values)
    rank = (len(values) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - rank) + values[hi] * (rank - lo))


if __name__ == "__main__":
    main()
