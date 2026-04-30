#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark JSON into the static site schema.")
    parser.add_argument("--results-dir", default="results", help="Directory containing result JSON files.")
    parser.add_argument("--output", default="site/data/results.json", help="Output site data JSON.")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke-test result rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs = list(iter_runs(results_dir, include_smoke=args.include_smoke))
    backfill_hybrid_components(runs)
    payload = {
        "schema_version": "site.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(results_dir),
        "runs": runs,
        "summary": {
            "rows": len(runs),
            "datasets": len({run["dataset"] for run in runs}),
            "models": len({run["model_id"] for run in runs}),
            "systems": len({run["system"] for run in runs}),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(runs)} rows")


def iter_runs(results_dir: Path, *, include_smoke: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("**/*.json")):
        if not include_smoke and "smoke" in path.parts:
            continue
        if "sweeps" in path.parts:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(result, dict) or result.get("schema_version") != "0.3":
            continue
        if "model" not in result or "dataset" not in result:
            continue
        if "quality" not in result:
            continue
        row = normalize_result(result, path)
        if row:
            rows.append(row)
    return rows


def normalize_result(result: dict[str, Any], path: Path) -> dict[str, Any] | None:
    model = result.get("model") or {}
    dataset = result.get("dataset") or {}
    retrieval_system = result.get("retrieval_system") or {}
    system_name = retrieval_system.get("name") or infer_system_name(model)
    benchmark = dataset.get("benchmark") or path.parts[-4] if len(path.parts) >= 4 else "unknown"
    dataset_name = dataset.get("name") or path.parts[-3] if len(path.parts) >= 3 else "unknown"
    storage = result.get("storage") or {}
    latency = result.get("latency") or {}
    quality = result.get("quality") or {}
    status = normalize_status(result.get("status"), latency, quality)
    index_bytes = pick_index_bytes(storage)

    return {
        "id": f"{benchmark}:{dataset_name}:{model.get('id', 'model')}:{system_name}:{path}",
        "benchmark": benchmark,
        "dataset": dataset_name,
        "dataset_label": dataset_label(benchmark, dataset_name),
        "model_id": model.get("id", "unknown-model"),
        "model_label": model_label(model.get("id", "unknown-model"), result),
        "family": retrieval_system.get("family") or model.get("family") or "unknown",
        "params_m": model.get("params_m"),
        "system": system_name,
        "system_label": system_label(system_name),
        "backend": retrieval_system.get("backend") or system_name,
        "compression": retrieval_system.get("compression") or storage_compression(storage),
        "exact": bool(retrieval_system.get("exact")),
        "status": status,
        "quality": {
            "ndcg_at_10": quality.get("ndcg_at_10"),
            "recall_at_100": quality.get("recall_at_100"),
            "mrr_at_10": quality.get("mrr_at_10"),
            "map_at_100": quality.get("map_at_100"),
        },
        "latency": {
            "e2e_query_ms_p50": latency.get("e2e_query_ms_p50"),
            "e2e_query_ms_p99": first_present(
                latency,
                "e2e_query_ms_p99",
                "serving_batch_1_e2e_ms_p99",
                "e2e_ms_p99",
            ),
            "tokenize_ms_p50": latency.get("tokenize_ms_p50"),
            "query_encode_ms_p50": latency.get("query_encode_ms_p50"),
            "retrieval_ms_p50_topk100": latency.get("retrieval_ms_p50_topk100"),
            "component_e2e_ms_p50": latency.get("component_e2e_ms_p50"),
            "latency_sample_size": latency.get("latency_sample_size"),
            "latency_query_ids": latency.get("latency_query_ids"),
            "status": latency.get("status"),
        },
        "storage": {
            "index_bytes": index_bytes,
            "index_bytes_per_doc": storage.get("index_bytes_per_doc"),
            "index_bytes_fp16": storage.get("index_bytes_fp16"),
            "index_bytes_int8": storage.get("index_bytes_int8"),
            "index_bytes_pq": storage.get("index_bytes_pq"),
            "index_bytes_backend": storage.get("index_bytes_backend"),
            "component_index_bytes": storage.get("component_index_bytes"),
        },
        "cost": {
            "cost_per_million_queries_usd": (result.get("cost") or {}).get(
                "cost_per_million_queries_usd"
            ),
            "cost_per_million_docs_indexed_usd": (result.get("cost") or {}).get(
                "cost_per_million_docs_indexed_usd"
            ),
            "reference_gpu_hourly_usd": (result.get("cost") or {}).get("reference_gpu_hourly_usd"),
        },
        "protocol": {
            "hardware": (result.get("run") or {}).get("hardware_label")
            or (result.get("system") or {}).get("platform")
            or "unknown",
            "precision": (result.get("run") or {}).get("precision")
            or result.get("precision")
            or "unknown",
            "latency_policy": "latency skipped"
            if latency.get("status") == "skipped"
            else "batch=1 isolated when measured",
            "created_at": (result.get("run") or {}).get("created_at"),
        },
        "component_sources": component_sources(retrieval_system.get("params") or {}),
        "result_path": str(path),
    }


def component_sources(params: dict[str, Any]) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for component in params.get("components") or []:
        if component == "sparse":
            sources["sparse"] = {"model": "bm25-py", "system": "sparse-bm25"}
        elif component == "dense":
            sources["dense"] = {
                "model": params.get("dense_model") or "",
                "system": params.get("dense_system") or "dense-hnsw",
            }
        elif component == "li":
            sources["li"] = {
                "model": params.get("li_model") or "",
                "system": params.get("li_system") or "li-fastplaid",
            }
    return sources


def backfill_hybrid_components(rows: list[dict[str, Any]]) -> None:
    by_key = {
        (row["benchmark"], row["dataset"], row["model_id"], row["system"]): row
        for row in rows
    }
    for row in rows:
        if row.get("family") != "hybrid":
            continue
        sources = row.get("component_sources") or {}
        latency_components = dict((row.get("latency") or {}).get("component_e2e_ms_p50") or {})
        storage_components = dict((row.get("storage") or {}).get("component_index_bytes") or {})
        component_rows: dict[str, dict[str, Any]] = {}
        for name, source in sources.items():
            component = by_key.get(
                (row["benchmark"], row["dataset"], source.get("model"), source.get("system"))
            )
            if not component:
                continue
            component_rows[name] = component
            if latency_components.get(name) is None:
                latency_components[name] = (component.get("latency") or {}).get("e2e_query_ms_p50")
            if storage_components.get(name) is None:
                storage_components[name] = (component.get("storage") or {}).get("index_bytes")

        row["latency"]["component_e2e_ms_p50"] = latency_components or None
        row["storage"]["component_index_bytes"] = storage_components or None

        required = list(sources)
        component_values = [latency_components.get(name) for name in required]
        if required and all(is_number(value) for value in component_values):
            total = float(sum(component_values))
            row["latency"]["e2e_query_ms_p50"] = total
            row["latency"]["retrieval_ms_p50_topk100"] = total
            row["latency"]["status"] = "derived"
            row["status"] = "completed"

        component_p99 = [
            (component_rows.get(name, {}).get("latency") or {}).get("e2e_query_ms_p99")
            for name in required
        ]
        if required and all(is_number(value) for value in component_p99):
            row["latency"]["e2e_query_ms_p99"] = float(sum(component_p99))

        component_encode_p50 = [
            (component_rows.get(name, {}).get("latency") or {}).get("query_encode_ms_p50")
            for name in required
        ]
        if required and all(is_number(value) for value in component_encode_p50):
            row["latency"]["query_encode_ms_p50"] = float(sum(component_encode_p50))

        component_retrieve_p50 = [
            (component_rows.get(name, {}).get("latency") or {}).get("retrieval_ms_p50_topk100")
            for name in required
        ]
        if required and all(is_number(value) for value in component_retrieve_p50):
            row["latency"]["retrieval_ms_p50_topk100"] = float(sum(component_retrieve_p50))


def normalize_status(status: str | None, latency: dict[str, Any], quality: dict[str, Any]) -> str:
    if status in {"not_applicable_small_corpus", "failed", "pending"}:
        return status
    if status in {"quality_only", "latency_skipped"}:
        return status
    if latency.get("status") == "skipped":
        return "quality_only"
    has_quality = any(is_number(value) for value in quality.values())
    has_latency = is_number(latency.get("e2e_query_ms_p50"))
    if status and status != "completed":
        return status
    if has_quality and not has_latency:
        return "quality_only"
    return "completed"


def first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def pick_index_bytes(storage: dict[str, Any]) -> int | float | None:
    for key in (
        "index_bytes_backend",
        "index_bytes_binary_rerank",
        "index_bytes_muvera",
        "index_bytes_pq",
        "index_bytes_hnsw",
        "index_bytes_fp16",
        "index_bytes_int8",
    ):
        value = storage.get(key)
        if is_number(value):
            return value
    return None


def storage_compression(storage: dict[str, Any]) -> str:
    if storage.get("index_bytes_binary_rerank") is not None:
        return "1-bit+rerank"
    if storage.get("index_bytes_muvera") is not None:
        return "muvera"
    if storage.get("index_bytes_pq") is not None:
        return "pq"
    if storage.get("index_bytes_hnsw") is not None:
        return "hnsw"
    if storage.get("index_bytes_int8") is not None:
        return "int8"
    return "fp16"


def infer_system_name(model: dict[str, Any]) -> str:
    if model.get("family") == "late_interaction":
        return "li-fastplaid" if model.get("search_mode") == "plaid" else "li-exact"
    return "dense-flat"


def model_label(model_id: str, result: dict[str, Any]) -> str:
    params = ((result.get("retrieval_system") or {}).get("params") or {})
    truncate_dim = params.get("truncate_dim") or (result.get("model") or {}).get("truncate_dim")
    label = shorten_model(model_id)
    return f"{label} @{truncate_dim}d" if truncate_dim else label


def shorten_model(model_id: str) -> str:
    name = model_id.split("/")[-1]
    replacements = {
        "-": " ",
        "bge": "BGE",
        "gte": "GTE",
        "nv": "NV",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    words = []
    for word in name.split():
        if word.upper() in {"BGE", "GTE", "NV", "E5"}:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def dataset_label(benchmark: str, dataset: str) -> str:
    if benchmark == "bright":
        return f"BRIGHT {dataset.replace('_', ' ').title()}"
    if benchmark == "limit":
        return "LIMIT"
    return dataset.replace("_", " ").title()


def system_label(system: str) -> str:
    return {
        "dense-flat": "Flat exact",
        "dense-hnsw": "HNSW",
        "dense-opq-ivfpq": "OPQ-IVF-PQ",
        "dense-rabitq": "RaBitQ",
        "dense-scann": "ScaNN",
        "dense-binary-rerank": "Binary rerank",
        "li-exact": "Exact MaxSim",
        "li-fastplaid": "FastPlaid",
        "li-muvera": "MUVERA",
        "hybrid": "Hybrid",
    }.get(system, system.replace("-", " ").title())


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    main()
