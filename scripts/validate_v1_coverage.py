#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = [
    ("beir", "nfcorpus"),
    ("beir", "scifact"),
    ("beir", "fiqa"),
    ("beir", "arguana"),
    ("beir", "scidocs"),
    ("beir", "trec-covid"),
    ("bright", "biology"),
    ("bright", "economics"),
    ("bright", "psychology"),
    ("bright", "robotics"),
    ("bright", "stackoverflow"),
    ("bright", "leetcode"),
    ("limit", "limit"),
]

INTENTIONAL_SKIP_STATUSES = {
    "not_applicable_small_corpus",
    "skipped_high_dim_opq_compute",
    "backend_oom_full8192_a100",
    "latency_oom",
    "component_latency_unavailable",
}

MAIN_DENSE_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "intfloat/e5-large-v2",
    "Qwen/Qwen3-Embedding-8B",
]

NV_MODEL = "nvidia/NV-Embed-v2"

LI_MODELS = [
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
    "lightonai/colbertv2.0",
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/Reason-ModernColBERT",
]

LI_EXACT_DATASETS = {
    ("beir", "nfcorpus"),
    ("beir", "scifact"),
    ("beir", "arguana"),
}

HYBRID_SYSTEMS = [
    "sparse-bm25",
    "hybrid-rrf-bm25-dense",
    "hybrid-rrf-bm25-li",
    "hybrid-rrf-bm25-dense-li",
    "hybrid-rrf-dense-li",
]


@dataclass(frozen=True)
class Requirement:
    benchmark: str
    dataset: str
    system: str
    model: str | None
    require_latency: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v1 result coverage and latency completeness.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--latency-sample-size", type=int, default=200)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when requirements are missing.")
    parser.add_argument("--show-missing", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.results_dir))
    requirements = build_requirements()
    missing: list[Requirement] = []
    bad_latency: list[tuple[Requirement, Path, str]] = []

    for requirement in requirements:
        candidates = find_candidates(rows, requirement)
        if not candidates:
            missing.append(requirement)
            continue
        if requirement.require_latency:
            candidate = choose_best_candidate(candidates)
            reason = latency_problem(candidate[1], args.latency_sample_size)
            if reason:
                bad_latency.append((requirement, candidate[0], reason))

    summarize_inventory(rows)
    print("")
    print(f"Required v1 cells: {len(requirements)}")
    print(f"Missing cells: {len(missing)}")
    print(f"Cells with missing/stale latency: {len(bad_latency)}")

    if missing:
        print("\nMissing examples:")
        for item in missing[: args.show_missing]:
            print(format_requirement(item))
        if len(missing) > args.show_missing:
            print(f"... {len(missing) - args.show_missing} more")

    if bad_latency:
        print("\nLatency problems:")
        for requirement, path, reason in bad_latency[: args.show_missing]:
            print(f"{format_requirement(requirement)} :: {reason} :: {path}")
        if len(bad_latency) > args.show_missing:
            print(f"... {len(bad_latency) - args.show_missing} more")

    if args.strict and (missing or bad_latency):
        raise SystemExit(1)


def load_rows(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name.endswith(".rankings.json"):
            continue
        if path.name == "hybrid_manifest.json":
            continue
        if "smoke" in path.parts or "sweeps" in path.parts:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("schema_version") == "0.3":
            rows.append((path, row))
    return rows


def build_requirements() -> list[Requirement]:
    requirements: list[Requirement] = []
    for benchmark, dataset in DATASETS:
        for model in MAIN_DENSE_MODELS:
            dense_systems = dense_systems_for_model(model)
            for system in dense_systems:
                requirements.append(Requirement(benchmark, dataset, system, model, True))
        for system in dense_systems_for_model(NV_MODEL):
            requirements.append(Requirement(benchmark, dataset, system, NV_MODEL, True))
        for model in LI_MODELS:
            for system in ("li-fastplaid", "li-muvera"):
                requirements.append(Requirement(benchmark, dataset, system, model, True))
            if (benchmark, dataset) in LI_EXACT_DATASETS:
                requirements.append(Requirement(benchmark, dataset, "li-exact", model, True))
        for system in HYBRID_SYSTEMS:
            requirements.append(Requirement(benchmark, dataset, system, None, system != "sparse-bm25"))
    return requirements


def dense_systems_for_model(model: str) -> list[str]:
    if model in {"Qwen/Qwen3-Embedding-8B", "nvidia/NV-Embed-v2"}:
        return ["dense-flat", "dense-hnsw", "dense-rabitq", "dense-binary-rerank"]
    return [
        "dense-flat",
        "dense-hnsw",
        "dense-opq-ivfpq",
        "dense-rabitq",
        "dense-scann",
        "dense-binary-rerank",
    ]


def find_candidates(
    rows: list[tuple[Path, dict[str, Any]]],
    requirement: Requirement,
) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path, row in rows:
        dataset = row.get("dataset") or {}
        system = row.get("retrieval_system") or {}
        model = row.get("model") or {}
        if dataset.get("benchmark") != requirement.benchmark:
            continue
        if dataset.get("name") != requirement.dataset:
            continue
        if system.get("name") != requirement.system:
            continue
        if requirement.model is not None and model.get("id") != requirement.model:
            continue
        candidates.append((path, row))
    return candidates


def choose_best_candidate(candidates: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]]:
    def score(item: tuple[Path, dict[str, Any]]) -> tuple[int, int, str]:
        path, row = item
        latency = row.get("latency") if isinstance(row.get("latency"), dict) else {}
        has_latency = latency.get("e2e_query_ms_p50") is not None
        sample_size = latency.get("latency_sample_size") or 0
        return (1 if has_latency else 0, int(sample_size), str(path))

    return sorted(candidates, key=score, reverse=True)[0]


def latency_problem(row: dict[str, Any], target_sample_size: int) -> str | None:
    status = row.get("status")
    if status in INTENTIONAL_SKIP_STATUSES:
        return None
    latency = row.get("latency") if isinstance(row.get("latency"), dict) else {}
    if latency.get("e2e_query_ms_p50") is None:
        return "missing e2e_query_ms_p50"
    query_rows = (row.get("dataset") or {}).get("query_rows")
    expected = min(target_sample_size, int(query_rows)) if query_rows else target_sample_size
    actual = latency.get("latency_sample_size")
    if actual != expected:
        return f"stale sample size {actual}, expected {expected}"
    return None


def summarize_inventory(rows: list[tuple[Path, dict[str, Any]]]) -> None:
    by_family = Counter()
    by_system = Counter()
    by_model = Counter()
    latency_missing = 0
    latency_stale = 0
    by_problem_system = defaultdict(Counter)
    for _, row in rows:
        system = row.get("retrieval_system") or {}
        model = row.get("model") or {}
        latency = row.get("latency") if isinstance(row.get("latency"), dict) else {}
        by_family[system.get("family") or "unknown"] += 1
        by_system[system.get("name") or "unknown"] += 1
        by_model[model.get("id") or "unknown"] += 1
        if system.get("family") in {"dense", "late_interaction"}:
            if latency.get("e2e_query_ms_p50") is None and row.get("status") not in INTENTIONAL_SKIP_STATUSES:
                latency_missing += 1
                by_problem_system[system.get("name") or "unknown"]["missing"] += 1
            elif latency.get("latency_sample_size") not in {None, 200, (row.get("dataset") or {}).get("query_rows")}:
                latency_stale += 1
                by_problem_system[system.get("name") or "unknown"]["stale"] += 1

    print(f"Inventory rows: {len(rows)}")
    print("Rows by family:", dict(sorted(by_family.items())))
    print("Rows by system:", dict(sorted(by_system.items())))
    print("Rows by model:", dict(sorted(by_model.items())))
    print(f"Direct rows missing latency: {latency_missing}")
    print(f"Direct rows with non-standard latency sample size: {latency_stale}")
    if by_problem_system:
        print("Latency problems by system:", {key: dict(value) for key, value in sorted(by_problem_system.items())})


def format_requirement(requirement: Requirement) -> str:
    model = requirement.model or "*"
    return f"{requirement.benchmark}/{requirement.dataset} {requirement.system} {model}"


if __name__ == "__main__":
    main()
