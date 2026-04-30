from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static Plotly benchmark dashboards.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="plots/output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.results_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        print("No result JSON files found yet.")
        return

    write_quality_latency(results, output_dir / "quality_latency.html")
    write_quality_storage(results, output_dir / "quality_storage.html")
    print(f"Wrote {output_dir}")


def load_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("**/*.json")):
        if "smoke" in path.parts or "sweeps" in path.parts:
            continue
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict) or "model" not in result:
            continue
        if result.get("schema_version") != "0.3":
            continue
        if result.get("status") not in {None, "completed"}:
            continue
        if "retrieval_system" not in result:
            continue
        rows.append(
            {
                "model": _display_model_name(result),
                "family": _family(result),
                "retrieval_system": _retrieval_system(result),
                "dataset": result["dataset"]["name"],
                "ndcg_at_10": result["quality"]["ndcg_at_10"],
                "recall_at_100": result["quality"]["recall_at_100"],
                "e2e_query_ms_p50": result["latency"].get("e2e_query_ms_p50", 0.0),
                "index_mb": _plot_index_bytes(result) / 1_000_000,
            }
        )
    return rows


def _display_model_name(result: dict) -> str:
    model_id = result["model"]["id"]
    retrieval_system = result.get("retrieval_system")
    if retrieval_system:
        system_name = retrieval_system["name"]
        params = retrieval_system.get("params") or {}
        if system_name == "li-fastplaid":
            return f"{model_id} ({system_name} nbits={params.get('nbits')}, probe={params.get('n_ivf_probe')})"
        if system_name == "dense-hnsw":
            return f"{model_id} ({system_name} M={params.get('m')}, ef={params.get('ef_search')})"
        if system_name in {"dense-opq-ivfpq", "dense-rabitq"}:
            return f"{model_id} ({system_name} nlist={params.get('nlist')})"
        if system_name in {"dense-scann", "dense-binary-rerank"}:
            return f"{model_id} ({system_name} candidates={params.get('candidates')})"
        if system_name == "li-muvera":
            return f"{model_id} ({system_name} candidates={params.get('candidates')})"
        return f"{model_id} ({system_name})"

    search_mode = result["model"].get("search_mode")
    if result["model"].get("family") == "late_interaction" and search_mode == "plaid":
        index = result.get("index", {})
        return (
            f"{model_id} (plaid "
            f"nbits={index.get('plaid_nbits')}, "
            f"probe={index.get('plaid_n_ivf_probe')})"
        )
    if result["model"].get("family") == "late_interaction" and search_mode:
        return f"{model_id} ({search_mode})"
    return model_id


def _family(result: dict) -> str:
    return result.get("retrieval_system", {}).get("family") or result["model"]["family"]


def _retrieval_system(result: dict) -> str:
    if "retrieval_system" in result:
        return result["retrieval_system"]["name"]
    family = result["model"].get("family")
    search_mode = result["model"].get("search_mode", "flat")
    if family == "dense":
        return "dense-flat"
    if family == "late_interaction" and search_mode == "plaid":
        return "li-fastplaid"
    if family == "late_interaction":
        return "li-exact"
    return search_mode


def _plot_index_bytes(result: dict) -> int:
    if result["storage"].get("index_bytes_backend") is not None:
        return result["storage"]["index_bytes_backend"]
    if result["storage"].get("index_bytes_binary_rerank") is not None:
        return result["storage"]["index_bytes_binary_rerank"]
    if result["storage"].get("index_bytes_muvera") is not None:
        return result["storage"]["index_bytes_muvera"]
    if result["storage"].get("index_bytes_pq") is not None:
        return result["storage"]["index_bytes_pq"]
    if result["storage"].get("index_bytes_hnsw") is not None:
        return result["storage"]["index_bytes_hnsw"]
    return result["storage"]["index_bytes_fp16"]


def write_quality_latency(rows: list[dict], output_path: Path) -> None:
    import plotly.express as px

    rows = [row for row in rows if row["e2e_query_ms_p50"] is not None]
    if not rows:
        output_path.write_text("<html><body>No latency results available yet.</body></html>\n", encoding="utf-8")
        return

    fig = px.scatter(
        rows,
        x="e2e_query_ms_p50",
        y="ndcg_at_10",
        color="family",
        symbol="retrieval_system",
        facet_col="dataset",
        hover_name="model",
        hover_data=["retrieval_system", "recall_at_100", "index_mb"],
        title="Quality vs Query Latency",
        labels={
            "e2e_query_ms_p50": "p50 end-to-end query latency (ms)",
            "ndcg_at_10": "nDCG@10",
            "family": "Model family",
            "retrieval_system": "Retrieval system",
        },
    )
    fig.update_layout(
        template="plotly_white",
        annotations=[
            {
                "text": "Each point is a model x dataset x retrieval-system result. Upper-left is the desirable Pareto region.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.22,
                "showarrow": False,
                "align": "left",
            }
        ],
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def write_quality_storage(rows: list[dict], output_path: Path) -> None:
    import plotly.express as px

    fig = px.scatter(
        rows,
        x="index_mb",
        y="ndcg_at_10",
        color="family",
        symbol="retrieval_system",
        facet_col="dataset",
        hover_name="model",
        hover_data=["retrieval_system", "recall_at_100", "e2e_query_ms_p50"],
        title="Quality vs Reported Index Size",
        labels={
            "index_mb": "Reported index size (MB)",
            "ndcg_at_10": "nDCG@10",
            "family": "Model family",
            "retrieval_system": "Retrieval system",
        },
    )
    fig.update_layout(
        template="plotly_white",
        annotations=[
            {
                "text": "Exact systems report FP16 representation size; HNSW and FastPlaid report their saved serving index when available.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.22,
                "showarrow": False,
                "align": "left",
            }
        ],
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    main()
