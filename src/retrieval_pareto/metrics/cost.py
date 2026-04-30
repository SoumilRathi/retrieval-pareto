from __future__ import annotations


REFERENCE_A100_SXM_40GB_HOURLY_USD = 1.99


def cost_summary(
    e2e_query_ms_p50: float | None,
    docs_indexed_per_second: float | None,
    hourly_rate_usd: float = REFERENCE_A100_SXM_40GB_HOURLY_USD,
    hardware_label: str = "Lambda A100 SXM 40GB",
    pricing_source: str = "https://lambda.ai/pricing",
) -> dict[str, float | None]:
    cost_per_million_queries = None
    if e2e_query_ms_p50 is not None:
        seconds_per_million_queries = (e2e_query_ms_p50 / 1000.0) * 1_000_000
        cost_per_million_queries = seconds_per_million_queries / 3600.0 * hourly_rate_usd

    cost_per_million_docs = None
    if docs_indexed_per_second and docs_indexed_per_second > 0:
        seconds_per_million_docs = 1_000_000 / docs_indexed_per_second
        cost_per_million_docs = seconds_per_million_docs / 3600.0 * hourly_rate_usd

    return {
        "reference_gpu_hourly_usd": hourly_rate_usd,
        "reference_gpu_hardware": hardware_label,
        "pricing_source": pricing_source,
        "cost_per_million_queries_usd": cost_per_million_queries,
        "cost_per_million_docs_indexed_usd": cost_per_million_docs,
    }
