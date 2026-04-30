from __future__ import annotations


def storage_summary(
    index_size_fp16: int,
    num_docs: int,
    index_size_pq: int | None = None,
) -> dict[str, float | int | None]:
    return {
        "index_bytes_fp16": index_size_fp16,
        "index_bytes_int8": index_size_fp16 // 2 if index_size_fp16 else 0,
        "index_bytes_pq": index_size_pq,
        "index_bytes_per_doc": index_size_fp16 / num_docs if num_docs else 0.0,
    }
