from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_pareto.adapters.dense_adapter import (
    DenseIndex,
    DenseQueryRepr,
    _build_faiss_index,
    sanitize_dense_embeddings,
)
from retrieval_pareto.adapters.late_interaction_adapter import (
    LateInteractionIndex,
    LateInteractionQueryRepr,
)


@dataclass(frozen=True)
class CacheInfo:
    enabled: bool
    hit: bool
    path: str | None
    key: str | None


def cache_path(
    cache_dir: Path,
    *,
    kind: str,
    family: str,
    model_id: str,
    dataset_id: str,
    dataset_name: str,
    split: str,
    query_limit: int | None = None,
    representation_key: str | None = None,
) -> Path:
    payload = {
        "schema_version": "0.2",
        "kind": kind,
        "family": family,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "split": split,
        "query_limit": query_limit,
        "representation_key": representation_key,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return cache_dir / family / kind / f"{digest}.npz"


def load_cached(path: Path, family: str, kind: str):
    if family == "dense" and kind == "corpus":
        return _load_dense_index(path)
    if family == "dense" and kind == "queries":
        return _load_dense_queries(path)
    if family == "late_interaction" and kind == "corpus":
        return _load_late_interaction_index(path)
    if family == "late_interaction" and kind == "queries":
        return _load_late_interaction_queries(path)
    raise ValueError(f"Unsupported cache type: {family}/{kind}")


def save_cached(path: Path, family: str, kind: str, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == "dense" and kind == "corpus":
        _save_dense_index(path, value)
        return
    if family == "dense" and kind == "queries":
        _save_dense_queries(path, value)
        return
    if family == "late_interaction" and kind == "corpus":
        _save_late_interaction_index(path, value)
        return
    if family == "late_interaction" and kind == "queries":
        _save_late_interaction_queries(path, value)
        return
    raise ValueError(f"Unsupported cache type: {family}/{kind}")


def cache_info(enabled: bool, hit: bool, path: Path | None) -> CacheInfo:
    return CacheInfo(
        enabled=enabled,
        hit=hit,
        path=str(path) if path is not None else None,
        key=path.stem if path is not None else None,
    )


def _save_dense_index(path: Path, index: DenseIndex) -> None:
    np.savez_compressed(
        path,
        doc_ids=np.asarray(index.doc_ids),
        embeddings=index.embeddings.astype("float32"),
    )


def _load_dense_index(path: Path) -> DenseIndex:
    with np.load(path) as data:
        embeddings = sanitize_dense_embeddings(data["embeddings"])
        doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
    return DenseIndex(doc_ids=doc_ids, embeddings=embeddings, faiss_index=_build_faiss_index(embeddings))


def _save_dense_queries(path: Path, query_repr: DenseQueryRepr) -> None:
    np.savez_compressed(
        path,
        query_ids=np.asarray(query_repr.query_ids),
        embeddings=query_repr.embeddings.astype("float32"),
    )


def _load_dense_queries(path: Path) -> DenseQueryRepr:
    with np.load(path) as data:
        embeddings = sanitize_dense_embeddings(data["embeddings"])
        query_ids = [str(query_id) for query_id in data["query_ids"].tolist()]
    return DenseQueryRepr(query_ids=query_ids, embeddings=embeddings)


def _save_late_interaction_index(path: Path, index: LateInteractionIndex) -> None:
    embeddings, lengths = _pack_token_embeddings(index.embeddings)
    np.savez_compressed(
        path,
        doc_ids=np.asarray(index.doc_ids),
        embeddings=embeddings,
        lengths=lengths,
    )


def _load_late_interaction_index(path: Path) -> LateInteractionIndex:
    with np.load(path) as data:
        doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
        embeddings = _unpack_token_embeddings(data["embeddings"], data["lengths"])
    return LateInteractionIndex(doc_ids=doc_ids, embeddings=embeddings)


def _save_late_interaction_queries(path: Path, query_repr: LateInteractionQueryRepr) -> None:
    embeddings, lengths = _pack_token_embeddings(query_repr.embeddings)
    np.savez_compressed(
        path,
        query_ids=np.asarray(query_repr.query_ids),
        embeddings=embeddings,
        lengths=lengths,
    )


def _load_late_interaction_queries(path: Path) -> LateInteractionQueryRepr:
    with np.load(path) as data:
        query_ids = [str(query_id) for query_id in data["query_ids"].tolist()]
        embeddings = _unpack_token_embeddings(data["embeddings"], data["lengths"])
    return LateInteractionQueryRepr(query_ids=query_ids, embeddings=embeddings)


def _pack_token_embeddings(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([value.shape[0] for value in values], dtype=np.int64)
    if values:
        embeddings = np.concatenate([value.astype("float32") for value in values], axis=0)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    return embeddings, lengths


def _unpack_token_embeddings(embeddings: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    offset = 0
    for length in lengths.tolist():
        next_offset = offset + int(length)
        values.append(embeddings[offset:next_offset].astype("float32"))
        offset = next_offset
    return values
