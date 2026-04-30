from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from retrieval_pareto.types import Document, Hit, Query


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


@dataclass
class DenseIndex:
    doc_ids: list[str]
    embeddings: np.ndarray
    faiss_index: Any | None = None
    hnsw_index: Any | None = None
    hnsw_index_path: str | None = None
    backend_index: Any | None = None
    backend_index_path: str | None = None
    backend_size_bytes: int | None = None
    binary_codes: np.ndarray | None = None


@dataclass
class DenseQueryRepr:
    query_ids: list[str]
    embeddings: np.ndarray


def _default_device() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def _model_kwargs_for_precision(precision: str, device: str | None) -> dict[str, Any]:
    if precision == "auto":
        return {}
    try:
        import torch
    except Exception:
        return {}
    if precision == "fp16":
        return {"torch_dtype": torch.float16}
    if precision == "bf16":
        return {"torch_dtype": torch.bfloat16}
    if precision == "fp32":
        return {"torch_dtype": torch.float32}
    raise ValueError("--precision must be one of: auto, fp16, bf16, fp32")


class DenseRetriever:
    family = "dense"

    def __init__(
        self,
        model_id: str,
        batch_size: int = 64,
        device: str | None = None,
        search_mode: str = "flat",
        index_folder: str = "indexes",
        index_name: str | None = None,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        faiss_nlist: int | None = None,
        faiss_pq_m: int = 32,
        binary_candidates: int = 1000,
        truncate_dim: int | None = None,
        max_seq_length: int | None = None,
        precision: str = "auto",
        load_model: bool = True,
    ):
        supported_modes = {"flat", "hnsw", "opq_ivf_pq", "rabitq", "scann", "binary_rerank"}
        if search_mode not in supported_modes:
            raise ValueError(f"search_mode must be one of: {', '.join(sorted(supported_modes))}")

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.search_mode = search_mode
        self.index_folder = index_folder
        self.index_name = index_name or "dense"
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.faiss_nlist = faiss_nlist
        self.faiss_pq_m = faiss_pq_m
        self.binary_candidates = binary_candidates
        self.truncate_dim = truncate_dim
        self.max_seq_length = max_seq_length
        self.precision = precision
        self.index_build_status = "not_applicable"
        self.index_build_spec: str | None = None
        self.model = None
        if not load_model:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Missing sentence-transformers. Run `uv sync` first.") from exc

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device:
            kwargs["device"] = self.device
        model_kwargs = _model_kwargs_for_precision(precision, self.device)
        constructor_params = inspect.signature(SentenceTransformer).parameters
        supports_model_kwargs = "model_kwargs" in constructor_params
        if model_kwargs and supports_model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        self.model = SentenceTransformer(model_id, **kwargs)
        if model_kwargs and not supports_model_kwargs:
            dtype = model_kwargs.get("torch_dtype")
            if dtype is not None and hasattr(self.model, "to"):
                self.model.to(dtype=dtype)
        if self.max_seq_length is not None:
            self.model.max_seq_length = self.max_seq_length

    def encode_corpus(self, documents: list[Document]) -> DenseIndex:
        if self.model is None:
            raise RuntimeError("DenseRetriever was constructed without a model; cached embeddings are required.")
        texts = [doc.text for doc in documents]
        encode_kwargs: dict[str, Any] = {}
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            **encode_kwargs,
        ).astype("float32")
        embeddings = sanitize_dense_embeddings(embeddings)
        return DenseIndex(
            doc_ids=[doc.doc_id for doc in documents],
            embeddings=embeddings,
            faiss_index=_build_faiss_index(embeddings),
        )

    def encode_queries(self, queries: list[Query]) -> DenseQueryRepr:
        if self.model is None:
            raise RuntimeError("DenseRetriever was constructed without a model; cached embeddings are required.")
        encode_kwargs: dict[str, Any] = {}
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim
        embeddings = self.model.encode(
            [query.text for query in queries],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **encode_kwargs,
        ).astype("float32")
        embeddings = sanitize_dense_embeddings(embeddings)
        return DenseQueryRepr(query_ids=[query.query_id for query in queries], embeddings=embeddings)

    def measure_query_encode_components(self, queries: list[Query]) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("DenseRetriever was constructed without a model; query latency source is required.")
        try:
            import torch
            from sentence_transformers.util import batch_to_device, truncate_embeddings
        except ImportError:
            _, total_ms = _timed(lambda: self.encode_queries(queries))
            return {"tokenize_ms": None, "forward_ms": None, "total_ms": total_ms}

        texts = [query.text for query in queries]
        features, tokenize_ms = _timed(lambda: self.model.tokenize(texts))
        device = getattr(self.model, "device", self.device)

        def forward_pass():
            encoded_features = batch_to_device(features, device)
            with torch.no_grad():
                output_features = self.model.forward(encoded_features)
                embeddings = output_features["sentence_embedding"]
                if self.truncate_dim:
                    embeddings = truncate_embeddings(embeddings, self.truncate_dim)
                torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
            return None

        _, forward_ms = _timed(forward_pass)
        return {
            "tokenize_ms": tokenize_ms,
            "forward_ms": forward_ms,
            "total_ms": tokenize_ms + forward_ms,
        }

    def search(self, query_repr: DenseQueryRepr, index: DenseIndex, k: int) -> list[list[Hit]]:
        top_k = min(k, len(index.doc_ids))
        index.embeddings = sanitize_dense_embeddings(index.embeddings)
        query_repr.embeddings = sanitize_dense_embeddings(query_repr.embeddings)
        if self.search_mode == "hnsw":
            if index.hnsw_index is None:
                index = self.prepare_index(index)
            positions, distances = index.hnsw_index.knn_query(query_repr.embeddings, k=top_k)
            scores = 1.0 - distances
        elif self.search_mode in {"opq_ivf_pq", "rabitq"}:
            if index.backend_index is None:
                index = self.prepare_index(index)
            scores, positions = index.backend_index.search(
                np.ascontiguousarray(query_repr.embeddings.astype("float32")),
                top_k,
            )
        elif self.search_mode == "scann":
            if index.backend_index is None:
                index = self.prepare_index(index)
            positions, scores = _scann_search(index.backend_index, query_repr.embeddings, top_k)
        elif self.search_mode == "binary_rerank":
            if index.binary_codes is None:
                index = self.prepare_index(index)
            scores, positions = _binary_hamming_rerank(
                index.embeddings,
                index.binary_codes,
                query_repr.embeddings,
                top_k,
                self.binary_candidates,
            )
        elif index.faiss_index is not None:
            scores, positions = index.faiss_index.search(query_repr.embeddings, top_k)
        else:
            scores, positions = _numpy_topk(query_repr.embeddings @ index.embeddings.T, top_k)

        rankings: list[list[Hit]] = []
        for row_scores, row_positions in zip(scores, positions):
            rankings.append(
                [
                    Hit(doc_id=index.doc_ids[int(pos)], score=float(score))
                    for score, pos in zip(row_scores, row_positions)
                    if int(pos) >= 0
                ]
            )
        return rankings

    def index_size_bytes(self, index: DenseIndex, compression: str = "fp16") -> int:
        if compression == "hnsw" and index.hnsw_index_path:
            return Path(index.hnsw_index_path).stat().st_size
        if compression == "backend" and index.backend_size_bytes is not None:
            return index.backend_size_bytes
        if compression == "binary_rerank" and index.binary_codes is not None:
            return int(index.binary_codes.nbytes + index.embeddings.shape[0] * index.embeddings.shape[1] * 2)
        if compression == "fp16":
            return int(index.embeddings.shape[0] * index.embeddings.shape[1] * 2)
        if compression == "int8":
            return int(index.embeddings.shape[0] * index.embeddings.shape[1])
        return int(index.embeddings.nbytes)

    def prepare_index(self, index: DenseIndex) -> DenseIndex:
        index.embeddings = sanitize_dense_embeddings(index.embeddings)
        if self.search_mode == "flat":
            self.index_build_status = "not_applicable"
            return index
        if index.hnsw_index is not None:
            self.index_build_status = "loaded"
            return index
        if index.backend_index is not None or index.binary_codes is not None:
            self.index_build_status = "loaded"
            return index

        if self.search_mode == "hnsw":
            return self._prepare_hnsw_index(index)
        if self.search_mode in {"opq_ivf_pq", "rabitq"}:
            return self._prepare_faiss_backend(index)
        if self.search_mode == "scann":
            return self._prepare_scann_index(index)
        if self.search_mode == "binary_rerank":
            return self._prepare_binary_rerank_index(index)
        raise ValueError(f"Unsupported search_mode: {self.search_mode}")

    def _prepare_hnsw_index(self, index: DenseIndex) -> DenseIndex:
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError("Missing hnswlib. Run `make install` first.") from exc

        index_path = Path(self.index_folder) / f"{self.index_name}.bin"
        hnsw_index = hnswlib.Index(space="cosine", dim=int(index.embeddings.shape[1]))
        if index_path.exists():
            hnsw_index.load_index(str(index_path), max_elements=len(index.doc_ids))
            self.index_build_status = "loaded"
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            hnsw_index.init_index(
                max_elements=len(index.doc_ids),
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_m,
            )
            hnsw_index.add_items(
                index.embeddings,
                np.arange(len(index.doc_ids), dtype=np.int64),
                num_threads=-1,
            )
            hnsw_index.save_index(str(index_path))
            self.index_build_status = "built"
        hnsw_index.set_ef(self.hnsw_ef_search)
        index.hnsw_index = hnsw_index
        index.hnsw_index_path = str(index_path)
        return index

    def _prepare_faiss_backend(self, index: DenseIndex) -> DenseIndex:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Missing faiss-cpu. Run `make install` first.") from exc

        index_path = Path(self.index_folder) / f"{self.index_name}.faiss"
        if index_path.exists():
            backend_index = faiss.read_index(str(index_path))
            self.index_build_status = "loaded"
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            spec = self._faiss_index_factory_spec(index.embeddings)
            self.index_build_spec = spec
            backend_index = faiss.index_factory(
                int(index.embeddings.shape[1]),
                spec,
                faiss.METRIC_INNER_PRODUCT,
            )
            train_embeddings = _training_sample(index.embeddings)
            if not backend_index.is_trained:
                backend_index.train(train_embeddings)
            backend_index.add(np.ascontiguousarray(index.embeddings.astype("float32")))
            faiss.write_index(backend_index, str(index_path))
            self.index_build_status = "built"
        index.backend_index = backend_index
        index.backend_index_path = str(index_path)
        index.backend_size_bytes = index_path.stat().st_size
        return index

    def _prepare_scann_index(self, index: DenseIndex) -> DenseIndex:
        try:
            import scann
        except ImportError as exc:
            raise RuntimeError("Missing scann. On Linux/A100 run `.venv/bin/pip install scann`.") from exc

        num_docs = len(index.doc_ids)
        leaves = min(max(1, int(np.sqrt(max(num_docs, 1)))), max(1, num_docs))
        leaves_to_search = min(max(1, leaves // 10), leaves)
        reorder = min(max(self.binary_candidates, 100), num_docs)
        builder = scann.scann_ops_pybind.builder(
            np.ascontiguousarray(index.embeddings.astype("float32")),
            reorder,
            "dot_product",
        )
        if num_docs >= 1000:
            builder = builder.tree(num_leaves=leaves, num_leaves_to_search=leaves_to_search)
        index.backend_index = builder.score_ah(2, anisotropic_quantization_threshold=0.2).reorder(reorder).build()
        index.backend_size_bytes = int(index.embeddings.nbytes)
        self.index_build_status = "built"
        self.index_build_spec = f"scann-ah2.leaves{leaves}.search{leaves_to_search}.reorder{reorder}"
        return index

    def _prepare_binary_rerank_index(self, index: DenseIndex) -> DenseIndex:
        index.binary_codes = _binary_codes(index.embeddings)
        index.backend_size_bytes = self.index_size_bytes(index, "binary_rerank")
        self.index_build_status = "built"
        self.index_build_spec = f"binary-hamming.candidates{self.binary_candidates}.fp16-rerank-store"
        return index

    def _faiss_index_factory_spec(self, embeddings: np.ndarray) -> str:
        num_docs, dim = embeddings.shape
        nlist = self.faiss_nlist or _auto_nlist(num_docs)
        if self.search_mode == "rabitq":
            return f"IVF{nlist},RaBitQ"

        pq_m = _compatible_pq_m(dim, self.faiss_pq_m, num_docs)
        self.faiss_pq_m = pq_m
        return f"OPQ{pq_m},IVF{nlist},PQ{pq_m}"

    def index_metadata(self, index: DenseIndex) -> dict[str, Any]:
        return {
            "search_mode": self.search_mode,
            "embedding_dim": int(index.embeddings.shape[1]),
            "truncate_dim": self.truncate_dim,
            "max_seq_length": self.max_seq_length,
            "precision": self.precision,
            "hnsw_index_path": index.hnsw_index_path,
            "hnsw_index_status": self.index_build_status if self.search_mode == "hnsw" else None,
            "hnsw_m": self.hnsw_m if self.search_mode == "hnsw" else None,
            "hnsw_ef_construction": self.hnsw_ef_construction if self.search_mode == "hnsw" else None,
            "hnsw_ef_search": self.hnsw_ef_search if self.search_mode == "hnsw" else None,
            "backend_index_path": index.backend_index_path,
            "backend_index_status": self.index_build_status
            if self.search_mode in {"opq_ivf_pq", "rabitq", "scann", "binary_rerank"}
            else None,
            "backend_index_spec": self.index_build_spec,
            "backend_size_bytes": index.backend_size_bytes,
            "faiss_nlist": self.faiss_nlist,
            "faiss_pq_m": self.faiss_pq_m if self.search_mode == "opq_ivf_pq" else None,
            "binary_candidates": self.binary_candidates
            if self.search_mode in {"binary_rerank", "scann"}
            else None,
        }


def _build_faiss_index(embeddings: np.ndarray):
    if os.environ.get("RETRIEVAL_PARETO_USE_FAISS") != "1":
        return None
    try:
        import faiss
    except ImportError:
        return None

    embeddings = sanitize_dense_embeddings(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def sanitize_dense_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    if np.isfinite(embeddings).all():
        return embeddings
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nonzero = norms.squeeze(axis=1) > 0
    embeddings[nonzero] = embeddings[nonzero] / norms[nonzero]
    return np.ascontiguousarray(embeddings.astype("float32"))


def _numpy_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.argpartition(-scores, kth=np.arange(k), axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, positions, axis=1)
    order = np.argsort(-top_scores, axis=1)
    positions = np.take_along_axis(positions, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    return top_scores, positions


def _auto_nlist(num_docs: int) -> int:
    if num_docs <= 1:
        return 1
    return max(1, min(max(16, int(4 * np.sqrt(num_docs))), 256, num_docs))


def _compatible_pq_m(dim: int, requested: int, num_docs: int) -> int:
    for candidate in [requested, 32, 24, 16, 12, 8, 6, 4, 2, 1]:
        if candidate <= 0:
            continue
        if dim % candidate == 0 and num_docs >= max(256, candidate * 128):
            return candidate
    for candidate in [requested, 32, 24, 16, 12, 8, 6, 4, 2, 1]:
        if candidate > 0 and dim % candidate == 0:
            return candidate
    return 1


def _training_sample(embeddings: np.ndarray, max_rows: int = 200_000) -> np.ndarray:
    embeddings = sanitize_dense_embeddings(embeddings)
    if len(embeddings) <= max_rows:
        return np.ascontiguousarray(embeddings.astype("float32"))
    rng = np.random.default_rng(13)
    rows = rng.choice(len(embeddings), size=max_rows, replace=False)
    return np.ascontiguousarray(embeddings[rows].astype("float32"))


def _binary_codes(embeddings: np.ndarray) -> np.ndarray:
    return np.packbits(embeddings >= 0, axis=1)


_BITCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _binary_hamming_rerank(
    doc_embeddings: np.ndarray,
    doc_codes: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_codes = _binary_codes(query_embeddings)
    candidate_count = min(max(candidates, k), len(doc_embeddings))
    all_positions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for query_embedding, query_code in zip(query_embeddings, query_codes):
        hamming = _BITCOUNT[np.bitwise_xor(doc_codes, query_code)].sum(axis=1)
        candidate_positions = np.argpartition(hamming, kth=np.arange(candidate_count))[:candidate_count]
        candidate_scores = doc_embeddings[candidate_positions] @ query_embedding
        order = np.argsort(-candidate_scores)[:k]
        all_positions.append(candidate_positions[order].astype(np.int64))
        all_scores.append(candidate_scores[order].astype(np.float32))
    return np.vstack(all_scores), np.vstack(all_positions)


def _scann_search(searcher: Any, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    output = searcher.search_batched(np.ascontiguousarray(embeddings.astype("float32")), final_num_neighbors=k)
    if isinstance(output, tuple):
        positions, scores = output
    else:
        positions, scores = output, np.zeros((len(embeddings), k), dtype=np.float32)
    return np.asarray(positions, dtype=np.int64), np.asarray(scores, dtype=np.float32)
