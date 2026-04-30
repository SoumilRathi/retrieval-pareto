from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable
from typing import Any

import numpy as np

from retrieval_pareto.types import Document, Hit, Query


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


@dataclass
class LateInteractionIndex:
    doc_ids: list[str]
    embeddings: list[np.ndarray]
    plaid_index: Any | None = None
    plaid_index_path: str | None = None
    muvera_embeddings: np.ndarray | None = None
    muvera_processor: Any | None = None
    muvera_size_bytes: int | None = None


@dataclass
class LateInteractionQueryRepr:
    query_ids: list[str]
    embeddings: list[np.ndarray]


class LateInteractionRetriever:
    """Late-interaction retriever supporting exact MaxSim, FastPlaid, and MUVERA.

    Exact MaxSim is used as the quality reference on small corpora. FastPlaid and MUVERA
    provide approximate serving points while preserving the same adapter interface.
    """

    family = "late_interaction"

    def __init__(
        self,
        model_id: str,
        batch_size: int = 16,
        device: str | None = None,
        document_length: int | None = None,
        search_mode: str = "exact",
        index_folder: str = "indexes",
        index_name: str | None = None,
        plaid_nbits: int = 4,
        plaid_n_ivf_probe: int = 8,
        plaid_n_full_scores: int = 8192,
        plaid_kmeans_niters: int = 4,
        plaid_search_batch_size: int = 262144,
        plaid_use_fast: bool = True,
        muvera_k_sim: int = 5,
        muvera_dim_proj: int = 16,
        muvera_r_reps: int = 20,
        muvera_candidates: int = 1000,
        load_model: bool = True,
    ):
        if search_mode not in {"exact", "plaid", "muvera"}:
            raise ValueError("search_mode must be one of: exact, plaid, muvera")

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.search_mode = search_mode
        self.index_folder = index_folder
        self.index_name = index_name or "colbert"
        self.plaid_nbits = plaid_nbits
        self.plaid_n_ivf_probe = plaid_n_ivf_probe
        self.plaid_n_full_scores = plaid_n_full_scores
        self.plaid_kmeans_niters = plaid_kmeans_niters
        self.plaid_search_batch_size = plaid_search_batch_size
        self.plaid_use_fast = plaid_use_fast
        self.muvera_k_sim = muvera_k_sim
        self.muvera_dim_proj = muvera_dim_proj
        self.muvera_r_reps = muvera_r_reps
        self.muvera_candidates = muvera_candidates
        self.index_build_status = "not_applicable"
        self.model = None
        if not load_model:
            return

        try:
            from pylate import models
        except ImportError as exc:
            raise RuntimeError("Missing PyLate. Run `uv sync` first.") from exc

        kwargs = {"model_name_or_path": model_id}
        if self.device:
            kwargs["device"] = self.device
        if document_length:
            kwargs["document_length"] = document_length
        self.model = models.ColBERT(**kwargs)

    def encode_corpus(self, documents: list[Document]) -> LateInteractionIndex:
        if self.model is None:
            raise RuntimeError("LateInteractionRetriever was constructed without a model; cached embeddings are required.")
        embeddings = self.model.encode(
            [doc.text for doc in documents],
            batch_size=self.batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        return LateInteractionIndex(
            doc_ids=[doc.doc_id for doc in documents],
            embeddings=list(_as_arrays(embeddings)),
        )

    def encode_queries(self, queries: list[Query]) -> LateInteractionQueryRepr:
        if self.model is None:
            raise RuntimeError("LateInteractionRetriever was constructed without a model; query latency source is required.")
        embeddings = self.model.encode(
            [query.text for query in queries],
            batch_size=self.batch_size,
            is_query=True,
            show_progress_bar=False,
        )
        return LateInteractionQueryRepr(
            query_ids=[query.query_id for query in queries],
            embeddings=list(_as_arrays(embeddings)),
        )

    def prepare_index(self, index: LateInteractionIndex) -> LateInteractionIndex:
        if self.search_mode == "exact":
            self.index_build_status = "not_applicable"
            return index
        if self.search_mode == "muvera":
            return self._prepare_muvera_index(index)

        try:
            from pylate import indexes
        except ImportError as exc:
            raise RuntimeError("Missing PyLate indexes. Run `make install` first.") from exc

        index_path = Path(self.index_folder) / self.index_name
        mapping_path = index_path / "documents_ids_to_plaid_ids.sqlite"
        exists = mapping_path.exists()
        plaid_index = indexes.PLAID(
            index_folder=self.index_folder,
            index_name=self.index_name,
            override=False,
            use_fast=self.plaid_use_fast,
            nbits=self.plaid_nbits,
            n_ivf_probe=self.plaid_n_ivf_probe,
            n_full_scores=self.plaid_n_full_scores,
            kmeans_niters=self.plaid_kmeans_niters,
            batch_size=self.plaid_search_batch_size,
            show_progress=False,
            device="cuda" if self.device == "cuda" else "cpu",
        )
        if not exists:
            plaid_index = plaid_index.add_documents(
                documents_ids=index.doc_ids,
                documents_embeddings=index.embeddings,
            )
            self.index_build_status = "built"
        else:
            self.index_build_status = "loaded"

        index.plaid_index = plaid_index
        index.plaid_index_path = str(index_path)
        return index

    def _prepare_muvera_index(self, index: LateInteractionIndex) -> LateInteractionIndex:
        if index.muvera_embeddings is not None:
            self.index_build_status = "loaded"
            return index
        try:
            from fastembed.postprocess import Muvera
        except ImportError as exc:
            raise RuntimeError(
                "Missing fastembed MUVERA support. On Linux/A100 run `.venv/bin/pip install fastembed`."
            ) from exc

        if not index.embeddings:
            index.muvera_embeddings = np.empty((0, 0), dtype=np.float32)
            self.index_build_status = "built"
            return index
        dim = int(index.embeddings[0].shape[1])
        processor = Muvera(
            dim=dim,
            k_sim=self.muvera_k_sim,
            dim_proj=self.muvera_dim_proj,
            r_reps=self.muvera_r_reps,
        )
        cache_path = self._muvera_cache_path()
        cached = self._load_muvera_cache(cache_path, index, dim, processor)
        if cached:
            self.index_build_status = "loaded"
            return index

        muvera_embeddings = [
            np.asarray(processor.process_document(doc_embedding), dtype=np.float32)
            for doc_embedding in index.embeddings
        ]
        index.muvera_embeddings = np.vstack(muvera_embeddings).astype("float32")
        index.muvera_processor = processor
        token_values = sum(int(np.prod(embedding.shape)) for embedding in index.embeddings)
        index.muvera_size_bytes = int(index.muvera_embeddings.nbytes + token_values * 2)
        self._save_muvera_cache(cache_path, index, dim)
        self.index_build_status = "built"
        return index

    def _muvera_cache_path(self) -> Path:
        return (
            Path(self.index_folder)
            / f"{self.index_name}.muvera"
            f".ksim{self.muvera_k_sim}"
            f".proj{self.muvera_dim_proj}"
            f".reps{self.muvera_r_reps}.npz"
        )

    def _muvera_cache_metadata(self, dim: int, doc_ids: list[str]) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "model_id": self.model_id,
            "index_name": self.index_name,
            "dim": dim,
            "k_sim": self.muvera_k_sim,
            "dim_proj": self.muvera_dim_proj,
            "r_reps": self.muvera_r_reps,
            "random_seed": 42,
            "doc_count": len(doc_ids),
        }

    def _load_muvera_cache(
        self,
        path: Path,
        index: LateInteractionIndex,
        dim: int,
        processor: Any,
    ) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                expected = self._muvera_cache_metadata(dim, index.doc_ids)
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        return False
                doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
                if doc_ids != index.doc_ids:
                    return False
                index.muvera_embeddings = data["muvera_embeddings"].astype("float32")
                index.muvera_size_bytes = int(data["muvera_size_bytes"].item())
                index.muvera_processor = processor
                index.plaid_index_path = str(path)
                return True
        except Exception:
            return False

    def _save_muvera_cache(self, path: Path, index: LateInteractionIndex, dim: int) -> None:
        if index.muvera_embeddings is None or index.muvera_size_bytes is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._muvera_cache_metadata(dim, index.doc_ids)
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            doc_ids=np.asarray(index.doc_ids),
            muvera_embeddings=index.muvera_embeddings.astype("float32"),
            muvera_size_bytes=np.asarray(index.muvera_size_bytes, dtype=np.int64),
        )

    def search(
        self, query_repr: LateInteractionQueryRepr, index: LateInteractionIndex, k: int
    ) -> list[list[Hit]]:
        if self.search_mode == "plaid":
            if index.plaid_index is None:
                index = self.prepare_index(index)
            return _plaid_rankings(query_repr, index, k)
        if self.search_mode == "muvera":
            if index.muvera_embeddings is None:
                index = self.prepare_index(index)
            return _muvera_rankings(query_repr, index, k, self.muvera_candidates)

        top_k = min(k, len(index.doc_ids))
        rankings: list[list[Hit]] = []
        for query_embedding in query_repr.embeddings:
            scores = np.fromiter(
                (_maxsim_score(query_embedding, doc_embedding) for doc_embedding in index.embeddings),
                dtype=np.float32,
                count=len(index.embeddings),
            )
            positions = np.argpartition(-scores, kth=np.arange(top_k))[:top_k]
            positions = positions[np.argsort(-scores[positions])]
            rankings.append(
                [
                    Hit(doc_id=index.doc_ids[int(pos)], score=float(scores[int(pos)]))
                    for pos in positions
                ]
            )
        return rankings

    def index_size_bytes(self, index: LateInteractionIndex, compression: str = "fp16") -> int:
        if compression == "pq" and index.plaid_index_path:
            return _directory_size_bytes(Path(index.plaid_index_path))
        if compression == "muvera" and index.muvera_size_bytes is not None:
            return index.muvera_size_bytes
        total_values = sum(int(np.prod(embedding.shape)) for embedding in index.embeddings)
        if compression == "fp16":
            return total_values * 2
        if compression == "int8":
            return total_values
        return sum(embedding.nbytes for embedding in index.embeddings)

    def index_metadata(self, index: LateInteractionIndex) -> dict[str, Any]:
        return {
            "search_mode": self.search_mode,
            "device": self.device,
            "plaid_index_path": index.plaid_index_path,
            "plaid_index_status": self.index_build_status,
            "plaid_nbits": self.plaid_nbits if self.search_mode == "plaid" else None,
            "plaid_n_ivf_probe": self.plaid_n_ivf_probe if self.search_mode == "plaid" else None,
            "plaid_n_full_scores": self.plaid_n_full_scores if self.search_mode == "plaid" else None,
            "plaid_search_batch_size": self.plaid_search_batch_size if self.search_mode == "plaid" else None,
            "muvera_index_status": self.index_build_status if self.search_mode == "muvera" else None,
            "muvera_k_sim": self.muvera_k_sim if self.search_mode == "muvera" else None,
            "muvera_dim_proj": self.muvera_dim_proj if self.search_mode == "muvera" else None,
            "muvera_r_reps": self.muvera_r_reps if self.search_mode == "muvera" else None,
            "muvera_candidates": self.muvera_candidates if self.search_mode == "muvera" else None,
            "muvera_size_bytes": index.muvera_size_bytes,
        }


def _as_arrays(embeddings: Iterable) -> Iterable[np.ndarray]:
    for embedding in embeddings:
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 2:
            array = np.reshape(array, (-1, array.shape[-1]))
        yield array


def _maxsim_score(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    token_scores = query_embedding @ doc_embedding.T
    return float(token_scores.max(axis=1).sum())


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _plaid_rankings(
    query_repr: LateInteractionQueryRepr, index: LateInteractionIndex, k: int
) -> list[list[Hit]]:
    query_batch_size = int(os.environ.get("PLAID_QUERY_BATCH", "8"))
    raw_results = []
    for start in range(0, len(query_repr.embeddings), query_batch_size):
        raw_results.extend(
            index.plaid_index(
                queries_embeddings=query_repr.embeddings[start : start + query_batch_size],
                k=k,
            )
        )
    rankings: list[list[Hit]] = []
    for query_results in raw_results:
        hits: list[Hit] = []
        for result in query_results:
            if isinstance(result, dict):
                doc_id = result.get("id")
                score = result.get("score")
            else:
                doc_id = result.id
                score = result.score
            hits.append(Hit(doc_id=str(doc_id), score=float(score)))
        rankings.append(hits)
    return rankings


def _muvera_rankings(
    query_repr: LateInteractionQueryRepr,
    index: LateInteractionIndex,
    k: int,
    candidates: int,
) -> list[list[Hit]]:
    top_k = min(k, len(index.doc_ids))
    candidate_count = min(max(candidates, top_k), len(index.doc_ids))
    rankings: list[list[Hit]] = []
    processor = index.muvera_processor
    if processor is None:
        raise RuntimeError("MUVERA index was not prepared.")
    for query_embedding in query_repr.embeddings:
        query_muvera = np.asarray(processor.process_query(query_embedding), dtype=np.float32)
        coarse_scores = index.muvera_embeddings @ query_muvera
        candidate_positions = np.argpartition(-coarse_scores, kth=np.arange(candidate_count))[
            :candidate_count
        ]
        rerank_scores = np.fromiter(
            (
                _maxsim_score(query_embedding, index.embeddings[int(position)])
                for position in candidate_positions
            ),
            dtype=np.float32,
            count=len(candidate_positions),
        )
        order = np.argsort(-rerank_scores)[:top_k]
        rankings.append(
            [
                Hit(doc_id=index.doc_ids[int(candidate_positions[int(pos)])], score=float(rerank_scores[int(pos)]))
                for pos in order
            ]
        )
    return rankings
