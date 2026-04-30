from __future__ import annotations

from dataclasses import dataclass

from retrieval_pareto.types import Document, Query


DATASET_IDS = {
    "nfcorpus": "mteb/nfcorpus",
    "scifact": "mteb/scifact",
    "fiqa": "mteb/fiqa",
    "arguana": "mteb/arguana",
    "scidocs": "mteb/scidocs",
    "trec-covid": "mteb/trec-covid",
}

LIMIT_DATASET_ID = "orionweller/LIMIT"
BRIGHT_DATASET_ID = "xlangai/BRIGHT"
BRIGHT_DATASETS = {
    "biology",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "leetcode",
}


@dataclass(frozen=True)
class RetrievalDataset:
    benchmark: str
    name: str
    hf_id: str
    split: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, float]]
    qrels_rows: int


def _row_id(row: dict, *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    raise KeyError(f"None of these id columns were present: {names}")


def _row_text(row: dict) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_beir_dataset(name: str, split: str = "test", limit: int | None = None) -> RetrievalDataset:
    """Load the BEIR-style datasets hosted by MTEB on Hugging Face.

    The MTEB mirrors expose qrels in the default config and separate corpus/queries configs.
    We evaluate only queries that have qrels in the requested split.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: `uv sync` or `uv pip install -e .`.") from exc

    if name not in DATASET_IDS:
        choices = ", ".join(sorted(DATASET_IDS))
        raise ValueError(f"Unsupported BEIR dataset '{name}'. Available: {choices}")

    hf_id = DATASET_IDS[name]
    corpus_rows = load_dataset(hf_id, "corpus", split="corpus")
    query_rows = load_dataset(hf_id, "queries", split="queries")
    qrel_rows = load_dataset(hf_id, "default", split=split)

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
    if limit is not None:
        queries = queries[:limit]
        keep = {query.query_id for query in queries}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in keep}
        qrels_rows = sum(len(rels) for rels in qrels.values())

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
        split=split,
        documents=documents,
        queries=queries,
        qrels=qrels,
        qrels_rows=qrels_rows,
    )


def load_limit_dataset(split: str = "test", limit: int | None = None) -> RetrievalDataset:
    """Load the LIMIT separation benchmark from Hugging Face.

    LIMIT uses the same corpus / queries / qrels shape as the MTEB BEIR mirrors, but it is
    semantically different enough to keep under its own benchmark namespace.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: `uv sync` or `uv pip install -e .`.") from exc

    corpus_rows = load_dataset(LIMIT_DATASET_ID, "corpus", split="corpus")
    query_rows = load_dataset(LIMIT_DATASET_ID, "queries", split="queries")
    qrel_rows = load_dataset(LIMIT_DATASET_ID, "default", split=split)

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
    if limit is not None:
        queries = queries[:limit]
        keep = {query.query_id for query in queries}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in keep}
        qrels_rows = sum(len(rels) for rels in qrels.values())

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
        split=split,
        documents=documents,
        queries=queries,
        qrels=qrels,
        qrels_rows=qrels_rows,
    )


def load_bright_dataset(name: str, limit: int | None = None) -> RetrievalDataset:
    """Load one BRIGHT split as a retrieval dataset.

    BRIGHT examples provide natural queries and gold chunk ids. The `documents` config exposes
    the chunked corpus whose ids match `gold_ids`.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: `uv sync` or `uv pip install -e .`.") from exc

    if name not in BRIGHT_DATASETS:
        choices = ", ".join(sorted(BRIGHT_DATASETS))
        raise ValueError(f"Unsupported BRIGHT dataset '{name}'. Available: {choices}")

    query_rows = load_dataset(BRIGHT_DATASET_ID, "examples", split=name)
    corpus_rows = load_dataset(BRIGHT_DATASET_ID, "documents", split=name)

    queries: list[Query] = []
    qrels: dict[str, dict[str, float]] = {}
    for row in query_rows:
        query_id = _row_id(row, "id")
        queries.append(Query(query_id=query_id, text=str(row["query"])))
        gold_ids = row.get("gold_ids") or []
        qrels[query_id] = {str(doc_id): 1.0 for doc_id in gold_ids}

    queries.sort(key=lambda query: query.query_id)
    if limit is not None:
        queries = queries[:limit]
        keep = {query.query_id for query in queries}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in keep}

    documents = [
        Document(
            doc_id=_row_id(row, "id"),
            text=str(row.get("content") or ""),
            title="",
        )
        for row in corpus_rows
    ]

    return RetrievalDataset(
        benchmark="bright",
        name=name,
        hf_id=BRIGHT_DATASET_ID,
        split=name,
        documents=documents,
        queries=queries,
        qrels=qrels,
        qrels_rows=sum(len(rels) for rels in qrels.values()),
    )
