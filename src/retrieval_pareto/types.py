from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(Protocol):
    model_id: str
    family: str

    def encode_corpus(self, documents: list[Document]):
        ...

    def encode_queries(self, queries: list[Query]):
        ...

    def search(self, query_repr, index, k: int) -> list[list[Hit]]:
        ...

    def index_size_bytes(self, index, compression: str = "fp16") -> int:
        ...
