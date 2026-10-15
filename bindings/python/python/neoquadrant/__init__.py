"""Embedded Qdrant: in-process vector search, no server.

>>> import neoquadrant
>>> with neoquadrant.Database("./vectors") as db:
...     docs = db.get_or_create_collection("docs", {"vectors": {"size": 4, "distance": "Cosine"}})
...     docs.upsert([{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"lang": "en"}}])
...     hits = docs.search([0.1, 0.2, 0.3, 0.4], limit=5,
...                        filter={"must": [{"key": "lang", "match": {"value": "en"}}]})

Requests and results use Qdrant's REST API shapes (filters, payload schemas, query, prefetch),
so Qdrant's documentation applies. Vectors may be lists or numpy arrays.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from ._native import (
    ClosedError,
    CollectionExistsError,
    CollectionNotFoundError,
    EngineError,
    InvalidArgumentError,
    LockedError,
    NativeDatabase as _NativeDatabase,
    NeoquadrantError,
)

__all__ = [
    "Database",
    "Collection",
    "NeoquadrantError",
    "EngineError",
    "LockedError",
    "CollectionExistsError",
    "CollectionNotFoundError",
    "ClosedError",
    "InvalidArgumentError",
]

PointId = Union[int, str]


def _plain(obj: Any) -> Any:
    """Convert numpy arrays/scalars (anything with ``tolist``) to plain Python, recursively."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return obj
    tolist = getattr(obj, "tolist", None)
    if tolist is not None:
        return tolist()
    return obj


def _selector(target: Any) -> dict:
    if isinstance(target, Mapping):
        if "points" in target or "filter" in target:
            return _plain(dict(target))
        raise InvalidArgumentError("expected a list of ids, {'points': [...]} or {'filter': {...}}")
    return {"points": _plain(list(target))}


def _drop_none(**kwargs: Any) -> dict:
    return {k: _plain(v) for k, v in kwargs.items() if v is not None}


class Collection:
    """A collection of points. Thread-safe; the GIL is released during engine calls."""

    __slots__ = ("_n",)

    def __init__(self, native: Any) -> None:
        self._n = native

    def __repr__(self) -> str:
        return f"Collection({self.name!r})"

    @property
    def name(self) -> str:
        return self._n.name

    # ---- writes

    def upsert(self, points: Iterable[Mapping[str, Any]]) -> None:
        """Insert or replace points: ``[{"id": 1, "vector": [...], "payload": {...}}, ...]``."""
        self._n.upsert(_plain(list(points)))

    def upsert_batch(
        self,
        ids: Sequence[PointId],
        vectors: Any,
        payloads: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        """Column-wise upsert, e.g. ``upsert_batch(ids, np_matrix, payloads)``."""
        ids = _plain(ids)
        vectors = _plain(vectors)
        if len(ids) != len(vectors) or (payloads is not None and len(payloads) != len(ids)):
            raise InvalidArgumentError("ids, vectors and payloads must have the same length")
        payloads = _plain(payloads) if payloads is not None else [None] * len(ids)
        self._n.upsert([{"id": i, "vector": v, "payload": p} for i, v, p in zip(ids, vectors, payloads)])

    def delete(self, target: Any) -> None:
        """Delete by ids (``[1, 2]``) or by filter (``{"filter": {...}}``)."""
        self._n.delete(_selector(target))

    def set_payload(self, payload: Mapping[str, Any], target: Any, *, key: Optional[str] = None) -> None:
        """Merge ``payload`` into the selected points (optionally under a nested ``key``)."""
        self._n.set_payload({**_selector(target), "payload": _plain(dict(payload)), **({"key": key} if key else {})})

    def overwrite_payload(self, payload: Mapping[str, Any], target: Any) -> None:
        self._n.overwrite_payload({**_selector(target), "payload": _plain(dict(payload))})

    def delete_payload(self, keys: Sequence[str], target: Any) -> None:
        self._n.delete_payload({**_selector(target), "keys": list(keys)})

    def clear_payload(self, target: Any) -> None:
        self._n.clear_payload(_selector(target))

    def create_payload_index(self, field_name: str, field_schema: Union[str, Mapping[str, Any]]) -> None:
        """Index a payload field (``"keyword"``, ``"integer"``, ``"float"``, ``"bool"``, ``"geo"``,
        ``"datetime"``, ``"text"``, ``"uuid"``) so filters on it are fast."""
        self._n.create_payload_index({"field_name": field_name, "field_schema": _plain(field_schema)})

    def delete_payload_index(self, field_name: str) -> None:
        self._n.delete_payload_index(field_name)

    # ---- reads

    def query(self, request: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> list[dict]:
        """Universal query (Qdrant ``/points/query``): nearest, hybrid prefetch + fusion, order_by."""
        return self._n.query(_plain({**(request or {}), **kwargs}))

    def search(
        self,
        vector: Any,
        limit: int = 10,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        using: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        score_threshold: Optional[float] = None,
        offset: int = 0,
        with_payload: Any = True,
        with_vector: Any = False,
    ) -> list[dict]:
        """Nearest neighbours of ``vector``."""
        return self._n.query(
            _drop_none(
                query=vector,
                limit=limit,
                filter=filter,
                using=using,
                params=params,
                score_threshold=score_threshold,
                offset=offset,
                with_payload=with_payload,
                with_vector=with_vector,
            )
        )

    def retrieve(self, ids: Sequence[PointId], *, with_payload: Any = True, with_vector: Any = False) -> list[dict]:
        return self._n.retrieve({"ids": _plain(list(ids)), "with_payload": with_payload, "with_vector": with_vector})

    def scroll(
        self,
        limit: int = 10,
        *,
        offset: Optional[PointId] = None,
        filter: Optional[Mapping[str, Any]] = None,
        order_by: Any = None,
        with_payload: Any = True,
        with_vector: Any = False,
    ) -> dict:
        """One page of points: ``{"points": [...], "next_page_offset": id | None}``."""
        return self._n.scroll(
            _drop_none(limit=limit, offset=offset, filter=filter, order_by=order_by, with_payload=with_payload, with_vector=with_vector)
        )

    def count(self, filter: Optional[Mapping[str, Any]] = None, *, exact: bool = True) -> int:
        return self._n.count(_drop_none(filter=filter, exact=exact))

    def facet(self, key: str, limit: int = 10, *, filter: Optional[Mapping[str, Any]] = None, exact: bool = True) -> list[dict]:
        """Value counts of an indexed keyword/integer/bool/uuid payload field."""
        return self._n.facet(_drop_none(key=key, limit=limit, filter=filter, exact=exact))

    # ---- maintenance

    def optimize(self) -> bool:
        """Run Qdrant's optimizers now (HNSW indexing, merging, vacuum). Blocks until done."""
        return self._n.optimize()

    def flush(self) -> None:
        self._n.flush()

    def info(self) -> dict:
        return self._n.info()


class Database:
    """An embedded Qdrant database: a directory holding collections.

    ``Database()`` or ``Database(":memory:")`` creates a temporary database deleted on close.

    :param durability: ``"full"`` (flush every write, default for files), ``"periodic"``
        (flush every ``flush_interval`` seconds) or ``"manual"`` (default for temporary DBs).
    :param auto_optimize: build HNSW indexes / merge segments in a background thread.
    """

    __slots__ = ("_n",)

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]", None] = None,
        *,
        durability: Optional[str] = None,
        flush_interval: float = 0.2,
        auto_optimize: bool = True,
        optimize_debounce: float = 0.5,
        max_indexing_threads: int = 0,
    ) -> None:
        temp = path is None or str(path) == ":memory:"
        if durability is None:
            durability = "manual" if temp else "full"
        self._n = _NativeDatabase.open(
            None if temp else os.fspath(path),
            durability,
            flush_interval,
            auto_optimize,
            optimize_debounce,
            max_indexing_threads,
        )

    def __repr__(self) -> str:
        try:
            return f"Database({str(self.path)!r})"
        except ClosedError:
            return "Database(<closed>)"

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def path(self) -> "os.PathLike[str]":
        return self._n.path

    def create_collection(self, name: str, config: Mapping[str, Any]) -> Collection:
        """Create a collection, e.g. ``{"vectors": {"size": 384, "distance": "Cosine"}}``."""
        return Collection(self._n.create_collection(name, _plain(dict(config))))

    def get_or_create_collection(self, name: str, config: Mapping[str, Any]) -> Collection:
        return Collection(self._n.get_or_create_collection(name, _plain(dict(config))))

    def collection(self, name: str) -> Collection:
        """Raises :class:`CollectionNotFoundError` if it does not exist."""
        return Collection(self._n.collection(name))

    def __getitem__(self, name: str) -> Collection:
        return self.collection(name)

    def __contains__(self, name: str) -> bool:
        return self._n.has_collection(name)

    def list_collections(self) -> list[str]:
        return self._n.list_collections()

    def drop_collection(self, name: str) -> bool:
        return self._n.drop_collection(name)

    def flush(self) -> None:
        self._n.flush()

    def close(self) -> None:
        """Stop background work, flush, and release the directory lock."""
        self._n.close()
