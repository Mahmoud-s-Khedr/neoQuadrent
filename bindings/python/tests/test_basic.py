import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import neoquadrant
from neoquadrant import Database


def test_crud_filters_payloads():
    with Database() as db:
        col = db.create_collection("docs", {"vectors": {"size": 4, "distance": "Cosine"}})
        col.upsert(
            [
                {"id": 1, "vector": [1, 0, 0, 0], "payload": {"city": "London", "price": 10}},
                {"id": 2, "vector": np.array([0.9, 0.1, 0, 0], dtype=np.float32), "payload": {"city": "Paris", "price": 20}},
                {"id": "5c56c793-69f3-4fbf-87e6-c4bf54c28c26", "vector": [0, 1, 0, 0], "payload": {"city": "London", "price": 30}},
            ]
        )
        hits = col.search(np.array([1, 0, 0, 0], dtype=np.float32), limit=2)
        assert [h["id"] for h in hits] == [1, 2]
        assert hits[0]["payload"]["city"] == "London"
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

        filtered = col.search(
            [1, 0, 0, 0],
            filter={"must": [{"key": "city", "match": {"value": "London"}}, {"key": "price", "range": {"gte": 20}}]},
        )
        assert [h["id"] for h in filtered] == ["5c56c793-69f3-4fbf-87e6-c4bf54c28c26"]

        assert col.count() == 3
        assert col.count({"must": [{"key": "city", "match": {"value": "London"}}]}) == 2

        col.set_payload({"stock": True}, [2])
        col.delete_payload(["price"], [2])
        (p2,) = col.retrieve([2], with_vector=True)
        assert p2["payload"] == {"city": "Paris", "stock": True}
        assert len(p2["vector"]) == 4

        col.delete({"filter": {"must": [{"key": "city", "match": {"value": "London"}}]}})
        assert col.count() == 1
        assert [p["id"] for p in col.scroll()["points"]] == [2]


def test_errors():
    db = Database()
    col = db.create_collection("e", {"vectors": {"size": 2, "distance": "Dot"}})
    with pytest.raises(neoquadrant.CollectionExistsError):
        db.create_collection("e", {"vectors": {"size": 2, "distance": "Dot"}})
    with pytest.raises(neoquadrant.CollectionNotFoundError):
        db["nope"]
    with pytest.raises(neoquadrant.EngineError):
        col.upsert([{"id": 1, "vector": [1, 2, 3]}])
    with pytest.raises(neoquadrant.InvalidArgumentError):
        col.search([1, 2], filter={"mustt": []})
    assert issubclass(neoquadrant.LockedError, neoquadrant.NeoquadrantError)
    db.close()
    with pytest.raises(neoquadrant.ClosedError):
        col.count()


def test_persistence_and_lock(tmp_path):
    db = Database(tmp_path)
    col = db.create_collection("a", {"vectors": {"size": 3, "distance": "Euclid"}})
    col.upsert([{"id": 7, "vector": [1, 2, 3], "payload": {"k": "v"}}])
    assert "a" in Database(tmp_path)  # same path, same process: shared handle

    code = f"""
import neoquadrant
try:
    neoquadrant.Database({str(tmp_path)!r}); print("opened")
except neoquadrant.LockedError:
    print("locked")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "locked"
    db.close()

    with Database(tmp_path) as db:
        (hit,) = db["a"].search([1, 2, 4], limit=1)
        assert hit["id"] == 7
        assert hit["score"] == pytest.approx(1.0, abs=1e-5)
        assert db.drop_collection("a")
        assert db.list_collections() == []


def test_numpy_batch_optimizer_and_gil_release():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5000, 32)).astype(np.float32)
    with Database(optimize_debounce=0.05) as db:
        col = db.create_collection(
            "v", {"vectors": {"size": 32, "distance": "Cosine"}, "optimizers_config": {"indexing_threshold": 50}}
        )
        col.upsert_batch(np.arange(5000), data, [{"even": bool(i % 2 == 0)} for i in range(5000)])
        assert col.count() == 5000

        # Searches from many threads at once: only works well if the GIL is released.
        results = []

        def worker(i):
            results.append(col.search(data[i], limit=5, filter={"must": [{"key": "even", "match": {"value": True}}]}))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 32 and all(len(r) == 5 and all(h["payload"]["even"] for h in r) for r in results)

        col.optimize()
        assert col.info()["indexed_vectors_count"] > 0


def test_hybrid_fusion():
    with Database() as db:
        col = db.create_collection(
            "h", {"vectors": {"dense": {"size": 3, "distance": "Cosine"}}, "sparse_vectors": {"text": {}}}
        )
        col.upsert(
            [
                {"id": 1, "vector": {"dense": [1, 0, 0], "text": {"indices": [1, 7], "values": [0.5, 1.0]}}},
                {"id": 2, "vector": {"dense": [0, 1, 0], "text": {"indices": [7], "values": [2.0]}}},
                {"id": 3, "vector": {"dense": [0.7, 0.7, 0], "text": {"indices": [3], "values": [1.0]}}},
            ]
        )
        fused = col.query(
            prefetch=[
                {"query": [1, 0, 0], "using": "dense", "limit": 3},
                {"query": {"indices": [7], "values": [1.0]}, "using": "text", "limit": 3},
            ],
            query={"fusion": "rrf"},
            limit=3,
        )
        assert fused[0]["id"] == 1


def test_payload_index_facet():
    with Database() as db:
        col = db.create_collection("f", {"vectors": {"size": 2, "distance": "Dot"}})
        col.create_payload_index("color", "keyword")
        col.upsert_batch(list(range(30)), np.ones((30, 2)), [{"color": ["red", "green", "blue"][i % 3]} for i in range(30)])
        facets = col.facet("color")
        assert sorted(h["value"] for h in facets) == ["blue", "green", "red"]
        assert all(h["count"] == 10 for h in facets)
