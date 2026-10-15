"""Runs the shared cases in tests/conformance/cases.json (also run by Rust and Node)."""

import json
from pathlib import Path

import pytest

import neoquadrant
from neoquadrant import Database

SPEC = json.loads((Path(__file__).resolve().parents[3] / "tests" / "conformance" / "cases.json").read_text())

CODES = {
    neoquadrant.EngineError: "engine",
    neoquadrant.InvalidArgumentError: "invalid_argument",
    neoquadrant.CollectionNotFoundError: "not_found",
    neoquadrant.CollectionExistsError: "exists",
    neoquadrant.ClosedError: "closed",
    neoquadrant.LockedError: "locked",
}


def run(db, col, config, op, args):
    if op == "upsert":
        return col.upsert(args)
    if op == "query":
        return col.query(args)
    if op == "count":
        return col.count(args.get("filter"), exact=args.get("exact", True))
    if op == "scroll":
        return col.scroll(**args)
    if op == "retrieve":
        return col.retrieve(args["ids"])
    if op == "facet":
        return col.facet(**args)
    if op == "delete":
        return col.delete(args)
    if op == "set_payload":
        return col.set_payload(args["payload"], args)
    if op == "overwrite_payload":
        return col.overwrite_payload(args["payload"], args)
    if op == "delete_payload":
        return col.delete_payload(args["keys"], args)
    if op == "clear_payload":
        return col.clear_payload(args)
    if op == "create_payload_index":
        return col.create_payload_index(args["field_name"], args["field_schema"])
    if op == "optimize":
        return col.optimize()
    if op == "create_collection_again":
        return db.create_collection("c", config)
    raise ValueError(f"unknown op {op}")


def ids_of(result):
    points = result["points"] if isinstance(result, dict) else result
    return [p["id"] for p in points]


@pytest.mark.parametrize("case", SPEC["cases"], ids=[c["name"] for c in SPEC["cases"]])
def test_conformance(case):
    with Database() as db:
        col = db.create_collection("c", case["config"])
        for i, step in enumerate(case["steps"]):
            ctx = f"step {i} ({step['op']})"
            args = step.get("args")
            if "expect_error" in step:
                with pytest.raises(neoquadrant.NeoquadrantError) as err:
                    run(db, col, case["config"], step["op"], args)
                assert CODES[type(err.value)] == step["expect_error"], ctx
                continue
            result = run(db, col, case["config"], step["op"], args)
            if "expect" in step:
                assert result == step["expect"], ctx
            if "expect_ids" in step:
                assert ids_of(result) == step["expect_ids"], ctx
            if "expect_next" in step:
                assert result["next_page_offset"] == step["expect_next"], ctx
            if "expect_scores" in step:
                assert [h["score"] for h in result] == pytest.approx(step["expect_scores"], abs=1e-4), ctx
            if "expect_payloads" in step:
                assert [p.get("payload") or {} for p in result] == step["expect_payloads"], ctx
