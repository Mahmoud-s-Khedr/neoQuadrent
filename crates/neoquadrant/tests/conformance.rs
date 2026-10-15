//! Runs the shared cases in `tests/conformance/cases.json` (also run by the Node and Python
//! bindings) against the Rust API.

use neoquadrant::requests::*;
use neoquadrant::{Collection, Database, Error};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn from<T: DeserializeOwned>(v: &Value) -> neoquadrant::Result<T> {
    Ok(serde_json::from_value(v.clone())?)
}

fn error_code(e: &Error) -> &'static str {
    match e {
        Error::Engine(_) => "engine",
        Error::InvalidArgument(_) => "invalid_argument",
        Error::CollectionNotFound(_) => "not_found",
        Error::CollectionExists(_) => "exists",
        Error::Closed(_) => "closed",
        Error::Locked(_) => "locked",
        Error::Io(_) => "io",
    }
}

/// Run one step; returns its JSON result.
fn run(
    db: &Database,
    col: &Collection,
    config: &Value,
    op: &str,
    args: &Value,
) -> neoquadrant::Result<Value> {
    let to = |v: &dyn erased::Ser| v.to_json();
    Ok(match op {
        "upsert" => {
            col.upsert(from(args)?)?;
            Value::Null
        }
        "query" => to(&col.query(from(args)?)?),
        "count" => json!(col.count(from(args)?)?),
        "scroll" => to(&col.scroll(from(args)?)?),
        "retrieve" => to(&col.retrieve(from(args)?)?),
        "facet" => to(&col.facet(from(args)?)?),
        "delete" => {
            col.delete(from(args)?)?;
            Value::Null
        }
        "set_payload" => {
            col.set_payload(from(args)?)?;
            Value::Null
        }
        "overwrite_payload" => {
            col.overwrite_payload(from(args)?)?;
            Value::Null
        }
        "delete_payload" => {
            col.delete_payload(from(args)?)?;
            Value::Null
        }
        "clear_payload" => {
            col.clear_payload(from(args)?)?;
            Value::Null
        }
        "create_payload_index" => {
            col.create_payload_index(from(args)?)?;
            Value::Null
        }
        "optimize" => json!(col.optimize()?),
        "create_collection_again" => {
            db.create_collection("c", from(config)?)?;
            Value::Null
        }
        other => panic!("unknown op {other}"),
    })
}

mod erased {
    pub trait Ser {
        fn to_json(&self) -> serde_json::Value;
    }
    impl<T: serde::Serialize> Ser for T {
        fn to_json(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }
}

fn ids_of(result: &Value) -> Vec<Value> {
    let list = result.get("points").unwrap_or(result);
    list.as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].clone())
        .collect()
}

#[test]
fn conformance_cases() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/conformance/cases.json"
    );
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = spec["cases"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let db = Database::open_temp().unwrap();
        let config = &case["config"];
        let col = db
            .create_collection("c", from::<CreateCollection>(config).unwrap())
            .unwrap();
        for (i, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let op = step["op"].as_str().unwrap();
            let ctx = format!("case `{name}` step {i} ({op})");
            let result = run(
                &db,
                &col,
                config,
                op,
                step.get("args").unwrap_or(&Value::Null),
            );
            if let Some(code) = step.get("expect_error") {
                let err = result.expect_err(&format!("{ctx}: expected error"));
                assert_eq!(error_code(&err), code, "{ctx}: {err}");
                continue;
            }
            let result = result.unwrap_or_else(|e| panic!("{ctx}: {e}"));
            if let Some(expected) = step.get("expect") {
                assert_eq!(&result, expected, "{ctx}");
            }
            if let Some(expected) = step.get("expect_ids") {
                assert_eq!(&Value::Array(ids_of(&result)), expected, "{ctx}");
            }
            if let Some(expected) = step.get("expect_next") {
                assert_eq!(&result["next_page_offset"], expected, "{ctx}");
            }
            if let Some(expected) = step.get("expect_scores") {
                for (got, want) in result
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected.as_array().unwrap())
                {
                    let (g, w) = (got["score"].as_f64().unwrap(), want.as_f64().unwrap());
                    assert!((g - w).abs() < 1e-4, "{ctx}: score {g} != {w}");
                }
            }
            if let Some(expected) = step.get("expect_payloads") {
                let got: Vec<Value> = result
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.get("payload").cloned().unwrap_or(json!({})))
                    .collect();
                assert_eq!(&Value::Array(got), expected, "{ctx}");
            }
        }
    }
    println!("{} conformance cases passed", cases.len());
}
