'use strict'

// Runs the shared cases in tests/conformance/cases.json (also run by Rust and Python).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { Database } = require('..')

const spec = require(path.resolve(__dirname, '../../../tests/conformance/cases.json'))

const CODES = {
  NEOQ_ENGINE: 'engine',
  NEOQ_INVALID_ARGUMENT: 'invalid_argument',
  NEOQ_NOT_FOUND: 'not_found',
  NEOQ_EXISTS: 'exists',
  NEOQ_CLOSED: 'closed',
  NEOQ_LOCKED: 'locked',
}

function run(db, col, config, op, args) {
  switch (op) {
    case 'upsert': return col.upsert(args)
    case 'query': return col.query(args)
    case 'count': return col.count(args)
    case 'scroll': return col.scroll(args)
    case 'retrieve': return col.retrieve(args.ids, args)
    case 'facet': return col.facet(args)
    case 'delete': return col.delete(args)
    case 'set_payload': return col.setPayload(args.payload, args)
    case 'overwrite_payload': return col.overwritePayload(args.payload, args)
    case 'delete_payload': return col.deletePayload(args.keys, args)
    case 'clear_payload': return col.clearPayload(args)
    case 'create_payload_index': return col.createPayloadIndex(args.field_name, args.field_schema)
    case 'optimize': return col.optimize()
    case 'create_collection_again': return db.createCollection('c', config)
    default: throw new Error(`unknown op ${op}`)
  }
}

const idsOf = (r) => (r.points ?? r).map((p) => p.id)

for (const c of spec.cases) {
  test(c.name, () => {
    const db = new Database()
    const col = db.createCollection('c', c.config)
    c.steps.forEach((step, i) => {
      const ctx = `step ${i} (${step.op})`
      if (step.expect_error) {
        assert.throws(() => run(db, col, c.config, step.op, step.args), (e) => CODES[e.code] === step.expect_error, ctx)
        return
      }
      const result = run(db, col, c.config, step.op, step.args)
      if ('expect' in step) assert.deepEqual(result, step.expect, ctx)
      if ('expect_ids' in step) assert.deepEqual(idsOf(result), step.expect_ids, ctx)
      if ('expect_next' in step) assert.deepEqual(result.next_page_offset, step.expect_next, ctx)
      if ('expect_scores' in step) {
        step.expect_scores.forEach((s, j) => assert.ok(Math.abs(result[j].score - s) < 1e-4, `${ctx}: ${result[j].score} != ${s}`))
      }
      if ('expect_payloads' in step) assert.deepEqual(result.map((p) => p.payload ?? {}), step.expect_payloads, ctx)
    })
    db.close()
  })
}
