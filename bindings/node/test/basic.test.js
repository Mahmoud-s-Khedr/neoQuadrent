'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { Database, NeoquadrantError } = require('..')

const tmpdir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'neoq-node-'))

test('crud, filters, payloads', () => {
  const db = new Database()
  const col = db.createCollection('docs', { vectors: { size: 4, distance: 'Cosine' } })
  col.upsert([
    { id: 1, vector: [1, 0, 0, 0], payload: { city: 'London', price: 10 } },
    { id: 2, vector: new Float32Array([0.9, 0.1, 0, 0]), payload: { city: 'Paris', price: 20 } },
    { id: '5c56c793-69f3-4fbf-87e6-c4bf54c28c26', vector: [0, 1, 0, 0], payload: { city: 'London', price: 30 } },
  ])

  const hits = col.search([1, 0, 0, 0], { limit: 2 })
  assert.deepEqual(hits.map((h) => h.id), [1, 2])
  assert.equal(hits[0].payload.city, 'London')
  assert.ok(Math.abs(hits[0].score - 1) < 1e-5)

  const filtered = col.query({
    query: new Float32Array([1, 0, 0, 0]),
    filter: { must: [{ key: 'city', match: { value: 'London' } }, { key: 'price', range: { gte: 20 } }] },
  })
  assert.deepEqual(filtered.map((h) => h.id), ['5c56c793-69f3-4fbf-87e6-c4bf54c28c26'])

  assert.equal(col.count(), 3)
  assert.equal(col.count({ filter: { must: [{ key: 'city', match: { value: 'London' } }] } }), 2)

  col.setPayload({ stock: true }, [2])
  col.deletePayload(['price'], [2])
  const [p2] = col.retrieve([2], { with_vector: true })
  assert.deepEqual(p2.payload, { city: 'Paris', stock: true })
  assert.equal(p2.vector.length, 4)

  col.delete({ filter: { must: [{ key: 'city', match: { value: 'London' } }] } })
  assert.equal(col.count(), 1)
  assert.deepEqual(col.scroll().points.map((p) => p.id), [2])
  db.close()
})

test('errors carry codes', () => {
  const db = new Database()
  const col = db.createCollection('e', { vectors: { size: 2, distance: 'Dot' } })
  assert.throws(() => db.createCollection('e', { vectors: { size: 2, distance: 'Dot' } }), { code: 'NEOQ_EXISTS' })
  assert.throws(() => db.collection('nope'), { code: 'NEOQ_NOT_FOUND' })
  assert.throws(() => col.upsert([{ id: 1, vector: [1, 2, 3] }]), { code: 'NEOQ_ENGINE' })
  assert.throws(() => col.query({ query: [1, 2], filter: { mustt: [] } }), (e) => e instanceof NeoquadrantError && e.code === 'NEOQ_INVALID_ARGUMENT')
  db.close()
  assert.throws(() => col.count(), { code: 'NEOQ_CLOSED' })
})

test('persists across reopen and locks the directory', () => {
  const dir = tmpdir()
  let db = new Database(dir)
  const col = db.createCollection('a', { vectors: { size: 3, distance: 'Euclid' } })
  col.upsert([{ id: 7, vector: [1, 2, 3], payload: { k: 'v' } }])
  // Same path in one process shares the handle
  const again = new Database(dir)
  assert.deepEqual(again.listCollections(), ['a'])
  db.close()

  db = new Database(dir)
  const hits = db.collection('a').search([1, 2, 4], { limit: 1 })
  assert.equal(hits[0].id, 7)
  assert.ok(Math.abs(hits[0].score - 1) < 1e-5)
  assert.equal(db.dropCollection('a'), true)
  assert.deepEqual(db.listCollections(), [])
  db.close()
  fs.rmSync(dir, { recursive: true })
})

test('lock blocks a second process', () => {
  const dir = tmpdir()
  const db = new Database(dir)
  const { spawnSync } = require('node:child_process')
  const script = `
    const { Database } = require(${JSON.stringify(path.resolve(__dirname, '..'))})
    try { new Database(${JSON.stringify(dir)}); console.log('opened') }
    catch (e) { console.log(e.code) }`
  const out = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8' })
  assert.equal(out.stdout.trim(), 'NEOQ_LOCKED')
  db.close()
  fs.rmSync(dir, { recursive: true })
})

test('async variants run off the event loop', async () => {
  const db = new Database(undefined, { optimizeDebounceMs: 50 })
  const dim = 32
  const col = db.createCollection('v', {
    vectors: { size: dim, distance: 'Dot' },
    optimizers_config: { indexing_threshold: 50 },
  })
  const points = Array.from({ length: 5000 }, (_, i) => ({
    id: i,
    vector: Float32Array.from({ length: dim }, (_, j) => Math.sin(i * 31 + j)),
    payload: { even: i % 2 === 0 },
  }))
  await col.upsertAsync(points)
  assert.equal(col.count(), 5000)

  let ticks = 0
  const timer = setInterval(() => ticks++, 1)
  const results = await Promise.all(
    Array.from({ length: 50 }, (_, i) =>
      col.searchAsync(points[i].vector, { limit: 5, filter: { must: [{ key: 'even', match: { value: true } }] } }),
    ),
  )
  clearInterval(timer)
  assert.equal(results.length, 50)
  assert.ok(results.every((r) => r.length === 5 && r.every((h) => h.payload.even)))

  await col.optimizeAsync()
  assert.ok(col.info().indexed_vectors_count > 0)
  db.close()
})

test('hybrid dense + sparse with RRF fusion', () => {
  const db = new Database()
  const col = db.createCollection('h', {
    vectors: { dense: { size: 3, distance: 'Cosine' } },
    sparse_vectors: { text: {} },
  })
  col.upsert([
    { id: 1, vector: { dense: [1, 0, 0], text: { indices: [1, 7], values: [0.5, 1] } } },
    { id: 2, vector: { dense: [0, 1, 0], text: { indices: [7], values: [2] } } },
    { id: 3, vector: { dense: [0.7, 0.7, 0], text: { indices: [3], values: [1] } } },
  ])
  const fused = col.query({
    prefetch: [
      { query: [1, 0, 0], using: 'dense', limit: 3 },
      { query: { indices: [7], values: [1] }, using: 'text', limit: 3 },
    ],
    query: { fusion: 'rrf' },
    limit: 3,
  })
  assert.equal(fused[0].id, 1)
  db.close()
})
