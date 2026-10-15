'use strict'

const path = require('node:path')
const fs = require('node:fs')

function loadNative() {
  const candidates = [
    `neoquadrant.${process.platform}-${process.arch}.node`,
    'neoquadrant.node',
  ]
  for (const file of candidates) {
    const full = path.join(__dirname, file)
    if (fs.existsSync(full)) return require(full)
  }
  throw new Error(
    `neoquadrant: no native binary for ${process.platform}-${process.arch}. Run \`npm run build\` in ${__dirname}.`,
  )
}

const native = loadNative()

class NeoquadrantError extends Error {
  constructor(code, message) {
    super(message)
    this.name = 'NeoquadrantError'
    this.code = code
  }
}

/** Native errors arrive as "CODE|message". */
function convertError(err) {
  if (err && typeof err.message === 'string') {
    const i = err.message.indexOf('|')
    if (i > 0 && err.message.startsWith('NEOQ_')) {
      return new NeoquadrantError(err.message.slice(0, i), err.message.slice(i + 1))
    }
  }
  return err
}

function call(fn) {
  try {
    return fn()
  } catch (err) {
    throw convertError(err)
  }
}

async function callAsync(fn) {
  try {
    return await fn()
  } catch (err) {
    throw convertError(err)
  }
}

// Typed arrays (Float32Array, …) serialize as plain arrays.
function replacer(_key, value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView) ? Array.from(value) : value
}
const enc = (value) => JSON.stringify(value === undefined ? {} : value, replacer)
const dec = (json) => JSON.parse(json)

/** Accept `[1, 2]`, `{points: [..]}` or `{filter: {..}}`. */
function selector(target) {
  if (Array.isArray(target)) return { points: target }
  if (target && (target.points || target.filter)) return target
  throw new NeoquadrantError('NEOQ_INVALID_ARGUMENT', 'expected an array of ids, {points: [...]} or {filter: {...}}')
}

class Collection {
  /** @param {any} native */
  constructor(native) {
    this._n = native
  }

  get name() {
    return this._n.name
  }

  upsert(points) {
    call(() => this._n.upsert(enc(points)))
  }

  async upsertAsync(points) {
    await callAsync(() => this._n.upsertAsync(enc(points)))
  }

  query(request) {
    return dec(call(() => this._n.query(enc(request))))
  }

  async queryAsync(request) {
    return dec(await callAsync(() => this._n.queryAsync(enc(request))))
  }

  search(vector, options = {}) {
    return this.query({ ...options, query: vector })
  }

  searchAsync(vector, options = {}) {
    return this.queryAsync({ ...options, query: vector })
  }

  retrieve(ids, options = {}) {
    return dec(call(() => this._n.retrieve(enc({ ...options, ids }))))
  }

  scroll(request = {}) {
    return dec(call(() => this._n.scroll(enc(request))))
  }

  count(request = {}) {
    return call(() => this._n.count(enc(request)))
  }

  facet(request) {
    return dec(call(() => this._n.facet(enc(request))))
  }

  delete(target) {
    call(() => this._n.delete(enc(selector(target))))
  }

  setPayload(payload, target) {
    call(() => this._n.setPayload(enc({ ...selector(target), payload })))
  }

  overwritePayload(payload, target) {
    call(() => this._n.overwritePayload(enc({ ...selector(target), payload })))
  }

  deletePayload(keys, target) {
    call(() => this._n.deletePayload(enc({ ...selector(target), keys })))
  }

  clearPayload(target) {
    call(() => this._n.clearPayload(enc(selector(target))))
  }

  createPayloadIndex(fieldName, fieldSchema) {
    call(() => this._n.createPayloadIndex(enc({ field_name: fieldName, field_schema: fieldSchema })))
  }

  deletePayloadIndex(fieldName) {
    call(() => this._n.deletePayloadIndex(fieldName))
  }

  optimize() {
    return call(() => this._n.optimize())
  }

  async optimizeAsync() {
    return dec(await callAsync(() => this._n.optimizeAsync()))
  }

  flush() {
    call(() => this._n.flush())
  }

  info() {
    return dec(call(() => this._n.info()))
  }
}

class Database {
  /**
   * @param {string} [location] directory path; omit or pass ":memory:" for a temporary database
   * @param {object} [options]
   */
  constructor(location, options) {
    this._n = call(() =>
      location === undefined || location === ':memory:'
        ? native.NativeDatabase.openTemp(options)
        : native.NativeDatabase.open(path.resolve(location), options),
    )
  }

  get path() {
    return call(() => this._n.path)
  }

  createCollection(name, config) {
    return new Collection(call(() => this._n.createCollection(name, enc(config))))
  }

  getOrCreateCollection(name, config) {
    return new Collection(call(() => this._n.getOrCreateCollection(name, enc(config))))
  }

  collection(name) {
    return new Collection(call(() => this._n.collection(name)))
  }

  hasCollection(name) {
    return call(() => this._n.hasCollection(name))
  }

  listCollections() {
    return call(() => this._n.listCollections())
  }

  dropCollection(name) {
    return call(() => this._n.dropCollection(name))
  }

  flush() {
    call(() => this._n.flush())
  }

  close() {
    call(() => this._n.close())
  }

  [Symbol.dispose]() {
    this.close()
  }
}

module.exports = { Database, Collection, NeoquadrantError }
