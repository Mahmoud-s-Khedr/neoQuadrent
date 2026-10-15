// Build the native module with cargo and copy it next to index.js as neoquadrant.<platform>-<arch>.node
'use strict'
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const debug = process.argv.includes('--debug')
const root = path.resolve(__dirname, '../..')
execFileSync('cargo', ['build', '-p', 'neoquadrant-node', ...(debug ? [] : ['--release'])], { cwd: root, stdio: 'inherit' })

const lib = { linux: 'libneoquadrant_node.so', darwin: 'libneoquadrant_node.dylib', win32: 'neoquadrant_node.dll' }[process.platform]
const src = path.join(root, 'target', debug ? 'debug' : 'release', lib)
const dest = path.join(__dirname, `neoquadrant.${process.platform}-${process.arch}.node`)
fs.copyFileSync(src, dest)
console.log(`built ${path.relative(process.cwd(), dest)}`)
