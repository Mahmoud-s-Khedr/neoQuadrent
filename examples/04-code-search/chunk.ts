// Source file → chunks. Syntax-aware with tree-sitter (functions, methods, types, top-level
// statements), falling back to line windows for files without a grammar or that fail to parse.
//
// web-tree-sitter is pinned to 0.25.10 and tree-sitter-wasms to 0.1.13: the prebuilt grammars in
// tree-sitter-wasms were built with tree-sitter-cli 0.20, which web-tree-sitter 0.25 still loads
// but 0.27 does not (Language.load fails).
import { createRequire } from 'node:module'
import path from 'node:path'
import { Language, type Node, Parser } from 'web-tree-sitter'

export interface CodeChunk {
  path: string
  language: string
  /** function | method | class | struct | enum | trait | interface | type | module | statement | window */
  kind: string
  /** Symbol name: `fn`, `Type::method`, `Class.method`; for statements and windows, a short label. */
  name: string
  start_line: number
  end_line: number
  text: string
  /** How the chunk was made: from the syntax tree, or line windows (no grammar / parse failure). */
  via: 'tree-sitter' | 'lines'
}

const GRAMMARS: Record<string, string> = {
  '.rs': 'rust',
  '.ts': 'typescript',
  '.mts': 'typescript',
  '.cts': 'typescript',
  '.tsx': 'tsx',
  '.js': 'javascript',
  '.mjs': 'javascript',
  '.cjs': 'javascript',
  '.py': 'python',
  '.pyi': 'python',
}
/** Indexed without a grammar, as line windows. */
const PLAIN: Record<string, string> = { '.md': 'markdown', '.toml': 'toml', '.prisma': 'prisma', '.sql': 'sql' }

export function languageOf(file: string): string | undefined {
  const ext = path.extname(file)
  return GRAMMARS[ext] ?? PLAIN[ext]
}

/** Chunks longer than this are split into windows (kind and name kept). */
const MAX_LINES = 80
const WINDOW = 60
const OVERLAP = 10
/** Single-line items (`pub mod x;`, `use ...;`) are not worth a chunk. */
const MIN_LINES = 2

let parsers: Promise<Map<string, Parser>> | undefined
async function getParsers(): Promise<Map<string, Parser>> {
  return (parsers ??= (async () => {
    await Parser.init()
    const dir = path.join(path.dirname(createRequire(import.meta.url).resolve('tree-sitter-wasms/package.json')), 'out')
    const map = new Map<string, Parser>()
    for (const lang of new Set(Object.values(GRAMMARS))) {
      const parser = new Parser()
      parser.setLanguage(await Language.load(path.join(dir, `tree-sitter-${lang}.wasm`)))
      map.set(lang, parser)
    }
    return map
  })())
}

/** Line windows over `lines` (0-based `from`..`to` inclusive), as 1-based chunk positions. */
function windows(lines: string[], from: number, to: number, base: Omit<CodeChunk, 'start_line' | 'end_line' | 'text'>): CodeChunk[] {
  const out: CodeChunk[] = []
  const total = to - from + 1
  const parts = total <= MAX_LINES ? 1 : Math.ceil((total - OVERLAP) / (WINDOW - OVERLAP))
  for (let p = 0, start = from; p < parts; p++, start += WINDOW - OVERLAP) {
    const end = parts === 1 ? to : Math.min(to, start + WINDOW - 1)
    const text = lines.slice(start, end + 1).join('\n')
    if (text.trim()) out.push({ ...base, name: parts > 1 ? `${base.name} (part ${p + 1}/${parts})` : base.name, start_line: start + 1, end_line: end + 1, text })
    if (end === to) break
  }
  return out
}

const COMMENTS = new Set(['comment', 'line_comment', 'block_comment', 'attribute_item', 'decorator'])

const KIND: Record<string, string> = {
  function_item: 'function',
  function_declaration: 'function',
  generator_function_declaration: 'function',
  function_definition: 'function',
  function_signature: 'function',
  struct_item: 'struct',
  enum_item: 'enum',
  enum_declaration: 'enum',
  union_item: 'struct',
  trait_item: 'trait',
  class_declaration: 'class',
  abstract_class_declaration: 'class',
  class_definition: 'class',
  interface_declaration: 'interface',
  type_alias_declaration: 'type',
  type_item: 'type',
  macro_definition: 'function',
  mod_item: 'module',
  module: 'module',
  internal_module: 'module',
}

/** Containers whose members become separate chunks. */
const MEMBER_BODIES: Record<string, string> = {
  impl_item: 'body',
  class_declaration: 'body',
  abstract_class_declaration: 'body',
  class_definition: 'body',
  mod_item: 'body',
  internal_module: 'body',
}

const MEMBERS = new Set(['function_item', 'method_definition', 'function_definition', 'decorated_definition', 'method_signature', 'abstract_method_signature', 'public_field_definition'])

function nameOf(node: Node): string {
  const n = node.childForFieldName('name')
  if (n) return n.text
  if (node.type === 'impl_item') {
    const type = node.childForFieldName('type')?.text ?? '?'
    const trait = node.childForFieldName('trait')?.text
    return trait ? `${trait} for ${type}` : type
  }
  return firstLine(node)
}

/** `router.post('/users/login', …)` → "router.post /users/login"; otherwise the first line. */
function statementName(node: Node): string {
  const call = node.namedChildren[0]
  if (call?.type === 'call_expression') {
    const fn = call.childForFieldName('function')?.text
    const arg = call.childForFieldName('arguments')?.namedChildren[0]
    if (fn && fn.length <= 40 && arg && (arg.type === 'string' || arg.type === 'template_string')) return `${fn} ${arg.text.slice(1, -1)}`.slice(0, 80)
  }
  return firstLine(node)
}

function firstLine(node: Node): string {
  return node.text.split('\n')[0].trim().slice(0, 60)
}

/** `export ...`, `declare ...`, `@decorator def ...`: the declaration inside. */
function unwrap(node: Node): Node {
  if (node.type === 'export_statement' || node.type === 'ambient_declaration') {
    const inner = node.childForFieldName('declaration') ?? node.namedChildren.find((c) => c && (KIND[c.type] || c.type === 'lexical_declaration' || c.type === 'variable_declaration'))
    if (inner) return unwrap(inner)
  }
  if (node.type === 'decorated_definition') {
    const inner = node.childForFieldName('definition')
    if (inner) return inner
  }
  return node
}

/** `const f = (…) => {…}` / `const f = function…`: a named function. */
function variableFunction(node: Node): { name: string; kind: string } | undefined {
  if (node.type !== 'lexical_declaration' && node.type !== 'variable_declaration') return undefined
  const decl = node.namedChildren.find((c) => c?.type === 'variable_declarator')
  const value = decl?.childForFieldName('value')
  if (!decl || !value) return undefined
  const name = decl.childForFieldName('name')?.text ?? '?'
  if (['arrow_function', 'function_expression', 'function'].includes(value.type)) return { name, kind: 'function' }
  return { name, kind: 'statement' }
}

export async function chunkFile(file: string, source: string): Promise<CodeChunk[]> {
  const language = languageOf(file)
  if (!language) return []
  const lines = source.split('\n')
  const grammar = GRAMMARS[path.extname(file)]
  const fallback = () => windows(lines, 0, lines.length - 1, { path: file, language, kind: 'window', name: path.basename(file), via: 'lines' })
  if (!grammar) return fallback()

  let tree
  try {
    tree = (await getParsers()).get(grammar)!.parse(source)
  } catch {
    tree = null
  }
  if (!tree) return fallback()
  // tree-sitter recovers from syntax it cannot parse (e.g. newer than the pinned grammar) with ERROR
  // nodes instead of failing. The walker would silently skip that code, so window the whole file.
  if (tree.rootNode.hasError) {
    tree.delete()
    return fallback()
  }

  const out: CodeChunk[] = []
  const emit = (node: Node, startRow: number, kind: string, name: string) => {
    const endRow = node.endPosition.row
    if (endRow - startRow + 1 < MIN_LINES) return
    out.push(...windows(lines, startRow, endRow, { path: file, language, kind, name, via: 'tree-sitter' }))
  }

  /** Walk the children of a container; `owner` prefixes member names (`Type::m`, `Class.m`). */
  const walk = (container: Node, owner?: string) => {
    let pendingStart: number | undefined // first row of the comments/attributes right above a node
    let prevEnd = -2
    for (const child of container.namedChildren) {
      if (!child) continue
      if (COMMENTS.has(child.type)) {
        if (pendingStart === undefined || child.startPosition.row > prevEnd + 1) pendingStart = child.startPosition.row
        prevEnd = child.endPosition.row
        continue
      }
      const start = pendingStart !== undefined && child.startPosition.row <= prevEnd + 1 ? pendingStart : child.startPosition.row
      pendingStart = undefined
      prevEnd = child.endPosition.row
      const node = unwrap(child)
      const sep = grammar === 'rust' ? '::' : '.'
      const qualified = (n: string) => (owner ? `${owner}${sep}${n}` : n)

      // Inline unit tests (`#[cfg(test)] mod tests`): skipped, which keeps the corpus (and
      // jina-code's CPU time) bounded. Test files and directories are excluded per repository.
      if (node.type === 'mod_item' && nameOf(node) === 'tests') continue

      const bodyField = MEMBER_BODIES[node.type]
      const body = bodyField ? node.childForFieldName(bodyField) : null
      if (body && (node.type === 'impl_item' || node.endPosition.row - node.startPosition.row + 1 > MAX_LINES || node.type.startsWith('class'))) {
        // Members as separate chunks; the container's own header goes with the first member's context.
        const name = nameOf(node)
        const before = out.length
        if (node.type === 'mod_item' || node.type === 'internal_module') walk(body, owner)
        else walk(body, qualified(name))
        // No member was big enough to index (a class with only fields, an impl of one-liners): keep it whole.
        if (out.length === before) emit(child, start, KIND[node.type] ?? 'impl', qualified(name))
        continue
      }
      if (owner && MEMBERS.has(node.type)) {
        emit(child, start, node.type === 'public_field_definition' ? 'statement' : 'method', qualified(nameOf(node)))
        continue
      }
      if (KIND[node.type]) {
        emit(child, start, owner && KIND[node.type] === 'function' ? 'method' : KIND[node.type], qualified(nameOf(node)))
        continue
      }
      const v = variableFunction(node)
      if (v) {
        emit(child, start, v.kind, qualified(v.name))
        continue
      }
      // Top-level statements that span several lines, e.g. `router.post('/users/login', async …)`.
      if (!owner && (node.type === 'expression_statement' || node.type === 'macro_invocation')) emit(child, start, 'statement', statementName(node))
    }
  }
  walk(tree.rootNode)
  tree.delete()
  return out.length > 0 ? out : fallback()
}
