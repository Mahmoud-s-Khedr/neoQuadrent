import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const EXAMPLES_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
/** Everything downloaded or generated at runtime (gitignored). */
export const DATA_DIR = path.join(EXAMPLES_DIR, '.data')
export const MODELS_DIR = path.join(DATA_DIR, 'models')
export const DATASETS_DIR = path.join(DATA_DIR, 'datasets')
export const DB_DIR = path.join(DATA_DIR, 'db')

export const args = process.argv.slice(2)
/** `--quick`: small dataset, used by `npm run e2e`. */
export const QUICK = args.includes('--quick')
export function flag(name: string): string | undefined {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : undefined
}
