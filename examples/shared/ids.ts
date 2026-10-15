import crypto from 'node:crypto'

/** UUID v5 (SHA-1, URL namespace): a stable point id derived from a name. */
export function uuidV5(name: string): string {
  const ns = Buffer.from('6ba7b8119dad11d180b400c04fd430c8', 'hex')
  const h = crypto.createHash('sha1').update(ns).update(name).digest()
  h[6] = (h[6] & 0x0f) | 0x50
  h[8] = (h[8] & 0x3f) | 0x80
  const x = h.subarray(0, 16).toString('hex')
  return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`
}
