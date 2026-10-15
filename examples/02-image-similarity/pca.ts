// PCA without dependencies: covariance matrix + cyclic Jacobi eigendecomposition. Exact, and
// fast enough for 512 dimensions (a few seconds).

export interface Pca {
  mean: Float64Array
  /** `k` principal axes, each of length `dim`, by decreasing variance. */
  components: Float64Array[]
  /** Fraction of the total variance the `k` components keep. */
  explained: number
  project(v: number[]): number[]
}

/** Eigenvalues and eigenvectors (columns of `vectors`, row-major n×n) of a symmetric matrix. */
export function symmetricEigen(matrix: Float64Array, n: number, maxSweeps = 50): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(matrix)
  const v = new Float64Array(n * n)
  for (let i = 0; i < n; i++) v[i * n + i] = 1
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0
    let diag = 0
    for (let i = 0; i < n; i++) {
      diag += a[i * n + i] ** 2
      for (let j = i + 1; j < n; j++) off += a[i * n + j] ** 2
    }
    if (off <= 1e-22 * diag) break
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q]
        if (Math.abs(apq) < 1e-300) continue
        const app = a[p * n + p]
        const aqq = a[q * n + q]
        const theta = (aqq - app) / (2 * apq)
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        // A ← Jᵀ A J, touching rows/columns p and q only.
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p]
          const akq = a[k * n + q]
          a[k * n + p] = c * akp - s * akq
          a[k * n + q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k]
          const aqk = a[q * n + k]
          a[p * n + k] = c * apk - s * aqk
          a[q * n + k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p]
          const vkq = v[k * n + q]
          v[k * n + p] = c * vkp - s * vkq
          v[k * n + q] = s * vkp + c * vkq
        }
      }
    }
  }
  const values = new Float64Array(n)
  for (let i = 0; i < n; i++) values[i] = a[i * n + i]
  return { values, vectors: v }
}

/** Fit PCA on `data` (rows) and keep `k` components. */
export function fitPca(data: number[][], k: number): Pca {
  const n = data[0].length
  const mean = new Float64Array(n)
  for (const row of data) for (let i = 0; i < n; i++) mean[i] += row[i] / data.length
  const cov = new Float64Array(n * n)
  const centered = new Float64Array(n)
  for (const row of data) {
    for (let i = 0; i < n; i++) centered[i] = row[i] - mean[i]
    for (let i = 0; i < n; i++) {
      const ci = centered[i]
      for (let j = i; j < n; j++) cov[i * n + j] += ci * centered[j]
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      cov[i * n + j] /= data.length - 1
      cov[j * n + i] = cov[i * n + j]
    }
  }
  const { values, vectors } = symmetricEigen(cov, n)
  const order = [...values.keys()].sort((x, y) => values[y] - values[x])
  const components = order.slice(0, k).map((col) => Float64Array.from({ length: n }, (_, row) => vectors[row * n + col]))
  const total = values.reduce((s, x) => s + Math.max(x, 0), 0)
  const kept = order.slice(0, k).reduce((s, i) => s + Math.max(values[i], 0), 0)
  return {
    mean,
    components,
    explained: kept / total,
    project: (v) => components.map((c) => c.reduce((s, ci, i) => s + ci * (v[i] - mean[i]), 0)),
  }
}
