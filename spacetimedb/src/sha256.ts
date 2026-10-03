const K: number[] = []
const H0: number[] = []
for (let n = 2; K.length < 64; n++) {
  let prime = true
  for (let d = 2; d * d <= n; d++) if (n % d === 0) { prime = false; break }
  if (!prime) continue
  if (H0.length < 8) H0.push(((Math.sqrt(n) % 1) * 2 ** 32) >>> 0)
  K.push(((Math.cbrt(n) % 1) * 2 ** 32) >>> 0)
}

const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n))

// Input must be ASCII (callers only pass hex strings).
export function sha256Hex(input: string): string {
  const bytes: number[] = []
  for (let i = 0; i < input.length; i++) bytes.push(input.charCodeAt(i) & 0xff)
  const bitLen = bytes.length * 8
  bytes.push(0x80)
  while (bytes.length % 64 !== 56) bytes.push(0)
  const hi = Math.floor(bitLen / 2 ** 32)
  for (const word of [hi, bitLen >>> 0]) {
    bytes.push((word >>> 24) & 0xff, (word >>> 16) & 0xff, (word >>> 8) & 0xff, word & 0xff)
  }

  const h = H0.slice()
  const w = new Array<number>(64)
  for (let off = 0; off < bytes.length; off += 64) {
    for (let i = 0; i < 16; i++) {
      const j = off + i * 4
      w[i] = ((bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3]) >>> 0
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0
    }
    let [a, b, c, d, e, f, g, hh] = h
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
      const ch = (e & f) ^ (~e & g)
      const t1 = (hh + S1 + ch + K[i] + w[i]) >>> 0
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
      const maj = (a & b) ^ (a & c) ^ (b & c)
      const t2 = (S0 + maj) >>> 0
      hh = g; g = f; f = e; e = (d + t1) >>> 0
      d = c; c = b; b = a; a = (t1 + t2) >>> 0
    }
    const next = [a, b, c, d, e, f, g, hh]
    for (let i = 0; i < 8; i++) h[i] = (h[i] + next[i]) >>> 0
  }
  return h.map(x => x.toString(16).padStart(8, '0')).join('')
}
