// SignalSense detection worker.
// Runs a YOLOv8 ONNX model with ONNX Runtime Web (WebAssembly) off the main thread,
// so the camera view stays smooth while the model works.
import * as ort from './ort/ort.wasm.min.mjs';

let session = null;
let S = 640;
let canvas = null;
let ctx = null;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

self.onmessage = async (e) => {
  const m = e.data;

  if (m.type === 'load') {
    try {
      ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
      ort.env.wasm.numThreads = 1;
      if (session) { try { await session.release(); } catch (_) {} session = null; }
      const t0 = performance.now();
      const src = m.buffer ? new Uint8Array(m.buffer) : new URL(m.url, m.base).href;
      session = await ort.InferenceSession.create(src, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      S = m.imgsz || 640;
      canvas = new OffscreenCanvas(S, S);
      ctx = canvas.getContext('2d', { willReadFrequently: true });
      const t1 = performance.now();
      const dummy = new ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]);
      const out = await session.run({ [session.inputNames[0]]: dummy });
      const o = out[session.outputNames[0]];
      self.postMessage({ type: 'ready', loadMs: t1 - t0, warmMs: performance.now() - t1, dims: o.dims, imgsz: S });
    } catch (err) {
      self.postMessage({ type: 'error', message: String((err && err.message) || err) });
    }
    return;
  }

  if (m.type === 'detect') {
    if (!session) { m.bitmap.close(); self.postMessage({ type: 'skip', id: m.id }); return; }
    try {
      const bm = m.bitmap;
      const W = bm.width, H = bm.height;
      const r = Math.min(S / W, S / H);
      const nw = Math.round(W * r), nh = Math.round(H * r);
      ctx.fillStyle = 'rgb(114,114,114)';
      ctx.fillRect(0, 0, S, S);
      ctx.drawImage(bm, 0, 0, nw, nh);
      bm.close();
      const px = ctx.getImageData(0, 0, S, S).data;
      const n = S * S;
      const f = new Float32Array(3 * n);
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        f[i] = px[j] / 255;
        f[i + n] = px[j + 1] / 255;
        f[i + 2 * n] = px[j + 2] / 255;
      }
      const t0 = performance.now();
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, S, S]) });
      const ms = performance.now() - t0;

      // YOLOv8 output: [1, 4 + classes, anchors]; rows are cx, cy, w, h, then class scores.
      const o = out[session.outputNames[0]];
      const C = o.dims[1], N = o.dims[2], d = o.data, nc = C - 4;
      const cand = [];
      for (let i = 0; i < N; i++) {
        let best = 0, bc = -1;
        for (let c = 0; c < nc; c++) {
          const s = d[(4 + c) * N + i];
          if (s > best) { best = s; bc = c; }
        }
        if (best < m.conf) continue;
        const x = d[i], y = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
        cand.push([(x - w / 2) / r, (y - h / 2) / r, (x + w / 2) / r, (y + h / 2) / r, bc, best]);
      }
      cand.sort((a, b) => b[5] - a[5]);
      const keep = [];
      for (const b of cand) {
        let ok = true;
        for (const k of keep) { if (iou(b, k) > m.iou) { ok = false; break; } }
        if (ok) { keep.push(b); if (keep.length >= 300) break; }
      }
      self.postMessage({ type: 'result', id: m.id, src: m.src, boxes: keep, ms, W, H, t: m.t });
    } catch (err) {
      self.postMessage({ type: 'error', message: String((err && err.message) || err), id: m.id });
    }
  }
};
