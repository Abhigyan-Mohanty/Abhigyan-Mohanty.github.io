// Almost Organic: 3D Game of Life demo. A CPU port of conways_3d.ipynb at a smaller scale. Each step every voxel
// counts the live cells around it: a dead voxel is created if the count is in the birth range, a live cell is
// sustained (and ages) if it is in the survival range, and otherwise dies of isolation or overcrowding.
// The window is either 3x3x3 (26 around) or 2x2x2 blocks (the other 7 in the block; the blocks shift by one
// voxel each step so patterns can spread). Drawn as voxels coloured by age, with a depth buffer.
(() => {
  const $ = id => document.getElementById(id);
  const cv = $('life-canvas');
  if (!cv) return;

  const PRESETS = {
    3: [['B5/S5-7: grows, then churns', [5, 5, 5, 7]], ['B5-6/S4-6: busy', [5, 6, 4, 6]], ['B5/S4-5: Bays 4555 (gliders; soups freeze)', [5, 5, 4, 5]],
        ['B6/S5-7: Bays 5766', [6, 6, 5, 7]], ['B4/S4-5: grows', [4, 4, 4, 5]], ['B5-7/S6-8: dense churn', [5, 7, 6, 8]]],
    2: [['B3-4/S2-4: busy', [3, 4, 2, 4]], ['B3/S2-3: Conway-like, dies out', [3, 3, 2, 3]],
        ['B3-4/S3-5: sparse churn', [3, 4, 3, 5]], ['B2-3/S2-4: dense', [2, 3, 2, 4]]],
  };
  const P = { L: 48, N: 2500, soup: 0.4, speed: 1, win: 3, wrap: true, rules: { 3: [5, 5, 5, 7], 2: [3, 4, 2, 4] }, op: 0.4 };
  let G, G2, A, B, L, t = 0, births = 0, deaths = 0;

  function reset() {
    L = P.L;
    const V = L * L * L, s = Math.max(1, Math.min(L, Math.round(P.soup * L))), o = (L - s) >> 1;
    G = new Uint8Array(V); G2 = new Uint8Array(V); A = new Uint8Array(V); B = new Uint8Array(V);
    const n = Math.min(P.N, s * s * s);
    const pick = new Set();                         // n distinct cells in the central cube
    while (pick.size < n) pick.add((Math.random() * s * s * s) | 0);
    for (const k of pick) { const x = k % s + o, y = ((k / s) | 0) % s + o, z = ((k / (s * s)) | 0) + o; G[(z * L + y) * L + x] = 1; }
    t = 0; births = deaths = 0;
  }
  const mask = (lo, hi) => { let m = 0; for (let c = Math.max(lo, 0); c <= Math.min(hi, 26); c++) m |= 1 << c; return m; };

  function sumAxis(src, dst, stride, first) {       // one pass of the separable 3x3x3 count
    const wrap = P.wrap, V = L * L * L;
    for (let i = 0; i < V; i++) {
      const c = ((i / stride) | 0) % L;
      let v = src[i], s = first ? (v ? 1 : 0) : v;
      let j = c > 0 ? i - stride : (wrap ? i + (L - 1) * stride : -1);
      if (j >= 0) { v = src[j]; s += first ? (v ? 1 : 0) : v; }
      j = c < L - 1 ? i + stride : (wrap ? i - (L - 1) * stride : -1);
      if (j >= 0) { v = src[j]; s += first ? (v ? 1 : 0) : v; }
      dst[i] = s;
    }
  }
  function apply(i, c, bm, sm) {
    const age = G[i];
    let nw;
    if (age) nw = (sm >> c) & 1 ? (age < 255 ? age + 1 : 255) : 0;
    else nw = (bm >> c) & 1;
    G2[i] = nw;
    if (!age && nw) births++; else if (age && !nw) deaths++;
  }
  function step() {
    const [b0, b1, s0, s1] = P.rules[P.win], bm = mask(b0, b1), sm = mask(s0, s1), wrap = P.wrap, LL = L * L;
    births = deaths = 0;
    if (P.win === 3) {
      sumAxis(G, A, 1, true); sumAxis(A, B, L, false);
      for (let z = 0, i = 0; z < L; z++) for (let y = 0; y < L; y++) for (let x = 0; x < L; x++, i++) {
        let s = B[i];
        if (z > 0) s += B[i - LL]; else if (wrap) s += B[i + (L - 1) * LL];
        if (z < L - 1) s += B[i + LL]; else if (wrap) s += B[i - (L - 1) * LL];
        apply(i, s - (G[i] ? 1 : 0), bm, sm);
      }
    } else {
      const o = t & 1;                              // 2x2x2 blocks, shifted by one voxel every other step
      for (let z = 0, i = 0; z < L; z++) for (let y = 0; y < L; y++) for (let x = 0; x < L; x++, i++) {
        const x0 = x - ((x + o) & 1), y0 = y - ((y + o) & 1), z0 = z - ((z + o) & 1);
        let s = 0;
        for (let dz = 0; dz < 2; dz++) {
          let zz = z0 + dz; if (zz < 0 || zz >= L) { if (!wrap) continue; zz = zz < 0 ? L - 1 : 0; }
          for (let dy = 0; dy < 2; dy++) {
            let yy = y0 + dy; if (yy < 0 || yy >= L) { if (!wrap) continue; yy = yy < 0 ? L - 1 : 0; }
            for (let dx = 0; dx < 2; dx++) {
              let xx = x0 + dx; if (xx < 0 || xx >= L) { if (!wrap) continue; xx = xx < 0 ? L - 1 : 0; }
              if (G[(zz * L + yy) * L + xx]) s++;
            }
          }
        }
        apply(i, s - (G[i] ? 1 : 0), bm, sm);
      }
    }
    [G, G2] = [G2, G];
    t++;
  }

  // ------------------------------------------------------------------ drawing: voxels into a depth buffer
  const ctx = cv.getContext('2d');
  let W = 0, img, pix, zb, az = 0.6, tilt = 0.4, zoom = 1;
  const rgb = (r, g, b) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  const BG = rgb(0x1b, 0x1d, 0x22), NSH = 8, LUT = new Uint32Array(64 * NSH);
  for (let q = 0; q < 64; q++) {                    // age -> newborn yellow ... old dark red, times a depth shade
    const tq = Math.min(1, q / 40), hd = (55 - 75 * tq + 360) % 360 / 60, l = 0.72 - 0.38 * tq, c = (1 - Math.abs(2 * l - 1)) * 0.95;
    const xx = c * (1 - Math.abs(hd % 2 - 1)), m = l - c / 2;
    const [r, g, b] = hd < 1 ? [c, xx, 0] : hd < 2 ? [xx, c, 0] : hd < 3 ? [0, c, xx] : hd < 4 ? [0, xx, c] : hd < 5 ? [xx, 0, c] : [c, 0, xx];
    for (let k = 0; k < NSH; k++) { const f = 0.45 + 0.55 * k / (NSH - 1); LUT[q * NSH + k] = rgb(((r + m) * 255 * f) | 0, ((g + m) * 255 * f) | 0, ((b + m) * 255 * f) | 0); }
  }
  const blend = (c0, c1, a) => rgb(((c0 & 255) * (1 - a) + (c1 & 255) * a) | 0, (((c0 >>> 8) & 255) * (1 - a) + ((c1 >>> 8) & 255) * a) | 0,
                                   (((c0 >>> 16) & 255) * (1 - a) + ((c1 >>> 16) & 255) * a) | 0);
  function fit() {
    const dpr = window.devicePixelRatio || 1;
    W = Math.max(64, Math.round(cv.clientWidth * dpr));
    if (cv.width !== W) { cv.width = cv.height = W; img = ctx.createImageData(W, W); pix = new Uint32Array(img.data.buffer); zb = new Float32Array(W * W); }
  }
  function draw() {
    fit();
    const h = (L - 1) / 2, sc = zoom * W / (L * 1.8), c2 = W / 2, dmax = L * 0.87, op = P.op, opaque = op >= 1;
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(tilt), se = Math.sin(tilt);
    pix.fill(BG); zb.fill(-Infinity);
    const sz = Math.max(1, Math.round(sc));
    for (let z = 0, i = 0; z < L; z++) for (let y = 0; y < L; y++) for (let x = 0; x < L; x++, i++) {
      const age = G[i];
      if (!age) continue;
      const dx = x - h, dy = y - h, dz = z - h, x1 = ca * dx - sa * dy, d1 = -(sa * dx + ca * dy), depth = se * dz + ce * d1;
      const sx = c2 + x1 * sc, sy = c2 - (ce * dz - se * d1) * sc;
      let k = ((depth / dmax + 1) * 0.5 * NSH) | 0; k = k < 0 ? 0 : k >= NSH ? NSH - 1 : k;
      const col = LUT[Math.min(63, age - 1) * NSH + k];
      const x0 = Math.round(sx - sz / 2), y0 = Math.round(sy - sz / 2);
      const xa = Math.max(0, x0), xb = Math.min(W, x0 + sz), ya = Math.max(0, y0), yb = Math.min(W, y0 + sz);
      for (let yy = ya; yy < yb; yy++) for (let o = yy * W + xa, oe = yy * W + xb; o < oe; o++) {
        if (opaque) { if (depth > zb[o]) { pix[o] = col; zb[o] = depth; } } else pix[o] = blend(pix[o], col, op);
      }
    }
    ctx.putImageData(img, 0, 0);
    const proj = (x, y, z) => { const dx = x - h, dy = y - h, dz = z - h, x1 = ca * dx - sa * dy, d1 = -(sa * dx + ca * dy); return [c2 + x1 * sc, c2 - (ce * dz - se * d1) * sc]; };
    ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1; ctx.beginPath();
    for (const [a, b] of [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]) {
      const pa = proj(a & 1 ? L - 1 : 0, a & 2 ? L - 1 : 0, a & 4 ? L - 1 : 0), pb = proj(b & 1 ? L - 1 : 0, b & 2 ? L - 1 : 0, b & 4 ? L - 1 : 0);
      ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();
  }

  // ------------------------------------------------------------------ controls and loop
  let running = true, frames = 0, rate = 0, tRate = performance.now(), stepsSince = 0, dirty = true, isVisible = true;
  new IntersectionObserver(es => { isVisible = es[0].isIntersecting; }).observe(cv);
  const rng = (a, b) => a > b ? 'none' : a === b ? `${a}` : `${a}-${b}`;
  function info() {
    let live = 0;
    for (let i = 0; i < G.length; i++) if (G[i]) live++;
    $('life-info').textContent = `step ${t.toLocaleString()}  ·  live cells ${live.toLocaleString()}  ·  last step: ${births.toLocaleString()} created, ${deaths.toLocaleString()} died  ·  ${rate} steps/s`;
  }
  function showRule() {
    const [b0, b1, s0, s1] = P.rules[P.win], nmax = P.win === 3 ? 26 : 7;
    ['life-b0', 'life-b1', 'life-s0', 'life-s1'].forEach((id, k) => { const el = $(id); el.max = nmax; el.value = P.rules[P.win][k]; });
    $('life-preset').innerHTML = '<option value="">presets…</option>' + PRESETS[P.win].map(([n], k) => `<option value="${k}">${n}</option>`).join('');
    $('life-rule').textContent = `B${rng(b0, b1)}/S${rng(s0, s1)}: counting the live cells around a cell (0 to ${nmax}), a dead cell is created at ${rng(b0, b1)}, ` +
      `a live cell is sustained at ${rng(s0, s1)}, and dies of isolation below ${s0} or of overcrowding above ${s1}.`;
    $('life-win').textContent = `Window: ${P.win === 3 ? '3×3×3' : '2×2×2'}`;
  }
  function loop() {
    if (running && isVisible) { for (let k = 0; k < P.speed; k++) step(); stepsSince += P.speed; dirty = true; }
    if (dirty && isVisible) { draw(); dirty = false; }
    if (++frames % 10 === 0) { const now = performance.now(); if (now > tRate) { rate = Math.round(stepsSince * 1000 / (now - tRate)); stepsSince = 0; tRate = now; } info(); }
    requestAnimationFrame(loop);
  }
  const bind = (id, key, fmt, conv = v => +v, after) => {
    const el = $(id), out = $(id + '-v');
    const set = () => { P[key] = conv(el.value); if (out) out.textContent = fmt(P[key]); if (after) after(); };
    el.addEventListener('input', set); set();
  };
  bind('life-speed', 'speed', v => v);
  bind('life-l', 'L', v => `${v}³`);
  bind('life-n', 'N', v => v.toLocaleString());
  bind('life-soup', 'soup', v => Math.round(v * 100) + '%', v => v / 100);
  bind('life-op', 'op', v => Math.round(v * 100) + '%', v => v / 100, () => { dirty = true; });
  ['life-b0', 'life-b1', 'life-s0', 'life-s1'].forEach((id, k) => $(id).addEventListener('change', e => {
    P.rules[P.win][k] = Math.max(0, Math.min(P.win === 3 ? 26 : 7, Math.round(+e.target.value || 0))); showRule();
  }));
  $('life-preset').addEventListener('change', e => { if (e.target.value === '') return; P.rules[P.win] = PRESETS[P.win][+e.target.value][1].slice(); showRule(); });
  $('life-win').addEventListener('click', () => { P.win = P.win === 3 ? 2 : 3; showRule(); });
  const wrapB = $('life-wrap'), showWrap = () => { wrapB.classList.toggle('on', P.wrap); wrapB.textContent = `Wrap edges: ${P.wrap ? 'on' : 'off'}`; };
  wrapB.addEventListener('click', () => { P.wrap = !P.wrap; showWrap(); }); showWrap();
  $('life-run').addEventListener('click', e => { running = !running; e.target.textContent = running ? 'Pause' : 'Play'; });
  $('life-step').addEventListener('click', () => { step(); dirty = true; info(); });
  $('life-reset').addEventListener('click', () => { reset(); dirty = true; info(); });
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!drag) return;
    az += (e.clientX - drag[0]) * 0.01; tilt = Math.max(-1.55, Math.min(1.55, tilt + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY]; dirty = true;
  });
  const end = () => { drag = null; };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
  cv.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(0.3, Math.min(8, zoom * Math.exp(-e.deltaY * 0.0015))); dirty = true; }, { passive: false });
  cv.addEventListener('dblclick', () => { az = 0.6; tilt = 0.4; zoom = 1; dirty = true; });
  showRule(); reset(); info(); requestAnimationFrame(loop);
})();
