// Almost Organic: 2D nucleation demo. A CPU port of the rules in nucleation_cuda2d.ipynb, at a smaller scale:
// Brownian particles bind into clusters (3 in a 2x2 window, or touching a cluster), clusters drift as rigid
// bodies with momentum, anything touching the central seed's system sticks to it, and floating masses at the
// mass ceiling eject a piece of themselves. Masses carry colours: a merge takes the biggest mass's colour, a
// mass formed from free particles gets a random one, ejected pieces keep their parent's. With the colour rule
// on, a mass ejects towards the nearest mass of another colour and away from one of its own colour, at a rate
// inversely proportional to the distance between them.
(() => {
  const $ = id => document.getElementById(id);
  const cv = $('nuc-canvas');
  if (!cv) return;

  const PAL = [[230, 25, 75], [60, 180, 75], [67, 99, 216], [255, 225, 25], [240, 50, 230], [66, 212, 244],
               [145, 30, 180], [250, 190, 212]];
  const P = {                                       // parameters (the controls write these)
    L: 160, N: 2500, speed: 3, cap: 40, erate: 0.05, efrac: 0.25, v0: 0.5, damp: 0.002, win: true,
    crule: true, ncol: 3, dref: 20, hl: true, hfrac: 0.8, anchor: false,
  };

  let S;                                            // the simulation state
  function reset() {
    const L = P.L, N = P.N, NP = N + (P.anchor ? 1 : 0);  // particle N: the seed (only with the anchor on)
    S = {
      L, N, NP, seed: P.anchor ? N : -1, t: 0, tLast: 1, ejections: 0,
      x: new Int16Array(NP), y: new Int16Array(NP),
      par: new Int32Array(NP), old: new Int32Array(NP), size: new Int32Array(NP).fill(1),
      fixed: new Uint8Array(NP), clus: new Uint8Array(NP), btime: new Int32Array(NP), col: new Uint8Array(NP),
      mx: new Float32Array(NP), my: new Float32Array(NP), ax: new Float32Array(NP), ay: new Float32Array(NP),
      gid: new Int32Array(NP), gtil: new Int32Array(NP),
      dir: new Int8Array(NP), blocked: new Uint8Array(NP),
      bx0: new Int16Array(NP), bx1: new Int16Array(NP), by0: new Int16Array(NP), by1: new Int16Array(NP),
      head: new Int32Array(L * L), next: new Int32Array(NP),
      act: new Uint8Array(NP).fill(1), moved: new Uint8Array(NP),  // who looks for new contacts this step
      // merge bookkeeping: the roots linked this step, and sums per new root
      links: new Int32Array(NP), nl: 0, tm: new Uint8Array(NP),
      ns: new Int32Array(NP), nmx: new Float32Array(NP), nmy: new Float32Array(NP), nfx: new Uint8Array(NP),
      bs: new Int32Array(NP), bc: new Uint8Array(NP),
      cx: new Float64Array(NP), cy: new Float64Array(NP), fl: new Int32Array(NP),   // ejection: centroids
      lab: new Uint8Array(NP),
    };
    for (let i = 0; i < NP; i++) {
      S.par[i] = i; S.x[i] = (Math.random() * L) | 0; S.y[i] = (Math.random() * L) | 0;
      S.col[i] = (Math.random() * P.ncol) | 0;
    }
    if (P.anchor) {                                 // the immobile seed: anything touching its system sticks
      S.x[N] = S.y[N] = L >> 1;
      S.fixed[N] = S.clus[N] = 1;
    }
  }

  function find(i) {
    const par = S.par;
    while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; }
    return i;
  }
  function union(a, b) {                            // the larger root goes under the smaller one
    a = find(a); b = find(b);
    if (a === b) return false;
    if (a > b) { const c = a; a = b; b = c; }
    S.par[b] = a;
    S.links[S.nl++] = b;                            // (each root is linked at most once)
    return true;
  }
  function rebuildGrid() {
    const { L, NP, head, next, x, y } = S;
    head.fill(-1);
    for (let i = 0; i < NP; i++) { const c = y[i] * L + x[i]; next[i] = head[c]; head[c] = i; }
  }

  function step() {
    const s = S, { L, NP, par, size, fixed, x, y, mx, my, ax, ay, dir, blocked, bx0, bx1, by0, by1 } = s;
    const t = ++s.t;
    // 1. each mobile root picks its hop: ballistic once its drift has built up a whole pixel, else Brownian
    //    (with probability 1/size); a cluster that would leave the box stays put and reflects its momentum
    for (let r = 0; r < NP; r++) {
      if (par[r] !== r || fixed[r]) continue;
      const m = size[r];
      let a0 = ax[r] + mx[r] / m, a1 = ay[r] + my[r] / m, d = -1, axis = -1;
      if (Math.abs(a0) >= 1 || Math.abs(a1) >= 1) { axis = Math.abs(a0) >= Math.abs(a1) ? 0 : 1; d = axis === 0 ? (a0 > 0 ? 0 : 1) : (a1 > 0 ? 2 : 3); }
      else if (Math.random() * m < 1) d = (Math.random() * 4) | 0;
      dir[r] = d; ax[r] = a0; ay[r] = a1;
      bx0[r] = by0[r] = L; bx1[r] = by1[r] = -1;
    }
    for (let i = 0; i < NP; i++) {                  // bounding boxes, for the wall check
      if (fixed[i]) continue;
      const r = par[i];
      if (x[i] < bx0[r]) bx0[r] = x[i]; if (x[i] > bx1[r]) bx1[r] = x[i];
      if (y[i] < by0[r]) by0[r] = y[i]; if (y[i] > by1[r]) by1[r] = y[i];
    }
    for (let r = 0; r < NP; r++) {
      if (par[r] !== r || fixed[r]) continue;
      const d = dir[r];
      const bl = (d === 0 && bx1[r] === L - 1) || (d === 1 && bx0[r] === 0) || (d === 2 && by1[r] === L - 1) || (d === 3 && by0[r] === 0);
      blocked[r] = bl ? 1 : 0;
      const ball = Math.abs(ax[r]) >= 1 || Math.abs(ay[r]) >= 1;
      if (ball) {                                   // the drift accumulator pays for a ballistic hop
        if (d < 2) { if (bl) { mx[r] = -mx[r]; ax[r] = 0; } else ax[r] -= d === 0 ? 1 : -1; }
        else { if (bl) { my[r] = -my[r]; ay[r] = 0; } else ay[r] -= d === 2 ? 1 : -1; }
      }
      const m = size[r], v = (Math.abs(mx[r]) + Math.abs(my[r])) / m, f = (1 - P.damp) / (v > 1 ? v : 1);
      mx[r] *= f; my[r] *= f;                       // damping; and no faster than one hop per step
    }
    for (let i = 0; i < NP; i++) {
      if (fixed[i]) continue;
      const r = par[i], d = dir[r];
      if (d < 0 || blocked[r]) continue;
      s.moved[r] = (t & 255) | 1;
      if (d === 0) x[i]++; else if (d === 1) x[i]--; else if (d === 2) y[i]++; else y[i]--;
    }
    rebuildGrid();
    // 2. binding: a 2x2 window holding 3 or more particles binds them; contact with a cluster binds
    const { head, next, clus, old, gid, gtil, act, moved } = s, mk = (t & 255) | 1;
    old.set(par);
    let merged = false;
    const nbh = new Int32Array(9);
    for (let i = 0; i < NP; i++) {                  // only particles that moved, are free or just changed:
      if (fixed[i]) continue;                       // any new contact involves one of them
      const ri = old[i];
      if (clus[i] && !act[i] && moved[ri] !== mk) continue;
      act[i] = 0;
      const xi = x[i], yi = y[i];
      let near = false;                             // anything from another cluster within one pixel?
      nbh.fill(0);
      for (let dy = -1; dy <= 1; dy++) {
        const yy = yi + dy; if (yy < 0 || yy >= L) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = xi + dx; if (xx < 0 || xx >= L) continue;
          for (let j = head[yy * L + xx]; j >= 0; j = next[j]) { nbh[(dy + 1) * 3 + dx + 1]++; if (old[j] !== ri) near = true; }
        }
      }
      if (!near) continue;
      let wmask = 0;
      if (P.win) for (let w = 0; w < 4; w++) {
        const wx = w & 1, wy = w >> 1;
        if (nbh[wy * 3 + wx] + nbh[wy * 3 + wx + 1] + nbh[(wy + 1) * 3 + wx] + nbh[(wy + 1) * 3 + wx + 1] >= 3) wmask |= 1 << w;
      }
      const gi = gid[ri], gt = gtil[ri];
      for (let dy = -1; dy <= 1; dy++) {
        const yy = yi + dy; if (yy < 0 || yy >= L) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = xi + dx; if (xx < 0 || xx >= L) continue;
          for (let j = head[yy * L + xx]; j >= 0; j = next[j]) {
            if (j === i || old[j] === ri) continue;
            let join = false;
            for (let w = 0; w < 4; w++) {
              if (!(wmask & (1 << w))) continue;
              const wx = (w & 1) - 1, wy = (w >> 1) - 1;
              if (dx >= wx && dx <= wx + 1 && dy >= wy && dy <= wy + 1) join = true;
            }
            if (dx * dx + dy * dy <= 1 && (clus[i] || clus[j])) join = true;
            if (join && gi !== 0 && gi === gid[old[j]] && t < gt) join = false;   // ejection grace
            if (join && union(i, j)) merged = true;
          }
        }
      }
    }
    if (merged) mergeBookkeeping(t);
    // 3. ejection: floating masses at the ceiling fire (first order); with the colour rule, at a rate ~ 1/distance
    if (P.cap < 1e9) eject(t);
  }

  function mergeBookkeeping(t) {
    const s = S, { NP, par, size, fixed, clus, btime, col, mx, my, ax, ay, act, links, tm, ns, nmx, nmy, nfx, bs, bc } = s;
    for (let i = 0; i < NP; i++) par[i] = find(i);  // flatten: every particle points at its new root
    const touched = [];
    for (let k = 0; k < s.nl; k++) {                // each linked root goes into its new root
      const l = links[k], n = par[l];
      if (!tm[n]) {
        tm[n] = 1; touched.push(n);
        ns[n] = size[n]; nmx[n] = mx[n]; nmy[n] = my[n]; nfx[n] = fixed[n]; bs[n] = size[n]; bc[n] = col[n];
      }
      ns[n] += size[l]; nmx[n] += mx[l]; nmy[n] += my[l];
      if (fixed[l]) nfx[n] = 1;
      if (size[l] > bs[n]) { bs[n] = size[l]; bc[n] = col[l]; }
      size[l] = 0; mx[l] = my[l] = ax[l] = ay[l] = 0;
    }
    s.nl = 0;
    for (const n of touched) {                      // a merged mass: the biggest one's colour, or random
      size[n] = ns[n]; mx[n] = nmx[n]; my[n] = nmy[n];
      col[n] = bs[n] >= 2 ? bc[n] : (Math.random() * P.ncol) | 0;
      if (nfx[n]) { fixed[n] = 1; mx[n] = my[n] = ax[n] = ay[n] = 0; }
    }
    for (let i = 0; i < NP; i++) {
      const r = par[i];
      if (!tm[r]) continue;
      if (fixed[r] && !fixed[i]) fixed[i] = 1;      // joined the seed's system: it stops
      if ((size[r] >= 2 || fixed[r]) && !clus[i]) { clus[i] = 1; btime[i] = t; s.tLast = t; act[i] = 1; }
    }
    for (const n of touched) tm[n] = 0;
  }

  function eject(t) {
    const s = S, { NP, par, size, fixed, x, y, cx, cy, fl } = s;
    const cand = [];
    let nf = 0;
    for (let r = 0; r < NP; r++) {                  // floating masses, and those at the ceiling
      if (par[r] !== r || fixed[r] || size[r] < 2) continue;
      fl[nf++] = r; cx[r] = cy[r] = 0;
      if (size[r] >= P.cap) cand.push(r);
    }
    if (!cand.length) return;
    if (P.crule) {                                  // their centroids
      for (let i = 0; i < NP; i++) { const r = par[i]; if (!fixed[i] && size[r] >= 2) { cx[r] += x[i]; cy[r] += y[i]; } }
      for (let q = 0; q < nf; q++) { const r = fl[q]; cx[r] /= size[r]; cy[r] /= size[r]; }
    }
    const fire = [];
    for (const r of cand) {
      let k = P.erate;
      if (P.crule) {                                // rate ~ 1 / distance to the nearest other centroid
        let best = Infinity;
        for (let q = 0; q < nf; q++) {
          const r2 = fl[q];
          if (r2 === r) continue;
          const d2 = (cx[r2] - cx[r]) ** 2 + (cy[r2] - cy[r]) ** 2;
          if (d2 < best) best = d2;
        }
        const d = best < Infinity ? Math.sqrt(best) : 1.414 * s.L;
        k *= P.dref / Math.max(d, 1);
      }
      if (Math.random() < k) fire.push(r);
    }
    if (!fire.length) return;
    const members = new Map();                      // the members of each mass that fires
    for (const r of fire) members.set(r, []);
    for (let i = 0; i < NP; i++) { const m = members.get(par[i]); if (m) m.push(i); }
    for (const r of fire) split(r, members.get(r), t);
  }

  function split(R, mem, t) {
    const s = S, { L, NP, par, size, fixed, clus, btime, x, y, col, mx, my, ax, ay, gid, gtil, lab, head, next } = s;
    const m = mem.length;
    let cx = 0, cy = 0;
    for (const i of mem) { cx += x[i]; cy += y[i]; }
    cx /= m; cy /= m;
    let best = Infinity, jn = -1;                   // the nearest particle of another floating mass
    for (let j = 0; j < NP; j++) {
      const r = par[j];
      if (r === R || fixed[j] || size[r] < 2) continue;
      const d2 = (x[j] - cx) ** 2 + (y[j] - cy) ** 2;
      if (d2 < best) { best = d2; jn = j; }
    }
    let ex, ey;
    if (jn >= 0) {                                  // away from it; with the colour rule, towards another colour
      ex = cx - x[jn]; ey = cy - y[jn];
      if (P.crule && col[R] !== col[par[jn]]) { ex = -ex; ey = -ey; }
    } else { ex = Math.random() - 0.5; ey = Math.random() - 0.5; }
    let n1 = Math.abs(ex) + Math.abs(ey);
    if (n1 < 1e-9) { ex = 1; ey = 0; n1 = 1; }
    ex /= n1; ey /= n1;
    let kk = Math.round(P.efrac * m); kk = Math.max(1, Math.min(m - 1, kk));
    const order = mem.slice().sort((a, b) => ((x[b] - cx) * ex + (y[b] - cy) * ey) - ((x[a] - cx) * ex + (y[a] - cy) * ey));
    for (let q = 0; q < m; q++) lab[order[q]] = q < kk ? 1 : 2;   // 1: ejecta (the side facing the direction), 2: rest
    const ke = kk, kr = m - kk, px = mx[R], py = my[R], jk = P.v0 * ke * kr / m;
    const vel = { 1: [(px * ke / m + ex * jk) / ke, (py * ke / m + ey * jk) / ke],
                  2: [(px * kr / m - ex * jk) / kr, (py * kr / m - ey * jk) / kr] };
    const colR = col[R], grace = Math.min(2000, Math.round(6 / Math.max(P.v0, 0.003)));
    for (const i of mem) par[i] = -1;               // regroup each part by contact (8-neighbourhood)
    for (const st of mem) {
      if (par[st] !== -1) continue;
      const piece = [st], lb = lab[st];
      par[st] = -2;
      for (let q = 0; q < piece.length; q++) {
        const i = piece[q];
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y[i] + dy; if (yy < 0 || yy >= L) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x[i] + dx; if (xx < 0 || xx >= L) continue;
            for (let j = head[yy * L + xx]; j >= 0; j = next[j]) if (par[j] === -1 && lab[j] === lb) { par[j] = -2; piece.push(j); }
          }
        }
      }
      let r = piece[0];
      for (const i of piece) if (i < r) r = i;
      for (const i of piece) { par[i] = r; size[i] = 0; mx[i] = my[i] = ax[i] = ay[i] = 0; S.act[i] = 1; }
      const sz = piece.length;
      size[r] = sz; mx[r] = vel[lb][0] * sz; my[r] = vel[lb][1] * sz;
      col[r] = colR; gid[r] = t; gtil[r] = t + grace;
      if (sz < 2) for (const i of piece) { clus[i] = 0; btime[i] = 0; }
    }
    s.ejections++;
  }

  // ------------------------------------------------------------------ drawing
  const ctx = cv.getContext('2d');
  let W = 0, img, pix;
  const rgb = (r, g, b) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  const BG = rgb(0xCC, 0x77, 0x22), WHITE = rgb(255, 255, 255);
  const GREY = Array.from({ length: 256 }, (_, g) => rgb(g, g, g)), PALC = PAL.map(c => rgb(...c));
  const blend = (c0, c1, a) => rgb(((c0 & 255) * (1 - a) + (c1 & 255) * a) | 0, (((c0 >>> 8) & 255) * (1 - a) + ((c1 >>> 8) & 255) * a) | 0,
                                   (((c0 >>> 16) & 255) * (1 - a) + ((c1 >>> 16) & 255) * a) | 0);
  const hue = r => { const h = (Math.imul(r + 1, 2654435761) >>> 0) % 360 / 60, c = 0.85 * 0.9, xx = c * (1 - Math.abs(h % 2 - 1)), m = 0.55 - c / 2;
    const [a, b, d] = h < 1 ? [c, xx, 0] : h < 2 ? [xx, c, 0] : h < 3 ? [0, c, xx] : h < 4 ? [0, xx, c] : h < 5 ? [xx, 0, c] : [c, 0, xx];
    return rgb(((a + m) * 255) | 0, ((b + m) * 255) | 0, ((d + m) * 255) | 0); };
  function fit() {
    const dpr = window.devicePixelRatio || 1, css = cv.clientWidth;
    W = Math.max(64, Math.round(css * dpr));
    if (cv.width !== W) { cv.width = cv.height = W; img = ctx.createImageData(W, W); pix = new Uint32Array(img.data.buffer); }
  }
  function draw() {
    fit();
    const { L, NP, x, y, par, size, fixed, clus, btime, col, seed, tLast } = S, sc = W / L, sz = Math.max(1, Math.round(sc));
    const crule = P.crule, ncol = P.ncol, gs = 230 / tLast;
    pix.fill(BG);
    const thr = P.cap < 1e9 ? Math.max(2, Math.ceil(P.hfrac * P.cap)) : Infinity;
    let hlOn = false;                               // colour masses near the ceiling, and dim the rest, but only
    if (P.hl && thr < Infinity)                     // while there is such a mass
      for (let r = 0; r < NP && !hlOn; r++) if (par[r] === r && !fixed[r] && size[r] >= thr) hlOn = true;
    for (let pass = 0; pass < 2; pass++) {          // translucent first, opaque on top
      for (let i = 0; i < NP; i++) {
        const r = par[i], free = !clus[i];
        const hl = hlOn && !fixed[i] && size[r] >= thr;
        let a = free ? 0.5 : 1;
        if (hlOn && !hl) a *= 0.3;
        if ((a < 1) !== (pass === 0)) continue;
        const c = hl ? (crule ? PALC[col[r] % ncol] : hue(r)) : free ? WHITE : GREY[i === seed ? 0 : Math.round(gs * btime[i])];
        const x0 = Math.floor(x[i] * sc), y0 = Math.floor((L - 1 - y[i]) * sc);   // y is up
        for (let yy = y0; yy < y0 + sz && yy < W; yy++)
          for (let o = yy * W + x0, oe = yy * W + Math.min(W, x0 + sz); o < oe; o++) pix[o] = pass ? c : blend(pix[o], c, a);
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // ------------------------------------------------------------------ controls and loop
  let running = true, rate = 0, frames = 0, tRate = performance.now(), stepsSince = 0;
  function info() {
    let free = 0, masses = 0, stuck = 0;
    for (let i = 0; i < S.NP; i++) {
      if (!S.clus[i]) free++;
      if (S.fixed[i]) stuck++;
      else if (S.par[i] === i && S.size[i] >= 2) masses++;
    }
    $('nuc-info').textContent = `step ${S.t.toLocaleString()}  ·  free ${free.toLocaleString()}  ·  floating masses ${masses}  ·  ` +
      (S.seed >= 0 ? `stuck to seed ${(stuck - 1).toLocaleString()}  ·  ` : '') + `ejections ${S.ejections.toLocaleString()}  ·  ${rate} steps/s`;
  }
  function loop() {
    if (running && isVisible) { for (let k = 0; k < P.speed; k++) step(); stepsSince += P.speed; }
    if (isVisible) draw();
    if (++frames % 10 === 0) {
      const now = performance.now();
      if (now > tRate) { rate = Math.round(stepsSince * 1000 / (now - tRate)); stepsSince = 0; tRate = now; }
      info();
    }
    requestAnimationFrame(loop);
  }
  let isVisible = true;                             // don't burn CPU while scrolled away
  new IntersectionObserver(es => { isVisible = es[0].isIntersecting; }).observe(cv);

  const bind = (id, key, fmt, conv = v => +v, after) => {
    const el = $(id), out = $(id + '-v');
    const set = () => { P[key] = conv(el.value); if (out) out.textContent = fmt(P[key]); if (after) after(); };
    el.addEventListener('input', set); set();
  };
  bind('nuc-speed', 'speed', v => v);
  bind('nuc-n', 'N', v => v.toLocaleString());
  bind('nuc-l', 'L', v => `${v} × ${v}`);
  bind('nuc-cap', 'cap', v => v >= 1e9 ? 'off' : v, v => +v >= 500 ? 1e9 : +v);
  bind('nuc-erate', 'erate', v => v.toPrecision(2) + ' /step', v => 10 ** +v);
  bind('nuc-efrac', 'efrac', v => Math.round(v * 100) + '%', v => v / 100);
  bind('nuc-v0', 'v0', v => v.toFixed(2));
  bind('nuc-ncol', 'ncol', v => v);
  bind('nuc-dref', 'dref', v => v + ' px');
  const tog = (id, key, label) => {
    const b = $(id), show = () => { b.classList.toggle('on', P[key]); b.textContent = `${label}: ${P[key] ? 'on' : 'off'}`; };
    b.addEventListener('click', () => { P[key] = !P[key]; show(); }); show();
  };
  tog('nuc-crule', 'crule', 'Colour rule');
  tog('nuc-hl', 'hl', 'Ceiling colours');
  tog('nuc-win', 'win', '2×2 binding');
  tog('nuc-anchor', 'anchor', 'Anchor');
  $('nuc-run').addEventListener('click', e => { running = !running; e.target.textContent = running ? 'Pause' : 'Play'; });
  $('nuc-reset').addEventListener('click', () => { reset(); info(); });
  reset(); info(); requestAnimationFrame(loop);
})();
