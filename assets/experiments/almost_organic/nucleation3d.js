// Almost Organic: 3D nucleation demo. A CPU port of the rules in nucleation_cuda.ipynb, at a smaller scale: the
// same model as the 2D demo in a cubic box (6 hop directions, 2x2x2 windows, face contact, pieces regrouped over
// the 26 neighbours), drawn into a depth buffer so it can be rotated.
(() => {
  const $ = id => document.getElementById(id);
  const cv = $('n3-canvas');
  if (!cv) return;

  const PAL = [[230, 25, 75], [60, 180, 75], [67, 99, 216], [255, 225, 25], [240, 50, 230], [66, 212, 244],
               [145, 30, 180], [250, 190, 212]];
  const P = {                                       // parameters (the controls write these)
    L: 60, N: 4000, speed: 2, cap: 40, erate: 0.05, efrac: 0.25, v0: 0.5, damp: 0.002, win: true,
    crule: true, ncol: 3, dref: 15, hl: true, hfrac: 0.8, anchor: false,
  };

  let S;                                            // the simulation state
  function reset() {
    const L = P.L, N = P.N, NP = N + (P.anchor ? 1 : 0);  // particle N: the seed (only with the anchor on)
    const f32 = () => new Float32Array(NP), i16 = () => new Int16Array(NP);
    S = {
      L, N, NP, seed: P.anchor ? N : -1, t: 0, tLast: 1, ejections: 0,
      x: i16(), y: i16(), z: i16(),
      par: new Int32Array(NP), old: new Int32Array(NP), size: new Int32Array(NP).fill(1),
      fixed: new Uint8Array(NP), clus: new Uint8Array(NP), btime: new Int32Array(NP), col: new Uint8Array(NP),
      mx: f32(), my: f32(), mz: f32(), ax: f32(), ay: f32(), az: f32(),
      gid: new Int32Array(NP), gtil: new Int32Array(NP),
      dir: new Int8Array(NP), blocked: new Uint8Array(NP),
      bx0: i16(), bx1: i16(), by0: i16(), by1: i16(), bz0: i16(), bz1: i16(),
      head: new Int32Array(L * L * L), next: new Int32Array(NP),
      act: new Uint8Array(NP).fill(1), moved: new Uint8Array(NP),  // who looks for new contacts this step
      // merge bookkeeping: the roots linked this step, and sums per new root
      links: new Int32Array(NP), nl: 0, tm: new Uint8Array(NP),
      ns: new Int32Array(NP), nmx: f32(), nmy: f32(), nmz: f32(), nfx: new Uint8Array(NP),
      bs: new Int32Array(NP), bc: new Uint8Array(NP),
      cx: new Float64Array(NP), cy: new Float64Array(NP), cz: new Float64Array(NP), fl: new Int32Array(NP),
      lab: new Uint8Array(NP),
    };
    for (let i = 0; i < NP; i++) {
      S.par[i] = i;
      S.x[i] = (Math.random() * L) | 0; S.y[i] = (Math.random() * L) | 0; S.z[i] = (Math.random() * L) | 0;
      S.col[i] = (Math.random() * P.ncol) | 0;
    }
    if (P.anchor) {                                 // the immobile seed: anything touching its system sticks
      S.x[N] = S.y[N] = S.z[N] = L >> 1;
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
    const { L, NP, head, next, x, y, z } = S;
    head.fill(-1);
    for (let i = 0; i < NP; i++) { const c = (z[i] * L + y[i]) * L + x[i]; next[i] = head[c]; head[c] = i; }
  }

  function step() {
    const s = S, { L, NP, par, size, fixed, x, y, z, mx, my, mz, ax, ay, az, dir, blocked, bx0, bx1, by0, by1, bz0, bz1 } = s;
    const t = ++s.t;
    // 1. each mobile root picks its hop: ballistic once its drift has built up a whole voxel, else Brownian
    //    (with probability 1/size); a cluster that would leave the box stays put and reflects its momentum
    for (let r = 0; r < NP; r++) {
      if (par[r] !== r || fixed[r]) continue;
      const m = size[r];
      const a0 = ax[r] + mx[r] / m, a1 = ay[r] + my[r] / m, a2 = az[r] + mz[r] / m;
      const b0 = Math.abs(a0), b1 = Math.abs(a1), b2 = Math.abs(a2);
      let d = -1;
      if (b0 >= 1 || b1 >= 1 || b2 >= 1) {
        if (b0 >= b1 && b0 >= b2) d = a0 > 0 ? 0 : 1; else if (b1 >= b2) d = a1 > 0 ? 2 : 3; else d = a2 > 0 ? 4 : 5;
      } else if (Math.random() * m < 1) d = (Math.random() * 6) | 0;
      dir[r] = d; ax[r] = a0; ay[r] = a1; az[r] = a2;
      bx0[r] = by0[r] = bz0[r] = L; bx1[r] = by1[r] = bz1[r] = -1;
    }
    for (let i = 0; i < NP; i++) {                  // bounding boxes, for the wall check
      if (fixed[i]) continue;
      const r = par[i];
      if (x[i] < bx0[r]) bx0[r] = x[i]; if (x[i] > bx1[r]) bx1[r] = x[i];
      if (y[i] < by0[r]) by0[r] = y[i]; if (y[i] > by1[r]) by1[r] = y[i];
      if (z[i] < bz0[r]) bz0[r] = z[i]; if (z[i] > bz1[r]) bz1[r] = z[i];
    }
    for (let r = 0; r < NP; r++) {
      if (par[r] !== r || fixed[r]) continue;
      const d = dir[r];
      const bl = (d === 0 && bx1[r] === L - 1) || (d === 1 && bx0[r] === 0) || (d === 2 && by1[r] === L - 1) ||
                 (d === 3 && by0[r] === 0) || (d === 4 && bz1[r] === L - 1) || (d === 5 && bz0[r] === 0);
      blocked[r] = bl ? 1 : 0;
      if (d >= 0 && (Math.abs(ax[r]) >= 1 || Math.abs(ay[r]) >= 1 || Math.abs(az[r]) >= 1)) {   // ballistic hop
        const sg = d & 1 ? -1 : 1;
        if (d < 2) { if (bl) { mx[r] = -mx[r]; ax[r] = 0; } else ax[r] -= sg; }
        else if (d < 4) { if (bl) { my[r] = -my[r]; ay[r] = 0; } else ay[r] -= sg; }
        else { if (bl) { mz[r] = -mz[r]; az[r] = 0; } else az[r] -= sg; }
      }
      const m = size[r], v = (Math.abs(mx[r]) + Math.abs(my[r]) + Math.abs(mz[r])) / m, f = (1 - P.damp) / (v > 1 ? v : 1);
      mx[r] *= f; my[r] *= f; mz[r] *= f;           // damping; and no faster than one hop per step
    }
    const moved = s.moved, mk = (t & 255) | 1;
    for (let i = 0; i < NP; i++) {
      if (fixed[i]) continue;
      const r = par[i], d = dir[r];
      if (d < 0 || blocked[r]) continue;
      moved[r] = mk;
      if (d === 0) x[i]++; else if (d === 1) x[i]--; else if (d === 2) y[i]++; else if (d === 3) y[i]--; else if (d === 4) z[i]++; else z[i]--;
    }
    rebuildGrid();
    // 2. binding: a 2x2x2 window holding 3 or more particles binds them; contact with a cluster binds
    const { head, next, clus, old, gid, gtil, act } = s;
    old.set(par);
    let merged = false;
    const nbh = new Int32Array(27);
    for (let i = 0; i < NP; i++) {                  // only particles that moved, are free or just changed:
      if (fixed[i]) continue;                       // any new contact involves one of them
      const ri = old[i];
      if (clus[i] && !act[i] && moved[ri] !== mk) continue;
      act[i] = 0;
      const xi = x[i], yi = y[i], zi = z[i];
      let near = false;                             // anything from another cluster within one voxel?
      nbh.fill(0);
      for (let dz = -1; dz <= 1; dz++) {
        const zz = zi + dz; if (zz < 0 || zz >= L) continue;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = yi + dy; if (yy < 0 || yy >= L) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = xi + dx; if (xx < 0 || xx >= L) continue;
            for (let j = head[(zz * L + yy) * L + xx]; j >= 0; j = next[j]) { nbh[((dz + 1) * 3 + dy + 1) * 3 + dx + 1]++; if (old[j] !== ri) near = true; }
          }
        }
      }
      if (!near) continue;
      let wmask = 0;
      if (P.win) for (let w = 0; w < 8; w++) {      // the 8 windows containing it
        const wx = w & 1, wy = (w >> 1) & 1, wz = w >> 2;
        let c = 0;
        for (let oz = 0; oz < 2; oz++) for (let oy = 0; oy < 2; oy++) for (let ox = 0; ox < 2; ox++) c += nbh[((wz + oz) * 3 + wy + oy) * 3 + wx + ox];
        if (c >= 3) wmask |= 1 << w;
      }
      const gi = gid[ri], gt = gtil[ri];
      for (let dz = -1; dz <= 1; dz++) {
        const zz = zi + dz; if (zz < 0 || zz >= L) continue;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = yi + dy; if (yy < 0 || yy >= L) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = xi + dx; if (xx < 0 || xx >= L) continue;
            for (let j = head[(zz * L + yy) * L + xx]; j >= 0; j = next[j]) {
              if (j === i || old[j] === ri) continue;
              let join = false;
              if (wmask) for (let w = 0; w < 8; w++) {
                if (!(wmask & (1 << w))) continue;
                const wx = (w & 1) - 1, wy = ((w >> 1) & 1) - 1, wz = (w >> 2) - 1;
                if (dx >= wx && dx <= wx + 1 && dy >= wy && dy <= wy + 1 && dz >= wz && dz <= wz + 1) join = true;
              }
              if (dx * dx + dy * dy + dz * dz <= 1 && (clus[i] || clus[j])) join = true;
              if (join && gi !== 0 && gi === gid[old[j]] && t < gt) join = false;   // ejection grace
              if (join && union(i, j)) merged = true;
            }
          }
        }
      }
    }
    if (merged) mergeBookkeeping(t);
    // 3. ejection: floating masses at the ceiling fire (first order); with the colour rule, at a rate ~ 1/distance
    if (P.cap < 1e9) eject(t);
  }

  function mergeBookkeeping(t) {
    const s = S, { NP, par, size, fixed, clus, btime, col, mx, my, mz, ax, ay, az, act, links, tm, ns, nmx, nmy, nmz, nfx, bs, bc } = s;
    for (let i = 0; i < NP; i++) par[i] = find(i);  // flatten: every particle points at its new root
    const touched = [];
    for (let k = 0; k < s.nl; k++) {                // each linked root goes into its new root
      const l = links[k], n = par[l];
      if (!tm[n]) {
        tm[n] = 1; touched.push(n);
        ns[n] = size[n]; nmx[n] = mx[n]; nmy[n] = my[n]; nmz[n] = mz[n]; nfx[n] = fixed[n]; bs[n] = size[n]; bc[n] = col[n];
      }
      ns[n] += size[l]; nmx[n] += mx[l]; nmy[n] += my[l]; nmz[n] += mz[l];
      if (fixed[l]) nfx[n] = 1;
      if (size[l] > bs[n]) { bs[n] = size[l]; bc[n] = col[l]; }
      size[l] = 0; mx[l] = my[l] = mz[l] = ax[l] = ay[l] = az[l] = 0;
    }
    s.nl = 0;
    for (const n of touched) {                      // a merged mass: the biggest one's colour, or random
      size[n] = ns[n]; mx[n] = nmx[n]; my[n] = nmy[n]; mz[n] = nmz[n];
      col[n] = bs[n] >= 2 ? bc[n] : (Math.random() * P.ncol) | 0;
      if (nfx[n]) { fixed[n] = 1; mx[n] = my[n] = mz[n] = ax[n] = ay[n] = az[n] = 0; }
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
    const s = S, { NP, par, size, fixed, x, y, z, cx, cy, cz, fl } = s;
    const cand = [];
    let nf = 0;
    for (let r = 0; r < NP; r++) {                  // floating masses, and those at the ceiling
      if (par[r] !== r || fixed[r] || size[r] < 2) continue;
      fl[nf++] = r; cx[r] = cy[r] = cz[r] = 0;
      if (size[r] >= P.cap) cand.push(r);
    }
    if (!cand.length) return;
    if (P.crule) {                                  // their centroids
      for (let i = 0; i < NP; i++) { const r = par[i]; if (!fixed[i] && size[r] >= 2) { cx[r] += x[i]; cy[r] += y[i]; cz[r] += z[i]; } }
      for (let q = 0; q < nf; q++) { const r = fl[q]; cx[r] /= size[r]; cy[r] /= size[r]; cz[r] /= size[r]; }
    }
    const fire = [];
    for (const r of cand) {
      let k = P.erate;
      if (P.crule) {                                // rate ~ 1 / distance to the nearest other centroid
        let best = Infinity;
        for (let q = 0; q < nf; q++) {
          const r2 = fl[q];
          if (r2 === r) continue;
          const d2 = (cx[r2] - cx[r]) ** 2 + (cy[r2] - cy[r]) ** 2 + (cz[r2] - cz[r]) ** 2;
          if (d2 < best) best = d2;
        }
        const d = best < Infinity ? Math.sqrt(best) : 1.732 * s.L;
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
    const s = S, { L, NP, par, size, fixed, clus, btime, x, y, z, col, mx, my, mz, ax, ay, az, gid, gtil, lab, head, next, act } = s;
    const m = mem.length;
    let cx = 0, cy = 0, cz = 0;
    for (const i of mem) { cx += x[i]; cy += y[i]; cz += z[i]; }
    cx /= m; cy /= m; cz /= m;
    let best = Infinity, jn = -1;                   // the nearest particle of another floating mass
    for (let j = 0; j < NP; j++) {
      const r = par[j];
      if (r === R || fixed[j] || size[r] < 2) continue;
      const d2 = (x[j] - cx) ** 2 + (y[j] - cy) ** 2 + (z[j] - cz) ** 2;
      if (d2 < best) { best = d2; jn = j; }
    }
    let ex, ey, ez;
    if (jn >= 0) {                                  // away from it; with the colour rule, towards another colour
      ex = cx - x[jn]; ey = cy - y[jn]; ez = cz - z[jn];
      if (P.crule && col[R] !== col[par[jn]]) { ex = -ex; ey = -ey; ez = -ez; }
    } else { ex = Math.random() - 0.5; ey = Math.random() - 0.5; ez = Math.random() - 0.5; }
    let n1 = Math.abs(ex) + Math.abs(ey) + Math.abs(ez);
    if (n1 < 1e-9) { ex = 1; ey = ez = 0; n1 = 1; }
    ex /= n1; ey /= n1; ez /= n1;
    let kk = Math.round(P.efrac * m); kk = Math.max(1, Math.min(m - 1, kk));
    const proj = i => (x[i] - cx) * ex + (y[i] - cy) * ey + (z[i] - cz) * ez;
    const order = mem.slice().sort((a, b) => proj(b) - proj(a));
    for (let q = 0; q < m; q++) lab[order[q]] = q < kk ? 1 : 2;   // 1: ejecta (the side facing the direction), 2: rest
    const ke = kk, kr = m - kk, px = mx[R], py = my[R], pz = mz[R], jk = P.v0 * ke * kr / m;
    const vel = { 1: [(px * ke / m + ex * jk) / ke, (py * ke / m + ey * jk) / ke, (pz * ke / m + ez * jk) / ke],
                  2: [(px * kr / m - ex * jk) / kr, (py * kr / m - ey * jk) / kr, (pz * kr / m - ez * jk) / kr] };
    const colR = col[R], grace = Math.min(2000, Math.round(6 / Math.max(P.v0, 0.003)));
    for (const i of mem) par[i] = -1;               // regroup each part by contact (26-neighbourhood)
    for (const st of mem) {
      if (par[st] !== -1) continue;
      const piece = [st], lb = lab[st];
      par[st] = -2;
      for (let q = 0; q < piece.length; q++) {
        const i = piece[q];
        for (let dz = -1; dz <= 1; dz++) {
          const zz = z[i] + dz; if (zz < 0 || zz >= L) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y[i] + dy; if (yy < 0 || yy >= L) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x[i] + dx; if (xx < 0 || xx >= L) continue;
              for (let j = head[(zz * L + yy) * L + xx]; j >= 0; j = next[j]) if (par[j] === -1 && lab[j] === lb) { par[j] = -2; piece.push(j); }
            }
          }
        }
      }
      let r = piece[0];
      for (const i of piece) if (i < r) r = i;
      for (const i of piece) { par[i] = r; size[i] = 0; mx[i] = my[i] = mz[i] = ax[i] = ay[i] = az[i] = 0; act[i] = 1; }
      const sz = piece.length;
      size[r] = sz; mx[r] = vel[lb][0] * sz; my[r] = vel[lb][1] * sz; mz[r] = vel[lb][2] * sz;
      col[r] = colR; gid[r] = t; gtil[r] = t + grace;
      if (sz < 2) for (const i of piece) { clus[i] = 0; btime[i] = 0; }
    }
    s.ejections++;
  }

  // ------------------------------------------------------------------ drawing: particles into a depth buffer
  const ctx = cv.getContext('2d');
  let W = 0, img, pix, zb, rot = 0.6, tilt = 0.4, zoom = 1;
  const rgb = (r, g, b) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  const BG = rgb(0xCC, 0x77, 0x22), WHITE = rgb(255, 255, 255);
  const GREY = Array.from({ length: 256 }, (_, g) => rgb(g, g, g)), PALC = PAL.map(c => rgb(...c));
  const blend = (c0, c1, a) => rgb(((c0 & 255) * (1 - a) + (c1 & 255) * a) | 0, (((c0 >>> 8) & 255) * (1 - a) + ((c1 >>> 8) & 255) * a) | 0,
                                   (((c0 >>> 16) & 255) * (1 - a) + ((c1 >>> 16) & 255) * a) | 0);
  const hue = r => { const h = (Math.imul(r + 1, 2654435761) >>> 0) % 360 / 60, c = 0.85 * 0.9, xx = c * (1 - Math.abs(h % 2 - 1)), m = 0.55 - c / 2;
    const [a, b, d] = h < 1 ? [c, xx, 0] : h < 2 ? [xx, c, 0] : h < 3 ? [0, c, xx] : h < 4 ? [0, xx, c] : h < 5 ? [xx, 0, c] : [c, 0, xx];
    return rgb(((a + m) * 255) | 0, ((b + m) * 255) | 0, ((d + m) * 255) | 0); };
  function fit() {
    const dpr = window.devicePixelRatio || 1;
    W = Math.max(64, Math.round(cv.clientWidth * dpr));
    if (cv.width !== W) { cv.width = cv.height = W; img = ctx.createImageData(W, W); pix = new Uint32Array(img.data.buffer); zb = new Float32Array(W * W); }
  }
  function draw() {
    fit();
    const { L, NP, x, y, z, par, size, fixed, clus, btime, col, seed, tLast } = S;
    const h = (L - 1) / 2, sc = zoom * W / (L * 1.8), c2 = W / 2, gs = 230 / tLast, crule = P.crule, ncol = P.ncol;
    const ca = Math.cos(rot), sa = Math.sin(rot), ce = Math.cos(tilt), se = Math.sin(tilt);
    pix.fill(BG); zb.fill(-Infinity);
    const fs = Math.max(1, Math.round(0.8 * sc)), bs = Math.max(2, Math.round(1.1 * sc));
    const thr = P.cap < 1e9 ? Math.max(2, Math.ceil(P.hfrac * P.cap)) : Infinity;
    let hlOn = false;                               // colour masses near the ceiling, and dim the rest, but only
    if (P.hl && thr < Infinity)                     // while there is such a mass
      for (let r = 0; r < NP && !hlOn; r++) if (par[r] === r && !fixed[r] && size[r] >= thr) hlOn = true;
    for (let pass = 0; pass < 2; pass++) {          // 0: opaque (writes depth), 1: translucent, blended in front
      for (let i = 0; i < NP; i++) {
        const r = par[i], free = !clus[i], hl = hlOn && !fixed[i] && size[r] >= thr;
        let a = free ? 0.5 : 1;
        if (hlOn && !hl) a *= 0.3;
        if ((a < 1) !== (pass === 1)) continue;
        const dx = x[i] - h, dy = y[i] - h, dz = z[i] - h;
        const x1 = ca * dx - sa * dy, d1 = -(sa * dx + ca * dy), depth = se * dz + ce * d1;   // z is up
        const sx = c2 + x1 * sc, sy = c2 - (ce * dz - se * d1) * sc;
        const c = hl ? (crule ? PALC[col[r] % ncol] : hue(r)) : free ? WHITE : GREY[i === seed ? 0 : Math.round(gs * btime[i])];
        const sz = free ? fs : bs, x0 = Math.round(sx - sz / 2), y0 = Math.round(sy - sz / 2);
        const xa = Math.max(0, x0), xb = Math.min(W, x0 + sz), ya = Math.max(0, y0), yb = Math.min(W, y0 + sz);
        for (let yy = ya; yy < yb; yy++) for (let o = yy * W + xa, oe = yy * W + xb; o < oe; o++) {
          if (depth > zb[o]) { if (pass === 0) { pix[o] = c; zb[o] = depth; } else pix[o] = blend(pix[o], c, a); }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    const pr = (px, py, pz) => { const dx = px - h, dy = py - h, dz = pz - h, x1 = ca * dx - sa * dy, d1 = -(sa * dx + ca * dy); return [c2 + x1 * sc, c2 - (ce * dz - se * d1) * sc]; };
    ctx.strokeStyle = 'rgba(60,30,0,0.45)'; ctx.lineWidth = 1; ctx.beginPath();          // cube outline
    for (const [a, b] of [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]) {
      const pa = pr(a & 1 ? L - 1 : 0, a & 2 ? L - 1 : 0, a & 4 ? L - 1 : 0), pb = pr(b & 1 ? L - 1 : 0, b & 2 ? L - 1 : 0, b & 4 ? L - 1 : 0);
      ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();
  }

  // ------------------------------------------------------------------ controls and loop
  let running = true, rate = 0, frames = 0, tRate = performance.now(), stepsSince = 0, isVisible = true;
  new IntersectionObserver(es => { isVisible = es[0].isIntersecting; }).observe(cv);   // idle while scrolled away
  function info() {
    let free = 0, masses = 0, stuck = 0;
    for (let i = 0; i < S.NP; i++) {
      if (!S.clus[i]) free++;
      if (S.fixed[i]) stuck++;
      else if (S.par[i] === i && S.size[i] >= 2) masses++;
    }
    $('n3-info').textContent = `step ${S.t.toLocaleString()}  ·  free ${free.toLocaleString()}  ·  floating masses ${masses}  ·  ` +
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
  const bind = (id, key, fmt, conv = v => +v) => {
    const el = $(id), out = $(id + '-v');
    const set = () => { P[key] = conv(el.value); if (out) out.textContent = fmt(P[key]); };
    el.addEventListener('input', set); set();
  };
  bind('n3-speed', 'speed', v => v);
  bind('n3-n', 'N', v => v.toLocaleString());
  bind('n3-l', 'L', v => `${v}³`);
  bind('n3-cap', 'cap', v => v >= 1e9 ? 'off' : v, v => +v >= 500 ? 1e9 : +v);
  bind('n3-erate', 'erate', v => v.toPrecision(2) + ' /step', v => 10 ** +v);
  bind('n3-efrac', 'efrac', v => Math.round(v * 100) + '%', v => v / 100);
  bind('n3-v0', 'v0', v => v.toFixed(2));
  bind('n3-ncol', 'ncol', v => v);
  bind('n3-dref', 'dref', v => v + ' voxels');
  const tog = (id, key, label) => {
    const b = $(id), show = () => { b.classList.toggle('on', P[key]); b.textContent = `${label}: ${P[key] ? 'on' : 'off'}`; };
    b.addEventListener('click', () => { P[key] = !P[key]; show(); }); show();
  };
  tog('n3-crule', 'crule', 'Colour rule');
  tog('n3-hl', 'hl', 'Ceiling colours');
  tog('n3-win', 'win', '2×2×2 binding');
  tog('n3-anchor', 'anchor', 'Anchor');
  $('n3-run').addEventListener('click', e => { running = !running; e.target.textContent = running ? 'Pause' : 'Play'; });
  $('n3-reset').addEventListener('click', () => { reset(); info(); });
  let drag = null;                                  // drag to rotate, scroll to zoom, double-click to reset
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!drag) return;
    rot += (e.clientX - drag[0]) * 0.01; tilt = Math.max(-1.55, Math.min(1.55, tilt + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
  });
  const end = () => { drag = null; };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
  cv.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(0.3, Math.min(8, zoom * Math.exp(-e.deltaY * 0.0015))); }, { passive: false });
  cv.addEventListener('dblclick', () => { rot = 0.6; tilt = 0.4; zoom = 1; });
  reset(); info(); requestAnimationFrame(loop);
})();
