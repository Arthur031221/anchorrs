(() => {
  'use strict';
  const RM = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DPR = Math.min(2, window.devicePixelRatio || 1);
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* ---------- trajectory viewer ---------- */
  const PI3 = 'π<sup>3</sup>';
  const PIMOS = 'π<span class="pi"><sup>3</sup><sub>mos</sub></span>-SLAM';
  function style(name) {
    if (name.startsWith('Ground')) return { key: 'gt', label: 'Ground truth', c: '#BFC6D4' };
    if (name.startsWith('Chained')) return { key: 'chain', label: 'Chained windows', c: '#8C6BB1' };
    if (name.startsWith('Anchored')) return { key: 'anc', label: 'Anchored windows', c: '#6F8FE6' };
    if (name.startsWith('AnchorRS')) return { key: 'ours', label: 'AnchorRS', c: '#364FC7' };
    if (name.startsWith('VGGT')) return { key: 'vggt', label: 'VGGT-Long (' + PI3 + ')', c: '#019E73', dash: true };
    return { key: 'mos', label: PIMOS, c: '#B5508C', dash: true };
  }
  function bytes(s) {
    const b = atob(s); const u = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
    return u;
  }
  function decode(d) {
    const q = 1 / d.q;
    const pos = (s) => { const i16 = new Int16Array(bytes(s).buffer); const f = new Float32Array(i16.length); for (let i = 0; i < i16.length; i++) f[i] = i16[i] * q; return f; };
    const clouds = d.clouds.map((c) => {
      const rgb = bytes(c.col); const col = new Uint32Array(c.n);
      for (let i = 0; i < c.n; i++) col[i] = (255 << 24) | (rgb[3 * i + 2] << 16) | (rgb[3 * i + 1] << 8) | rgb[3 * i];
      return { n: c.n, p: pos(c.pos), col };
    });
    const trajs = d.trajs.map((t) => Object.assign({ n: t.n, p: pos(t.pos), ate: t.ate, on: true }, style(t.name)));
    /* frame the trajectories and the bulk of the points */
    const ax = [[], [], []];
    for (const t of trajs) for (let i = 0; i < t.p.length; i++) ax[i % 3].push(t.p[i]);
    const cl = clouds[clouds.length - 1].p;
    for (let i = 0; i < cl.length; i += 9) { ax[0].push(cl[i]); ax[1].push(cl[i + 1]); ax[2].push(cl[i + 2]); }
    const ctr = [], half = [];
    for (const a of ax) { a.sort((x, y) => x - y); const lo = a[Math.floor(a.length * 0.03)], hi = a[Math.floor(a.length * 0.97)]; ctr.push((lo + hi) / 2); half.push((hi - lo) / 2); }
    const r = Math.max(half[0], half[1], half[2] * 1.3) || 1;
    const norm = (f) => { for (let i = 0; i < f.length; i++) f[i] = (f[i] - ctr[i % 3]) / r; };
    clouds.forEach((c) => norm(c.p)); trajs.forEach((t) => norm(t.p));
    const zs = []; for (let i = 2; i < cl.length; i += 3) zs.push(cl[i]); zs.sort((a, b) => a - b);
    return { id: d.id, nframes: d.nframes, clouds, trajs, floor: zs[Math.floor(zs.length * 0.03)] || -0.6 };
  }

  class Viewer {
    constructor(fig) {
      this.fig = fig;
      this.mode = fig.dataset.mode;
      this.ids = fig.dataset.scenes.split(',');
      this.names = fig.dataset.names.split('|');
      this.box = $('.v-stage', fig);
      this.cv = $('canvas', this.box);
      this.ctx = this.cv.getContext('2d');
      this.off = document.createElement('canvas');
      this.octx = this.off.getContext('2d');
      this.meta = $('.v-meta', fig);
      this.cache = new Map();
      this.d = null; this.fade = null;
      this.yaw = -0.75; this.pitch = 0.6; this.t = 0; this.dur = 2600;
      this.playing = false; this.visible = false; this.started = false;
      this.scene = +(fig.dataset.start || 0);
      this.buildChips();
      this.bind();
      new ResizeObserver(() => this.resize()).observe(this.box);
      new IntersectionObserver((es) => {
        this.visible = es[0].isIntersecting;
        if (this.visible) { if (!this.started) this.start(); this.last = performance.now(); this.req(); }
      }, { rootMargin: '200px 0px' }).observe(fig);
      if (this.mode === 'stages') this.load(this.scene);
    }
    buildChips() {
      const box = $('.chips', this.fig);
      if (!box) { this.chips = []; return; }
      const groups = (box.dataset.groups || '').split('|').filter(Boolean).map((g) => g.split(':'));
      const rows = [];
      if (groups.length) {
        groups.forEach((g) => {
          const wrap = document.createElement('div'); wrap.className = 'chip-group';
          const lab = document.createElement('span'); lab.className = 'chip-lab'; lab.textContent = g[0];
          const row = document.createElement('div'); row.className = 'chip-row';
          wrap.appendChild(lab); wrap.appendChild(row); box.appendChild(wrap);
          for (let k = 0; k < +g[1]; k++) rows.push(row);
        });
      }
      this.chips = this.ids.map((id, i) => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'chip'; b.setAttribute('role', 'radio');
        b.setAttribute('aria-checked', String(i === this.scene)); b.tabIndex = i === this.scene ? 0 : -1;
        b.textContent = this.names[i];
        b.addEventListener('click', () => this.pick(i));
        b.addEventListener('keydown', (e) => {
          const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
          if (!k) return; e.preventDefault();
          const j = (i + k + this.ids.length) % this.ids.length; this.chips[j].focus(); this.pick(j);
        });
        (rows[i] || box).appendChild(b); return b;
      });
    }
    bind() {
      let drag = null;
      this.box.tabIndex = 0;
      this.box.addEventListener('pointerdown', (e) => {
        drag = { x: e.clientX, y: e.clientY, id: e.pointerId, touch: e.pointerType !== 'mouse' };
        this.box.setPointerCapture(e.pointerId);
      });
      this.box.addEventListener('pointermove', (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        this.yaw += (e.clientX - drag.x) * 0.0085;
        if (!drag.touch) this.pitch = Math.max(0.08, Math.min(1.42, this.pitch + (e.clientY - drag.y) * 0.006));
        drag.x = e.clientX; drag.y = e.clientY; this.dirty = true; this.req();
      });
      const end = () => { drag = null; };
      this.box.addEventListener('pointerup', end);
      this.box.addEventListener('pointercancel', end);
      this.box.addEventListener('keydown', (e) => {
        const k = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
        if (!k) return; e.preventDefault();
        this.yaw += k[0]; this.pitch = Math.max(0.08, Math.min(1.42, this.pitch + k[1])); this.dirty = true; this.req();
      });
    }
    setPlaying(on) {
      this.playing = on;
      if (on) { this.last = performance.now(); this.req(); }
    }
    start() {
      this.started = true;
      if (this.mode === 'methods') this.load(this.scene);
      this.t = 1; this.setPlaying(false);
      this.dirty = true;
    }
    async get(i) {
      const id = this.ids[i];
      if (!this.cache.has(id)) this.cache.set(id, fetch('data/' + id + '.json').then((r) => r.json()).then(decode));
      return this.cache.get(id);
    }
    async load(i) {
      this.fig.classList.remove('ready');
      const d = await this.get(i);
      if (i !== this.scene) return;
      this.d = d; this.fig.classList.add('ready');
      if (this.mode === 'stages') this.buildStages(); else this.buildLegend();
      if (this.meta) this.meta.textContent = this.names[i] + ', ' + d.nframes + ' frames, ATE in cm';
      this.reveal();
      const nx = this.ids[i + 1]; if (nx && this.visible) setTimeout(() => this.get(i + 1), 1200);
    }
    pick(i) {
      if (i === this.scene) return;
      this.scene = i;
      this.chips.forEach((c, j) => { c.setAttribute('aria-checked', String(j === i)); c.tabIndex = j === i ? 0 : -1; });
      this.load(i);
    }
    cloudNow() { return this.d.clouds[this.mode === 'stages' ? this.d.clouds.length - 1 : 0]; }
    reveal() {
      if (RM) { this.fade = null; this.dirty = true; this.req(); return; }
      this.fade = { t0: performance.now(), p: 0 }; this.req();
    }
    buildStages() {
      const box = $('.stages', this.fig); box.textContent = '';
      this.stageBtns = this.d.trajs.map((t, i) => {
        const b = document.createElement('div');
        b.className = 'stage';
        b.style.setProperty('--c', t.c);
        b.classList.toggle('ours', t.key === 'ours');
        b.classList.toggle('gt', t.key === 'gt');
        b.innerHTML = '<span class="s-name">' + t.label + '</span>' + (t.key === 'gt' ? '' : '<span class="s-ate">' + t.ate + '<small>cm</small></span>');
        box.appendChild(b); return b;
      });
    }
    buildLegend() {
      const ul = $('.legend', this.fig); ul.textContent = '';
      this.d.trajs.forEach((t) => {
        const li = document.createElement('li');
        li.className = 'lg' + (t.key === 'ours' ? ' ours' : '') + (t.dash ? ' dash' : '') + (t.key === 'gt' ? ' gt' : '');
        li.style.setProperty('--c', t.c);
        li.innerHTML = '<span class="sw"></span><span>' + t.label + '</span><span class="mono">' + (t.ate !== undefined && t.ate !== null && t.ate !== '' ? t.ate + ' cm' : '') + '</span>';
        ul.appendChild(li);
      });
    }
    resize() {
      const r = this.box.getBoundingClientRect();
      const W = Math.max(2, Math.round(r.width * DPR)), H = Math.max(2, Math.round(r.height * DPR));
      if (W === this.W && H === this.H) return;
      this.W = W; this.H = H;
      this.cv.width = this.off.width = W; this.cv.height = this.off.height = H;
      this.img = this.octx.createImageData(W, H);
      this.buf = new Uint32Array(this.img.data.buffer);
      this.zb = new Float32Array(W * H);
      this.f = Math.min(W, H * 1.15) * 0.38 * 3.1;
      this.dirty = true; this.req();
    }
    req() { if (!this.raf) this.raf = requestAnimationFrame((n) => this.tick(n)); }
    tick(now) {
      this.raf = 0;
      if (!this.visible || document.hidden) return;
      const dt = Math.min(64, now - (this.last || now)); this.last = now;
      let need = this.dirty;
      if (this.playing && this.d) {
        this.t = Math.min(1, this.t + dt / this.dur);
        if (this.t >= 1) this.setPlaying(false);
        need = true;
      }
      if (this.fade) { this.fade.p = (now - this.fade.t0) / 380; if (this.fade.p >= 1) this.fade = null; need = true; }
      if (need) { this.render(); this.dirty = false; }
      if (this.playing || this.fade) this.req();
    }
    render() {
      const d = this.d; if (!d || !this.W) return;
      const W = this.W, H = this.H, ctx = this.ctx, f = this.f, D = 3.1, cx = W / 2, cy = H * 0.5;
      const c = Math.cos(this.yaw), s = Math.sin(this.yaw), ce = Math.cos(this.pitch), se = Math.sin(this.pitch);
      const buf = this.buf, zb = this.zb; buf.fill(0); zb.fill(1e9);
      const sz = DPR >= 2 ? 4 : 3;
      const splat = (cl) => {
        const p = cl.p, col = cl.col, n = cl.n;
        for (let i = 0; i < n; i++) {
          const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
          const x1 = x * c - y * s, y1 = x * s + y * c;
          const dep = D + y1 * ce - z * se; if (dep < 0.3) continue;
          const px = (cx + f * x1 / dep) | 0, py = (cy - f * (y1 * se + z * ce) / dep) | 0;
          if (px < 0 || py < 0 || px > W - sz || py > H - sz) continue;
          const k = col[i];
          for (let yy = 0; yy < sz; yy++) {
            let o = (py + yy) * W + px;
            for (let xx = 0; xx < sz; xx++, o++) if (dep < zb[o]) { zb[o] = dep; buf[o] = k; }
          }
        }
      };
      splat(this.cloudNow());
      this.octx.putImageData(this.img, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const P = (x, y, z) => {
        const x1 = x * c - y * s, y1 = x * s + y * c, dep = D + y1 * ce - z * se;
        return [cx + f * x1 / dep, cy - f * (y1 * se + z * ce) / dep, dep];
      };
      /* floor grid */
      ctx.lineWidth = DPR; ctx.strokeStyle = 'rgba(54,79,199,.13)';
      ctx.beginPath();
      for (let g = -1.2; g <= 1.201; g += 0.2) {
        const a = P(g, -1.2, d.floor), b = P(g, 1.2, d.floor), a2 = P(-1.2, g, d.floor), b2 = P(1.2, g, d.floor);
        if (a[2] > 0.3 && b[2] > 0.3) { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
        if (a2[2] > 0.3 && b2[2] > 0.3) { ctx.moveTo(a2[0], a2[1]); ctx.lineTo(b2[0], b2[1]); }
      }
      ctx.stroke();
      ctx.globalAlpha = this.fade ? Math.max(0, this.fade.p) : 1;
      ctx.drawImage(this.off, 0, 0);
      ctx.globalAlpha = 1;
      /* trajectories */
      let list;
      if (this.mode === 'stages') list = d.trajs;
      else list = d.trajs.filter((t) => t.on).sort((a, b) => (a.key === 'gt' ? -1 : a.key === 'ours' ? 1 : 0) - (b.key === 'gt' ? -1 : b.key === 'ours' ? 1 : 0));
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      let head = null;
      for (const t of list) {
        const faint = this.mode === 'stages' && (t.key === 'chain' || t.key === 'anc');
        ctx.globalAlpha = faint ? 0.7 : 1;
        const last = this.t * (t.n - 1), k = Math.floor(last), fr = last - k;
        if (last <= 0) continue;
        ctx.beginPath();
        for (let i = 0; i <= k; i++) {
          const q = P(t.p[3 * i], t.p[3 * i + 1], t.p[3 * i + 2]);
          if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]);
        }
        let hx, hy;
        if (k < t.n - 1) {
          const j = 3 * k, j2 = j + 3;
          const q = P(t.p[j] + (t.p[j2] - t.p[j]) * fr, t.p[j + 1] + (t.p[j2 + 1] - t.p[j + 1]) * fr, t.p[j + 2] + (t.p[j2 + 2] - t.p[j + 2]) * fr);
          ctx.lineTo(q[0], q[1]); hx = q[0]; hy = q[1];
        } else { const q = P(t.p[3 * k], t.p[3 * k + 1], t.p[3 * k + 2]); hx = q[0]; hy = q[1]; }
        if (t.key === 'gt') { ctx.setLineDash([]); ctx.strokeStyle = 'rgba(150,160,180,.55)'; ctx.lineWidth = 5 * DPR; }
        else if (t.dash) { ctx.setLineDash([5 * DPR, 4 * DPR]); ctx.strokeStyle = t.c; ctx.lineWidth = 1.8 * DPR; }
        else { ctx.setLineDash([]); ctx.strokeStyle = t.c; ctx.lineWidth = 2.4 * DPR; }
        ctx.stroke();
        if (t.key !== 'gt' && !t.dash && !faint) head = [hx, hy, t.c];
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
      if (head && this.t < 1) {
        ctx.beginPath(); ctx.arc(head[0], head[1], 4.5 * DPR, 0, 6.2832);
        ctx.fillStyle = head[2]; ctx.fill(); ctx.lineWidth = 2 * DPR; ctx.strokeStyle = '#fff'; ctx.stroke();
      }
    }
  }
  /* ---------- 4D replay: three panes, one WebGL context, one shared camera ---------- */
  function initReplay() {
    const fig = $('#v4d'); if (!fig) return;
    const video = $('video', fig), stage = $('.v4-stage', fig), pre = $('.v4-pre', fig), cv = $('canvas', stage);
    const chipBox = $('.chips', fig), bar = $('.v4-bar', fig), hint = $('.v4-hint', fig);
    const playBtn = $('.v4-play', fig), range = $('.v4-range', fig), labs = $$('.v4-lab', stage);
    const ids = fig.dataset.scenes.split(','), names = fig.dataset.names.split('|');
    const KEYS = ['ours', 'p3m', 'vgl'], COLS = [[0.212, 0.31, 0.78], [0.71, 0.314, 0.549], [0.004, 0.62, 0.451]];
    const FOVF = 1 / Math.tan(0.4), FPK = 1.4;
    let gl = null;
    try { gl = cv.getContext('webgl', { antialias: false, alpha: false, preserveDrawingBuffer: true }); } catch (e) { gl = null; }
    if (!gl) return;
    let dead = false;
    const fallback = () => {
      if (dead) return; dead = true;
      fig.classList.remove('on');
      [chipBox, stage, bar, hint].forEach((n) => { n.hidden = true; });
      video.hidden = false;
    };
    const show = () => {
      fig.classList.add('on');
      [chipBox, stage, bar, hint].forEach((n) => { n.hidden = false; });
      video.hidden = true;
    };
    cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); fallback(); });
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader'); return s; };
    const prog = (vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link'); return p; };
    let P1, P2;
    try {
      P1 = prog('attribute vec3 p;attribute vec3 c;attribute float s;uniform mat4 M;uniform float K,lo,hi;varying vec3 v;void main(){vec4 q=M*vec4(p,1.);gl_Position=q;gl_PointSize=clamp(s*K/max(q.w,1e-3),lo,hi);v=c;}',
        'precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1.);}');
      P2 = prog('attribute vec3 p;uniform mat4 M;uniform float z;void main(){gl_Position=M*vec4(p,1.);gl_PointSize=z;}',
        'precision mediump float;uniform vec4 col;void main(){vec2 d=gl_PointCoord-.5;float r=dot(d,d);if(r>.25)discard;gl_FragColor=r>.15?vec4(1.,1.,1.,1.):col;}');
    } catch (e) { return; }
    const L1 = { p: gl.getAttribLocation(P1, 'p'), c: gl.getAttribLocation(P1, 'c'), s: gl.getAttribLocation(P1, 's'), M: gl.getUniformLocation(P1, 'M'), K: gl.getUniformLocation(P1, 'K'), lo: gl.getUniformLocation(P1, 'lo'), hi: gl.getUniformLocation(P1, 'hi') };
    const L2 = { p: gl.getAttribLocation(P2, 'p'), M: gl.getUniformLocation(P2, 'M'), z: gl.getUniformLocation(P2, 'z'), col: gl.getUniformLocation(P2, 'col') };
    const buf = () => { const b = gl.createBuffer(); return b; };
    const up = (b, data) => { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); };

    /* ----- data: decoded as recon27's viewer does ----- */
    const raw = new Map();
    const getRaw = (id) => {
      const ok = (r) => { if (!r.ok) throw new Error('http'); return r; };
      if (!raw.has(id)) raw.set(id, fetch('media/4d/' + id + '.json').then(ok).then((r) => r.json()).then((d) =>
        fetch('media/4d/' + d.bin).then(ok).then((r) => r.arrayBuffer()).then((ab) => {
          /* each array field is [byte offset, byte length] into the .bin file */
          const cut = (o, k) => { const [s, n] = o[k]; o[k] = new Uint8Array(ab.slice(s, s + n)); };
          ['off', 'scl', 'pts', 'col'].forEach((k) => cut(d, k));
          KEYS.forEach((m) => { cut(d[m], 's'); cut(d[m], 'c2w'); });
          return d;
        })));
      return raw.get(id);
    };
    const b64 = (s, T) => { const u = typeof s === 'string' ? bytes(s) : s; return T === Uint8Array ? u : new T(u.buffer, 0, Math.floor(u.length / T.BYTES_PER_ELEMENT)); };
    function build(d) {
      const off = b64(d.off, Float32Array), scl = b64(d.scl, Float32Array), pts = b64(d.pts, Int16Array), col = b64(d.col, Uint8Array);
      const nf = d.n.length, st = new Int32Array(nf + 1);
      for (let q = 0; q < nf; q++) st[q + 1] = st[q] + d.n[q];
      const tot = st[nf], alpha = new Float32Array(nf);
      for (let q = 0; q < nf; q++) {
        const nq = d.n[q], s0 = st[q], sc = scl[q], ox = off[3 * q], oy = off[3 * q + 1], oz = off[3 * q + 2], a = [];
        let qx = 0, qy = 0, qz = 0;
        for (let i = 0; i < nq; i++) {
          const j = 3 * (s0 + i), x = pts[j] * sc + ox, y = pts[j + 1] * sc + oy, z = pts[j + 2] * sc + oz;
          if (i > 0 && z > 1e-6) { const dx = x - qx, dy = y - qy, dz = z - qz; a.push(Math.sqrt(dx * dx + dy * dy + dz * dz) / z); }
          qx = x; qy = y; qz = z;
        }
        a.sort((u, v) => u - v); alpha[q] = a.length ? a[a.length >> 1] : 0;
      }
      const meth = KEYS.map((k) => {
        const S = b64(d[k].s, Float32Array), M = b64(d[k].c2w, Float32Array);
        const pos = new Float32Array(3 * tot), siz = new Float32Array(tot), path = new Float32Array(3 * nf);
        for (let q = 0; q < nf; q++) {
          const s = S[q], m = 12 * q, n = d.n[q], ps = st[q], sc = scl[q], ox = off[3 * q], oy = off[3 * q + 1], oz = off[3 * q + 2];
          for (let i = 0; i < n; i++) {
            const c = 3 * (ps + i), x = (pts[c] * sc + ox) * s, y = (pts[c + 1] * sc + oy) * s, z = (pts[c + 2] * sc + oz) * s;
            pos[c] = M[m] * x + M[m + 1] * y + M[m + 2] * z + M[m + 3];
            pos[c + 1] = M[m + 4] * x + M[m + 5] * y + M[m + 6] * z + M[m + 7];
            pos[c + 2] = M[m + 8] * x + M[m + 9] * y + M[m + 10] * z + M[m + 11];
            siz[ps + i] = FPK * alpha[q] * z;
          }
          path[3 * q] = M[m + 3]; path[3 * q + 1] = M[m + 7]; path[3 * q + 2] = M[m + 11];
        }
        /* camera path: each pose and three points between it and the next, so the line reads as one stroke */
        const sub = 4, pp = new Float32Array(3 * ((nf - 1) * sub + 1));
        for (let q = 0; q < nf - 1; q++) for (let u = 0; u < sub; u++) for (let a = 0; a < 3; a++) pp[3 * (q * sub + u) + a] = path[3 * q + a] + (path[3 * q + 3 + a] - path[3 * q + a]) * (u / sub);
        for (let a = 0; a < 3; a++) pp[3 * (nf - 1) * sub + a] = path[3 * (nf - 1) + a];
        return { pos, siz, pp, path };
      });
      return { nf, st, col, meth, ctr: d.center, rad: d.radius, up: d.up };
    }

    /* ----- GPU state for the scene on screen ----- */
    let G = null; // { nf, st, bufs:[{pos,siz,pp}], colb, path, ctr, rad, up }
    const free = () => {
      if (!G) return;
      gl.deleteBuffer(G.colb); G.bufs.forEach((b) => { gl.deleteBuffer(b.pos); gl.deleteBuffer(b.siz); gl.deleteBuffer(b.pp); });
      G = null;
    };
    function upload(B) {
      free();
      const colb = buf(); up(colb, B.col);
      const bufs = B.meth.map((m) => { const o = { pos: buf(), siz: buf(), pp: buf() }; up(o.pos, m.pos); up(o.siz, m.siz); up(o.pp, m.pp); return o; });
      G = { nf: B.nf, st: B.st, colb, bufs, path: B.meth.map((m) => m.path), ctr: B.ctr, rad: B.rad, up: B.up };
    }

    /* ----- shared camera ----- */
    const cam = { az: 0.6, el: 0.85, dist: 5 };
    const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dotv = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    function resetCam() { cam.az = 0.6; cam.el = 0.85; cam.dist = G ? G.rad * 3.8 : 5; }
    function matrix(aspect) {
      const u = norm(G.up), r = Math.abs(u[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0], e1 = norm(cross(u, r)), e2 = cross(u, e1);
      const ce = Math.cos(cam.el), se = Math.sin(cam.el), ca = Math.cos(cam.az), sa = Math.sin(cam.az), eye = [];
      for (let i = 0; i < 3; i++) eye.push(G.ctr[i] + cam.dist * (ce * (ca * e1[i] + sa * e2[i]) + se * u[i]));
      const f = norm([G.ctr[0] - eye[0], G.ctr[1] - eye[1], G.ctr[2] - eye[2]]), s = norm(cross(f, u)), v = cross(s, f);
      const near = Math.max(G.rad * 0.02, cam.dist * 0.02), far = cam.dist + G.rad * 6, fx = FOVF / aspect, fy = FOVF;
      const a = (far + near) / (near - far), b = 2 * far * near / (near - far);
      /* column-major view-projection */
      const tx = -dotv(s, eye), ty = -dotv(v, eye), tz = dotv(f, eye);
      return new Float32Array([
        fx * s[0], fy * v[0], a * -f[0] + 0, f[0],
        fx * s[1], fy * v[1], a * -f[1], f[1],
        fx * s[2], fy * v[2], a * -f[2], f[2],
        fx * tx, fy * ty, a * tz + b, -tz]);
    }

    /* ----- drawing ----- */
    let frame = 99, W = 0, H = 0, dirty = true, raf = 0, vis = false, auto = false, playing = false, held = false, last = 0, acc = 0;
    function panes() {
      const sr = stage.getBoundingClientRect(), k = W / Math.max(1, sr.width);
      return labs.map((l) => { const r = l.getBoundingClientRect(); return { x: Math.round((r.left - sr.left) * k), y: Math.round((r.top - sr.top) * k), w: Math.round(r.width * k), h: Math.round(r.height * k) }; });
    }
    function draw() {
      dirty = false;
      if (!G || !W) return;
      gl.disable(gl.SCISSOR_TEST); gl.clearColor(0.933, 0.945, 0.969, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.SCISSOR_TEST);
      const ps = panes(), q = frame, n = G.st[q + 1], dpr = W / Math.max(1, stage.clientWidth);
      ps.forEach((p, m) => {
        const y = H - p.y - p.h;
        gl.viewport(p.x, y, p.w, p.h); gl.scissor(p.x, y, p.w, p.h);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        const M = matrix(p.w / p.h), b = G.bufs[m];
        gl.enable(gl.DEPTH_TEST);
        gl.useProgram(P1);
        gl.uniformMatrix4fv(L1.M, false, M); gl.uniform1f(L1.K, FOVF * p.h / 2); gl.uniform1f(L1.lo, 2 * dpr); gl.uniform1f(L1.hi, 22 * dpr);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.pos); gl.enableVertexAttribArray(L1.p); gl.vertexAttribPointer(L1.p, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, G.colb); gl.enableVertexAttribArray(L1.c); gl.vertexAttribPointer(L1.c, 3, gl.UNSIGNED_BYTE, true, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.siz); gl.enableVertexAttribArray(L1.s); gl.vertexAttribPointer(L1.s, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, n);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(P2); gl.disableVertexAttribArray(L1.c); gl.disableVertexAttribArray(L1.s);
        gl.uniformMatrix4fv(L2.M, false, M);
        gl.bindBuffer(gl.ARRAY_BUFFER, b.pp); gl.enableVertexAttribArray(L2.p); gl.vertexAttribPointer(L2.p, 3, gl.FLOAT, false, 0, 0);
        const c = COLS[m]; gl.uniform4f(L2.col, c[0], c[1], c[2], 1);
        gl.uniform1f(L2.z, 5 * dpr); gl.drawArrays(gl.POINTS, 0, q * 4 + 1);
        gl.uniform1f(L2.z, 12 * dpr); gl.drawArrays(gl.POINTS, q * 4, 1);
        gl.disableVertexAttribArray(L2.p);
      });
    }
    function req() { if (!raf) raf = requestAnimationFrame(tick); }
    function tick(now) {
      raf = 0;
      if (dead) return;
      if (!vis || document.hidden) { last = 0; return; }
      if (playing && G) {
        if (!last) last = now;
        acc += Math.min(100, now - last) / 1000 * 10; last = now;
        if (acc >= 1) { const k = Math.floor(acc); acc -= k; frame = frame + k > G.nf - 1 ? 0 : frame + k; range.value = frame; dirty = true; }
      } else last = 0;
      if (dirty) draw();
      if (playing) req();
    }
    function setPlaying(on) {
      playing = on; playBtn.textContent = on ? 'Pause' : 'Play'; playBtn.setAttribute('aria-pressed', String(on));
      if (on && G && frame >= G.nf - 1) { frame = 0; range.value = 0; dirty = true; }
      last = 0; acc = 0; req();
    }
    function resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1), r = stage.getBoundingClientRect();
      const w = Math.max(2, Math.round(r.width * dpr)), h = Math.max(2, Math.round(r.height * dpr));
      if (w === W && h === H) return;
      W = w; H = h; cv.width = w; cv.height = h; dirty = true; req();
    }
    new ResizeObserver(resize).observe(stage);

    /* ----- scene choice ----- */
    let scene = 0, started = false, token = 0;
    const chips = ids.map((id, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', String(i === 0)); b.tabIndex = i === 0 ? 0 : -1; b.textContent = names[i];
      b.addEventListener('click', () => pick(i));
      b.addEventListener('keydown', (e) => {
        const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (!k) return; e.preventDefault();
        const j = (i + k + ids.length) % ids.length; chips[j].focus(); pick(j);
      });
      chipBox.appendChild(b); return b;
    });
    async function load(i) {
      const t = ++token;
      try {
        const d = await getRaw(ids[i]);
        if (t !== token || dead) return;
        const B = build(d);
        upload(B); resetCam();
        frame = B.nf - 1; range.max = B.nf - 1; range.value = frame;
        pre.hidden = true; resize(); dirty = true;
        if (!started) { started = true; if (auto && !held && !RM) setPlaying(true); else setPlaying(false); }
        req();
      } catch (e) { if (t === token) fallback(); }
    }
    function pick(i) {
      if (i === scene) return;
      scene = i;
      chips.forEach((c, j) => { c.setAttribute('aria-checked', String(j === i)); c.tabIndex = j === i ? 0 : -1; });
      load(i);
    }

    /* ----- controls: drag or arrows turn, wheel or pinch zoom ----- */
    const clampD = () => { if (G) cam.dist = Math.max(G.rad * 0.4, Math.min(G.rad * 6, cam.dist)); };
    const ptrs = new Map();
    let pinch = 0;
    stage.addEventListener('pointerdown', (e) => {
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, touch: e.pointerType !== 'mouse' });
      try { stage.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ }
      if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    });
    stage.addEventListener('pointermove', (e) => {
      const p = ptrs.get(e.pointerId); if (!p || !G) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      if (ptrs.size >= 2) {
        p.x = e.clientX; p.y = e.clientY;
        const [a, b] = [...ptrs.values()], dd = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0 && dd > 0) { cam.dist *= pinch / dd; clampD(); }
        pinch = dd;
        cam.el = Math.max(-1.5, Math.min(1.5, cam.el + dy * 0.003));
      } else {
        p.x = e.clientX; p.y = e.clientY;
        cam.az -= dx * 0.0085;
        if (!p.touch) cam.el = Math.max(-1.5, Math.min(1.5, cam.el + dy * 0.006));
      }
      dirty = true; req();
    });
    const end = (e) => { ptrs.delete(e.pointerId); pinch = 0; };
    stage.addEventListener('pointerup', end); stage.addEventListener('pointercancel', end);
    stage.addEventListener('wheel', (e) => {
      if (!G) return; e.preventDefault();
      cam.dist *= Math.exp(Math.max(-100, Math.min(100, e.deltaY)) * 0.0015); clampD(); dirty = true; req();
    }, { passive: false });
    stage.addEventListener('keydown', (e) => {
      const k = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
      if (!k || !G) return; e.preventDefault();
      cam.az += k[0]; cam.el = Math.max(-1.5, Math.min(1.5, cam.el + k[1])); dirty = true; req();
    });

    /* ----- time axis ----- */
    range.addEventListener('input', () => { held = true; setPlaying(false); frame = +range.value; dirty = true; req(); });
    playBtn.addEventListener('click', () => { if (playing) { held = true; setPlaying(false); } else { held = false; setPlaying(true); } });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) { last = 0; dirty = true; req(); } });

    /* ----- lazy load and autoplay ----- */
    show();
    pre.hidden = false;
    setPlaying(false);
    new IntersectionObserver((es) => { if (es[0].isIntersecting && !G && !dead && !started) { load(scene); } }, { rootMargin: '600px 0px' }).observe(fig);
    new IntersectionObserver((es) => {
      const e = es[es.length - 1];
      vis = e.isIntersecting; auto = e.intersectionRatio >= 0.4;
      if (vis) { dirty = true; req(); }
      if (!G || RM) return;
      if (auto) { if (!held && !playing) setPlaying(true); }
      else if (playing) setPlaying(false);
    }, { threshold: [0, 0.4] }).observe(stage);
  }
  try { initReplay(); } catch (e) { const f = $('#v4d'); if (f) { $$('.chips, .v4-stage, .v4-bar, .v4-hint', f).forEach((n) => { n.hidden = true; }); const v = $('video', f); if (v) v.hidden = false; } }

  const viewers = $$('.viewer').map((f) => new Viewer(f));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    viewers.forEach((v) => { v.last = performance.now(); v.dirty = true; v.req(); });
  });

  /* ---------- videos play while at least 40% is on screen; a video the reader paused stays paused ---------- */
  if (!RM && 'IntersectionObserver' in window) {
    const vio = new IntersectionObserver((es) => {
      for (const e of es) {
        const v = e.target;
        if (e.isIntersecting) { if (!v.dataset.held) { const p = v.play(); if (p) p.catch(() => {}); } }
        else if (!v.paused) { v.dataset.auto = '1'; v.pause(); }
      }
    }, { threshold: 0.4 });
    $$('video').forEach((v) => {
      v.addEventListener('pause', () => { if (v.dataset.auto) delete v.dataset.auto; else if (!v.ended) v.dataset.held = '1'; });
      v.addEventListener('play', () => { delete v.dataset.held; });
      vio.observe(v);
    });
  }

  /* contents: mark the section being read */
  (() => {
    const toc = document.querySelector('.toc');
    if (!toc) return;
    const list = toc.querySelector('ol');
    const links = [...toc.querySelectorAll('a[href^="#"]')];
    const secs = links.map((a) => document.getElementById(a.getAttribute('href').slice(1)));
    if (secs.some((s) => !s)) return;
    const still = matchMedia('(prefers-reduced-motion: reduce)');
    let cur = -1, queued = false;
    const set = (i) => {
      if (i === cur) return;
      cur = i;
      links.forEach((a, k) => { if (k === i) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
      // the phone strip scrolls sideways: keep the current label in view
      if (i >= 0 && list.scrollWidth > list.clientWidth + 1) {
        const a = links[i], left = a.offsetLeft - (parseFloat(getComputedStyle(list).paddingLeft) || 0);
        if (a.offsetLeft < list.scrollLeft || a.offsetLeft + a.offsetWidth > list.scrollLeft + list.clientWidth) list.scrollTo({ left, behavior: still.matches ? 'auto' : 'smooth' });
      }
    };
    const pick = () => {
      queued = false;
      const line = innerHeight * 0.35;
      let i = -1;
      secs.forEach((s, k) => { if (s.getBoundingClientRect().top <= line) i = k; });
      if (innerHeight + scrollY >= document.documentElement.scrollHeight - 40) i = secs.length - 1;
      set(i);
    };
    const ask = () => { if (!queued) { queued = true; requestAnimationFrame(pick); } };
    addEventListener('scroll', ask, { passive: true });
    addEventListener('resize', ask);
    pick();
  })();
})();
