/* modelos.js — uma animação SVG curta por modelo (3 a 6 s), repetível com um clique.
   Regras: o estado final é o próprio desenho (Web Animations com fill "backwards"), então a impressão e o modo
   sem movimento saem completos. Números exibidos só vêm de DADOS; as figuras sem dados reais são esquemas e
   dizem isso no canto ("esquema"). Paleta e funções de desenho: graficos.js (G). */
"use strict";
const M = (() => {
  const { s, t, COR, F, corMembro } = G;
  const W = 360, H = 200;
  const VIOLETA = "#6250d6";

  /* ---------------------------------------------------------------- utilidades */
  function rng(seed) {                                  // pseudoaleatório determinístico (mulberry32)
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let x = a;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }
  function nuvem(r, n, cx, cy, sx, sy) {
    const p = [];
    for (let i = 0; i < n; i++) {
      const u = Math.max(r(), 1e-6), v = r(), rr = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * v;
      p.push([cx + sx * rr * Math.cos(th), cy + sy * rr * Math.sin(th)]);
    }
    return p.map(([x, y]) => [Math.max(12, Math.min(W - 12, x)), Math.max(14, Math.min(H - 14, y))]);
  }
  const sig = x => 1 / (1 + Math.exp(-x));
  const lig = { v: true };
  function an(el, quadros, op) {                         // anima só quando ligado (não na impressão)
    if (!lig.v || G.semMovimento || !el || !el.animate) return null;
    return el.animate(quadros, Object.assign({ fill: "backwards", easing: "cubic-bezier(.2,.7,.2,1)" }, op));
  }
  const pop = (el, atraso, dur = 360) => an(el, [{ opacity: 0, transform: "scale(.2)" }, { opacity: 1, transform: "scale(1)" }], { duration: dur, delay: atraso });
  const surge = (el, atraso, dur = 300) => an(el, [{ opacity: 0 }, { opacity: 1 }], { duration: dur, delay: atraso });
  function caixaPropria(el) { el.style.transformBox = "fill-box"; el.style.transformOrigin = "50% 50%"; return el; }
  function traco(el, atraso, dur) {                      // desenha uma linha/caminho do começo ao fim
    el.setAttribute("pathLength", 1);
    return an(el, [{ strokeDasharray: "1 1", strokeDashoffset: 1 }, { strokeDasharray: "1 1", strokeDashoffset: 0 }], { duration: dur, delay: atraso, easing: "ease-in-out" });
  }
  function texto(pai, x, y, txt, cls, at) { return t(pai, x, y, txt, Object.assign({ class: cls || "m-rot" }, at)); }
  const seg = (pai, x1, y1, x2, y2, at) => s("path", Object.assign({ d: `M${x1.toFixed(1)},${y1.toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}`, fill: "none" }, at), pai);
  function chip(pai, x, y, txt, cor) {
    const g = s("g", null, pai), w = txt.length * 6.4 + 14;
    s("rect", { x: x - w / 2, y: y - 9, width: w, height: 18, rx: 9, fill: cor, "fill-opacity": 0.16, stroke: cor, "stroke-width": 1 }, g);
    texto(g, x, y, txt, "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
    return g;
  }
  const esquema = (sv, emCima) => emCima ? texto(sv, W - 6, 11, "esquema", "m-nota", { "text-anchor": "end" }) : texto(sv, 8, H - 5, "esquema", "m-nota");
  function palavras(D) {
    const sp = D.palavras.spam.map(p => p.palavra).filter(p => p !== "paliourg");
    const re = D.palavras.email_real.map(p => p.palavra);
    const lr = Object.fromEntries([...D.palavras.spam, ...D.palavras.email_real].map(p => [p.palavra, p.log_razao]));
    return { sp, re, lr };
  }

  /* ---------------------------------------------------------------- Regressão Logística: a balança de palavras */
  function lr(sv, D) {
    const { sp, re, lr: lg } = palavras(D);
    const seq = [re[0], sp[0], sp[1], sp[2]];
    const px = 180, py = 70, L = 112, T0 = 400, dt = 750;
    let soma = 0;
    const angs = seq.map(w => (soma += lg[w], Math.max(-12, Math.min(12, soma * 1.4))));
    s("path", { d: `M${px - 18},150 L${px + 18},150 L${px},${py + 4} Z`, fill: COR.eixo }, sv);
    s("line", { x1: px - 46, x2: px + 46, y1: 150, y2: 150, stroke: COR.eixo, "stroke-width": 2 }, sv);
    const viga = s("line", { x1: px - L, x2: px + L, y1: py, y2: py, stroke: COR.tinta, "stroke-width": 3, "stroke-linecap": "round" }, sv);
    viga.style.transformBox = "view-box"; viga.style.transformOrigin = `${px}px ${py}px`;
    const fim = angs[angs.length - 1];
    viga.style.transform = `rotate(${fim}deg)`;
    const dur = T0 + dt * seq.length + 300;
    const quadrosA = [{ transform: "rotate(0deg)", offset: 0 }, { transform: "rotate(0deg)", offset: T0 / dur }];
    angs.forEach((a, i) => quadrosA.push({ transform: `rotate(${a}deg)`, offset: (T0 + dt * (i + 1)) / dur }));
    quadrosA[quadrosA.length - 1].offset = Math.min(1, quadrosA[quadrosA.length - 1].offset);
    if (quadrosA[quadrosA.length - 1].offset < 1) quadrosA.push({ transform: `rotate(${fim}deg)`, offset: 1 });
    an(viga, quadrosA, { duration: dur, easing: "ease-in-out" });
    const prato = (x, lado, cor, rotulo) => {
      const g = s("g", null, sv);
      s("line", { x1: x, x2: x - 40, y1: py, y2: py + 52, stroke: COR.muda }, g);
      s("line", { x1: x, x2: x + 40, y1: py, y2: py + 52, stroke: COR.muda }, g);
      s("path", { d: `M${x - 46},${py + 52} Q${x},${py + 70} ${x + 46},${py + 52} Z`, fill: cor, "fill-opacity": 0.18, stroke: cor, "stroke-width": 1.5 }, g);
      texto(g, x, py + 74, rotulo, "m-rot forte", { "text-anchor": "middle" });
      const dy = a => lado * L * Math.sin(a * Math.PI / 180);
      g.style.transform = `translateY(${dy(fim)}px)`;
      const q = [{ transform: "translateY(0px)", offset: 0 }, { transform: "translateY(0px)", offset: T0 / dur }];
      angs.forEach((a, i) => q.push({ transform: `translateY(${dy(a)}px)`, offset: (T0 + dt * (i + 1)) / dur }));
      if (q[q.length - 1].offset < 1) q.push({ transform: `translateY(${dy(fim)}px)`, offset: 1 });
      an(g, q, { duration: dur, easing: "ease-in-out" });
      return g;
    };
    const pr = prato(px - L, -1, COR.real, "e-mail real"), ps = prato(px + L, 1, COR.spam, "spam");
    const pos = { real: 0, spam: 0 };
    seq.forEach((w, i) => {
      const ehSpam = sp.includes(w), alvo = ehSpam ? ps : pr, n = pos[ehSpam ? "spam" : "real"]++;
      const x = (ehSpam ? px + L : px - L) + (n % 2 ? 20 : -20), y = py + 42 - Math.floor(n / 2) * 20;
      const c = chip(alvo, x, y, w, ehSpam ? COR.spam : COR.real);
      an(c, [{ opacity: 0, transform: "translateY(-70px)" }, { opacity: 1, transform: "translateY(0px)" }], { duration: 420, delay: T0 + dt * i + dt - 450 });
    });
    const p = sig(soma * 0.4), x0 = 60, x1 = 300;
    texto(sv, x0, 189, "nota de spam", "m-rot", { "dominant-baseline": "central" });
    s("rect", { x: x0 + 78, y: 184, width: x1 - x0 - 78, height: 10, rx: 5, fill: COR.grade }, sv);
    const b = s("rect", { x: x0 + 78, y: 184, width: (x1 - x0 - 78) * p, height: 10, rx: 5, fill: COR.spam }, sv);
    b.style.transformBox = "fill-box"; b.style.transformOrigin = "0% 50%";
    an(b, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 700, delay: dur });
    texto(sv, 8, 14, "palavras típicas de spam e de e-mail real caem nos pratos", "m-nota");
    texto(sv, W - 6, 11, "esquema", "m-nota", { "text-anchor": "end" });
  }

  /* ---------------------------------------------------------------- LinearSVC: a reta com a maior folga */
  function lsvc(sv) {
    const r = rng(7), c1 = [112, 128], c2 = [250, 74];
    const real = nuvem(r, 15, c1[0], c1[1], 20, 16), spam = nuvem(r, 15, c2[0], c2[1], 20, 16);
    const m = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2], dx = c2[0] - c1[0], dy = c2[1] - c1[1], nn = Math.hypot(dx, dy);
    const n = [dx / nn, dy / nn], dist0 = p => (p[0] - m[0]) * n[0] + (p[1] - m[1]) * n[1];
    // centra a reta entre o e-mail real e o spam mais próximos (a maior folga nessa direção)
    const dr = Math.min(...real.map(p => -dist0(p))), ds = Math.min(...spam.map(p => dist0(p))), desl = (ds - dr) / 2;
    m[0] += n[0] * desl; m[1] += n[1] * desl;
    const dist = p => (p[0] - m[0]) * n[0] + (p[1] - m[1]) * n[1];
    const margem = Math.max(6, (dr + ds) / 2 - 1.5);
    const ang = Math.atan2(n[1], n[0]) * 180 / Math.PI + 90;
    const ponto = (p, cor) => s("circle", { cx: p[0], cy: p[1], r: 4.5, fill: cor, stroke: COR.fundo, "stroke-width": 1.5 }, sv);
    // candidatas ruins: aparecem e somem
    [-30, 26].forEach((d, i) => {
      const g = s("g", { transform: `translate(${m[0]},${m[1]}) rotate(${ang + d})` }, sv);
      const l = s("line", { x1: -230, x2: 230, y1: 0, y2: 0, stroke: COR.muda, "stroke-width": 1.5, "stroke-dasharray": "4 4", opacity: 0 }, g);
      an(l, [{ opacity: 0 }, { opacity: 0.9, offset: 0.3 }, { opacity: 0.9, offset: 0.7 }, { opacity: 0 }], { duration: 1100, delay: 300 + i * 1000 });
    });
    const g = s("g", { transform: `translate(${m[0]},${m[1]}) rotate(${ang})` }, sv);
    const faixa = s("rect", { x: -240, y: -margem, width: 480, height: 2 * margem, fill: COR.tinta, "fill-opacity": 0.07 }, g);
    caixaPropria(faixa);
    an(faixa, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 900, delay: 2400 });
    [-margem, margem].forEach(y => { const l = s("line", { x1: -240, x2: 240, y1: y, y2: y, stroke: COR.tinta2, "stroke-dasharray": "3 3" }, g); surge(l, 3100); });
    const reta = seg(g, -240, 0, 240, 0, { stroke: COR.tinta, "stroke-width": 2.2 });
    traco(reta, 2300, 700);
    real.forEach(p => ponto(p, COR.real)); spam.forEach(p => ponto(p, COR.spam));
    [...real.map(p => [p, -dist(p)]), ...spam.map(p => [p, dist(p)])].filter(([, d]) => d < margem + 4).forEach(([p]) => {
      const c = s("circle", { cx: p[0], cy: p[1], r: 8, fill: "none", stroke: COR.tinta, "stroke-width": 1.4 }, sv);
      pop(caixaPropria(c), 3500);
    });
    const tx = texto(sv, 8, 14, "a reta escolhida deixa a rua mais larga entre os lados", "m-nota");
    surge(tx, 3300);
    esquema(sv);
  }

  /* ---------------------------------------------------------------- kNN: os vizinhos votam */
  function knn(sv, D) {
    const r = rng(21), real = nuvem(r, 18, 120, 112, 34, 30), spam = nuvem(r, 18, 238, 92, 34, 30);
    const k = parseInt(((D.der.mod.knn || {}).cfg_padrao_d || {}).k, 10) || 5;
    const q = [192, 96];
    const todos = [...real.map(p => [p, 0]), ...spam.map(p => [p, 1])].map(([p, c]) => ({ p, c, d: Math.hypot(p[0] - q[0], p[1] - q[1]) })).sort((a, b) => a.d - b.d);
    const viz = todos.slice(0, k), nSpam = viz.filter(v => v.c).length, ehSpam = nSpam * 2 > k;
    const raio = viz[viz.length - 1].d + 6;
    const anel = s("circle", { cx: q[0], cy: q[1], r: raio, fill: COR.tinta, "fill-opacity": 0.05, stroke: COR.tinta2, "stroke-dasharray": "3 3" }, sv);
    pop(caixaPropria(anel), 400, 700);
    viz.forEach((v, i) => { const l = seg(sv, q[0], q[1], v.p[0], v.p[1], { stroke: v.c ? COR.spam : COR.real, "stroke-width": 1.6 }); traco(l, 1100 + i * 260, 260); });
    todos.forEach(v => s("circle", { cx: v.p[0], cy: v.p[1], r: 4.5, fill: v.c ? COR.spam : COR.real, stroke: COR.fundo, "stroke-width": 1.5 }, sv));
    viz.forEach((v, i) => { const c = s("circle", { cx: v.p[0], cy: v.p[1], r: 8, fill: "none", stroke: COR.tinta, "stroke-width": 1.3 }, sv); pop(caixaPropria(c), 1100 + i * 260 + 200); });
    const fim = 1100 + k * 260 + 400;
    const novo = s("circle", { cx: q[0], cy: q[1], r: 8, fill: ehSpam ? COR.spam : COR.real, stroke: COR.tinta, "stroke-width": 2 }, sv);
    an(novo, [{ fill: COR.fundo }, { fill: COR.fundo, offset: 0.6 }, { fill: ehSpam ? COR.spam : COR.real }], { duration: fim + 300 });
    const qm = texto(sv, q[0], q[1], "?", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
    an(qm, [{ opacity: 1 }, { opacity: 1, offset: 0.9 }, { opacity: 0 }], { duration: fim, fill: "both" });
    qm.style.opacity = 0;
    const tx = texto(sv, 8, 14, `e-mail novo: os ${k} vizinhos mais parecidos votam; a maioria decide`, "m-nota");
    surge(tx, fim);
    esquema(sv);
  }

  /* ---------------------------------------------------------------- SVM RBF: a fronteira curva */
  function svm(sv) {
    const r = rng(11), cx = 182, cy = 102, spam = [], real = [];
    for (let i = 0; i < 16; i++) { const a = r() * 2 * Math.PI, d = 30 * Math.sqrt(r()); spam.push([cx + d * Math.cos(a), cy + d * 0.85 * Math.sin(a)]); }
    for (let i = 0; i < 24; i++) { const a = r() * 2 * Math.PI, d = 58 + 30 * r(); real.push([cx + d * 1.35 * Math.cos(a), cy + d * 0.85 * Math.sin(a)]); }
    const reta = s("line", { x1: 120, y1: 196, x2: 250, y2: 6, stroke: COR.muda, "stroke-width": 1.6, "stroke-dasharray": "5 4", opacity: 0 }, sv);
    an(reta, [{ opacity: 0 }, { opacity: 1, offset: 0.25 }, { opacity: 1, offset: 0.75 }, { opacity: 0 }], { duration: 1500, delay: 300 });
    const xr = texto(sv, 252, 22, "reta: não separa", "m-nota", { opacity: 0 });
    an(xr, [{ opacity: 0 }, { opacity: 1, offset: 0.3 }, { opacity: 1, offset: 0.75 }, { opacity: 0 }], { duration: 1500, delay: 300 });
    const pts = [];
    for (let i = 0; i < 16; i++) { const a = i / 16 * 2 * Math.PI, rr = 46 + 5 * Math.sin(3 * a + 0.6); pts.push([cx + rr * 1.15 * Math.cos(a), cy + rr * 0.92 * Math.sin(a)]); }
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length; i++) {
      const p0 = pts[(i - 1 + pts.length) % pts.length], p1 = pts[i], p2 = pts[(i + 1) % pts.length], p3 = pts[(i + 2) % pts.length];
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6], c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    const area = s("path", { d: d + "Z", fill: COR.spam, "fill-opacity": 0.1 }, sv);
    surge(area, 3100, 600);
    const curva = s("path", { d: d + "Z", fill: "none", stroke: COR.tinta, "stroke-width": 2.2 }, sv);
    traco(curva, 1900, 1300);
    real.forEach(p => s("circle", { cx: p[0], cy: p[1], r: 4.5, fill: COR.real, stroke: COR.fundo, "stroke-width": 1.5 }, sv));
    spam.forEach(p => s("circle", { cx: p[0], cy: p[1], r: 4.5, fill: COR.spam, stroke: COR.fundo, "stroke-width": 1.5 }, sv));
    const tx = texto(sv, 8, 14, "com os “temas” do SVD, a fronteira pode se curvar", "m-nota");
    surge(tx, 3200);
    esquema(sv);
  }

  /* ---------------------------------------------------------------- Naive Bayes: as chances multiplicadas */
  function nb(sv, D) {
    const { sp, re, lr: lg } = palavras(D);
    const seq = [sp[0], re[0], sp[1], sp[2]];
    // Esquema: parte do meio (o Complement não dá desvantagem ao spam por ser minoria) e usa palavras típicas.
    const p0 = 0.5;
    let lo = 0;
    const ps = seq.map(w => (lo += 0.45 * lg[w], sig(lo)));
    const T0 = 400, dt = 800;
    texto(sv, 8, 14, "cada palavra típica, sozinha, multiplica a chance", "m-nota");
    texto(sv, W - 6, 11, "esquema", "m-nota", { "text-anchor": "end" });
    seq.forEach((w, i) => {
      const y = 38 + i * 38, ehSpam = sp.includes(w);
      const g = s("g", null, sv);
      chip(g, 50, y, w, ehSpam ? COR.spam : COR.real);
      const ps_ = sig(0.6 * lg[w]);
      s("rect", { x: 92, y: y - 9, width: 40 * (1 - ps_), height: 7, rx: 2, fill: COR.real }, g);
      s("rect", { x: 92, y: y + 2, width: 40 * ps_, height: 7, rx: 2, fill: COR.spam }, g);
      if (i) texto(g, 22, y - 19, "×", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
      an(g, [{ opacity: 0, transform: "translateX(-20px)" }, { opacity: 1, transform: "translateX(0px)" }], { duration: 380, delay: T0 + dt * i });
    });
    const bx = 178, bw = 166, by = 92;
    texto(sv, bx, by - 22, "chance de spam até aqui", "m-rot forte");
    s("rect", { x: bx, y: by, width: bw, height: 20, rx: 4, fill: COR.real }, sv);
    const lar = s("rect", { x: bx, y: by, width: bw, height: 20, rx: 4, fill: COR.spam }, sv);
    lar.style.transformBox = "fill-box"; lar.style.transformOrigin = "100% 50%";
    const fim = ps[ps.length - 1], dur = T0 + dt * seq.length + 200;
    lar.style.transform = `scaleX(${fim})`;
    const q = [{ transform: `scaleX(${p0})`, offset: 0 }, { transform: `scaleX(${p0})`, offset: T0 / dur }];
    ps.forEach((p, i) => q.push({ transform: `scaleX(${p})`, offset: (T0 + dt * i + 400) / dur }));
    q.push({ transform: `scaleX(${fim})`, offset: 1 });
    an(lar, q, { duration: dur, easing: "ease-in-out" });
    texto(sv, bx, by + 36, "e-mail real", "m-rot", { fill: COR.real });
    texto(sv, bx + bw, by + 36, "spam", "m-rot", { "text-anchor": "end" });
    texto(sv, bx, by + 60, "começa no meio: o spam, minoria,", "m-nota");
    texto(sv, bx, by + 74, "não sai em desvantagem", "m-nota");
  }

  /* ---------------------------------------------------------------- MLP: neurônios acendendo */
  function mlp(sv, D) {
    const { sp, re } = palavras(D);
    const ent = [sp[0], re[0], sp[1], re[1], sp[2]], tem = [1, 0, 1, 0, 1];
    const xi = 98, xh = 210, xo = 316, yi = ent.map((_, i) => 32 + i * 34), yh = [46, 82, 118, 154], yo = 100;
    const forca = [0.95, 0.35, 0.8, 0.55];
    yi.forEach(y => yh.forEach(z => s("line", { x1: xi, y1: y, x2: xh, y2: z, stroke: COR.grade, "stroke-width": 1 }, sv)));
    yi.forEach((y, i) => { if (!tem[i]) return; yh.forEach((z, j) => { const l = s("line", { x1: xi, y1: y, x2: xh, y2: z, stroke: COR.spam, "stroke-width": 1.4, opacity: 0 }, sv); an(l, [{ opacity: 0 }, { opacity: 0.9, offset: 0.4 }, { opacity: 0 }], { duration: 900, delay: 700 + i * 80 }); }); });
    yh.forEach((z, j) => { s("line", { x1: xh, y1: z, x2: xo, y2: yo, stroke: COR.grade }, sv); const l = s("line", { x1: xh, y1: z, x2: xo, y2: yo, stroke: COR.spam, "stroke-width": 1 + 2 * forca[j], opacity: 0 }, sv); an(l, [{ opacity: 0 }, { opacity: 0.9, offset: 0.4 }, { opacity: 0 }], { duration: 900, delay: 2000 + j * 90 }); });
    ent.forEach((w, i) => {
      chip(sv, 44, yi[i], w, tem[i] ? COR.spam : COR.muda);
      const c = s("circle", { cx: xi, cy: yi[i], r: 7, fill: tem[i] ? COR.spam : COR.fundo, stroke: COR.tinta2, "stroke-width": 1.2 }, sv);
      if (tem[i]) an(c, [{ fill: COR.fundo }, { fill: COR.spam }], { duration: 300, delay: 300 + i * 90 });
    });
    yh.forEach((z, j) => {
      const c = s("circle", { cx: xh, cy: z, r: 10, fill: COR.spam, "fill-opacity": forca[j], stroke: COR.tinta2, "stroke-width": 1.2 }, sv);
      an(c, [{ fillOpacity: 0 }, { fillOpacity: forca[j] }], { duration: 500, delay: 1500 + j * 90 });
    });
    const o = s("circle", { cx: xo, cy: yo, r: 14, fill: COR.spam, stroke: COR.tinta, "stroke-width": 1.5 }, sv);
    an(o, [{ fill: COR.fundo }, { fill: COR.spam }], { duration: 500, delay: 2900 });
    const tx = texto(sv, xo, yo + 30, "spam", "m-rot forte", { "text-anchor": "middle" });
    surge(tx, 3200);
    texto(sv, xi, 12, "palavras", "m-nota", { "text-anchor": "middle" });
    texto(sv, xh, 12, "camada oculta", "m-nota", { "text-anchor": "middle" });
    texto(sv, xo, 12, "nota", "m-nota", { "text-anchor": "middle" });
    esquema(sv);
  }

  /* ---------------------------------------------------------------- árvores pequenas (Extra Trees e LightGBM) */
  function arvore(pai, x, y, esc, caminho, folhaCor, atraso) {
    const g = s("g", null, pai), dx = 13 * esc, dy = 20 * esc;
    const nos = [[x, y], [x - dx, y + dy], [x + dx, y + dy], [x - 1.6 * dx, y + 2 * dy], [x - 0.5 * dx, y + 2 * dy], [x + 0.5 * dx, y + 2 * dy], [x + 1.6 * dx, y + 2 * dy]];
    [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]].forEach(([a, b]) => s("line", { x1: nos[a][0], y1: nos[a][1], x2: nos[b][0], y2: nos[b][1], stroke: COR.eixo, "stroke-width": 1.3 }, g));
    if (caminho) {
      const [a, b] = caminho;   // índices do filho e da folha
      [[0, a], [a, b]].forEach(([p, q], k) => { const l = seg(g, nos[p][0], nos[p][1], nos[q][0], nos[q][1], { stroke: COR.tinta, "stroke-width": 2.2 }); traco(l, atraso + k * 220, 220); });
    }
    nos.forEach((n, i) => {
      const folha = i >= 3, alvo = caminho && i === caminho[1];
      const c = s("circle", { cx: n[0], cy: n[1], r: folha ? 4.2 * esc : 3.4 * esc, fill: alvo ? folhaCor : (folha ? COR.fundo : COR.tinta2), stroke: COR.tinta2, "stroke-width": 1 }, g);
      if (alvo) an(c, [{ fill: COR.fundo }, { fill: folhaCor }], { duration: 250, delay: atraso + 440 });
    });
    return { g, folha: caminho ? nos[caminho[1]] : nos[0] };
  }

  /* ---------------------------------------------------------------- Extra Trees: o júri de árvores */
  function et(sv) {
    const votos = [1, 1, 0, 1, 1, 0, 1], cam = [[1, 3], [2, 5], [1, 4], [2, 6], [1, 3], [2, 5], [2, 6]];
    const bins = { 0: { x: 92, n: 0 }, 1: { x: 228, n: 0 } };
    texto(sv, 8, 12, "cada árvore faz perguntas sorteadas e vota", "m-nota");
    votos.forEach((v, i) => {
      const x = 32 + i * 49, at = 300 + i * 380;
      const { folha } = arvore(sv, x, 28, 0.82, cam[i], v ? COR.spam : COR.real, at);
      const b = bins[v], fx = b.x + (b.n % 4) * 14, fy = 160 - Math.floor(b.n / 4) * 14; b.n++;
      const d = s("circle", { cx: fx, cy: fy, r: 5.5, fill: v ? COR.spam : COR.real }, sv);
      an(d, [{ opacity: 0, transform: `translate(${folha[0] - fx}px, ${folha[1] - fy}px)` }, { opacity: 1, transform: `translate(${folha[0] - fx}px, ${folha[1] - fy}px)`, offset: 0.2 }, { opacity: 1, transform: "translate(0px, 0px)" }], { duration: 650, delay: at + 600 });
    });
    s("line", { x1: 70, x2: 300, y1: 170, y2: 170, stroke: COR.eixo }, sv);
    texto(sv, 112, 186, "e-mail real", "m-rot", { "text-anchor": "middle" });
    texto(sv, 248, 186, "spam", "m-rot", { "text-anchor": "middle" });
    const maior = votos.filter(Boolean).length * 2 > votos.length;
    const r = s("rect", { x: maior ? 214 : 78, y: 132, width: 70, height: 60, rx: 8, fill: "none", stroke: COR.tinta, "stroke-width": 1.6 }, sv);
    surge(r, 300 + votos.length * 380 + 900);
    const tx = texto(sv, maior ? 249 : 113, 124, "a maioria decide", "m-rot forte", { "text-anchor": "middle" });
    surge(tx, 300 + votos.length * 380 + 900);
    esquema(sv);
  }

  /* ---------------------------------------------------------------- LightGBM: árvores em sequência corrigindo erros */
  function lgbm(sv) {
    const erros = [1, 0.62, 0.4, 0.26, 0.17], n = erros.length, x0 = 40, passo = 70, hmax = 64, base = 172;
    texto(sv, 8, 12, "cada árvore corrige um pouco o erro que sobra", "m-nota");
    erros.forEach((e, i) => {
      const x = x0 + i * passo, at = 300 + i * 700;
      const { g } = arvore(sv, x, 30, 0.95, null, null, at);
      pop(caixaPropria(g), at);
      if (i) { const p = texto(sv, x - passo / 2, 52, "+", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" }); surge(p, at); }
      const b = s("rect", { x: x - 12, y: base - hmax * e, width: 24, height: hmax * e, rx: 3, fill: COR.spam, "fill-opacity": 0.85 }, sv);
      b.style.transformBox = "fill-box"; b.style.transformOrigin = "50% 100%";
      an(b, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 450, delay: at + 300 });
    });
    s("line", { x1: 20, x2: W - 20, y1: base, y2: base, stroke: COR.eixo }, sv);
    texto(sv, x0 - 12, base + 16, "erro que sobra depois de cada árvore", "m-rot");
    esquema(sv, true);
  }

  /* ---------------------------------------------------------------- stacking: sete notas → presidente → limiar (e-mail real do JSON) */
  function stacking(sv, D, o) {
    const ex = D.exemplos[o.exemplo ?? 0], fk = o.filtro || "stacking7_padrao", et_ = ex.etapas[fk];
    const membros = Object.keys(et_.notas_membros);
    // pesos do filtro entregue (cor e espessura das linhas): Stacking 7 padrão ou Especialista
    const pesos = D.finais[fk === "especialista" ? "especialista" : "generalista_sem_chave"].pesos_meta;
    const y = i => 22 + i * 23, xb = 52, wb = 44;
    texto(sv, 8, 10, `um ${ex.rotulo} da caixa ${ex.origem.replace(/ \(.*\)/, "")}`, "m-nota");
    const caixa = [160, 68, 76, 64];
    membros.forEach((m, i) => {
      const g = s("g", null, sv), v = et_.notas_membros[m], c = pesos[m];
      s("circle", { cx: 10, cy: y(i) + 4, r: 4.5, fill: corMembro(m) }, g);
      texto(g, 19, y(i) + 4, (G.MEMBRO[m] || {}).sigla || m, "m-rot", { "dominant-baseline": "central" });
      s("rect", { x: xb, y: y(i), width: wb, height: 8, rx: 3, fill: COR.grade }, g);
      s("rect", { x: xb, y: y(i), width: wb * Math.max(0, Math.min(1, v)), height: 8, rx: 3, fill: COR.tinta2 }, g);
      texto(g, xb + wb + 4, y(i) + 4, F.dec(v, 2), "m-num", { "dominant-baseline": "central" });
      an(g, [{ opacity: 0, transform: "translateX(-14px)" }, { opacity: 1, transform: "translateX(0px)" }], { duration: 300, delay: 200 + i * 160 });
      const l = seg(sv, 128, y(i) + 4, caixa[0], caixa[1] + caixa[3] / 2, { stroke: c >= 0 ? COR.spam : COR.real, "stroke-width": 0.8 + Math.min(4, Math.abs(c) * 1.8), "stroke-opacity": 0.75 });
      traco(l, 1500 + i * 60, 400);
    });
    const bx = s("rect", { x: caixa[0], y: caixa[1], width: caixa[2], height: caixa[3], rx: 10, fill: COR.fundo, stroke: COR.tinta, "stroke-width": 1.8 }, sv);
    an(caixaPropria(bx), [{ transform: "scale(1)" }, { transform: "scale(1.08)" }, { transform: "scale(1)" }], { duration: 500, delay: 2100 });
    texto(sv, caixa[0] + caixa[2] / 2, caixa[1] + 26, o.presidente || "presidente", "m-rot forte", { "text-anchor": "middle" });
    texto(sv, caixa[0] + caixa[2] / 2, caixa[1] + 42, "pesa as notas", "m-nota", { "text-anchor": "middle" });
    const gx = 258, gy0 = 26, gh = 132, p = et_.prob_spam, lim = et_.limiar;
    s("line", { x1: caixa[0] + caixa[2], y1: caixa[1] + caixa[3] / 2, x2: gx - 6, y2: caixa[1] + caixa[3] / 2, stroke: COR.tinta2, "stroke-width": 1.5 }, sv);
    s("rect", { x: gx, y: gy0, width: 16, height: gh, rx: 4, fill: COR.grade }, sv);
    const enc = s("rect", { x: gx, y: gy0 + gh * (1 - p), width: 16, height: gh * p, rx: 4, fill: et_.spam ? COR.spam : COR.real }, sv);
    enc.style.transformBox = "fill-box"; enc.style.transformOrigin = "50% 100%";
    an(enc, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 800, delay: 2400 });
    const yl = gy0 + gh * (1 - lim);
    s("line", { x1: gx - 4, x2: gx + 20, y1: yl, y2: yl, stroke: COR.tinta, "stroke-width": 2 }, sv);
    texto(sv, gx + 24, yl, "limiar " + F.dec(lim, 2), "m-nota", { "dominant-baseline": "central" });
    const tp = texto(sv, gx + 24, gy0 + gh * (1 - p) - 2, "P(spam) " + F.pct(p, 0), "m-num", { "dominant-baseline": "central" });
    surge(tp, 3100);
    const dec = chip(sv, 286, 172, et_.spam ? "vai para a lixeira" : "fica na caixa", et_.spam ? COR.spam : COR.real);
    pop(caixaPropria(dec), 3400);
    // duas linhas de legenda abaixo dos membros; a 1ª é curta para não encostar no chip da decisão
    texto(sv, 8, 184, "linha = peso (laranja +, azul −)", "m-nota");
    texto(sv, 8, 197, "nota do SVM: distância à fronteira, não probabilidade", "m-nota");
  }

  /* ---------------------------------------------------------------- algoritmo genético: gerações evoluindo (histórico do JSON) */
  function ga(sv, D) {
    const h = D.ga.historico, n = h.length, pad0 = D.ga.mcc_padrao, novas = pad0 + D.ga.ganho_novas;
    const vals = [...h.flatMap(g => [g.melhor, g.media]), pad0], lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.12;
    const L = 44, R = 108, T = 52, B = 30;
    const fx = i => L + i / (n - 1) * (W - L - R), fy = v => T + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - T - B);
    [lo, hi].forEach(v => { s("line", { x1: L, x2: W - R, y1: fy(v), y2: fy(v), stroke: COR.grade }, sv); texto(sv, L - 4, fy(v), F.dec(v, 3), "m-nota", { "text-anchor": "end", "dominant-baseline": "central" }); });
    // a linha do padrão e, à direita, os dois ganhos contra ele: na busca e em divisões novas
    const xb = W - R + 62, ul0 = h[n - 1];
    s("line", { x1: L, x2: xb, y1: fy(pad0), y2: fy(pad0), stroke: COR.tinta2, "stroke-dasharray": "4 3" }, sv);
    texto(sv, L + 2, fy(pad0) + 10, "configuração padrão", "m-nota");
    const ganhos = s("g", null, sv);
    s("line", { x1: xb, x2: xb, y1: fy(pad0), y2: fy(ul0.melhor), stroke: VIOLETA, "stroke-width": 2 }, ganhos);
    texto(ganhos, xb + 4, (fy(pad0) + fy(ul0.melhor)) / 2 - 6, F.sinal(D.ga.ganho_busca, 4), "m-num", { "dominant-baseline": "central" });
    texto(ganhos, xb + 4, (fy(pad0) + fy(ul0.melhor)) / 2 + 6, "na busca", "m-nota", { "dominant-baseline": "central" });
    s("circle", { cx: xb, cy: fy(novas), r: 4, fill: COR.spam, stroke: COR.fundo, "stroke-width": 1.5 }, ganhos);
    texto(ganhos, xb - 8, fy(novas) - 10, F.sinal(D.ga.ganho_novas, 4) + " em divisões novas", "m-nota", { "text-anchor": "end", "dominant-baseline": "central" });
    surge(ganhos, 4300, 500);
    s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: COR.eixo }, sv);
    texto(sv, L, H - B + 14, "geração " + h[0].geracao, "m-nota");
    texto(sv, W - R, H - B + 14, "geração " + h[n - 1].geracao, "m-nota", { "text-anchor": "end" });
    const caminho = campo => h.map((g, i) => (i ? "L" : "M") + fx(i).toFixed(1) + "," + fy(g[campo]).toFixed(1)).join("");
    const med = s("path", { d: caminho("media"), fill: "none", stroke: VIOLETA, "stroke-opacity": 0.45, "stroke-width": 2 }, sv);
    const mel = s("path", { d: caminho("melhor"), fill: "none", stroke: VIOLETA, "stroke-width": 2.4 }, sv);
    traco(med, 600, 3600); traco(mel, 600, 3600);
    const ul = h[n - 1];
    const t1 = texto(sv, W - R + 6, fy(ul.melhor), "melhor", "m-rot forte", { "dominant-baseline": "central" });
    const t2 = texto(sv, W - R + 6, fy(ul.media) + 4, "média", "m-rot", { "dominant-baseline": "central" });
    surge(t1, 4200); surge(t2, 4200);
    ["seleção", "cruzamento", "mutação"].forEach((e, i) => {
      const c = chip(sv, 80 + i * 92, 18, e, VIOLETA);
      an(c, [{ opacity: 0.35 }, { opacity: 1 }, { opacity: 0.35 }], { duration: 900, delay: 600 + i * 300, iterations: 4, fill: "none" });
    });
    texto(sv, L, T - 6, "MCC na busca", "m-nota");
  }

  /* ---------------------------------------------------------------- TabPFN: tabela de treino + e-mail novo → uma passada → nota */
  function tabpfn(sv) {
    const linhas = [0, 1, 0, 0, 1], x0 = 18, y0 = 34, cw = 20, ch = 16;
    texto(sv, x0, 22, "treino: e-mails e rótulos", "m-nota");
    linhas.forEach((c, i) => {
      const g = s("g", null, sv);
      for (let j = 0; j < 4; j++) s("rect", { x: x0 + j * (cw + 2), y: y0 + i * (ch + 3), width: cw, height: ch, rx: 2, fill: COR.grade }, g);
      s("rect", { x: x0 + 4 * (cw + 2) + 4, y: y0 + i * (ch + 3), width: cw, height: ch, rx: 2, fill: c ? COR.spam : COR.real }, g);
      an(g, [{ opacity: 0 }, { opacity: 1 }], { duration: 250, delay: 200 + i * 120 });
    });
    const yn = y0 + linhas.length * (ch + 3) + 16;
    texto(sv, x0, yn - 5, "e-mail novo", "m-nota");
    for (let j = 0; j < 4; j++) s("rect", { x: x0 + j * (cw + 2), y: yn, width: cw, height: ch, rx: 2, fill: COR.eixo }, sv);
    const alvo = s("rect", { x: x0 + 4 * (cw + 2) + 4, y: yn, width: cw, height: ch, rx: 2, fill: COR.spam, stroke: COR.tinta, "stroke-width": 1.2 }, sv);
    an(alvo, [{ fill: COR.fundo }, { fill: COR.fundo, offset: 0.85 }, { fill: COR.spam }], { duration: 3900 });
    const q = texto(sv, x0 + 4 * (cw + 2) + 4 + cw / 2, yn + ch / 2, "?", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
    q.style.opacity = 0;
    an(q, [{ opacity: 1 }, { opacity: 1, offset: 0.9 }, { opacity: 0 }], { duration: 3700, fill: "both" });
    const bx = [176, 52, 104, 96];
    [[x0 + 5 * (cw + 2) + 8, y0 + 40], [x0 + 5 * (cw + 2) + 8, yn + ch / 2]].forEach(([x, y], i) => { const l = seg(sv, x, y, bx[0], bx[1] + bx[3] / 2, { stroke: COR.tinta2, "stroke-width": 1.5 }); traco(l, 900 + i * 300, 400); });
    const caixa = s("rect", { x: bx[0], y: bx[1], width: bx[2], height: bx[3], rx: 12, fill: COR.fundo, stroke: COR.tinta, "stroke-width": 1.8 }, sv);
    const varre = s("rect", { x: bx[0] + 4, y: bx[1] + 4, width: 18, height: bx[3] - 8, rx: 6, fill: "#1baf7a", "fill-opacity": 0.35 }, sv);
    varre.style.opacity = 0;
    an(varre, [{ opacity: 0.9, transform: "translateX(0px)" }, { opacity: 0.9, transform: `translateX(${bx[2] - 26}px)`, offset: 0.95 }, { opacity: 0, transform: `translateX(${bx[2] - 26}px)` }], { duration: 1300, delay: 1700, fill: "both" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 40, "TabPFN", "m-rot forte", { "text-anchor": "middle" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 56, "pré-treinado em", "m-nota", { "text-anchor": "middle" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 68, "tabelas sintéticas", "m-nota", { "text-anchor": "middle" });
    void caixa;
    s("line", { x1: bx[0] + bx[2], y1: bx[1] + bx[3] / 2, x2: 304, y2: bx[1] + bx[3] / 2, stroke: COR.tinta2, "stroke-width": 1.5 }, sv);
    s("rect", { x: 308, y: 50, width: 16, height: 100, rx: 4, fill: COR.grade }, sv);
    const nota = s("rect", { x: 308, y: 64, width: 16, height: 86, rx: 4, fill: COR.spam }, sv);
    nota.style.transformBox = "fill-box"; nota.style.transformOrigin = "50% 100%";
    an(nota, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 600, delay: 3000 });
    texto(sv, 316, 166, "nota", "m-nota", { "text-anchor": "middle" });
    // frase em duas linhas, alinhada à direita; o "esquema" fica no canto de cima para não disputar o rodapé
    const tx = s("g", null, sv);
    texto(tx, W - 8, H - 21, "sem treinar nesta base:", "m-rot forte", { "text-anchor": "end" });
    texto(tx, W - 8, H - 7, "o treino entra como contexto", "m-rot forte", { "text-anchor": "end" });
    surge(tx, 3200);
    esquema(sv, true);
  }

  const ANIM = { lr, lsvc, knn, svm, nb, mlp, et, lgbm, stacking, ga, tabpfn };

  /* ---------------------------------------------------------------- gráficos do conselho (dados do JSON) */
  // nomes de exibição: o 1º lugar do torneio é um stacking de 4 membros; Extra Trees não é bagging
  const NOME_TORNEIO = { "Stacking (meta LR)": "Stacking de 4 membros (LR+NB+RF+LGBM)" };
  const FAMILIA_NOME = { bagging: "árvores aleatórias (média)", "combinação": "combinação (stacking)" };
  function pesos(cont, D, o = {}) {
    const ps = D.stacking.pesos, hi = Math.max(...ps.map(p => p.peso + p.dp)), lo = Math.min(0, ...ps.map(p => p.peso - p.dp));
    G.barras(cont, {
      itens: ps.map(p => ({ rotulo: p.membro, marca: corMembro(p.membro), valor: p.peso, erro: p.dp, txt: `${F.dec(p.peso, 2)} ± ${F.dec(p.dp, 2)}`,
        cor: corMembro(p.membro), forte: p.membro === ps[0].membro,
        dica: `${G.mk(corMembro(p.membro))}<b>${G.esc(p.membro)}</b><br>peso médio ${F.dec(p.peso, 2)} ± ${F.dec(p.dp, 2)} (desvio padrão entre os folds)` })),
      dom: [lo, hi * 1.1], fmt: v => F.dec(v, 2), margemE: 132, margemD: 78, espessura: 11, animar: o.animar,
      atraso: i => (o.atraso ?? 0) + i * 90, rotulo: "peso médio de cada membro no meta-modelo",
    });
  }
  function junto(cont, D, o = {}) {
    const js = D.der.junto;
    G.barras(cont, {
      itens: js.map(j => ({ rotulo: j.membro, marca: corMembro(j.membro), valor: j.p, cor: corMembro(j.membro), forte: j === js[0],
        dica: `${G.mk(corMembro(j.membro))}<b>${G.esc(j.membro)}</b><br>quando outro membro erra, erra junto em ${F.pct(j.p, 1)} das vezes` })),
      dom: [0, 1], fmt: v => F.pct(v, 0), margemE: 132, espessura: 11, animar: o.animar, rotulo: "quanto cada membro erra junto com os outros",
    });
  }
  function salva(cont, D, o = {}) {
    const rs = D.der.resgate, mx = Math.max(...rs.map(r => r.n));
    G.barras(cont, {
      itens: rs.map(r => ({ rotulo: r.membro, marca: corMembro(r.membro), valor: r.n, cor: corMembro(r.membro), forte: r === rs[0],
        dica: `${G.mk(corMembro(r.membro))}<b>${G.esc(r.membro)}</b><br>acerta ${F.int(r.n)} e-mails em que a maioria erra` })),
      dom: [0, mx * 1.2], fmt: F.int, margemE: 132, espessura: 11, animar: o.animar, rotulo: "e-mails salvos quando a maioria erra",
    });
  }
  function s7s13(cont, D, o = {}) {
    const vs = D.stacking.variantes.filter(v => /^Stacking (7|13) \(/.test(v.nome));
    G.barras(cont, {
      itens: vs.map(v => ({ rotulo: v.nome, valor: v.mcc, cor: G.FAMILIA["combinação"], forte: v.nome.startsWith("Stacking 7") })),
      dom: [0, 1], fmt: F.mcc, margemE: 180, margemD: 48, espessura: 10, animar: o.animar, atraso: i => 300 + i * 120, rotulo: "Stacking 7 contra Stacking 13",
    });
  }
  function mapaErros(cont, leg, D, o = {}) {
    const mods = D.stacking.erros_modelos;
    if (leg) leg.textContent = "";
    G.mapaTriangular(cont, mods, D.stacking.p_erra, { completo: true, animar: o.animar, destaque: D.der.junto[0].membro, legendaEm: leg,
      rotulo: "P(a linha erra | a coluna erra)",
      fmtDica: (i, j, v) => `<b>${G.esc(mods[i])}</b> erra em ${F.pct(v, 0)} das vezes em que <b>${G.esc(mods[j])}</b> erra` });
  }

  /* ---------------------------------------------------------------- montagem: desenha e repete com clique */
  function montar(cont, tipo, D, o = {}) {
    if (!ANIM[tipo]) throw new Error("animação desconhecida: " + tipo);
    const desenhar = animar => {
      cont.textContent = "";
      const sv = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", class: "m-svg", role: "img", "aria-label": o.rotulo || tipo }, cont);
      lig.v = !!animar;
      try { ANIM[tipo](sv, D, o); } finally { lig.v = true; }
      if (!o.mini) G.h("span", { class: "repetir no-print" }, cont, "↻ clique para repetir");
    };
    desenhar(o.animar);
    if (!o.mini) cont.onclick = () => desenhar(true);
  }
  return { montar, tipos: Object.keys(ANIM), pesos, junto, salva, s7s13, mapaErros, NOME_TORNEIO, FAMILIA_NOME };
})();
