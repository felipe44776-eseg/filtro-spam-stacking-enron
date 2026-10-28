/* modelos.js — uma animação SVG curta por modelo (3 a 6 s), repetível com um clique.
   Regras: o estado final é o próprio desenho (Web Animations com fill "backwards"), então a impressão e o modo
   sem movimento saem completos. Números exibidos só vêm de DADOS; as figuras sem dados reais são esquemas e
   dizem isso no canto ("esquema"). Paleta e funções de desenho: graficos.js (G).
   Gramática comum (v2): legenda de cores no alto ("● e-mail real ● spam" e o que é cada marca), os dois lados
   nomeados no desenho, eixos com palavras reais da base, e o passo a passo numerado à direita da cena (nos
   cartões) ou embaixo (stacking, GA, TabPFN), que aparece junto com cada etapa e fica inteiro no fim.
   As mini (slide do mapa) desenham só a cena e a legenda de cores. */
"use strict";
const M = (() => {
  const { s, t, COR, F, corMembro } = G;
  // viewBox: nos cartões, a cena fica à esquerda (CENA de largura) e o passo a passo à direita (a partir de PX);
  // as mini (slide do mapa) desenham só a cena; stacking, GA e TabPFN têm caixa própria, mais alta.
  const DIM = { card: [430, 210], mini: [270, 210], stacking: [360, 266], ga: [360, 254], tabpfn: [360, 248] };
  const CENA = 265, PX = CENA + 8;
  let W = 360, H = 200, nClip = 0;
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
  // nuvem gaussiana; o ponto que cai fora de `lim` ou em área reservada (`livre` falso) é sorteado de novo
  function nuvem(r, n, cx, cy, sx, sy, lim, livre) {
    const [x0, y0, x1, y1] = lim || [12, 14, W - 12, H - 14], p = [];
    for (let tent = 0; p.length < n && tent < n * 80; tent++) {
      const u = Math.max(r(), 1e-6), v = r(), rr = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * v;
      const q = [cx + sx * rr * Math.cos(th), cy + sy * rr * Math.sin(th)];
      if (q[0] < x0 || q[0] > x1 || q[1] < y0 || q[1] > y1 || (livre && !livre(q))) continue;
      if (p.some(o => Math.hypot(o[0] - q[0], o[1] - q[1]) < 7)) continue;   // sem pontos encavalados
      p.push(q);
    }
    return p;
  }
  // caixas reservadas para texto: [x0, y0, x1, y1]; devolve o teste "o ponto está livre?"
  const longe = (caixas, folga = 6) => p => caixas.every(([a, b, c, d]) => p[0] < a - folga || p[0] > c + folga || p[1] < b - folga || p[1] > d + folga);
  const larg = (txt, fs = 10.5) => txt.length * fs * 0.5;            // largura estimada (Segoe UI, com folga)
  function recorte(sv, x, y, w, h) {                                   // grupo recortado num retângulo
    const id = "m-rec-" + (++nClip), cp = s("clipPath", { id }, s("defs", null, sv));
    s("rect", { x, y, width: w, height: h }, cp);
    return s("g", { "clip-path": `url(#${id})` }, sv);
  }
  const ponto = (pai, p, cor, r = 4.5) => s("circle", { cx: p[0], cy: p[1], r, fill: cor, stroke: COR.fundo, "stroke-width": 1.5 }, pai);
  // legenda de cores no alto da cena: "● e-mail real ● spam" e, nos cartões, o que é cada marca
  function topo(sv, o, extra, nota = "esquema", itens) {
    const g = s("g", null, sv);
    let x = 6;
    for (const [cor, nome] of itens || [[COR.real, "e-mail real"], [COR.spam, "spam"]]) {
      s("circle", { cx: x + 4, cy: 11, r: 4, fill: cor }, g);
      texto(g, x + 11, 11, nome, "m-leg", { "dominant-baseline": "central" });
      x += 11 + larg(nome) + 6;
    }
    if (extra && !o.mini) texto(g, x + 8, 11, "· " + extra, "m-nota", { "dominant-baseline": "central" });
    if (nota) texto(sv, W - 6, 11, nota, "m-nota", { "text-anchor": "end", "dominant-baseline": "central" });
  }
  // passo a passo à direita da cena (só nos cartões): número, texto em linhas, aparece no tempo `em`.
  // O último passo (a conclusão) sai em negrito; { nota: true } é um comentário sem número.
  function etapas(sv, o, passos, y0 = 30) {
    if (o.mini) return;
    let y = y0, n = 0;
    const ultimo = passos.map(p => !p.nota).lastIndexOf(true);
    passos.forEach((p, i) => {
      const g = s("g", null, sv), forte = i === ultimo;
      if (!p.nota) {
        s("circle", { cx: PX + 6.5, cy: y, r: 6.5, fill: forte ? COR.tinta : COR.tinta2 }, g);
        texto(g, PX + 6.5, y + 0.5, String(++n), "m-passo-n", { "text-anchor": "middle", "dominant-baseline": "central" });
      }
      p.l.forEach((ln, k) => texto(g, PX + 17, y + k * 12.5, ln, p.nota ? "m-nota" : "m-passo" + (forte ? " forte" : ""), { "dominant-baseline": "central" }));
      surge(g, p.em ?? 0, 400);
      y += p.l.length * 12.5 + (p.nota ? 5 : 7);
    });
    return y;
  }
  // eixos em L com o sentido escrito: P = { x0, y0, x1, y1 } (área do gráfico)
  function eixos(sv, P, rx, ry) {
    s("path", { d: `M${P.x0},${P.y0} V${P.y1} H${P.x1}`, fill: "none", stroke: COR.eixo, "stroke-width": 1.2 }, sv);
    texto(sv, P.x1, P.y1 + 11, rx, "m-eixo", { "text-anchor": "end", "dominant-baseline": "central" });
    texto(sv, P.x0 + 2, P.y0 - 8, ry, "m-eixo", { "dominant-baseline": "central" });
  }
  const rotulo = (pai, x, y, txt, at) => texto(pai, x, y, txt, "m-rot forte halo", Object.assign({ "dominant-baseline": "central" }, at));
  // o e-mail novo: aparece em `chega` com "?", e no fim (`fim`) ganha a cor da decisão
  // lado: 1 = rótulo à direita, -1 = à esquerda, ou [dx, dy] para centrar o rótulo em q + (dx, dy)
  function emailNovo(pai, q, cor, chega, fim, o, lado = 1) {
    const g = s("g", null, pai);
    const c = s("circle", { cx: q[0], cy: q[1], r: 7.5, fill: cor, stroke: COR.tinta, "stroke-width": 2 }, g);
    an(c, [{ fill: COR.fundo }, { fill: COR.fundo, offset: 0.92 }, { fill: cor }], { duration: fim });
    const qm = texto(g, q[0], q[1] + 0.5, "?", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
    qm.style.opacity = 0;
    an(qm, [{ opacity: 1 }, { opacity: 1, offset: 0.92 }, { opacity: 0 }], { duration: fim, fill: "both" });
    if (!o.mini) {
      if (Array.isArray(lado)) {
        // rótulo longe do marco: linha-guia do rótulo até o círculo
        const d = Math.hypot(lado[0], lado[1]), u = [lado[0] / d, lado[1] / d], ry = Math.abs(u[1]) > 0.3 ? 8 : 0, rx = ry ? 0 : 34;
        seg(g, q[0] + u[0] * 10, q[1] + u[1] * 10, q[0] + lado[0] - u[0] * rx, q[1] + lado[1] - Math.sign(u[1]) * ry, { stroke: COR.tinta, "stroke-width": 1, "stroke-dasharray": "2 2" });
        rotulo(g, q[0] + lado[0], q[1] + lado[1], "e-mail novo", { "text-anchor": "middle" });
      }
      else rotulo(g, q[0] + lado * 12, q[1], "e-mail novo", { "text-anchor": lado > 0 ? "start" : "end" });
    }
    surge(g, chega, 350);
    return g;
  }
  // o lugar mais vazio para um marco e seu rótulo "e-mail novo" (à direita, ou à esquerda se não couber antes
  // de xmax): a caixa do rótulo também precisa estar livre de pontos. Devolve [x, y, lado].
  function lugarLivre(cands, pontos, caixas, xmax = Infinity, rotW = 66) {
    let melhor = [...cands[0], 1], nota = -Infinity;
    for (const q of cands) {
      const d = Math.min(...pontos.map(p => Math.hypot(p[0] - q[0], p[1] - q[1])));
      const ld = q[0] + 12 + rotW <= xmax ? 1 : -1;
      const rot = ld > 0 ? [q[0] + 8, q[1] - 8, q[0] + 10 + rotW, q[1] + 8] : [q[0] - 10 - rotW, q[1] - 8, q[0] - 8, q[1] + 8];
      const noRot = pontos.some(p => p[0] > rot[0] - 5 && p[0] < rot[2] + 5 && p[1] > rot[1] - 5 && p[1] < rot[3] + 5);
      const u = [Math.min(rot[0], q[0] - 10), q[1] - 10, Math.max(rot[2], q[0] + 10), q[1] + 10];
      const bate = caixas.some(([a, b, c, e]) => !(u[2] < a || u[0] > c || u[3] < b || u[1] > e));
      const v = Math.min(d, 26) - (bate ? 100 : 0) - (noRot ? 40 : 0);
      if (v > nota) { nota = v; melhor = [q[0], q[1], ld]; }
    }
    return melhor;
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
  function palavras(D) {
    const sp = D.palavras.spam.map(p => p.palavra).filter(p => p !== "paliourg");
    const re = D.palavras.email_real.map(p => p.palavra);
    const lr = Object.fromEntries([...D.palavras.spam, ...D.palavras.email_real].map(p => [p.palavra, p.log_razao]));
    return { sp, re, lr };
  }

  /* ---------------------------------------------------------------- Regressão Logística: a balança de palavras */
  function lr(sv, D, o) {
    const { sp, re, lr: lg } = palavras(D);
    const seq = [re[0], sp[0], sp[1], sp[2]];
    const px = 132, py = 62, L = 86, T0 = 500, dt = 750;
    topo(sv, o, "cada ficha é uma palavra", o.mini ? "esquema" : "esquema · pesos ilustrativos");
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
      s("line", { x1: x, x2: x - 38, y1: py, y2: py + 52, stroke: COR.muda }, g);
      s("line", { x1: x, x2: x + 38, y1: py, y2: py + 52, stroke: COR.muda }, g);
      s("path", { d: `M${x - 44},${py + 52} Q${x},${py + 70} ${x + 44},${py + 52} Z`, fill: cor, "fill-opacity": 0.18, stroke: cor, "stroke-width": 1.5 }, g);
      texto(g, x, py + 76, rotulo, "m-rot forte", { "text-anchor": "middle" });
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
      // uma ficha por posição: duas lado a lado e a terceira em cima, sem encostar
      const x = (ehSpam ? px + L : px - L) + (n === 2 ? 0 : n % 2 ? 21 : -21), y = py + 42 - Math.floor(n / 2) * 20;
      const c = chip(alvo, x, y, w, ehSpam ? COR.spam : COR.real);
      an(c, [{ opacity: 0, transform: "translateY(-44px)" }, { opacity: 1, transform: "translateY(0px)" }], { duration: 420, delay: T0 + dt * i + dt - 450 });
    });
    // a inclinação vira a chance de spam (barra de 0% a 100%; o valor da figura é ilustrativo e não é escrito)
    const p = sig(soma * 0.4), x0 = 92, x1 = 258, yb = 174;
    texto(sv, 6, yb + 5, "chance de spam", "m-rot", { "dominant-baseline": "central" });
    s("rect", { x: x0, y: yb, width: x1 - x0, height: 10, rx: 5, fill: COR.grade }, sv);
    const b = s("rect", { x: x0, y: yb, width: (x1 - x0) * p, height: 10, rx: 5, fill: COR.spam }, sv);
    b.style.transformBox = "fill-box"; b.style.transformOrigin = "0% 50%";
    an(b, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 700, delay: dur });
    texto(sv, x0, yb + 22, F.pct(0, 0), "m-eixo", { "dominant-baseline": "central" });
    texto(sv, x1, yb + 22, F.pct(1, 0), "m-eixo", { "text-anchor": "end", "dominant-baseline": "central" });
    etapas(sv, o, [
      { l: ["dois pratos: o do e-mail", "real e o do spam"], em: 0 },
      { l: ["cada palavra do e-mail cai", "num prato, com o peso", "aprendido no treino"], em: T0 },
      { l: ["a balança pende para o", "lado mais pesado"], em: T0 + dt * 2 },
      { l: ["a inclinação vira a chance", `de spam: aqui, ${p > 0.5 ? "alta → spam" : "baixa → e-mail real"}`], em: dur },
    ]);
  }

  /* ---------------------------------------------------------------- LinearSVC: a reta com a maior folga */
  // área do gráfico das figuras de pontos (igual nos cartões e nas mini) e os eixos com palavras reais da base
  const AREA = { x0: 10, y0: 40, x1: 262, y1: 188 };
  function eixosPalavras(sv, D, o) {
    if (o.mini) return;
    const { sp, re } = palavras(D);
    eixos(sv, AREA, `mais palavras de propaganda (${sp[0]}, ${sp[1]}) →`, `↑ mais palavras de trabalho (${re[0]}, ${re[1]})`);
  }

  function lsvc(sv, D, o) {
    topo(sv, o, "cada ponto é um e-mail");
    eixosPalavras(sv, D, o);
    const A = AREA, rotR = [A.x0 + 6, A.y0 + 3, A.x0 + 118, A.y0 + 15], rotS = [A.x1 - 82, A.y1 - 17, A.x1 - 2, A.y1 - 5];
    const livre = o.mini ? null : longe([rotR, rotS]);
    const r = rng(7), c1 = [82, 90], c2 = [196, 142];
    const real = nuvem(r, 14, c1[0], c1[1], 17, 12, [A.x0 + 10, A.y0 + 8, A.x1 - 10, A.y1 - 8], livre);
    const spam = nuvem(r, 14, c2[0], c2[1], 17, 12, [A.x0 + 10, A.y0 + 8, A.x1 - 10, A.y1 - 8], livre);
    const m = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2], dx = c2[0] - c1[0], dy = c2[1] - c1[1], nn = Math.hypot(dx, dy);
    const n = [dx / nn, dy / nn], dir = [-n[1], n[0]], dist0 = p => (p[0] - m[0]) * n[0] + (p[1] - m[1]) * n[1];
    // centra a reta entre o e-mail real e o spam mais próximos (a maior folga nessa direção)
    const dr = Math.min(...real.map(p => -dist0(p))), ds = Math.min(...spam.map(p => dist0(p))), desl = (ds - dr) / 2;
    m[0] += n[0] * desl; m[1] += n[1] * desl;
    const dist = p => (p[0] - m[0]) * n[0] + (p[1] - m[1]) * n[1];
    const margem = Math.max(6, (dr + ds) / 2 - 1.5);
    const ang = Math.atan2(n[1], n[0]) * 180 / Math.PI + 90;
    const T_RETA = 2300, T_BEIRA = 3400, T_NOVO = 4100, T_FIM = 5000;
    const cena = recorte(sv, A.x0 + 1, A.y0 - 2, A.x1 - A.x0, A.y1 - A.y0 + 1);
    // os dois bairros: meio-planos de cada lado da avenida, com a cor clarinha da classe
    const lado = sinal => {
      const a = [m[0] + dir[0] * 600, m[1] + dir[1] * 600], b = [m[0] - dir[0] * 600, m[1] - dir[1] * 600];
      const f = v => v.map(z => z.toFixed(1)).join(",");
      return `M${f(a)}L${f(b)}L${f([b[0] + sinal * n[0] * 600, b[1] + sinal * n[1] * 600])}L${f([a[0] + sinal * n[0] * 600, a[1] + sinal * n[1] * 600])}Z`;
    };
    [[-1, COR.real], [1, COR.spam]].forEach(([sn, cor]) => surge(s("path", { d: lado(sn), fill: cor, "fill-opacity": 0.07 }, cena), T_RETA + 300, 700));
    // candidatas: aparecem e somem
    [-30, 26].forEach((d, i) => {
      const g = s("g", { transform: `translate(${m[0]},${m[1]}) rotate(${ang + d})` }, cena);
      const l = s("line", { x1: -230, x2: 230, y1: 0, y2: 0, stroke: COR.muda, "stroke-width": 1.5, "stroke-dasharray": "4 4", opacity: 0 }, g);
      an(l, [{ opacity: 0 }, { opacity: 0.9, offset: 0.3 }, { opacity: 0.9, offset: 0.7 }, { opacity: 0 }], { duration: 1100, delay: 300 + i * 1000 });
    });
    const g = s("g", { transform: `translate(${m[0]},${m[1]}) rotate(${ang})` }, cena);
    const faixa = s("rect", { x: -240, y: -margem, width: 480, height: 2 * margem, fill: COR.tinta, "fill-opacity": 0.07 }, g);
    caixaPropria(faixa);
    an(faixa, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 900, delay: T_RETA + 100 });
    [-margem, margem].forEach(y => { const l = s("line", { x1: -240, x2: 240, y1: y, y2: y, stroke: COR.tinta2, "stroke-dasharray": "3 3" }, g); surge(l, T_RETA + 800); });
    const reta = seg(g, -240, 0, 240, 0, { stroke: COR.tinta, "stroke-width": 2.2 });
    traco(reta, T_RETA, 700);
    if (!o.mini) {
      surge(rotulo(sv, rotR[0], (rotR[1] + rotR[3]) / 2, "bairro do e-mail real"), T_RETA + 300);
      surge(rotulo(sv, rotS[2], (rotS[1] + rotS[3]) / 2, "bairro do spam", { "text-anchor": "end" }), T_RETA + 300);
    }
    real.forEach(p => ponto(sv, p, COR.real)); spam.forEach(p => ponto(sv, p, COR.spam));
    // as casas da beira da avenida (vetores de suporte)
    [...real.map(p => [p, -dist(p)]), ...spam.map(p => [p, dist(p)])].filter(([, d]) => d < margem + 4).forEach(([p]) => {
      const c = s("circle", { cx: p[0], cy: p[1], r: 8, fill: "none", stroke: COR.tinta, "stroke-width": 1.4 }, sv);
      pop(caixaPropria(c), T_BEIRA);
    });
    // e-mail novo do lado do spam, no lugar mais vazio; o tracejado até o meio da avenida é a nota
    const cands = [];
    for (let k = margem + 16; k <= margem + 46; k += 6) for (let tt = -70; tt <= 70; tt += 7) {
      const q = [m[0] + n[0] * k + dir[0] * tt, m[1] + n[1] * k + dir[1] * tt];
      if (q[0] > A.x0 + 14 && q[0] < A.x1 - 12 && q[1] > A.y0 + 14 && q[1] < A.y1 - 14) cands.push(q);
    }
    const [qx, qy, ld] = lugarLivre(cands, [...real, ...spam], [rotR, rotS], A.x1 - 2), q = [qx, qy];
    const pe = [q[0] - n[0] * dist(q), q[1] - n[1] * dist(q)];
    const tr = seg(sv, q[0], q[1], pe[0], pe[1], { stroke: COR.tinta, "stroke-width": 1.4, "stroke-dasharray": "2 3" });
    surge(tr, T_FIM - 300);
    emailNovo(sv, q, COR.spam, T_NOVO, T_FIM, o, ld);
    etapas(sv, o, [
      { l: ["cada ponto é um e-mail de", "treino: uma casa no bairro"], em: 0 },
      { l: ["várias retas separam os", "dois bairros"], em: 300 },
      { l: ["a escolhida abre a avenida", "mais larga; as casas da", "beira (circuladas) decidem"], em: T_RETA },
      { l: ["e-mail novo: caiu do lado", "do bairro do spam → spam"], em: T_NOVO },
      { l: ["nota: a distância até o meio", "(tracejado), não uma chance"], em: T_FIM - 300, nota: true },
    ]);
  }

  /* ---------------------------------------------------------------- kNN: os vizinhos votam */
  function knn(sv, D, o) {
    topo(sv, o, "cada ponto é um e-mail");
    eixosPalavras(sv, D, o);
    const A = AREA, lim = [A.x0 + 10, A.y0 + 8, A.x1 - 10, A.y1 - 8];
    const r = rng(21), real = nuvem(r, 17, 92, 92, 30, 22, lim), spam = nuvem(r, 17, 182, 138, 30, 22, lim);
    const k = parseInt(((D.der.mod.knn || {}).cfg_padrao_d || {}).k, 10) || 5;
    const todos = [...real.map(p => [p, 0]), ...spam.map(p => [p, 1])];
    // o e-mail novo fica entre os dois grupos, no primeiro lugar em que a maioria é spam mas há voto contrário
    const vizDe = q => todos.map(([p, c]) => ({ p, c, d: Math.hypot(p[0] - q[0], p[1] - q[1]) })).sort((a, b) => a.d - b.d).slice(0, k);
    let q = [137, 115];
    for (let u = 0.42; u <= 0.7; u += 0.02) {
      const c = [92 + (182 - 92) * u, 92 + (138 - 92) * u], v = vizDe(c), ns = v.filter(x => x.c).length;
      if (ns * 2 > k && ns < k && v[0].d > 9) { q = c; break; }
    }
    const viz = vizDe(q), nSpam = viz.filter(v => v.c).length, nReal = k - nSpam, ehSpam = nSpam * 2 > k;
    const raio = viz[viz.length - 1].d + 6;
    const T_ANEL = 900, T_VIZ = 1600, fim = T_VIZ + k * 260 + 500;
    const anel = s("circle", { cx: q[0], cy: q[1], r: raio, fill: COR.tinta, "fill-opacity": 0.05, stroke: COR.tinta2, "stroke-dasharray": "3 3" }, sv);
    pop(caixaPropria(anel), T_ANEL, 700);
    viz.forEach((v, i) => { const l = seg(sv, q[0], q[1], v.p[0], v.p[1], { stroke: v.c ? COR.spam : COR.real, "stroke-width": 1.6 }); traco(l, T_VIZ + i * 260, 260); });
    todos.forEach(([p, c]) => ponto(sv, p, c ? COR.spam : COR.real));
    viz.forEach((v, i) => { const c = s("circle", { cx: v.p[0], cy: v.p[1], r: 8, fill: "none", stroke: COR.tinta, "stroke-width": 1.3 }, sv); pop(caixaPropria(c), T_VIZ + i * 260 + 200); });
    // rótulo do e-mail novo fora do anel dos vizinhos (dentro dele, as linhas de voto), onde cobre menos pontos
    let rot = [0, -raio - 10], pior = Infinity;
    for (const g_ of [-90, -120, -60, 90, 120, 60, 180, 0]) {
      const a = g_ * Math.PI / 180, dx = Math.cos(a) * (raio + 36 * Math.abs(Math.cos(a)) + 10), dy = Math.sin(a) * (raio + 10);
      const c = [q[0] + dx, q[1] + dy], cx_ = [c[0] - 33, c[1] - 8, c[0] + 33, c[1] + 8];
      if (cx_[0] < A.x0 + 2 || cx_[2] > A.x1 || cx_[1] < A.y0 - 2 || cx_[3] > A.y1) continue;
      const n_ = todos.filter(([p]) => p[0] > cx_[0] - 5 && p[0] < cx_[2] + 5 && p[1] > cx_[1] - 5 && p[1] < cx_[3] + 5).length;
      if (n_ < pior) { pior = n_; rot = [dx, dy]; }
    }
    emailNovo(sv, q, ehSpam ? COR.spam : COR.real, 500, fim, o, rot);
    const dec = ehSpam ? "spam" : "e-mail real";
    if (o.mini) {
      const tx = texto(sv, 8, H - 8, `${nSpam} spam × ${nReal} e-mail real → ${dec}`, "m-rot forte halo", { "dominant-baseline": "central" });
      surge(tx, fim);
    }
    etapas(sv, o, [
      { l: ["cada ponto é um e-mail de", "treino, já com a resposta"], em: 0 },
      { l: ["chega um e-mail novo (?)"], em: 500 },
      { l: [`ele procura os ${k} e-mails`, "mais parecidos: os que usam", "as mesmas palavras", "(medido pelo cosseno)"], em: T_ANEL },
      { l: ["votos dos vizinhos:", `${nSpam} spam × ${nReal} e-mail real`, `→ vale a maioria: ${dec}`], em: fim },
    ]);
  }

  /* ---------------------------------------------------------------- SVM RBF: a fronteira curva */
  function svm(sv, D, o) {
    topo(sv, o, "cada ponto é um e-mail");
    const A = AREA;
    if (!o.mini) eixos(sv, A, "tema A do SVD →", "↑ tema B do SVD (temas: palavras que andam juntas)");
    const r = rng(11), cx = 138, cy = 112, spam = [], real = [];
    const q = [cx + 22, cy - 22];
    const rotS = [cx - 33, cy - 1, cx + 33, cy + 11], rotR = [A.x0 + 6, A.y0 + 3, A.x0 + 112, A.y0 + 15];
    const rotQ = [q[0] + 8, q[1] - 8, q[0] + 76, q[1] + 8], rotX = [A.x1 - 100, A.y0 + 3, A.x1 - 2, A.y0 + 15];
    const livre = longe(o.mini ? [rotS] : [rotS, rotR, rotQ, rotX], 5);
    const dentro = (p, f) => Math.hypot((p[0] - cx) / (55 * f), (p[1] - cy) / (41 * f)) < 1;
    for (let t_ = 0; spam.length < 15 && t_ < 2000; t_++) {
      const a = r() * 2 * Math.PI, d = 40 * Math.sqrt(r()), p = [cx + d * 1.1 * Math.cos(a), cy + d * 0.82 * Math.sin(a)];
      if (livre(p) && Math.hypot(p[0] - q[0], p[1] - q[1]) > 13 && !spam.some(z => Math.hypot(z[0] - p[0], z[1] - p[1]) < 8)) spam.push(p);
    }
    for (let t_ = 0; real.length < 26 && t_ < 4000; t_++) {
      const a = r() * 2 * Math.PI, d = 58 + 34 * r(), p = [cx + d * 1.35 * Math.cos(a), cy + d * 0.85 * Math.sin(a)];
      if (p[0] < A.x0 + 9 || p[0] > A.x1 - 9 || p[1] < A.y0 + 7 || p[1] > A.y1 - 7 || dentro(p, 1.2) || !livre(p)) continue;
      if (!real.some(z => Math.hypot(z[0] - p[0], z[1] - p[1]) < 9)) real.push(p);
    }
    const T_RETA = 500, T_CERCA = 2100, T_LADOS = 3300, T_FIM = 4300;
    const reta = s("line", { x1: 92, y1: A.y1, x2: 196, y2: A.y0, stroke: COR.muda, "stroke-width": 1.6, "stroke-dasharray": "5 4", opacity: 0 }, sv);
    an(reta, [{ opacity: 0 }, { opacity: 1, offset: 0.25 }, { opacity: 1, offset: 0.75 }, { opacity: 0 }], { duration: 1500, delay: T_RETA });
    if (!o.mini) {
      const xr = texto(sv, rotX[2], (rotX[1] + rotX[3]) / 2, "uma reta não separa", "m-nota halo", { "text-anchor": "end", "dominant-baseline": "central", opacity: 0 });
      an(xr, [{ opacity: 0 }, { opacity: 1, offset: 0.25 }, { opacity: 1, offset: 0.75 }, { opacity: 0 }], { duration: 1500, delay: T_RETA });
    }
    const pts = [];
    for (let i = 0; i < 16; i++) { const a = i / 16 * 2 * Math.PI, rr = 46 + 4 * Math.sin(3 * a + 0.6); pts.push([cx + rr * 1.2 * Math.cos(a), cy + rr * 0.9 * Math.sin(a)]); }
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length; i++) {
      const p0 = pts[(i - 1 + pts.length) % pts.length], p1 = pts[i], p2 = pts[(i + 1) % pts.length], p3 = pts[(i + 2) % pts.length];
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6], c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    // fora da cerca: o lado do e-mail real (fundo azul clarinho na área); dentro: o lado do spam
    const fundo = s("path", { d: `M${A.x0 + 1},${A.y0 - 2}H${A.x1}V${A.y1 - 1}H${A.x0 + 1}Z` + d + "Z", fill: COR.real, "fill-opacity": 0.06, "fill-rule": "evenodd" }, sv);
    surge(fundo, T_LADOS, 600);
    const area = s("path", { d: d + "Z", fill: COR.spam, "fill-opacity": 0.1 }, sv);
    surge(area, T_LADOS, 600);
    const curva = s("path", { d: d + "Z", fill: "none", stroke: COR.tinta, "stroke-width": 2.2 }, sv);
    traco(curva, T_CERCA, 1300);
    real.forEach(p => ponto(sv, p, COR.real));
    spam.forEach(p => ponto(sv, p, COR.spam));
    if (!o.mini) {
      surge(rotulo(sv, cx, cy + 5, "lado do spam", { "text-anchor": "middle" }), T_LADOS);
      surge(rotulo(sv, rotR[0], (rotR[1] + rotR[3]) / 2, "lado do e-mail real"), T_LADOS);
    }
    emailNovo(sv, q, COR.spam, T_LADOS + 200, T_FIM, o);
    etapas(sv, o, [
      { l: ["antes, as palavras viram", "“temas” (SVD): grupos de", "palavras que andam juntas"], em: 0 },
      { l: ["aqui o spam forma uma ilha:", "nenhuma reta separa"], em: T_RETA },
      { l: ["o kernel RBF desenha uma", "cerca curva em volta dela"], em: T_CERCA },
      { l: ["dentro: lado do spam; fora:", "lado do e-mail real"], em: T_LADOS },
      { l: ["e-mail novo dentro da", "cerca → spam"], em: T_FIM },
    ]);
  }

  /* ---------------------------------------------------------------- Naive Bayes: as chances multiplicadas */
  // Uma linha por pista: o placar depois de cada palavra. O estado final mostra a história inteira.
  function nb(sv, D, o) {
    const { sp, re, lr: lg } = palavras(D);
    const seq = [sp[0], re[0], sp[1], sp[2]];
    // Esquema: parte do meio (o Complement não dá desvantagem ao spam por ser minoria) e usa palavras típicas.
    let lo = 0;
    const ps = [0.5, ...seq.map(w => (lo += 0.45 * lg[w], sig(lo)))];
    const T0 = 500, dt = 800, dur = T0 + dt * seq.length;
    topo(sv, o, "cada ficha é uma palavra-pista", o.mini ? "esquema" : "esquema · pesos ilustrativos");
    const xa = 106, xb = 258, xm = (xa + xb) / 2, X = p => xa + (xb - xa) * p, yl = i => 58 + i * 30, y0 = 44, y1 = yl(seq.length) + 12;
    // os dois lados do placar
    s("rect", { x: xa, y: y0, width: xm - xa, height: y1 - y0, rx: 6, fill: COR.real, "fill-opacity": 0.07 }, sv);
    s("rect", { x: xm, y: y0, width: xb - xm, height: y1 - y0, rx: 6, fill: COR.spam, "fill-opacity": 0.07 }, sv);
    s("line", { x1: xm, x2: xm, y1: y0, y2: y1, stroke: COR.eixo, "stroke-dasharray": "3 3" }, sv);
    [["lado do", (xa + xm) / 2], ["lado do", (xm + xb) / 2]].forEach(([a, x], k) => {
      texto(sv, x, 25, a, "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
      texto(sv, x, 37.5, k ? "spam" : "e-mail real", "m-rot forte", { "text-anchor": "middle", "dominant-baseline": "central" });
    });
    texto(sv, 44, 37.5, "pista", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    const cor = p => (p > 0.5 ? COR.spam : p < 0.5 ? COR.real : COR.fundo);   // no meio: neutro
    ps.forEach((p, i) => {
      const y = yl(i), g = s("g", null, sv), em = i ? T0 + dt * (i - 1) : 0;
      s("line", { x1: xa + 4, x2: xb - 4, y1: y, y2: y, stroke: COR.grade, "stroke-width": 2 }, g);
      if (i) {
        const w = seq[i - 1], ehSpam = sp.includes(w), xp = X(ps[i - 1]), xn = X(p);
        chip(g, 44, y, w, ehSpam ? COR.spam : COR.real);
        // de onde o placar estava (anel vazio) até onde a pista o empurrou (seta)
        s("circle", { cx: xp, cy: y, r: 5, fill: "none", stroke: COR.muda, "stroke-width": 1.2 }, g);
        const sx = Math.sign(xn - xp), seta = seg(g, xp + sx * 6, y, xn - sx * 8, y, { stroke: COR.tinta2, "stroke-width": 1.4 });
        traco(seta, em + 350, 350);
        s("path", { d: `M${xn - sx * 8},${y - 3.5}L${xn - sx * 3},${y}L${xn - sx * 8},${y + 3.5}`, fill: "none", stroke: COR.tinta2, "stroke-width": 1.4 }, g);
      } else texto(g, 44, y, "começo", "m-nota", { "text-anchor": "middle", "dominant-baseline": "central" });
      const mk = s("circle", { cx: X(p), cy: y, r: 6, fill: cor(p), stroke: COR.tinta, "stroke-width": 1.2 }, g);
      if (i) an(mk, [{ transform: `translateX(${X(ps[i - 1]) - X(p)}px)`, fill: cor(ps[i - 1]) }, { transform: `translateX(${X(ps[i - 1]) - X(p)}px)`, fill: cor(ps[i - 1]), offset: 0.4 }, { transform: "translateX(0px)", fill: cor(p) }], { duration: 800, delay: em });
      an(g, [{ opacity: 0, transform: "translateY(-6px)" }, { opacity: 1, transform: "translateY(0px)" }], { duration: 350, delay: em });
    });
    const fim = ps[ps.length - 1], dec = fim > 0.5 ? "spam" : "e-mail real";
    if (o.mini) surge(texto(sv, xm, H - 8, `fim: lado do ${dec} → ${dec}`, "m-rot forte", { "text-anchor": "middle", "dominant-baseline": "central" }), dur);
    etapas(sv, o, [
      { l: ["o placar começa no meio,", "entre os dois lados"], em: 0 },
      { l: ["cada palavra-pista empurra", "o placar para um lado"], em: T0 },
      { l: ["cada pista conta sozinha,", "como se não tivesse nada", "a ver com as outras: por", "isso, “ingênuo”"], em: T0 + dt },
      { l: ["no fim, o placar ficou", `do lado do ${dec} → ${dec}`], em: dur },
    ]);
  }

  /* ---------------------------------------------------------------- MLP: neurônios acendendo */
  // três colunas: palavras do e-mail → detetives (camada oculta) → chefe (nota de spam)
  function mlp(sv, D, o) {
    const { sp, re } = palavras(D);
    const ent = [sp[0], re[0], sp[1], re[1], sp[2]], tem = [1, 0, 1, 0, 1], combo = [0, 2];   // o detetive 1 reconhece sp[0] + sp[1]
    const xi = 76, xh = 148, xo = 228, yi = ent.map((_, i) => 58 + i * 28), yh = [66, 98, 130, 162], yo = 114;
    const forca = [0.95, 0.35, 0.8, 0.55];
    const T_ENT = 300, T_DET = 1300, T_CHEFE = 2100, T_NOTA = 2900;
    topo(sv, o, "ficha cinza: palavra ausente");
    [[44, "palavras", "do e-mail"], [xh, "detetives", "(camada oculta)"], [xo, "chefe:", "nota de spam"]].forEach(([x, a, b]) => {
      texto(sv, x, 26, a, "m-rot forte", { "text-anchor": "middle", "dominant-baseline": "central" });
      texto(sv, x, 39, b, "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    });
    yi.forEach(y => yh.forEach(z => s("line", { x1: xi, y1: y, x2: xh, y2: z, stroke: COR.grade, "stroke-width": 1 }, sv)));
    // a combinação que o detetive 1 aprendeu: as duas palavras de propaganda juntas (fica desenhada no fim)
    combo.forEach(i => { const l = seg(sv, xi, yi[i], xh, yh[0], { stroke: COR.spam, "stroke-width": 2.6, "stroke-opacity": 0.85 }); traco(l, T_DET, 450); });
    yi.forEach((y, i) => { if (!tem[i]) return; yh.forEach(z => { const l = s("line", { x1: xi, y1: y, x2: xh, y2: z, stroke: COR.spam, "stroke-width": 1.4, opacity: 0 }, sv); an(l, [{ opacity: 0 }, { opacity: 0.9, offset: 0.4 }, { opacity: 0 }], { duration: 900, delay: T_ENT + 400 + i * 80 }); }); });
    // do detetive ao chefe: espessura = quanto o chefe ouve aquele detetive
    yh.forEach((z, j) => {
      const l = s("line", { x1: xh, y1: z, x2: xo, y2: yo, stroke: COR.tinta2, "stroke-width": 0.8 + 3 * forca[j], "stroke-opacity": 0.45 }, sv);
      traco(l, T_CHEFE + j * 90, 400);
    });
    ent.forEach((w, i) => {
      chip(sv, 34, yi[i], w, tem[i] ? (sp.includes(w) ? COR.spam : COR.real) : COR.muda);
      const c = s("circle", { cx: xi, cy: yi[i], r: 6.5, fill: tem[i] ? COR.spam : COR.fundo, stroke: COR.tinta2, "stroke-width": 1.2 }, sv);
      if (tem[i]) an(c, [{ fill: COR.fundo }, { fill: COR.spam }], { duration: 300, delay: T_ENT + i * 90 });
    });
    yh.forEach((z, j) => {
      const c = s("circle", { cx: xh, cy: z, r: 10, fill: COR.spam, "fill-opacity": forca[j], stroke: j === 0 ? COR.tinta : COR.tinta2, "stroke-width": j === 0 ? 2 : 1.2 }, sv);
      an(c, [{ fillOpacity: 0 }, { fillOpacity: forca[j] }], { duration: 500, delay: T_DET + 200 + j * 90 });
    });
    const sai = s("circle", { cx: xo, cy: yo, r: 14, fill: COR.spam, stroke: COR.tinta, "stroke-width": 1.5 }, sv);
    an(sai, [{ fill: COR.fundo }, { fill: COR.spam }], { duration: 500, delay: T_NOTA });
    surge(texto(sv, xo, yo + 28, "spam", "m-rot forte", { "text-anchor": "middle", "dominant-baseline": "central" }), T_NOTA + 300);
    etapas(sv, o, [
      { l: ["entram as palavras do", "e-mail (cinza: ausente)"], em: 0 },
      { l: ["cada detetive reconhece", "uma combinação de palavras", `(ex.: ${ent[combo[0]]} e ${ent[combo[1]]} juntas)`], em: T_DET },
      { l: ["o chefe junta as opiniões;", "linha grossa = opinião", "que pesa mais"], em: T_CHEFE },
      { l: ["nota de spam alta → spam"], em: T_NOTA },
    ]);
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
  // À esquerda, um jurado por inteiro (as perguntas escritas e o caminho do e-mail); à direita, os outros
  // jurados em miniatura e as duas urnas. Votos de esquema; a contagem é calculada dos votos desenhados.
  function et(sv, D, o) {
    const { sp, re } = palavras(D);
    const votos = [1, 1, 0, 1, 1, 0, 1], cam = [null, [2, 5], [1, 4], [2, 6], [1, 3], [2, 5], [2, 6]];
    const T_CAM = 600, T_JURI = 2100, passo = 300, T_FIM = T_JURI + (votos.length - 1) * passo + 900;
    topo(sv, o, "cada folha é um voto");
    // o jurado por inteiro: tem sp[0]? sim → tem re[0]? não → folha spam
    const pergunta = (x, y, w) => {
      const tx = `tem “${w}”?`, lw = larg(tx, 9.5) + 8, g = s("g", null, sv);
      s("rect", { x: x - lw / 2, y: y - 8, width: lw, height: 16, rx: 8, fill: COR.papel || "#fff", stroke: COR.tinta2, "stroke-width": 1.1 }, g);
      texto(g, x, y + 0.5, tx, "m-no", { "text-anchor": "middle", "dominant-baseline": "central" });
    };
    const N = { raiz: [80, 48], e: [42, 94], d: [118, 94] }, F_ = [[22, 140, 0], [62, 140, 1], [98, 140, 1], [138, 140, 0]];
    texto(sv, N.raiz[0], 30, "um jurado", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    const aresta = (a, b, forte) => seg(sv, a[0], a[1] + 8, b[0], b[1] - (b[1] > 120 ? 6 : 8), { stroke: forte ? COR.tinta : COR.eixo, "stroke-width": forte ? 2.4 : 1.3 });
    aresta(N.raiz, N.e); aresta(N.raiz, N.d);
    [[N.e, F_[0]], [N.e, F_[1]], [N.d, F_[2]], [N.d, F_[3]]].forEach(([a, b]) => aresta(a, b));
    traco(aresta(N.raiz, N.e, true), T_CAM, 300); traco(aresta(N.e, F_[1], true), T_CAM + 300, 300);
    [["sim", 57, 70, "end"], ["não", 103, 70, "start"], ["sim", 29, 118, "end"], ["não", 55, 118, "start"], ["sim", 105, 118, "end"], ["não", 131, 118, "start"]]
      .forEach(([tx, x, y, a]) => texto(sv, x, y, tx, "m-eixo", { "text-anchor": a, "dominant-baseline": "central" }));
    pergunta(...N.raiz, sp[0]); pergunta(...N.e, re[0]); pergunta(...N.d, sp[1]);
    F_.forEach(([x, y, c], i) => {
      const f = s("rect", { x: x - 7, y: y - 6, width: 14, height: 12, rx: 3, fill: c ? COR.spam : COR.real, stroke: i === 1 ? COR.tinta : "none", "stroke-width": 2 }, sv);
      if (i === 1) an(f, [{ transform: "scale(1)" }, { transform: "scale(1.35)" }, { transform: "scale(1)" }], { duration: 400, delay: T_CAM + 600 });
      if (i === 1) caixaPropria(f);
    });
    surge(texto(sv, F_[1][0], 158, "voto: spam", "m-rot forte halo", { "text-anchor": "middle", "dominant-baseline": "central" }), T_CAM + 600);
    // os outros jurados e as urnas
    texto(sv, 213, 30, "outros jurados", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    const urna = { 0: { x: 188, n: 0 }, 1: { x: 240, n: 0 } }, base = 178;
    votos.forEach((v, i) => {
      let fl, at;
      if (i === 0) { fl = [F_[1][0], F_[1][1]]; at = T_CAM + 900; }
      else {
        const k = i - 1, x = 182 + (k % 3) * 31, y = 44 + Math.floor(k / 3) * 40;
        at = T_JURI + k * passo;
        const a = arvore(sv, x, y, 0.62, cam[i], v ? COR.spam : COR.real, at);
        pop(caixaPropria(a.g), at); fl = a.folha; at += 500;
      }
      const b = urna[v], fx = b.x + ((b.n % 3) - 1) * 11, fy = base - 7 - Math.floor(b.n / 3) * 11; b.n++;
      const d = s("circle", { cx: fx, cy: fy, r: 4.8, fill: v ? COR.spam : COR.real }, sv);
      an(d, [{ opacity: 0, transform: `translate(${fl[0] - fx}px, ${fl[1] - fy}px)` }, { opacity: 1, transform: `translate(${fl[0] - fx}px, ${fl[1] - fy}px)`, offset: 0.2 }, { opacity: 1, transform: "translate(0px, 0px)" }], { duration: 650, delay: at });
    });
    s("line", { x1: 163, x2: 262, y1: base, y2: base, stroke: COR.eixo }, sv);
    texto(sv, urna[0].x, base + 11, "e-mail real", "m-rot", { "text-anchor": "middle", "dominant-baseline": "central" });
    texto(sv, urna[1].x, base + 11, "spam", "m-rot", { "text-anchor": "middle", "dominant-baseline": "central" });
    const nS = votos.filter(Boolean).length, nR = votos.length - nS, maior = nS * 2 > votos.length, dec = maior ? "spam" : "e-mail real";
    const ub = urna[maior ? 1 : 0].x;
    surge(s("rect", { x: ub - 22, y: base - 36, width: 44, height: 56, rx: 7, fill: "none", stroke: COR.tinta, "stroke-width": 1.6 }, sv), T_FIM);
    surge(texto(sv, ub, base - 44, "maioria", "m-rot forte halo", { "text-anchor": "middle", "dominant-baseline": "central" }), T_FIM);
    etapas(sv, o, [
      { l: ["cada jurado é uma árvore", "de perguntas: o e-mail", "tem esta palavra?"], em: 0 },
      { l: ["o e-mail desce pelas", "respostas, sim ou não, até", "uma folha: o voto"], em: T_CAM },
      { l: ["os cortes foram sorteados:", "cada jurado pensa diferente"], em: T_JURI },
      { l: ["votos do júri:", `${nS} spam × ${nR} e-mail real`, `→ vale a maioria: ${dec}`], em: T_FIM },
    ]);
  }

  /* ---------------------------------------------------------------- LightGBM: árvores em sequência corrigindo erros */
  // árvores em fila (palpite, correção 1, 2, 3 …) e, embaixo de cada uma, o erro que ainda sobra
  function lgbm(sv, D, o) {
    const erros = [1, 0.58, 0.34, 0.2], nomes = ["palpite", "correção 1", "correção 2", "correção 3"];
    const x0 = 32, passo = 62, hmax = 66, base = 170, dt = 700, T0 = 300;
    topo(sv, o, null, "esquema", []);
    const topos = [];
    erros.forEach((e, i) => {
      const x = x0 + i * passo, at = T0 + i * dt;
      const { g } = arvore(sv, x, 30, 0.85, null, null, at);
      pop(caixaPropria(g), at);
      if (i) surge(texto(sv, x - passo / 2, 47, "+", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" }), at);
      surge(texto(sv, x, 80, nomes[i], i ? "m-leg" : "m-rot forte", { "text-anchor": "middle", "dominant-baseline": "central" }), at);
      const b = s("rect", { x: x - 13, y: base - hmax * e, width: 26, height: hmax * e, rx: 3, fill: VIOLETA, "fill-opacity": 0.7 }, sv);
      b.style.transformBox = "fill-box"; b.style.transformOrigin = "50% 100%";
      an(b, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 450, delay: at + 300 });
      topos.push([x, base - hmax * e]);
    });
    const xr = x0 + erros.length * passo - 14;
    surge(texto(sv, xr, 47, "…", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" }), T0 + erros.length * dt);
    const curva = s("path", { d: topos.map((p, i) => (i ? "L" : "M") + p[0] + "," + (p[1] - 4)).join(""), fill: "none", stroke: COR.tinta2, "stroke-width": 1.2, "stroke-dasharray": "3 3" }, sv);
    traco(curva, T0 + 750, dt * (erros.length - 1));      // chega a cada barra quando ela termina de subir
    s("line", { x1: 10, x2: CENA - 6, y1: base, y2: base, stroke: COR.eixo }, sv);
    texto(sv, (10 + CENA - 6) / 2, base + 14, "erro que sobra depois de cada rodada", "m-leg", { "text-anchor": "middle", "dominant-baseline": "central" });
    const T_FIM = T0 + erros.length * dt + 300;
    etapas(sv, o, [
      { l: ["a primeira árvore dá", "um palpite"], em: T0 },
      { l: ["a próxima olha onde ela", "errou e corrige um pouco"], em: T0 + dt },
      { l: ["cada correção mira o erro", "que ainda sobra: a barra", "diminui a cada rodada"], em: T0 + 2 * dt },
      { l: ["a soma do palpite com as", "correções vira a nota"], em: T_FIM },
    ]);
  }

  /* ---------------------------------------------------------------- stacking: sete notas → presidente → limiar (e-mail real do JSON) */
  function stacking(sv, D, o) {
    const ex = D.exemplos[o.exemplo ?? 0], fk = o.filtro || "stacking7_padrao", et_ = ex.etapas[fk];
    const membros = Object.keys(et_.notas_membros);
    // pesos do filtro entregue (cor e espessura das linhas): Stacking 7 padrão ou Especialista
    const pesos = D.finais[fk === "especialista" ? "especialista" : "generalista_sem_chave"].pesos_meta;
    const y = i => 54 + i * 24, xb = 52, wb = 44, T_PRES = 1500, T_LIM = 2400, T_DEC = 3400;
    texto(sv, 8, 10, `um ${ex.rotulo} da caixa ${ex.origem.replace(/ \(.*\)/, "")}, que o filtro nunca viu`, "m-nota", { "dominant-baseline": "central" });
    // três etapas, numeradas no alto de cada coluna e sincronizadas com a animação
    const selo = (x, yy, n, a, b, em, anc = "start") => {
      const g = s("g", null, sv), xt = anc === "start" ? x + 16 : x;
      s("circle", { cx: anc === "start" ? x + 6.5 : x - larg(a, 11.5) / 2 - 10, cy: yy, r: 6.5, fill: COR.tinta }, g);
      texto(g, anc === "start" ? x + 6.5 : x - larg(a, 11.5) / 2 - 10, yy + 0.5, String(n), "m-passo-n", { "text-anchor": "middle", "dominant-baseline": "central" });
      texto(g, xt, yy, a, "m-rot forte", { "text-anchor": anc, "dominant-baseline": "central" });
      texto(g, xt, yy + 13, b, "m-eixo", { "text-anchor": anc, "dominant-baseline": "central" });
      surge(g, em, 400);
    };
    const caixa = [160, 98, 76, 64];
    selo(4, 28, 1, "nota de cada membro", "0 = e-mail real · 1 = spam", 0);
    const pres = o.presidente || "presidente";
    selo(caixa[0] + caixa[2] / 2 + 4, 66, 2, "o " + pres, "pesa as notas", T_PRES, "middle");
    selo(248, 28, 3, "chance de spam", "contra o limiar", T_LIM);
    membros.forEach((m, i) => {
      const g = s("g", null, sv), v = et_.notas_membros[m], c = pesos[m];
      s("circle", { cx: 10, cy: y(i) + 4, r: 4.5, fill: corMembro(m) }, g);
      texto(g, 19, y(i) + 4, (G.MEMBRO[m] || {}).sigla || m, "m-rot", { "dominant-baseline": "central" });
      s("rect", { x: xb, y: y(i), width: wb, height: 8, rx: 3, fill: COR.grade }, g);
      s("rect", { x: xb, y: y(i), width: wb * Math.max(0, Math.min(1, v)), height: 8, rx: 3, fill: COR.tinta2 }, g);
      texto(g, xb + wb + 4, y(i) + 4, F.dec(v, 2), "m-num", { "dominant-baseline": "central" });
      an(g, [{ opacity: 0, transform: "translateX(-14px)" }, { opacity: 1, transform: "translateX(0px)" }], { duration: 300, delay: 200 + i * 160 });
      const l = seg(sv, 128, y(i) + 4, caixa[0], caixa[1] + caixa[3] / 2, { stroke: c >= 0 ? COR.spam : COR.real, "stroke-width": 0.8 + Math.min(4, Math.abs(c) * 1.8), "stroke-opacity": 0.75 });
      traco(l, T_PRES + i * 60, 400);
    });
    const bx = s("rect", { x: caixa[0], y: caixa[1], width: caixa[2], height: caixa[3], rx: 10, fill: COR.fundo, stroke: COR.tinta, "stroke-width": 1.8 }, sv);
    an(caixaPropria(bx), [{ transform: "scale(1)" }, { transform: "scale(1.08)" }, { transform: "scale(1)" }], { duration: 500, delay: T_PRES + 600 });
    texto(sv, caixa[0] + caixa[2] / 2, caixa[1] + 26, pres, "m-rot forte", { "text-anchor": "middle" });
    texto(sv, caixa[0] + caixa[2] / 2, caixa[1] + 42, "(Reg. Logística)", "m-nota", { "text-anchor": "middle" });
    const gx = 258, gy0 = 54, gh = 154, p = et_.prob_spam, lim = et_.limiar;
    s("line", { x1: caixa[0] + caixa[2], y1: caixa[1] + caixa[3] / 2, x2: gx - 6, y2: caixa[1] + caixa[3] / 2, stroke: COR.tinta2, "stroke-width": 1.5 }, sv);
    s("rect", { x: gx, y: gy0, width: 16, height: gh, rx: 4, fill: COR.grade }, sv);
    const enc = s("rect", { x: gx, y: gy0 + gh * (1 - p), width: 16, height: gh * p, rx: 4, fill: et_.spam ? COR.spam : COR.real }, sv);
    enc.style.transformBox = "fill-box"; enc.style.transformOrigin = "50% 100%";
    an(enc, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 800, delay: T_LIM });
    const yl = gy0 + gh * (1 - lim), yp = gy0 + gh * (1 - p);
    s("line", { x1: gx - 4, x2: gx + 20, y1: yl, y2: yl, stroke: COR.tinta, "stroke-width": 2 }, sv);
    // rótulos à direita da barra: o do P(spam) acima do limiar, sem encostar
    const ytp = Math.min(yp, yl - 14);
    texto(sv, gx + 24, yl, "limiar " + F.dec(lim, 2), "m-nota", { "dominant-baseline": "central" });
    texto(sv, gx + 24, yl + 12, "acima: lixeira", "m-eixo", { "dominant-baseline": "central" });
    const tp = texto(sv, gx + 24, ytp, "P(spam) " + F.pct(p, 0), "m-num", { "dominant-baseline": "central" });
    surge(tp, T_LIM + 700);
    const dec = chip(sv, 290, 228, et_.spam ? "vai para a lixeira" : "fica na caixa", et_.spam ? COR.spam : COR.real);
    pop(caixaPropria(dec), T_DEC);
    // o que são as linhas (cor e espessura) e a ressalva da nota do SVM; à esquerda do chip da decisão
    ["cor da linha: laranja puxa para spam;", "azul puxa para e-mail real",
     `espessura: quanto o ${pres} confia`, "nota do SVM: distância, não probabilidade"].forEach((tx, i) =>
      texto(sv, 8, 218 + i * 12.5, tx, "m-nota", { "dominant-baseline": "central" }));
  }

  /* ---------------------------------------------------------------- algoritmo genético: gerações evoluindo (histórico do JSON) */
  function ga(sv, D) {
    const h = D.ga.historico, n = h.length, pad0 = D.ga.mcc_padrao, novas = pad0 + D.ga.ganho_novas;
    const vals = [...h.flatMap(g => [g.melhor, g.media]), pad0], lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.12;
    const L = 44, R = 108, T = 52, B = 92;      // embaixo do gráfico, o passo a passo da analogia
    // folga maior embaixo: o rótulo "configuração padrão" cabe entre a linha tracejada e o eixo
    const padB = (hi - lo) * 0.3;
    const fx = i => L + i / (n - 1) * (W - L - R), fy = v => T + (1 - (v - lo + padB) / (hi - lo + pad + padB)) * (H - T - B);
    [lo, hi].forEach(v => { s("line", { x1: L, x2: W - R, y1: fy(v), y2: fy(v), stroke: COR.grade }, sv); texto(sv, L - 4, fy(v), F.dec(v, 3), "m-nota", { "text-anchor": "end", "dominant-baseline": "central" }); });
    // a linha do padrão e, à direita, os dois ganhos contra ele: na busca e em divisões novas
    const xb = W - R + 62, ul0 = h[n - 1];
    s("line", { x1: L, x2: xb, y1: fy(pad0), y2: fy(pad0), stroke: COR.tinta2, "stroke-dasharray": "4 3" }, sv);
    texto(sv, L + 2, fy(pad0) + 10, "configuração padrão", "m-nota");
    const ganhos = s("g", null, sv);
    s("line", { x1: xb, x2: xb, y1: fy(pad0), y2: fy(ul0.melhor), stroke: VIOLETA, "stroke-width": 2 }, ganhos);
    texto(ganhos, xb + 4, (fy(pad0) + fy(ul0.melhor)) / 2 - 7, F.sinal(D.ga.ganho_busca, 4), "m-num", { "dominant-baseline": "central" });
    texto(ganhos, xb + 4, (fy(pad0) + fy(ul0.melhor)) / 2 + 7, "na busca", "m-nota", { "dominant-baseline": "central" });
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
    texto(sv, L, T - 6, "MCC na busca (a “velocidade” do cavalo)", "m-nota");
    etapasLinha(sv, H - B + 36, [
      { l: "cada cavalo é uma combinação de ajustes; a velocidade é o MCC", em: 0 },
      { l: "os mais rápidos cruzam; os filhos sofrem pequenas mutações", em: 600 },
      { l: "geração após geração, o melhor (linha cheia) e a média sobem", em: 2400 },
    ]);
  }
  // passo a passo em linhas inteiras, embaixo da figura (stacking, GA, TabPFN)
  function etapasLinha(sv, y0, passos) {
    passos.forEach((p, i) => {
      const g = s("g", null, sv), y = y0 + i * 16, forte = i === passos.length - 1;
      s("circle", { cx: 14.5, cy: y, r: 6.5, fill: forte ? COR.tinta : COR.tinta2 }, g);
      texto(g, 14.5, y + 0.5, String(i + 1), "m-passo-n", { "text-anchor": "middle", "dominant-baseline": "central" });
      texto(g, 26, y, p.l, "m-passo" + (forte ? " forte" : ""), { "dominant-baseline": "central" });
      surge(g, p.em ?? 0, 400);
    });
  }

  /* ---------------------------------------------------------------- TabPFN: tabela de treino + e-mail novo → uma passada → nota */
  function tabpfn(sv, D, o) {
    const linhas = [0, 1, 0, 0, 1], x0 = 18, y0 = 48, cw = 20, ch = 16;
    topo(sv, o, "cada linha da tabela é um e-mail");
    texto(sv, x0, 26, "e-mails de treino, já respondidos", "m-rot forte", { "dominant-baseline": "central" });
    texto(sv, x0 + 2 * (cw + 2) - 1, 39, "palavras", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    texto(sv, x0 + 4 * (cw + 2) + 4 + cw / 2, 39, "resposta", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    linhas.forEach((c, i) => {
      const g = s("g", null, sv);
      for (let j = 0; j < 4; j++) s("rect", { x: x0 + j * (cw + 2), y: y0 + i * (ch + 3), width: cw, height: ch, rx: 2, fill: COR.grade }, g);
      s("rect", { x: x0 + 4 * (cw + 2) + 4, y: y0 + i * (ch + 3), width: cw, height: ch, rx: 2, fill: c ? COR.spam : COR.real }, g);
      an(g, [{ opacity: 0 }, { opacity: 1 }], { duration: 250, delay: 200 + i * 120 });
    });
    const yn = y0 + linhas.length * (ch + 3) + 18;
    texto(sv, x0, yn - 7, "e-mail novo", "m-rot forte", { "dominant-baseline": "central" });
    for (let j = 0; j < 4; j++) s("rect", { x: x0 + j * (cw + 2), y: yn, width: cw, height: ch, rx: 2, fill: COR.eixo }, sv);
    const alvo = s("rect", { x: x0 + 4 * (cw + 2) + 4, y: yn, width: cw, height: ch, rx: 2, fill: COR.spam, stroke: COR.tinta, "stroke-width": 1.2 }, sv);
    an(alvo, [{ fill: COR.fundo }, { fill: COR.fundo, offset: 0.85 }, { fill: COR.spam }], { duration: 3900 });
    const q = texto(sv, x0 + 4 * (cw + 2) + 4 + cw / 2, yn + ch / 2, "?", "m-chip", { "text-anchor": "middle", "dominant-baseline": "central" });
    q.style.opacity = 0;
    an(q, [{ opacity: 1 }, { opacity: 1, offset: 0.9 }, { opacity: 0 }], { duration: 3700, fill: "both" });
    const bx = [166, 70, 120, 96];
    [[x0 + 5 * (cw + 2) + 8, y0 + 40], [x0 + 5 * (cw + 2) + 8, yn + ch / 2]].forEach(([x, y], i) => { const l = seg(sv, x, y, bx[0], bx[1] + bx[3] / 2, { stroke: COR.tinta2, "stroke-width": 1.5 }); traco(l, 900 + i * 300, 400); });
    // antes de tudo: a pilha de "provas de mentira" (tabelas inventadas) que ele já estudou
    const pilha = s("g", null, sv);
    for (let k = 3; k >= 0; k--) s("rect", { x: bx[0] + 40 + k * 5, y: 32 - k * 4, width: 34, height: 22, rx: 3, fill: COR.fundo, stroke: COR.muda, "stroke-width": 1 }, pilha);
    texto(pilha, bx[0] + 100, 30, "milhões de tabelas", "m-eixo", { "dominant-baseline": "central" });
    texto(pilha, bx[0] + 100, 42, "inventadas", "m-eixo", { "dominant-baseline": "central" });
    s("path", { d: `M${bx[0] + 57},${56} V${bx[1] - 2}`, stroke: COR.muda, "stroke-dasharray": "2 3", fill: "none" }, pilha);
    const caixa = s("rect", { x: bx[0], y: bx[1], width: bx[2], height: bx[3], rx: 12, fill: COR.fundo, stroke: COR.tinta, "stroke-width": 1.8 }, sv);
    const varre = s("rect", { x: bx[0] + 4, y: bx[1] + 4, width: 18, height: bx[3] - 8, rx: 6, fill: "#1baf7a", "fill-opacity": 0.35 }, sv);
    varre.style.opacity = 0;
    an(varre, [{ opacity: 0.9, transform: "translateX(0px)" }, { opacity: 0.9, transform: `translateX(${bx[2] - 26}px)`, offset: 0.95 }, { opacity: 0, transform: `translateX(${bx[2] - 26}px)` }], { duration: 1300, delay: 1700, fill: "both" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 36, "TabPFN", "m-rot forte", { "text-anchor": "middle" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 53, "já estudou antes;", "m-nota", { "text-anchor": "middle" });
    texto(sv, bx[0] + bx[2] / 2, bx[1] + 66, "aqui não treina", "m-nota", { "text-anchor": "middle" });
    void caixa;
    const xn = 312, yn0 = 66, hn = 100;
    s("line", { x1: bx[0] + bx[2], y1: bx[1] + bx[3] / 2, x2: xn - 6, y2: bx[1] + bx[3] / 2, stroke: COR.tinta2, "stroke-width": 1.5 }, sv);
    s("rect", { x: xn, y: yn0, width: 16, height: hn, rx: 4, fill: COR.grade }, sv);
    const nota = s("rect", { x: xn, y: yn0 + 14, width: 16, height: hn - 14, rx: 4, fill: COR.spam }, sv);
    nota.style.transformBox = "fill-box"; nota.style.transformOrigin = "50% 100%";
    an(nota, [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 600, delay: 3000 });
    texto(sv, xn + 8, yn0 + hn + 13, "nota de", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    texto(sv, xn + 8, yn0 + hn + 25, "spam", "m-eixo", { "text-anchor": "middle", "dominant-baseline": "central" });
    etapasLinha(sv, 208, [
      { l: "antes de ver esta base, ele estudou milhões de tabelas inventadas", em: 0 },
      { l: "recebe de uma vez os e-mails de treino, já respondidos, e o novo", em: 900 },
      { l: "responde sem estudar de novo: o treino entra como contexto", em: 3000 },
    ]);
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
    // o nome do meta-modelo muda com o deck: "presidente" (banca) ou data-presidente="chefe" (equipe)
    if (!o.presidente && cont.dataset && cont.dataset.presidente) o = Object.assign({}, o, { presidente: cont.dataset.presidente });
    const desenhar = animar => {
      cont.textContent = "";
      [W, H] = DIM[tipo] || (o.mini ? DIM.mini : DIM.card);
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
