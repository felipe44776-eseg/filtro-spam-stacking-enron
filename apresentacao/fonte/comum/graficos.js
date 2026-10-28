/* graficos.js — gráficos SVG animados do Projeto 3, em JS puro (sem bibliotecas).
   Regras: todo número exibido vem de DADOS (nada é digitado aqui); a cor fica na marca, nunca no texto;
   barras finas com ponta arredondada e base reta; grade em linha fina; legenda sempre que há duas séries ou mais.
   Animação = Web Animations com fill "backwards": o estado final é o próprio desenho, então impressão e
   modo sem movimento saem completos. */
"use strict";
const G = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const COR = {
    fundo: "#fcfcfb", tinta: "#0b0b0b", tinta2: "#52514e", muda: "#898781", grade: "#e1e0d9", eixo: "#c3c2b7",
    real: "#2a78d6", spam: "#eb6834",
  };
  const FAMILIA = {
    linear: "#2a78d6", kernel: "#eb6834", vizinhos: "#1baf7a", "rede neural": "#eda100", "probabilística": "#e87ba4",
    bagging: "#008300", boosting: "#6250d6", "combinação": "#e34948", "referência": COR.eixo, "árvore": COR.muda,
  };
  const ORDEM_FAMILIAS = ["linear", "kernel", "vizinhos", "rede neural", "probabilística", "bagging", "boosting", "combinação"];
  const MEMBRO = {
    "Reg. Logística": { fam: "linear", sigla: "LR" }, "SVM RBF": { fam: "kernel", sigla: "SVM" },
    "kNN cosseno": { fam: "vizinhos", sigla: "kNN" }, "MLP": { fam: "rede neural", sigla: "MLP" },
    "Naive Bayes": { fam: "probabilística", sigla: "NB" }, "Extra Trees": { fam: "bagging", sigla: "ET" },
    "LightGBM": { fam: "boosting", sigla: "LGBM" },
  };
  const corMembro = m => FAMILIA[(MEMBRO[m] || {}).fam] || COR.muda;
  // Mesmas cores das figuras do projeto (scripts/09_modelos_finais.py: CATEGORICA[0], [1], [6], [2] e TINTA_MUDA).
  // Chaves = chaves de DADOS.finais (as antigas: generalista = Stacking 8; generalista_sem_chave = Stacking 7 padrão).
  // O nome exibido é o `rotulo` de cada entrada de DADOS.finais; `nome` aqui é só o reserva.
  const FILTRO = {
    especialista: { nome: "Especialista (S7 do GA)", curto: "Especialista", sub: "Stacking 7 do GA · o menor custo nos folds (empate no teste trancado)", cor: "#2a78d6" },
    generalista_sem_chave: { nome: "Stacking 7 padrão", curto: "Stacking 7 padrão", sub: "sem chave · usado na demo", cor: "#eb6834" },
    generalista: { nome: "Stacking 8 (+TabPFN)", curto: "Stacking 8", sub: "Stacking 7 padrão + TabPFN · o TabPFN barateia o stacking fora da caixa", cor: "#6250d6" },
    tabpfn: { nome: "TabPFN-3.5 sozinho", curto: "TabPFN sozinho", sub: "modelo de fundação, sem ajuste · o mais conservador fora da caixa", cor: "#1baf7a" },
    linha_de_base: { nome: "LinearSVC", curto: "LinearSVC", sub: "linha de base", cor: COR.muda },
    nao_filtrar: { nome: "não filtrar", curto: "não filtrar", sub: "referência: todo spam passa", cor: COR.tinta2 },
  };
  if (typeof DADOS !== "undefined" && DADOS.finais) {
    for (const k in FILTRO) if (DADOS.finais[k] && DADOS.finais[k].rotulo) FILTRO[k].nome = DADOS.finais[k].rotulo;
    if (DADOS.protocolo) FILTRO.especialista.sub = `Stacking 7 do GA · o menor custo nos ${DADOS.protocolo.folds} folds (empate no teste trancado)`;
  }
  // Ordem dos filtros nos gráficos e tabelas: azul, laranja, violeta e verde nunca lado a lado com o vizinho
  // parecido (validado: dataviz/validate_palette.js). "não filtrar" é referência: linha tracejada, não barra.
  const ORDEM_FILTROS = ["especialista", "generalista_sem_chave", "generalista", "tabpfn", "linha_de_base"];
  const REFERENCIA = "nao_filtrar";
  // Nome de método no JSON → chave do filtro final correspondente (ou null se não for um filtro final).
  function chaveMetodo(m) {
    if (/[Nn]ão filtrar/.test(m)) return "nao_filtrar";
    if (/^Stacking 8 padrão/.test(m)) return "generalista";
    if (/^Stacking 7 padrão$/.test(m)) return "generalista_sem_chave";
    if (/^Stacking 7 (especialista|do GA)/.test(m)) return "especialista";
    if (/^TabPFN-3\.5-Plus( sozinho)?$/.test(m)) return "tabpfn";
    if (/^LinearSVC( padrão)?$/.test(m)) return "linha_de_base";
    return null;
  }
  const nomeMetodo = (m, curto) => { const k = chaveMetodo(m); return k ? FILTRO[k][curto ? "curto" : "nome"] : m; };
  const ordemMetodo = m => { const k = chaveMetodo(m); const i = k === "nao_filtrar" ? 99 : ORDEM_FILTROS.indexOf(k); return i < 0 ? 50 : i; };
  function corMetodo(nome) {
    if (/[Nn]ão filtrar/.test(nome)) return FILTRO.nao_filtrar.cor;
    if (/Stacking 8 especialista/.test(nome)) return "#e87ba4";
    if (/Stacking 8/.test(nome)) return FILTRO.generalista.cor;
    if (/Stacking 7 (especialista|do GA)/.test(nome)) return FILTRO.especialista.cor;
    if (/Stacking 7/.test(nome)) return FILTRO.generalista_sem_chave.cor;
    if (/TabPFN/.test(nome)) return FILTRO.tabpfn.cor;
    if (/LinearSVC/.test(nome)) return COR.muda;
    if (/Reg\. Log/.test(nome)) return FAMILIA.linear;
    return COR.eixo;
  }
  const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

  /* ---------------------------------------------------------------- formatação pt-BR */
  const F = {
    dec(x, c) {
      if (x == null || Number.isNaN(x)) return "–";
      const s = Math.abs(x).toFixed(c), neg = x < 0 && Number(s) !== 0;
      return (neg ? "−" : "") + s.replace(".", ",");
    },
    mcc: x => F.dec(x, 3),
    pct: (x, c = 1) => F.dec(100 * x, c) + "%",
    custo: x => F.dec(x, 1),
    int(n) {
      const v = Math.round(n), a = String(Math.abs(v));
      return (v < 0 ? "−" : "") + (a.length >= 5 ? a.replace(/\B(?=(\d{3})+(?!\d))/g, ".") : a);
    },
    sinal: (x, c) => (x > 0 && Number(x.toFixed(c)) !== 0 ? "+" : "") + F.dec(x, c),
  };

  /* ---------------------------------------------------------------- utilidades */
  const api = { semMovimento: false };
  const vivos = new Set();
  function s(tag, at, pai) {
    const e = document.createElementNS(NS, tag);
    for (const k in at || {}) if (at[k] != null) e.setAttribute(k, at[k]);
    if (pai) pai.appendChild(e);
    return e;
  }
  function h(tag, at, pai, texto) {
    const e = document.createElement(tag);
    for (const k in at || {}) if (at[k] != null) e.setAttribute(k, at[k]);
    if (texto != null) e.textContent = texto;
    if (pai) pai.appendChild(e);
    return e;
  }
  function t(pai, x, y, texto, at) {
    const e = s("text", Object.assign({ x, y }, at), pai);
    e.textContent = texto;
    return e;
  }
  function svgEm(cont, W, H, rotulo) {
    cont.textContent = "";
    return s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "g-svg", role: "img", "aria-label": rotulo || null }, cont);
  }
  const medidas = (cont, o) => [o.largura || cont.clientWidth || 400, o.altura || cont.clientHeight || 260];
  const esc = x => String(x).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const mk = cor => `<span class="m" style="background:${cor}"></span>`;
  function anim(el, quadros, op, ligado) {
    if (!ligado || api.semMovimento || !el || !el.animate) return null;
    return el.animate(quadros, Object.assign({ fill: "backwards", easing: "cubic-bezier(.2,.7,.2,1)" }, op));
  }
  function contar(el, ate, fmt, op, ligado) {
    if (!ligado || api.semMovimento) { el.textContent = fmt(ate); return; }
    const de = op.de ?? 0, dur = op.duracao ?? 900, t0 = performance.now() + (op.atraso ?? 0);
    const tw = { fim() { el.textContent = fmt(ate); vivos.delete(tw); } };
    vivos.add(tw);
    el.textContent = fmt(de);
    const passo = agora => {
      if (!vivos.has(tw)) return;
      const u = Math.min(1, Math.max(0, (agora - t0) / dur));
      el.textContent = fmt(de + (ate - de) * (1 - Math.pow(1 - u, 3)));
      if (u < 1) requestAnimationFrame(passo); else tw.fim();
    };
    requestAnimationFrame(passo);
  }
  api.finalizar = () => [...vivos].forEach(tw => tw.fim());

  let dicaEl = null;
  function dicaMostrar(ev, html) {
    if (!dicaEl) dicaEl = h("div", { class: "g-dica no-print", role: "tooltip" }, document.body);
    dicaEl.innerHTML = html;
    dicaEl.hidden = false;
    const r = dicaEl.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
    dicaEl.style.left = x + "px";
    dicaEl.style.top = y + "px";
  }
  const dicaEsconder = () => { if (dicaEl) dicaEl.hidden = true; };
  function dica(alvo, conteudo) {
    const f = ev => dicaMostrar(ev, typeof conteudo === "function" ? conteudo() : conteudo);
    alvo.addEventListener("pointerenter", f);
    alvo.addEventListener("pointermove", f);
    alvo.addEventListener("pointerleave", dicaEsconder);
  }

  const hexRgb = x => { const n = parseInt(x.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const rgbHex = r => "#" + r.map(v => Math.round(v).toString(16).padStart(2, "0")).join("");
  function rampa(u) {
    u = Math.max(0, Math.min(1, u));
    const p = u * (SEQ.length - 1), i = Math.min(SEQ.length - 2, Math.floor(p)), f = p - i;
    const a = hexRgb(SEQ[i]), b = hexRgb(SEQ[i + 1]);
    return rgbHex(a.map((v, k) => v + (b[k] - v) * f));
  }
  function luminancia(hex) {
    const c = hexRgb(hex).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  function passoBom(x) {
    const e = Math.pow(10, Math.floor(Math.log10(x))), f = x / e;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * e;
  }
  function ticksLin(a, b, n = 4) {
    const p = passoBom((b - a) / n), r = [];
    for (let v = Math.ceil(a / p) * p; v <= b + p * 1e-9; v += p) r.push(Number(v.toFixed(10)));
    return r;
  }
  function ticksLog(a, b) {
    const r = [];
    for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++)
      for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= a && v <= b) r.push(v); }
    return r;
  }
  function caminhoBarra(x0, x1, y, alt, r = 4) {
    const w = Math.abs(x1 - x0);
    if (w < 0.5) return `M${x0},${y}v${alt}`;
    r = Math.min(r, w, alt / 2);
    if (x1 >= x0) return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + alt - r}Q${x1},${y + alt} ${x1 - r},${y + alt}H${x0}Z`;
    return `M${x0},${y}H${x1 + r}Q${x1},${y} ${x1},${y + r}V${y + alt - r}Q${x1},${y + alt} ${x1 + r},${y + alt}H${x0}Z`;
  }
  const truncar = (txt, n) => (txt.length > n ? txt.slice(0, n).replace(/\s+\S*$/, "") + " …" : txt);

  /* ---------------------------------------------------------------- legenda (HTML) */
  function legenda(cont, itens, classe) {
    const d = h("div", { class: "g-legenda" + (classe ? " " + classe : "") }, cont);
    for (const it of itens) {
      const e = h("span", { class: "g-leg-item" }, d);
      const m = h("span", { class: "g-leg-marca " + (it.forma || "ponto") }, e);
      m.style.background = it.cor;
      h("span", null, e, it.nome);
    }
    return d;
  }
  const legendaFamilias = (cont, fams) =>
    legenda(cont, ORDEM_FAMILIAS.filter(f => fams.includes(f)).map(f => ({ nome: f, cor: FAMILIA[f], forma: "quadro" })));

  /* ---------------------------------------------------------------- barras horizontais */
  function barras(cont, o) {
    const [W, H] = medidas(cont, o), A = o.animar;
    const sv = svgEm(cont, W, H, o.rotulo);
    const L = o.margemE ?? 170, R = o.margemD ?? 56, T = o.margemT ?? (o.refs ? 18 : 2), B = o.eixo ? 22 : 2;
    const n = o.itens.length, banda = (H - T - B) / n;
    const esp = Math.min(o.espessura ?? 16, banda - 3);
    const [d0, d1] = o.dom;
    const x = v => L + (Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0) * (W - L - R);
    const x0 = x(d0 <= 0 && d1 >= 0 ? 0 : d0);
    if (o.eixo) for (const tk of o.eixo.ticks) {
      s("line", { x1: x(tk), x2: x(tk), y1: T, y2: H - B, class: "g-grade" }, sv);
      t(sv, x(tk), H - 6, o.eixo.fmt(tk), { class: "g-tick", "text-anchor": "middle" });
    }
    s("line", { x1: x0, x2: x0, y1: T, y2: H - B, class: "g-base" }, sv);
    o.itens.forEach((it, i) => {
      const yc = T + banda * (i + 0.5), g = s("g", { class: it.apagado ? "apagado" : null }, sv);
      let lx = o.rotuloX ?? 0;
      if (it.marca) { s("circle", { cx: lx + 5, cy: yc, r: 5, fill: it.marca }, g); lx += 15; }
      if (L > 8) t(g, lx, yc, it.rotulo, { class: "g-rot" + (it.forte ? " forte" : ""), "dominant-baseline": "central" });
      const xv = x(it.valor), atraso = o.atraso ? o.atraso(i, it) : i * 45;
      const p = s("path", { d: caminhoBarra(x0, xv, yc - esp / 2, esp), fill: it.cor || COR.tinta2, class: "g-barra" }, g);
      p.style.transformOrigin = xv >= x0 ? "0% 50%" : "100% 50%";
      anim(p, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 650, delay: atraso }, A);
      let xt = xv;
      if (it.erro) {                                    // ± desvio padrão: traço fino sobre a barra
        const a = x(it.valor - it.erro), b = x(it.valor + it.erro);
        const w = s("g", { class: "g-erro" }, g);
        s("line", { x1: a, x2: b, y1: yc, y2: yc, stroke: COR.tinta, "stroke-width": 1.2 }, w);
        [a, b].forEach(xx => s("line", { x1: xx, x2: xx, y1: yc - 4, y2: yc + 4, stroke: COR.tinta, "stroke-width": 1.2 }, w));
        anim(w, [{ opacity: 0 }, { opacity: 1 }], { duration: 250, delay: atraso + 500 }, A);
        xt = Math.max(xv, b);
      }
      if (o.fmt) {
        const v = t(g, xt + (xv >= x0 ? 6 : -6), yc, it.txt ?? o.fmt(it.valor),
          { class: "g-val" + (it.forte ? " forte" : ""), "text-anchor": xv >= x0 ? "start" : "end", "dominant-baseline": "central" });
        anim(v, [{ opacity: 0 }, { opacity: 1 }], { duration: 250, delay: atraso + 450 }, A);
      }
      if (it.dica) dica(s("rect", { x: 0, y: yc - banda / 2, width: W, height: banda, class: "g-alvo" }, sv), it.dica);
    });
    for (const r of o.refs || []) {
      const xr = x(r.valor);
      s("line", { x1: xr, x2: xr, y1: T - 4, y2: H - B, class: "g-ref" }, sv);
      if (r.rotulo) t(sv, xr + (r.esquerda ? -4 : 4), T - 8, r.rotulo, { class: "g-ref-rot", "text-anchor": r.esquerda ? "end" : "start" });
    }
    return sv;
  }

  /* ---------------------------------------------------------------- corrida: barras que crescem e se ordenam */
  // o.itens: [{ rotulo, valor, cor, dica?, forte? }]; o.inicio: ordem de partida (índices), se faltar usa a dada.
  // O desenho final é o ranking ordenado (impressão e modo sem movimento saem completos).
  // o.anterior: { valores, posicoes } de um desenho anterior, para animar a troca (botão antes × depois).
  function corrida(cont, o) {
    const [W, H] = medidas(cont, o), A = o.animar && !api.semMovimento, sv = svgEm(cont, W, H, o.rotulo);
    const L = o.margemE ?? 200, R = o.margemD ?? 54, T = 2, B = 22, n = o.itens.length, banda = (H - T - B) / n;
    const esp = Math.min(o.espessura ?? 12, banda - 3), [d0, d1] = o.dom;
    const x = v => L + (Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0) * (W - L - R);
    const ordem = o.itens.map((_, i) => i).sort((a, b) => o.itens[b].valor - o.itens[a].valor || a - b);
    const pos = []; ordem.forEach((idx, r) => { pos[idx] = r; });
    const ini = o.anterior ? o.anterior.posicoes : (() => { const p = []; (o.inicio || o.itens.map((_, i) => i)).forEach((idx, r) => { p[idx] = r; }); return p; })();
    (o.eixo || []).forEach(tk => {
      s("line", { x1: x(tk), x2: x(tk), y1: T, y2: H - B, class: "g-grade" }, sv);
      t(sv, x(tk), H - 6, (o.fmtEixo || o.fmt)(tk), { class: "g-tick", "text-anchor": "middle" });
    });
    s("line", { x1: x(d0), x2: x(d0), y1: T, y2: H - B, class: "g-base" }, sv);
    const crescer = o.anterior ? 0 : 1500, mover = 900;
    o.itens.forEach((it, i) => {
      const y = T + banda * (pos[i] + 0.5), externo = s("g", { transform: `translate(0,${y.toFixed(1)})` }, sv);
      const g = s("g", null, externo);
      const dy = (ini[i] - pos[i]) * banda;
      if (A && dy) anim(g, [{ transform: `translateY(${dy}px)` }, { transform: `translateY(${dy}px)`, offset: crescer / (crescer + mover) }, { transform: "translateY(0px)" }],
        { duration: crescer + mover, easing: "ease-in-out" }, true);
      t(g, 0, 0, it.rotulo, { class: "g-rot" + (it.forte ? " forte" : ""), "dominant-baseline": "central" });
      const xv = x(it.valor), p = s("path", { d: caminhoBarra(x(d0), xv, -esp / 2, esp), fill: it.cor, class: "g-barra" }, g);
      p.style.transformOrigin = "0% 50%";
      if (A && !o.anterior) anim(p, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 650, delay: (o.atraso ? o.atraso(i) : ini[i] * 50) }, true);
      if (A && o.anterior && o.anterior.valores[i] !== it.valor) {
        const r = Math.max(0.001, (x(o.anterior.valores[i]) - x(d0)) / Math.max(0.001, xv - x(d0)));
        anim(p, [{ transform: `scaleX(${r})` }, { transform: "scaleX(1)" }], { duration: mover, easing: "ease-in-out" }, true);
      }
      const v = t(g, xv + 6, 0, o.fmt(it.valor), { class: "g-val" + (it.forte ? " forte" : ""), "dominant-baseline": "central" });
      if (A && !o.anterior) anim(v, [{ opacity: 0 }, { opacity: 1 }], { duration: 250, delay: crescer }, true);
      if (it.anel) s("rect", { x: -4, y: -banda / 2 + 1, width: W - 2, height: banda - 2, rx: 4, fill: "none", stroke: COR.tinta, "stroke-width": 1.2 }, g);
      if (it.dica) dica(s("rect", { x: 0, y: -banda / 2, width: W, height: banda, class: "g-alvo" }, g), it.dica);
    });
    return { posicoes: pos, valores: o.itens.map(it => it.valor) };
  }

  /* ---------------------------------------------------------------- halteres (duas classes por linha) */
  function halteres(cont, linhas, o) {
    const [W, H] = medidas(cont, o), A = o.animar, sv = svgEm(cont, W, H, o.rotulo);
    const L = o.margemE ?? 92, R = 14, T = 4, B = 22, banda = (H - T - B) / linhas.length;
    const x = v => L + v * (W - L - R);
    for (const tk of [0, 0.25, 0.5, 0.75, 1]) {
      s("line", { x1: x(tk), x2: x(tk), y1: T, y2: H - B, class: "g-grade" }, sv);
      t(sv, x(tk), H - 6, F.pct(tk, 0), { class: "g-tick", "text-anchor": "middle" });
    }
    linhas.forEach((ln, i) => {
      const y = T + banda * (i + 0.5), a = x(ln.a), b = x(ln.b), junto = Math.abs(a - b) < 8;
      t(sv, 0, y, ln.rotulo, { class: "g-rot" + (ln.forte ? " forte" : ""), "dominant-baseline": "central" });
      const seg = s("line", { x1: Math.min(a, b), x2: Math.max(a, b), y1: y, y2: y, stroke: COR.eixo, "stroke-width": 2 }, sv);
      const ca = s("circle", { cx: a, cy: junto ? y - 4 : y, r: 6, fill: COR.real, stroke: COR.fundo, "stroke-width": 2, class: "g-pop" }, sv);
      const cb = s("circle", { cx: b, cy: junto ? y + 4 : y, r: 6, fill: COR.spam, stroke: COR.fundo, "stroke-width": 2, class: "g-pop" }, sv);
      anim(ca, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 350, delay: 150 + i * 90 }, A);
      anim(cb, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 350, delay: 330 + i * 90 }, A);
      anim(seg, [{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 450 + i * 90 }, A);
      dica(s("rect", { x: 0, y: y - banda / 2, width: W, height: banda, class: "g-alvo" }, sv),
        `<b>${esc(ln.rotulo)}</b><br>${mk(COR.real)}e-mails reais: ${F.pct(ln.a)}<br>${mk(COR.spam)}spams: ${F.pct(ln.b)}`);
    });
    return sv;
  }

  /* ---------------------------------------------------------------- dispersão (SVD em 2D) */
  function dispersao(cont, pts, o) {
    const [W, H] = medidas(cont, o), A = o.animar, sv = svgEm(cont, W, H, o.rotulo);
    const m = o.margem ?? 8, passo = o.passo ?? 1;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    const k = Math.min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0));
    const ox = m + (W - 2 * m - k * (x1 - x0)) / 2, oy = m + (H - 2 * m - k * (y1 - y0)) / 2;
    const X = v => (ox + (v - x0) * k).toFixed(1), Y = v => (H - oy - (v - y0) * k).toFixed(1);
    if (o.cruz) {
      s("line", { x1: X(0), x2: X(0), y1: m, y2: H - m, class: "g-grade" }, sv);
      s("line", { x1: m, x2: W - m, y1: Y(0), y2: Y(0), class: "g-grade" }, sv);
    }
    const op = o.opacidade ?? 0.5;
    const grupos = [COR.real, COR.spam].map(c => s("g", { fill: c, "fill-opacity": op }, sv));
    const r = o.raio ?? 2.2;
    for (let i = 0; i < pts.length; i += passo) s("circle", { cx: X(pts[i][0]), cy: Y(pts[i][1]), r }, grupos[pts[i][2]]);
    grupos.forEach((g, c) => anim(g, [
      { opacity: 0, fill: COR.eixo }, { opacity: 1, fill: COR.eixo, offset: 0.3 }, { opacity: 1, fill: COR.eixo, offset: 0.55 },
      { opacity: 1, fill: c ? COR.spam : COR.real }], { duration: o.duracao ?? 2600, easing: "ease-in-out" }, A));
    return sv;
  }

  /* ---------------------------------------------------------------- mapa triangular de erros em comum */
  // o.completo: matriz assimétrica inteira (sem a diagonal), linha = "este erra", coluna = "quando este erra".
  function mapaTriangular(cont, modelos, M, o) {
    const [W, H] = medidas(cont, o), A = o.animar, sv = svgEm(cont, W, H, o.rotulo);
    const n = modelos.length, L = 62, B = 34, T = 2, cheio = !!o.completo;
    const nl = cheio ? n : n - 1, nc = cheio ? n : n - 1;
    const c = Math.min((W - L - 4) / nc, (H - T - B) / nl);
    const celulas = [];
    for (let i = cheio ? 0 : 1; i < n; i++) for (let j = 0; j < (cheio ? n : i); j++) if (i !== j) celulas.push([i, j]);
    const vals = celulas.map(([i, j]) => M[i][j]), vmin = Math.min(...vals), vmax = Math.max(...vals);
    const sig = m => (MEMBRO[m] || {}).sigla || m;
    const fmtD = o.fmtDica || ((i, j, v) => `<b>${esc(modelos[i])} × ${esc(modelos[j])}</b><br>Jaccard dos erros: ${F.dec(v, 2)}`);
    for (let i = cheio ? 0 : 1; i < n; i++) {
      const y = T + (cheio ? i : i - 1) * c, forte = modelos[i] === o.destaque;
      s("circle", { cx: 6, cy: y + c / 2, r: 5, fill: corMembro(modelos[i]) }, sv);
      t(sv, 16, y + c / 2, sig(modelos[i]), { class: "g-rot" + (forte ? " forte" : ""), "dominant-baseline": "central" });
    }
    celulas.forEach(([i, j]) => {
      const y = T + (cheio ? i : i - 1) * c, x = L + j * c, v = M[i][j], cor = rampa((v - vmin) / (vmax - vmin));
      const g = s("g", { class: "g-pop" }, sv);
      const ret = s("rect", { x: x + 1, y: y + 1, width: c - 2, height: c - 2, rx: 3, fill: cor }, g);
      const tx = t(g, x + c / 2, y + c / 2, F.dec(v, 2), { "text-anchor": "middle", "dominant-baseline": "central", class: "g-cel" });
      tx.style.fill = luminancia(cor) < 0.3 ? "#ffffff" : COR.tinta;
      anim(g, [{ opacity: 0, transform: "scale(.5)" }, { opacity: 1, transform: "scale(1)" }], { duration: 420, delay: Math.abs(i - j) * 160 + j * 25 }, A);
      dica(ret, fmtD(i, j, v));
    });
    for (let j = 0; j < (cheio ? n : n - 1); j++) {
      const xc = L + j * c + c / 2, yb = T + nl * c + 10, forte = modelos[j] === o.destaque;
      s("circle", { cx: xc, cy: yb, r: 4, fill: corMembro(modelos[j]) }, sv);
      t(sv, xc, yb + 15, sig(modelos[j]), { class: "g-rot" + (forte ? " forte" : ""), "text-anchor": "middle" });
    }
    if (o.legendaEm) {
      const d = h("div", { class: "g-rampa" }, o.legendaEm);
      h("span", null, d, F.dec(vmin, 2) + " · erram pouco juntos");
      const i = h("i", null, d);
      i.style.background = `linear-gradient(90deg, ${SEQ.join(",")})`;
      h("span", null, d, "erram muito juntos · " + F.dec(vmax, 2));
    }
    return sv;
  }

  /* ---------------------------------------------------------------- curvas (com rótulos no fim e cruz de leitura) */
  function curvas(cont, o) {
    const [W, H] = medidas(cont, o), A = o.animar, sv = svgEm(cont, W, H, o.rotulo);
    const L = o.margemE ?? 52, R = o.margemD ?? 16, T = o.margemT ?? 22, B = o.margemB ?? 38;
    const lx = o.x.log;
    const fx = v => { const [a, b] = o.x.dom; const u = lx ? (Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a)) : (v - a) / (b - a); return L + u * (W - L - R); };
    const fy = v => { const [a, b] = o.y.dom; return T + (1 - (v - a) / (b - a)) * (H - T - B); };
    o.y.ticks.forEach(tk => {
      s("line", { x1: L, x2: W - R, y1: fy(tk), y2: fy(tk), class: "g-grade" }, sv);
      t(sv, L - 8, fy(tk), o.y.fmt(tk), { class: "g-tick", "text-anchor": "end", "dominant-baseline": "central" });
    });
    s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "g-base" }, sv);
    o.x.ticks.forEach(tk => {
      s("line", { x1: fx(tk), x2: fx(tk), y1: H - B, y2: H - B + 4, class: "g-base" }, sv);
      t(sv, fx(tk), H - B + 17, o.x.fmt(tk), { class: "g-tick", "text-anchor": "middle" });
    });
    if (o.x.titulo) t(sv, L + (W - L - R) / 2, H - 3, o.x.titulo, { class: "g-eixo-tit", "text-anchor": "middle" });
    if (o.y.titulo) t(sv, L - 8, 10, o.y.titulo, { class: "g-eixo-tit", "text-anchor": "start" });
    (o.refs || []).forEach(r => {
      s("line", { x1: L, x2: W - R, y1: fy(r.y), y2: fy(r.y), class: "g-ref" }, sv);
      t(sv, r.esquerda ? L + 4 : W - R, fy(r.y) + (r.abaixo ? 13 : -6), r.rotulo, { class: "g-ref-rot", "text-anchor": r.esquerda ? "start" : "end" });
    });
    const fins = [], dur = o.duracao ?? 1800;
    o.series.forEach((se, k) => {
      const atraso = (o.defasagem ?? 0) * k;
      const d = se.pts.map((p, i) => (i ? "L" : "M") + fx(p[0]).toFixed(1) + "," + fy(p[1]).toFixed(1)).join("");
      const p = s("path", { d, fill: "none", stroke: se.cor, "stroke-width": se.esp ?? 2, "stroke-linejoin": "round", "stroke-linecap": "round", pathLength: 1 }, sv);
      anim(p, [{ strokeDasharray: "1 1", strokeDashoffset: 1 }, { strokeDasharray: "1 1", strokeDashoffset: 0 }], { duration: dur, delay: atraso, easing: "ease-in-out" }, A);
      if (se.marcadores) se.pts.forEach((pt, i) => {
        const c = s("circle", { cx: fx(pt[0]), cy: fy(pt[1]), r: 4, fill: se.cor, stroke: COR.fundo, "stroke-width": 2, class: "g-pop" }, sv);
        anim(c, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 300, delay: atraso + dur * i / Math.max(1, se.pts.length - 1) }, A);
      });
      const u = se.pts[se.pts.length - 1];
      if (o.rotulosFim) {
        if (!se.marcadores) {
          const c = s("circle", { cx: fx(u[0]), cy: fy(u[1]), r: 4, fill: se.cor, stroke: COR.fundo, "stroke-width": 2 }, sv);
          anim(c, [{ opacity: 0 }, { opacity: 1 }], { duration: 200, delay: atraso + dur }, A);
        }
        fins.push({ x: fx(u[0]), y: fy(u[1]), texto: se.rotuloFim ?? se.nome, atraso: atraso + dur });
      }
    });
    if (fins.length) {                                   // afasta rótulos que colidem e liga com linha-guia
      fins.sort((a, b) => a.y - b.y);
      fins.forEach((f, i) => { f.ly = i && f.y < fins[i - 1].ly + 16 ? fins[i - 1].ly + 16 : f.y; });
      const excesso = fins[fins.length - 1].ly - (H - B);
      if (excesso > 0) fins.forEach(f => { f.ly -= excesso; });
      fins.forEach(f => {
        const xl = f.x + (o.recuoRotulo ?? 14);
        if (Math.abs(f.ly - f.y) > 1) s("line", { x1: f.x + 5, y1: f.y, x2: xl - 3, y2: f.ly, stroke: COR.eixo }, sv);
        const e = t(sv, xl, f.ly, f.texto, { class: "g-rot forte", "dominant-baseline": "central" });
        anim(e, [{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: f.atraso }, A);
      });
    }
    if (o.dica !== false) {                              // cruz de leitura: linha vertical + valores de todas as séries
      const cursor = s("line", { y1: T, y2: H - B, class: "g-cursor", visibility: "hidden" }, sv);
      const alvo = s("rect", { x: L, y: T, width: W - L - R, height: H - T - B, class: "g-alvo" }, sv);
      const fmtD = o.y.fmtDica || o.y.fmt, fmtX = o.x.fmtDica || o.x.fmt;
      alvo.addEventListener("pointermove", ev => {
        const pt = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(sv.getScreenCTM().inverse());
        const perto = o.series.map(se => {
          let melhor = se.pts[0], dmin = Infinity;
          for (const p of se.pts) { const d = Math.abs(fx(p[0]) - pt.x); if (d < dmin) { dmin = d; melhor = p; } }
          return [se, melhor];
        });
        const xr = fx(perto[0][1][0]);
        cursor.setAttribute("x1", xr); cursor.setAttribute("x2", xr); cursor.setAttribute("visibility", "visible");
        dicaMostrar(ev, `<b>${esc(o.x.nome || "")} ${esc(fmtX(perto[0][1][0]))}</b><br>` +
          perto.map(([se, p]) => `${mk(se.cor)}${esc(se.nome)}: ${esc(fmtD(p[1]))}`).join("<br>"));
      });
      alvo.addEventListener("pointerleave", () => { cursor.setAttribute("visibility", "hidden"); dicaEsconder(); });
    }
    return { sv, fx, fy };
  }

  /* ---------------------------------------------------------------- pontos de permutação (tira) */
  function tira(cont, o) {
    const [W, H] = medidas(cont, o), A = o.animar, sv = svgEm(cont, W, H, o.rotulo);
    const L = 10, R = 10, B = 22, [d0, d1] = o.dom, yc = (H - B) / 2;
    const x = v => L + (v - d0) / (d1 - d0) * (W - L - R);
    ticksLin(d0, d1, 5).forEach(tk => {
      s("line", { x1: x(tk), x2: x(tk), y1: 4, y2: H - B, class: "g-grade" }, sv);
      t(sv, x(tk), H - 6, F.dec(tk, 1), { class: "g-tick", "text-anchor": "middle" });
    });
    // Muitos pontos (centenas de permutações): faixa vertical maior, pontos menores e animação total limitada.
    const n = o.pontos.length, muitos = n > 60, faixas = muitos ? 11 : 5, passo = muitos ? 3.2 : 7;
    const atrasoP = Math.min(30, 1400 / Math.max(1, n));
    const grupo = s("g", { "fill-opacity": muitos ? 0.45 : 1 }, sv);
    o.pontos.forEach((v, i) => {
      const c = s("circle", { cx: x(v), cy: yc + ((i % faixas) - (faixas - 1) / 2) * passo, r: muitos ? 2.4 : 4, fill: COR.muda,
        stroke: muitos ? null : COR.fundo, "stroke-width": muitos ? null : 2, class: "g-pop" }, grupo);
      anim(c, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 250, delay: i * atrasoP }, A);
    });
    const d = o.destaque, cd = s("circle", { cx: x(d.valor), cy: yc, r: 7, fill: COR.tinta, stroke: COR.fundo, "stroke-width": 2, class: "g-pop" }, sv);
    anim(cd, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 400, delay: n * atrasoP + 200 }, A);
    t(sv, x(d.valor), yc - 16, d.rotulo, { class: "g-rot forte", "text-anchor": "end" });
    t(sv, x(o.pontos.reduce((a, b) => Math.max(a, b))) + 12, yc - 22, o.rotuloPontos, { class: "g-rot" });
    return sv;
  }

  /* ---------------------------------------------------------------- envelopes: um e-mail real = λ spams */
  function envelopes(cont, o) {
    const n = o.n, A = o.animar, w = 34, hh = 24, gap = 8, x0 = 150;
    const W = x0 + n * (w + gap), H = hh + 26;
    const sv = svgEm(cont, W, H, o.rotulo);
    const env = (x, cor, g) => {
      s("rect", { x, y: 2, width: w, height: hh, rx: 3, fill: cor }, g);
      s("polyline", { points: `${x + 3},${5} ${x + w / 2},${2 + hh * 0.55} ${x + w - 3},${5}`, fill: "none", stroke: "#fff", "stroke-width": 1.6, "stroke-linejoin": "round" }, g);
    };
    const g0 = s("g", { class: "g-pop" }, sv);
    env(40, COR.real, g0);
    anim(g0, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 350 }, A);
    const ig = t(sv, (40 + w + x0) / 2, 2 + hh / 2, "=", { class: "g-val", "text-anchor": "middle", "dominant-baseline": "central" });
    ig.style.fontSize = "24px";
    for (let i = 0; i < n; i++) {
      const g = s("g", { class: "g-pop" }, sv);
      env(x0 + i * (w + gap), COR.spam, g);
      anim(g, [{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 300, delay: 400 + i * 110 }, A);
    }
    t(sv, 40 + w / 2, H - 2, o.rotuloReal, { class: "g-tick", "text-anchor": "middle" });
    t(sv, x0, H - 2, o.rotuloSpam, { class: "g-tick" });
    return sv;
  }

  /* ---------------------------------------------------------------- barra empilhada 100% (HTML) */
  function empilhada(cont, partes, o) {
    cont.textContent = "";
    const total = partes.reduce((a, p) => a + p.valor, 0);
    const barra = h("div", { style: `display:flex;gap:2px;height:${o.altura ?? 22}px` }, cont);
    partes.forEach((p, i) => {
      const seg = h("div", { style: `width:${100 * p.valor / total}%;background:${p.cor};border-radius:${i === 0 ? "4px 0 0 4px" : i === partes.length - 1 ? "0 4px 4px 0" : "0"}` }, barra);
      seg.style.transformOrigin = "0 50%";
      anim(seg, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 600, delay: i * 500 }, o.animar);
      dica(seg, `${mk(p.cor)}<b>${esc(p.rotulo)}</b>: ${F.int(p.valor)} (${F.pct(p.valor / total)})`);
    });
    legenda(cont, partes.map(p => ({ nome: `${p.rotulo}: ${F.int(p.valor)} (${F.pct(p.valor / total)})`, cor: p.cor }))).style.marginTop = "8px";
  }

  /* ---------------------------------------------------------------- tabela HTML com destaque por célula */
  // cab: [{ rot, sub? }]; linhas: [[célula, ...], ...]; célula = texto ou { t, marca?, forte?, ruim?, apagado? }.
  // A primeira coluna fica à esquerda; as demais, numéricas, à direita. "ruim" = fundo rosado (pior que a referência).
  function tabela(cont, cab, linhas, o = {}) {
    const tb = h("table", { class: "tabela" + (o.classe ? " " + o.classe : ""), style: o.estilo || null }, cont);
    const th = h("tr", null, h("thead", null, tb));
    const num = i => i > 0 && !o.texto;                    // o.texto: todas as colunas alinhadas à esquerda
    cab.forEach((c, i) => {
      const e = h("th", { class: num(i) ? "n nb" : "nb" }, th, c.rot);
      if (c.sub) { h("br", null, e); h("span", { style: "font-weight:400" }, e, c.sub); }
    });
    const bd = h("tbody", null, tb);
    linhas.forEach(l => {
      const tr = h("tr", null, bd);
      l.forEach((c, i) => {
        const cel = typeof c === "object" && c !== null ? c : { t: c };
        const td = h("td", { class: num(i) ? "n" : (o.quebra ? null : "nb") }, tr);
        if (cel.marca) td.innerHTML = mk(cel.marca);
        td.append(cel.t ?? "–");
        if (cel.forte) { td.style.fontWeight = "700"; td.style.color = COR.tinta; }
        if (cel.ruim) { td.style.background = "#fde3d6"; td.style.color = COR.tinta; td.style.fontWeight = td.style.fontWeight || "650"; }
        if (cel.apagado) td.style.color = COR.muda;
      });
    });
    return tb;
  }

  /* ---------------------------------------------------------------- o caminho de um e-mail pelo filtro */
  function caminho(cont, D, o) {
    const exemplos = D.exemplos;
    let ex = o.exemplo ?? 0, fil = o.filtro ?? "stacking7_padrao";
    cont.textContent = "";
    const raiz = h("div", { class: "cam" }, cont);
    const ctl = h("div", { class: "cam-controles no-print" }, raiz);
    const g1 = h("div", { class: "botoes", role: "group" }, ctl);
    h("span", { class: "rot" }, g1, "e-mail");
    const bEx = exemplos.map((e, i) => {
      const b = h("button", { type: "button", title: e.origem }, g1, e.rotulo);
      b.addEventListener("click", () => { ex = i; desenhar(true); });
      return b;
    });
    const g2 = h("div", { class: "botoes", role: "group" }, ctl);
    h("span", { class: "rot" }, g2, "filtro");
    const NOMES = { stacking7_padrao: "Stacking 7 padrão (demo)", especialista: "Especialista" };
    const bF = Object.keys(NOMES).map(k => {
      const b = h("button", { type: "button", "data-k": k }, g2, NOMES[k]);
      b.addEventListener("click", () => { fil = k; desenhar(true); });
      return b;
    });
    h("button", { type: "button", class: "bt" }, ctl, "↻ repetir").addEventListener("click", () => desenhar(true));
    const grade = h("div", { class: "cam-grade" }, raiz);

    function desenhar(A) {
      A = A && !api.semMovimento;
      bEx.forEach((b, i) => b.setAttribute("aria-pressed", String(i === ex)));
      bF.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.k === fil)));
      grade.textContent = "";
      const e = exemplos[ex], et = e.etapas[fil];
      let k = 0;
      const card = (titulo, sub) => {
        const c = h("div", { class: "cam-card" }, grade), t4 = h("h4", null, c);
        h("span", { class: "passo" }, t4, String(++k));
        h("span", null, t4, titulo);
        if (sub) h("span", { class: "sub" }, t4, sub);
        anim(c, [{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 420, delay: (k - 1) * 520 }, A);
        return c;
      };
      // 1. texto
      const c1 = card("Texto do e-mail", e.origem);
      h("pre", null, c1, truncar(e.texto, 300));
      // 2. formato AUEB
      const c2 = card("Formato AUEB", "minúsculas, pontuação separada");
      h("pre", null, c2, truncar(e.texto_aueb, 300));
      if (e.texto_aueb === e.texto) h("div", { class: "rodape-card" }, c2, "Igual ao original: este e-mail já veio pré-processado pela AUEB.");
      // 3. contagem
      const c3 = card("Contagem por substring", `${F.int(et.colunas_nao_zero)} de ${F.int(D.base.n_palavras)} colunas não zero`);
      const chips = h("div", { class: "chips" }, c3);
      const pares = Object.entries(e.vetor_nao_zero);
      const palavras = pares.filter(([w]) => w.length >= 3 && w !== "ect").sort((a, b) => b[1] - a[1]).slice(0, 9);
      const letras = pares.filter(([w]) => w.length === 1).sort((a, b) => b[1] - a[1]).slice(0, 4);
      const chip = ([w, n], cls) => { const c = h("span", { class: "chip" + (cls ? " " + cls : "") }, chips); h("b", null, c, w); c.append(" " + F.int(n)); };
      const ect = pares.find(([w]) => w === "ect");
      if (ect) chip(ect, "destaque");
      palavras.forEach(p => chip(p));
      letras.forEach(p => chip(p, "letra"));
      if (ect) h("div", { class: "rodape-card" }, c3, "“ect” está dentro de “Subject”; letras soltas também são colunas.");
      h("div", { class: "rodape-card" }, c3, `Cobertura: ${F.pct(e.cobertura)} das palavras do texto existem no vocabulário.`);
      // 4. sete notas
      const c4 = card("Sete notas", "uma por família");
      const nm = h("div", { class: "notas-membros" }, c4);
      Object.entries(et.notas_membros).forEach(([m, v], i) => {
        const pt = h("span", { class: "m" }, nm); pt.style.background = corMembro(m);
        h("span", null, nm, m);
        const vv = h("span", { class: "v" }, nm);
        contar(vv, v, x => F.dec(x, 3), { atraso: 3 * 520 + i * 60, duracao: 500 }, A);
      });
      h("div", { class: "rodape-card" }, c4, "Probabilidade de spam; no SVM, a distância à fronteira.");
      // 5. meta-modelo
      const c5 = card("Meta-modelo", "peso × nota, somados");
      const gw = h("div", { class: "g", style: "flex:1;min-height:0" }, c5);
      h("div", { class: "rodape-card" }, c5, "À direita empurra para spam; à esquerda, para e-mail real.");
      const linhas = [...Object.entries(et.contribuicoes).map(([m, v]) => ({ rotulo: (MEMBRO[m] || {}).sigla || m, valor: v, cor: corMembro(m), dica: `${mk(corMembro(m))}<b>${esc(m)}</b>: ${F.sinal(v, 3)}` })),
        { rotulo: "intercepto", valor: et.intercepto, cor: COR.eixo, dica: `intercepto: ${F.sinal(et.intercepto, 3)}` },
        { rotulo: "soma", valor: et.logit, cor: COR.tinta, forte: true, dica: `<b>soma (logit)</b>: ${F.sinal(et.logit, 3)}` }];
      const M = Math.max(...linhas.map(l => Math.abs(l.valor))) * 1.5;   // folga para o valor escrito na ponta
      barras(gw, {
        itens: linhas, dom: [-M, M], fmt: v => F.sinal(v, 2), margemE: 78, margemD: 4, espessura: 10, animar: A,
        atraso: i => 4 * 520 + i * 70, rotulo: "contribuição de cada membro para o logit",
      });
      // 6. decisão
      const c6 = card("Decisão", "P(spam) contra o limiar");
      const dc = h("div", { class: "decisao" }, c6);
      const big = h("div", { class: "numero" }, dc);
      contar(big, et.prob_spam, x => F.pct(x), { atraso: 5 * 520 + 200, duracao: 900 }, A);
      h("div", { class: "numero-rot" }, dc, `probabilidade de spam · limiar ${F.dec(et.limiar, 3)}`);
      const trilho = h("div", { style: "position:relative;height:12px;border-radius:6px;background:#eceae4;margin:4px 0 10px" }, dc);
      const ench = h("div", { style: `position:absolute;left:0;top:0;bottom:0;width:${100 * et.prob_spam}%;border-radius:6px;background:${COR.tinta2}` }, trilho);
      ench.style.transformOrigin = "0 50%";
      anim(ench, [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], { duration: 900, delay: 5 * 520 + 200 }, A);
      h("div", { style: `position:absolute;left:${100 * et.limiar}%;top:-4px;bottom:-4px;width:2px;background:${COR.tinta}` }, trilho);
      const ver = h("div", { class: "veredito" }, dc);
      const mm = h("span", { class: "m" }, ver); mm.style.background = et.spam ? COR.spam : COR.real;
      h("span", null, ver, et.spam ? "spam" : "e-mail real");
      const verdade = e.rotulo.startsWith("spam");
      h("div", { class: "rodape-card" }, dc, (verdade === et.spam ? "Acertou: o e-mail era " : "Errou: o e-mail era ") + e.rotulo + ".");
      anim(ver, [{ opacity: 0 }, { opacity: 1 }], { duration: 300, delay: 5 * 520 + 1100 }, A);
    }
    desenhar(o.animar);
  }

  /* ---------------------------------------------------------------- simulador de limiar */
  function simulador(cont, D, o) {
    const tab = D.simulador, nomes = ["especialista", "generalista"], n = tab.especialista.length;
    const lims = tab.especialista.map(r => r.limiar);
    const perto = v => lims.reduce((bi, l, i) => (Math.abs(l - v) < Math.abs(lims[bi] - v) ? i : bi), 0);
    const fr = D.base.n_email_real / D.base.n_emails, fs = 1 - fr, lam = D.meta.custo_fp / D.meta.custo_fn;
    cont.textContent = "";
    const raiz = h("div", { class: "sim" }, cont);
    const topo = h("div", { class: "sim-topo" }, raiz);
    const id = "sim-" + Math.random().toString(36).slice(2, 8);
    h("label", { for: id }, topo, "Limiar de P(spam):");
    const inp = h("input", { type: "range", min: 0, max: n - 1, step: 1, id, class: "sim-range", "aria-label": "limiar" }, topo);
    const out = h("output", { class: "sim-lim", for: id }, topo, F.dec(lims[0], 2));   // já com texto: a altura não muda depois
    const bts = h("div", { class: "botoes no-print" }, topo);
    nomes.forEach(k => {
      h("button", { type: "button" }, bts, "limiar do treino · " + FILTRO[k].nome)
        .addEventListener("click", () => mover(perto(D.finais[k].limiar_custo)));
    });
    const pain = h("div", { class: "sim-paineis" }, raiz);
    const P = nomes.map(k => painel(k));

    function painel(k) {
      const dados = tab[k], el = h("div", { class: "sim-painel cartao" }, pain);
      const tt = h("div", { class: "tit-g" }, el, FILTRO[k].nome);
      h("span", { style: "font-weight:400;color:#898781;margin-left:8px;font-size:13px" }, tt, FILTRO[k].sub);
      // Monta e preenche tudo antes de medir: o gráfico fica com a altura que sobra no cartão.
      const gw = h("div", { class: "g", style: "flex:1;min-height:0" }, el);
      const lei = h("div", { class: "sim-leituras" }, el);
      const leitura = (cor, rot) => {
        const d = h("div", null, lei), v = h("div", { class: "v" }, d, "–"), r = h("div", { class: "r" }, d);
        r.innerHTML = (cor ? mk(cor) : "") + esc(rot);
        return v;
      };
      const vR = leitura(COR.real, "e-mails reais na lixeira"), vS = leitura(COR.spam, "spam que passou"), vC = leitura(null, "custo por mil e-mails");
      const comp = h("div", { class: "sim-comp" }, el);
      const cR = h("span", { style: `background:${COR.real}` }, comp), cS = h("span", { style: `background:${COR.spam}` }, comp);
      const leg = h("div", { class: "legenda-g", style: "min-height:2.7em" }, el);
      const W = gw.clientWidth || 520, H = Math.max(150, gw.clientHeight || 240), L = 46, R = 14, T = 34, B = 26;
      const sv = svgEm(gw, W, H, "custo por mil e-mails conforme o limiar");
      const cs = dados.map(r => r.custo_por_mil), y0 = Math.min(...cs) * 0.8, y1 = Math.max(...cs) * 1.15;
      const fx = v => L + v * (W - L - R);
      const fy = v => T + (1 - (Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) * (H - T - B);
      ticksLog(y0, y1).forEach(tk => {
        s("line", { x1: L, x2: W - R, y1: fy(tk), y2: fy(tk), class: "g-grade" }, sv);
        t(sv, L - 6, fy(tk), F.int(tk), { class: "g-tick", "text-anchor": "end", "dominant-baseline": "central" });
      });
      [0, 0.25, 0.5, 0.75, 1].forEach(tk => t(sv, fx(tk), H - 8, F.dec(tk, 2), { class: "g-tick", "text-anchor": "middle" }));
      s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "g-base" }, sv);
      t(sv, 0, 10, "custo por mil e-mails (escala log) conforme o limiar", { class: "g-eixo-tit" });
      s("path", { d: dados.map((r, i) => (i ? "L" : "M") + fx(r.limiar).toFixed(1) + "," + fy(r.custo_por_mil).toFixed(1)).join(""), fill: "none", stroke: COR.tinta2, "stroke-width": 2, "stroke-linejoin": "round" }, sv);
      const iMin = cs.indexOf(Math.min(...cs));
      s("circle", { cx: fx(lims[iMin]), cy: fy(cs[iMin]), r: 4, fill: COR.fundo, stroke: COR.tinta2, "stroke-width": 1.5 }, sv);
      t(sv, fx(lims[iMin]), fy(cs[iMin]) - 10, "mínimo", { class: "g-tick", "text-anchor": "middle" });
      const lt = D.finais[k].limiar_custo, xt = fx(lt);       // limiar escolhido no treino: triângulo no topo + linha fina
      s("line", { x1: xt, x2: xt, y1: T, y2: H - B, class: "g-ref" }, sv);
      s("path", { d: `M${xt},${T}l-5,-8h10z`, fill: COR.tinta }, sv);
      t(sv, xt + 8, T - 3, "limiar do treino " + F.dec(lt, 3), { class: "g-tick" });
      const cur = s("line", { y1: T, y2: H - B, class: "g-cursor" }, sv);
      const dot = s("circle", { r: 5, fill: COR.tinta, stroke: COR.fundo, "stroke-width": 2 }, sv);
      return {
        atualizar(i) {
          const r = dados[i], x = fx(r.limiar);
          cur.setAttribute("x1", x); cur.setAttribute("x2", x);
          dot.setAttribute("cx", x); dot.setAttribute("cy", fy(r.custo_por_mil));
          vR.textContent = F.pct(r.real_lixeira);
          vS.textContent = F.pct(r.spam_passou);
          vC.textContent = F.custo(r.custo_por_mil);
          const a = lam * r.real_lixeira * fr, b = r.spam_passou * fs, tot = a + b || 1;
          cR.style.width = (100 * a / tot) + "%";
          cS.style.width = (100 * b / tot) + "%";
          leg.textContent = `De onde vem o custo: ${F.pct(a / tot)} de e-mails reais na lixeira (cada um vale ${F.int(lam)}) e ${F.pct(b / tot)} de spam que passou.`;
        },
      };
    }
    function mover(i) { inp.value = i; out.textContent = F.dec(lims[i], 2); P.forEach(p => p.atualizar(i)); }
    inp.addEventListener("input", () => mover(Number(inp.value)));
    const alvo = perto(D.finais.especialista.limiar_custo);
    if (o.animar && !api.semMovimento) {            // varredura inicial: o cursor percorre do limiar baixo ao do treino
      mover(0);                                     // estado válido já no primeiro quadro
      const t0 = performance.now(), dur = 1600;
      const tw = { fim() { mover(alvo); vivos.delete(tw); } };
      vivos.add(tw);
      const passo = agora => {
        if (!vivos.has(tw)) return;
        const u = Math.max(0, Math.min(1, (agora - t0) / dur));   // o carimbo do quadro pode ser anterior a t0
        mover(Math.round(alvo * (1 - Math.pow(1 - u, 3))));
        if (u < 1) requestAnimationFrame(passo); else tw.fim();
      };
      requestAnimationFrame(passo);
    } else mover(alvo);
  }

  /* ---------------------------------------------------------------- grande final: três avaliações, escalas separadas */
  function finalTres(cont, D, o) {
    // "não filtrar" entra como linha de referência tracejada (custo e TCR), nunca como barra colorida.
    const NF = D.finais[REFERENCIA];
    const MET = {
      custo: { rot: "custo por mil e-mails", campo: "custo_por_mil", fmt: F.custo, melhor: Math.min, nota: "menor é melhor",
               ref: k => ({ valor: NF[k].custo_por_mil, rotulo: "não filtrar " + F.custo(NF[k].custo_por_mil) }) },
      tcr: { rot: "TCR", campo: "tcr", fmt: v => F.dec(v, 2), melhor: Math.max, nota: "custo de não filtrar ÷ custo do filtro; acima de um, filtrar compensa",
             ref: k => ({ valor: NF[k].tcr, rotulo: "não filtrar " + F.dec(NF[k].tcr, 0) }) },
      mcc: { rot: "MCC", campo: "mcc_custo", fmt: F.mcc, melhor: Math.max, nota: "maior é melhor; não filtrar tem MCC zero", ref: null },
    };
    let met = o.metrica || "custo";
    cont.textContent = "";
    const raiz = h("div", { style: "display:flex;flex-direction:column;gap:10px;height:100%" }, cont);
    const ctl = h("div", { class: "linha" }, raiz);
    const bts = h("div", { class: "botoes no-print" }, ctl);
    const bb = Object.keys(MET).map(k => {
      const b = h("button", { type: "button", "data-k": k }, bts, MET[k].rot + " (limiar de custo)");
      b.addEventListener("click", () => { met = k; desenhar(true); });
      return b;
    });
    const nota = h("span", { class: "legenda-g" }, ctl);
    const linha = h("div", { class: "fin-paineis" }, raiz);
    const avals = [
      ["cv20", "Esta caixa", `validação cruzada, ${F.int(D.protocolo.folds)} folds`],
      ["externo", "Outras caixas da Enron", "Enron2–6, média das cinco caixas"],
      ["spamassassin", "SpamAssassin", "decidiu a regra D10 e, na D11, o destaque"],
    ];
    function desenhar(A) {
      bb.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.k === met)));
      const M = MET[met];
      nota.textContent = `${M.rot} · ${M.nota} · linha tracejada: não filtrar · cada painel tem a sua escala`;
      if (!M.ref) nota.textContent = `${M.rot} · ${M.nota} · cada painel tem a sua escala`;
      linha.textContent = "";
      avals.forEach(([k, tit, sub], j) => {
        const p = h("div", { class: "fin-painel" }, linha);
        const tt = h("div", { class: "fin-tit" }, p, tit);
        h("span", null, tt, sub);
        const g = h("div", { class: "fin-g" }, p);
        const vals = ORDEM_FILTROS.map(f => D.finais[f][k][M.campo]), best = M.melhor(...vals);
        const ref = M.ref ? M.ref(k) : null, dmax = Math.max(...vals, ref ? ref.valor : 0) * 1.22;
        if (ref) ref.esquerda = ref.valor / dmax > 0.5;     // rótulo da referência para dentro do painel
        barras(g, {
          itens: ORDEM_FILTROS.map((f, i) => ({
            rotulo: FILTRO[f].curto, marca: null, valor: vals[i], cor: FILTRO[f].cor, forte: vals[i] === best,
            dica: `${mk(FILTRO[f].cor)}<b>${esc(FILTRO[f].nome)}</b> · ${esc(tit)}<br>${M.rot}: ${M.fmt(vals[i])}<br>` +
              `e-mails reais na lixeira: ${F.pct(D.finais[f][k].real_lixeira_custo)} · spam que passou: ${F.pct(D.finais[f][k].spam_passou_custo)}<br>` +
              `TCR ${F.dec(D.finais[f][k].tcr, 2)} (${D.finais[f][k].tcr >= 1 ? "compensa filtrar" : "não filtrar sai mais barato"})`,
          })),
          dom: [0, dmax], fmt: M.fmt, margemE: j === 0 ? 122 : 4, margemD: 52, espessura: 18,
          refs: ref ? [ref] : null, margemT: ref ? 20 : 2,
          animar: A && !api.semMovimento, atraso: i => j * 380 + i * 70, rotulo: `${M.rot}: ${tit}`,
        });
      });
    }
    desenhar(o.animar);
  }

  return Object.assign(api, {
    COR, FAMILIA, ORDEM_FAMILIAS, MEMBRO, FILTRO, ORDEM_FILTROS, REFERENCIA, F, corMembro, corMetodo,
    chaveMetodo, nomeMetodo, ordemMetodo, tabela,
    h, s, t, svgEm, anim, contar, dica, mk, esc, legenda, legendaFamilias, ticksLin, ticksLog,
    barras, corrida, halteres, dispersao, mapaTriangular, curvas, tira, envelopes, empilhada, caminho, simulador, finalTres,
  });
})();
