/* deck.js — navegação, escala, notas do apresentador, cronômetro, glossário, abas e impressão.
   Teclas: → ← espaço PageUp PageDown Home End · F tela cheia · N notas · G glossário · T linha do tempo
           R zera o cronômetro · P imprimir · ? atalhos · Esc fecha painéis.
   URL: #5 abre o slide 5 · ?imprimir desenha tudo sem animação · ?verificar mede estouro de geometria. */
"use strict";
const Deck = (() => {
  const W = 1280, H = 720;
  const params = new URLSearchParams(location.search);
  let slides = [], quadros = [], atual = -1, inicio = Date.now();
  const ganchos = {}, desenhados = new Set();
  let hud, painel, ajuda, glossario = null, tRolagem = 0;

  const el = (tag, at, pai, html) => {
    const e = document.createElement(tag);
    for (const k in at || {}) e.setAttribute(k, at[k]);
    if (html != null) e.innerHTML = html;
    if (pai) pai.appendChild(e);
    return e;
  };

  /* ------------------------------------------------------------ gráficos por slide */
  function registrar(id, fn) { ganchos[id] = fn; }
  function desenhar(sl, animar) {
    const fn = ganchos[sl.dataset.id];
    if (!fn) return;
    try { fn(sl, { animar: animar && !G.semMovimento }); }
    catch (e) { console.error("erro ao desenhar o slide", sl.dataset.id, e); }
    desenhados.add(sl);
  }

  /* ------------------------------------------------------------ escala e posição */
  function escalar() {
    const q = quadros.find(x => x.offsetWidth) || quadros[0];
    if (!q) return;
    const e = Math.min(q.clientWidth / W, q.clientHeight / H);
    document.documentElement.style.setProperty("--escala", e.toFixed(4));
  }
  const indiceDaRolagem = () =>
    Math.max(0, Math.min(slides.length - 1, Math.round(window.scrollY / Math.max(1, window.innerHeight))));

  function ir(i) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    quadros[i].scrollIntoView({ block: "start", behavior: "instant" });
    marcar(i);
  }
  function irPara(id) {
    const i = slides.findIndex(s => s.dataset.id === id);
    if (i >= 0) ir(i);
  }
  function marcar(i) {
    if (i === atual) return;
    atual = i;
    history.replaceState(null, "", location.pathname + location.search + "#" + (i + 1));
    desenhar(slides[i], true);
    atualizarHud();
    atualizarNotas();
  }

  /* ------------------------------------------------------------ HUD, cronômetro, notas, ajuda */
  const relogio = () => {
    const s = Math.floor((Date.now() - inicio) / 1000);
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  };
  const titulo = sl => (sl.querySelector("h1, h2") || {}).textContent?.trim().replace(/\s+/g, " ") || "";

  function atualizarHud() {
    if (!hud) return;
    hud.querySelector(".hud-pos").textContent = `${atual + 1} / ${slides.length}`;
    hud.querySelector(".hud-rel").textContent = relogio();
    if (painel) painel.querySelector(".pn-relogio").textContent = relogio();
  }
  function atualizarNotas() {
    if (!painel || atual < 0) return;
    const sl = slides[atual], n = sl.querySelector(".notas");
    painel.querySelector(".pn-pos").textContent = `slide ${atual + 1} / ${slides.length}`;
    painel.querySelector(".pn-tit").textContent = titulo(sl);
    painel.querySelector(".pn-texto").innerHTML = n ? n.innerHTML : "<p>(sem notas)</p>";
    const prox = slides[atual + 1];
    painel.querySelector(".pn-prox").innerHTML = prox ? "Próximo: <b></b>" : "Último slide.";
    if (prox) painel.querySelector(".pn-prox b").textContent = titulo(prox);
  }
  function alternarNotas() {
    document.body.classList.toggle("com-notas");
    escalar();
    ir(atual);
    atualizarNotas();
  }
  function telaCheia() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
    else document.exitFullscreen?.();
  }

  /* ------------------------------------------------------------ glossário */
  function montarGlossario() {
    const fonte = document.getElementById("glossario-dados");
    if (!fonte) return null;
    const brutos = JSON.parse(fonte.textContent);
    const ordemGrupos = [...new Set(brutos.map(i => i.g))];          // grupos na ordem em que foram escritos
    const itens = [...brutos].sort((a, b) => a.t.localeCompare(b.t, "pt-BR", { sensitivity: "base" }));
    const p = el("aside", { id: "glossario", class: "no-print", "aria-label": "Glossário" }, document.body,
      '<div class="gp-topo"><h3>Glossário</h3><input type="search" placeholder="buscar termo…" aria-label="buscar no glossário"></div><div class="gp-lista"></div>');
    const lista = p.querySelector(".gp-lista"), busca = p.querySelector("input");
    const linhas = itens.map(it => {
      const d = el("div", { class: "gp-item", "data-t": it.t }, lista);
      el("b", {}, d).textContent = it.t;
      if (it.n) el("i", {}, d).textContent = it.n;
      el("div", {}, d).textContent = it.d;
      return [it, d];
    });
    busca.addEventListener("input", () => {
      const q = busca.value.trim().toLowerCase();
      linhas.forEach(([it, d]) => { d.hidden = q && !(it.t + " " + (it.n || "") + " " + it.d).toLowerCase().includes(q); });
    });
    const abrir = termo => {
      document.body.classList.add("com-glossario");
      busca.value = ""; linhas.forEach(([, d]) => { d.hidden = false; d.classList.remove("alvo"); });
      const alvo = linhas.find(([it]) => it.t === termo);
      if (alvo) { alvo[1].classList.add("alvo"); alvo[1].scrollIntoView({ block: "center" }); }
    };
    document.addEventListener("click", e => {
      const a = e.target.closest("[data-g]");
      if (a) { e.preventDefault(); abrir(a.dataset.g); }
    });

    // índice clicável num slide (.gl-indice) + área de definição (.gl-def)
    document.querySelectorAll(".gl-indice").forEach(ind => {
      const def = ind.closest(".slide").querySelector(".gl-def");
      const grupos = ordemGrupos;
      const mostrar = (it, bt) => {
        ind.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === bt)));
        def.querySelector(".termo").textContent = it.t;
        def.querySelector(".nome").textContent = it.n || "";
        def.querySelector(".def").textContent = it.d;
      };
      let primeiro = null;
      grupos.forEach(g => {
        const bloco = el("div", { class: "gl-grupo" }, ind);
        el("h4", {}, bloco).textContent = g;
        const bts = el("div", { class: "botoes" }, bloco);
        itens.filter(i => i.g === g).forEach(it => {
          const b = el("button", { type: "button" }, bts);
          b.textContent = it.t;
          b.addEventListener("click", () => mostrar(it, b));
          if (!primeiro || it === brutos[0]) primeiro = [it, b];
        });
      });
      if (primeiro) mostrar(...primeiro);
    });
    // listas completas para impressão (.gl-lista): os itens são divididos entre as listas, em ordem
    const listas = [...document.querySelectorAll(".gl-lista")];
    const passo = Math.ceil(itens.length / Math.max(1, listas.length));
    listas.forEach((ls, k) => {
      const fatia = itens.slice(k * passo, (k + 1) * passo);
      fatia.forEach(it => {
        const d = el("div", {}, ls);
        el("b", {}, d).textContent = it.t;
        d.append(it.n ? ` (${it.n}). ` : ". ", it.d);
      });
      const faixa = ls.closest(".slide").querySelector(".gl-faixa");
      if (faixa && fatia.length) faixa.textContent = `de ${fatia[0].t} a ${fatia[fatia.length - 1].t}`;
    });
    return { abrir, alternar: () => document.body.classList.toggle("com-glossario") };
  }

  /* ------------------------------------------------------------ abas genéricas e links internos */
  function montarAbas() {
    document.querySelectorAll("[data-abas]").forEach(grupo => {
      const botoes = [...grupo.querySelectorAll("[data-aba]")];
      const paineis = [...grupo.querySelectorAll("[data-painel]")];
      const ativar = k => {
        botoes.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.aba === k)));
        paineis.forEach(p => p.classList.toggle("ativo", p.dataset.painel === k));
      };
      botoes.forEach(b => b.addEventListener("click", () => ativar(b.dataset.aba)));
      if (botoes[0]) ativar(botoes[0].dataset.aba);
    });
    document.addEventListener("click", e => {
      const a = e.target.closest("[data-ir]");
      if (a) { e.preventDefault(); irPara(a.dataset.ir); }
      const b = e.target.closest("button");
      if (b && e.detail > 0) b.blur();          // clique de mouse não deixa o foco preso no botão
    });
    document.addEventListener("pointerup", e => {
      if (e.target.matches?.('input[type="range"]')) setTimeout(() => e.target.blur(), 0);
    });
  }

  /* ------------------------------------------------------------ teclado */
  function tecla(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const alvo = e.target, k = e.key;
    if (alvo.matches?.("input, textarea, select")) { if (k === "Escape") alvo.blur(); return; }
    if (alvo.matches?.("button") && (k === " " || k === "Enter")) return;
    if (["ArrowRight", "ArrowDown", "PageDown"].includes(k) || (k === " " && !e.shiftKey)) { e.preventDefault(); ir(atual + 1); }
    else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(k) || (k === " " && e.shiftKey)) { e.preventDefault(); ir(atual - 1); }
    else if (k === "Home") { e.preventDefault(); ir(0); }
    else if (k === "End") { e.preventDefault(); ir(slides.length - 1); }
    else if (k === "f" || k === "F") telaCheia();
    else if (k === "n" || k === "N") alternarNotas();
    else if ((k === "g" || k === "G") && glossario) glossario.alternar();
    else if (k === "t" || k === "T") irPara("linha-do-tempo");
    else if (k === "r" || k === "R") { inicio = Date.now(); atualizarHud(); }
    else if (k === "p" || k === "P") window.print();
    else if (k === "?" || k === "h" || k === "H") ajuda.classList.toggle("aberto");
    else if (k === "Escape") {
      ajuda.classList.remove("aberto");
      document.body.classList.remove("com-glossario");
    }
  }

  /* ------------------------------------------------------------ impressão */
  function prepararImpressao() {
    G.finalizar();
    document.getAnimations?.().forEach(a => { try { a.finish(); } catch (_) { /* animação infinita */ } });
    slides.forEach(sl => { if (!desenhados.has(sl)) desenhar(sl, false); });
  }

  /* ------------------------------------------------------------ verificação de geometria (?verificar) */
  function verificarGeometria() {
    document.documentElement.style.setProperty("--escala", "1");
    const linhas = [];
    slides.forEach((sl, i) => {
      const r0 = sl.getBoundingClientRect(), corpo = sl.querySelector(".corpo");
      const rc = corpo ? corpo.getBoundingClientRect() : null;
      const problemas = new Set();
      sl.querySelectorAll("*").forEach(e => {
        if (e.closest(".notas") || e.ownerSVGElement) return;
        const r = e.getBoundingClientRect();
        if (!r.width && !r.height) return;
        const nome = e.tagName.toLowerCase() + (e.className && typeof e.className === "string" ? "." + e.className.trim().split(/\s+/).join(".") : "");
        if (r.right - r0.right > 1 || r.bottom - r0.bottom > 1 || r0.left - r.left > 1 || r0.top - r.top > 1) {
          problemas.add(`fora do slide: ${nome} (${Math.round(r.right - r0.left)}×${Math.round(r.bottom - r0.top)})`);
          e.classList.add("estoura");
        } else if (rc && corpo.contains(e) && r.bottom - rc.bottom > 1) {
          problemas.add(`invade o rodapé: ${nome} (+${Math.round(r.bottom - rc.bottom)}px)`);
          e.classList.add("estoura");
        }
        const ov = getComputedStyle(e).overflow;
        if (ov !== "visible" && e.tagName !== "svg" && (e.scrollHeight - e.clientHeight > 2 || e.scrollWidth - e.clientWidth > 2)) {
          problemas.add(`conteúdo cortado: ${nome} (${e.scrollWidth}×${e.scrollHeight} em ${e.clientWidth}×${e.clientHeight})`);
        }
      });
      linhas.push(`${i + 1}\t${sl.dataset.id}\t${problemas.size ? [...problemas].join(" | ") : "ok"}`);
    });
    const pre = el("pre", { id: "relatorio-geometria" }, document.body);
    pre.textContent = `slides=${slides.length}\n` + linhas.join("\n");
    console.log(pre.textContent);
  }

  /* ------------------------------------------------------------ início */
  function iniciar() {
    slides = [...document.querySelectorAll(".slide")];
    const rodape = document.body.dataset.rodape || "";
    // capítulos da história: body[data-capitulos="nome|nome|…"] e slide[data-capitulo="k"] (k a partir de 1)
    const caps = (document.body.dataset.capitulos || "").split("|").filter(Boolean);
    slides.forEach(sl => {
      const q = el("div", { class: "quadro" });
      sl.before(q);
      q.appendChild(sl);
      quadros.push(q);
      const k = parseInt(sl.dataset.capitulo, 10);
      if (caps.length && k >= 1 && k <= caps.length) {
        const txt = `Capítulo ${k} de ${caps.length} · ${caps[k - 1]}`;
        sl.querySelectorAll(".cap-num").forEach(c => { c.textContent = `Capítulo ${k} de ${caps.length}`; });
        if (!sl.classList.contains("sem-rodape")) {
          const r = el("footer", { class: "rod" }, sl);
          const p = el("span", { class: "cap-prog" }, r);
          const segs = el("span", { class: "segs", "aria-hidden": "true" }, p);
          caps.forEach((nome, i) => { const s = el("i", { class: i + 1 < k ? "feito" : i + 1 === k ? "atual" : "", title: nome }, segs); void s; });
          el("b", {}, p).textContent = txt;
          el("span", { class: "pag" }, r);
        }
      } else if (!sl.classList.contains("sem-rodape")) {
        const r = el("footer", { class: "rod" }, sl);
        el("span", {}, r).textContent = rodape;
        el("span", { class: "pag" }, r);
      }
    });
    quadros[quadros.length - 1].classList.add("ultimo");
    slides.forEach((sl, i) => { const p = sl.querySelector(".pag"); if (p) p.textContent = `${i + 1} / ${slides.length}`; });

    hud = el("div", { class: "hud no-print" }, document.body,
      '<span class="hud-pos"></span> · <span class="hud-rel"></span> · <button type="button" title="atalhos (?)">?</button>');
    hud.querySelector("button").addEventListener("click", () => ajuda.classList.toggle("aberto"));
    painel = el("aside", { id: "painel-notas", class: "no-print" }, document.body,
      '<div class="pn-topo"><span class="pn-pos"></span><span>R zera</span></div><div class="pn-relogio"></div>' +
      '<p class="pn-tit" style="font-weight:650;margin:10px 0 12px;color:#fff"></p><div class="pn-texto"></div><div class="pn-prox"></div>');
    glossario = montarGlossario();
    const temLinha = !!document.querySelector('[data-id="linha-do-tempo"]');
    ajuda = el("div", { id: "ajuda", class: "no-print" }, document.body,
      '<div class="caixa"><h3>Atalhos</h3><dl>' +
      "<dt>→ espaço</dt><dd>próximo slide</dd><dt>← ⇧espaço</dt><dd>slide anterior</dd>" +
      "<dt>Home End</dt><dd>primeiro / último</dd><dt>F</dt><dd>tela cheia</dd>" +
      "<dt>N</dt><dd>notas do apresentador e cronômetro</dd><dt>R</dt><dd>zera o cronômetro</dd>" +
      (glossario ? "<dt>G</dt><dd>glossário (ou clique numa sigla sublinhada)</dd>" : "") +
      (temLinha ? "<dt>T</dt><dd>volta à linha do tempo</dd>" : "") +
      "<dt>P</dt><dd>imprimir / salvar PDF (margens: nenhuma; gráficos de fundo: ligado)</dd><dt>Esc</dt><dd>fecha painéis</dd></dl></div>");
    ajuda.addEventListener("click", e => { if (e.target === ajuda) ajuda.classList.remove("aberto"); });
    montarAbas();

    escalar();
    addEventListener("resize", () => { escalar(); if (atual >= 0) quadros[atual].scrollIntoView({ block: "start", behavior: "instant" }); });
    document.addEventListener("fullscreenchange", () => { escalar(); setTimeout(() => quadros[atual]?.scrollIntoView({ block: "start", behavior: "instant" }), 60); });
    addEventListener("keydown", tecla);
    addEventListener("beforeprint", prepararImpressao);

    // Todos os gráficos são desenhados já no início, sem animação: a impressão e a rolagem nunca encontram
    // slide vazio. Ao entrar num slide, o gráfico é redesenhado com animação.
    G.semMovimento = params.has("imprimir") || params.has("verificar") || matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (params.has("notas")) { document.body.classList.add("com-notas"); escalar(); }   // ?notas abre já em modo apresentador
    slides.forEach(sl => desenhar(sl, false));
    if (params.has("slide")) {                     // ?slide=N mostra só um slide (exportar imagem por slide)
      const n = parseInt(params.get("slide"), 10) - 1;
      quadros.forEach((q, i) => { if (i !== n) q.style.display = "none"; });
      atual = n; atualizarHud(); atualizarNotas();
      desenhar(slides[n], true);
      document.body.classList.add("ocioso");
      return;
    }
    const alvo = parseInt(location.hash.slice(1), 10);
    ir(Number.isFinite(alvo) ? alvo - 1 : 0);
    addEventListener("scroll", () => { clearTimeout(tRolagem); tRolagem = setTimeout(() => marcar(indiceDaRolagem()), 90); }, { passive: true });
    setInterval(atualizarHud, 1000);
    // HUD e cursor somem após alguns segundos parados (não cobrem o número da página ao apresentar)
    let tOcioso = 0;
    const acordar = () => { document.body.classList.remove("ocioso"); clearTimeout(tOcioso); tOcioso = setTimeout(() => document.body.classList.add("ocioso"), 2500); };
    ["pointermove", "keydown", "pointerdown"].forEach(ev => addEventListener(ev, acordar, { passive: true }));
    acordar();
    if (params.has("verificar")) setTimeout(verificarGeometria, 200);
  }

  return { registrar, iniciar, ir, irPara, get atual() { return atual; } };
})();
