#!/usr/bin/env python3
"""Monta os decks HTML autocontidos do Projeto 3 (só biblioteca padrão).

Uso, na raiz do projeto:
    .venv\\Scripts\\python.exe apresentacao\\build.py

Entrada
    resultados/metricas_finais.json   fonte única dos números
    docs/REGISTRO_DECISOES.md         só os dois limites da regra pré-registrada D10
    apresentacao/fonte/               templates editáveis (comum/estilo.css, comum/deck.js, comum/graficos.js,
                                      banca.html, equipe.html, index.html)
Saída
    apresentacao/banca/index.html     deck técnico (banca)
    apresentacao/equipe/index.html    deck didático (equipe)
    apresentacao/index.html           página de entrada (GitHub Pages)

Regras dos templates
    * Número exibido entra como {{caminho|formato}}; o caminho é resolvido em DADOS na hora do build.
      Caminho inexistente interrompe o build. Formatos: int, mcc, d1..d4, pct, custo, s3, s4, spct, p, n, txt.
    * <link rel="stylesheet" href="..."> e <script src="..."></script> são trocados pelo conteúdo do arquivo.
    * <!--@DADOS@--> vira  <script>const DADOS = {...};</script>.

Depois de gerar, o build verifica: tokens pendentes, recursos externos, a palavra em inglês para "e-mail real"
fora do glossário e dígitos digitados à mão no texto dos slides. Qualquer falha devolve código de saída 1.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FONTE = AQUI / "fonte"
METRICAS = RAIZ / "resultados" / "metricas_finais.json"
REGISTRO = RAIZ / "docs" / "REGISTRO_DECISOES.md"

DECKS = {"banca": "banca.html", "equipe": "equipe.html"}


# --------------------------------------------------------------------------------------------- extração
def arred(o, sig: int = 6):
    """Arredonda floats a `sig` algarismos significativos (reduz o tamanho do JSON embutido)."""
    if isinstance(o, float):
        return float(f"{o:.{sig}g}")
    if isinstance(o, list):
        return [arred(x, sig) for x in o]
    if isinstance(o, dict):
        return {k: arred(v, sig) for k, v in o.items()}
    return o


def regra_d10() -> dict:
    """Os dois limites da regra de D10 não estão no JSON: são lidos do registro de decisões."""
    texto = REGISTRO.read_text(encoding="utf-8")
    sec = texto[texto.index("### D10"):]
    p = re.search(r"P\(redução\)\s*≥\s*([0-9]+,[0-9]+)", sec)
    t = re.search(r"não piora o MCC[^.]*?mais de\s*([0-9]+,[0-9]+)", sec)
    if not (p and t):
        raise SystemExit("ERRO: não achei os limites da regra D10 em docs/REGISTRO_DECISOES.md")
    return {"p_min": float(p.group(1).replace(",", ".")), "tol_mcc": float(t.group(1).replace(",", "."))}


def extrair(R: dict) -> dict:
    b, ns, st, ge, tb, ce = (R["base"], R["nao_supervisionado"], R["stacking"], R["genetico"],
                             R["terceira_base"], R["contra_estudo"])
    pr = R["protocolo"]
    comp15 = {c["B"]: c for c in st["comparacoes_15_folds"]}
    comp_ga = {(c["A"], c["B"]): c for c in ge["comparacoes"]}
    D = {
        "meta": {k: R["meta"][k] for k in ("custo_fp", "custo_fn", "seed_teste")},
        "base": {k: v for k, v in b.items() if k not in ("origem", "contagem")},
        "dummy_acuracia": R["metricas_didaticas"]["dummy_acuracia"],
        "palavras": R["palavras"],
        "dominio": R["dominio"],
        "dominio_d": {d["palavra"]: d for d in R["dominio"]},
        "protocolo": {
            "teste_n": pr["teste_trancado"]["n"], "teste_real": pr["teste_trancado"]["email_real"],
            "teste_spam": pr["teste_trancado"]["spam"], "teste_usos": pr["teste_trancado"]["usos"],
            "folds": pr["folds_confirmacao"],
            "piso": [{"folds": int(k), "dif": v} for k, v in pr["piso_mcc_detectavel"].items()],
        },
        "ns": {
            "svd2d": [[round(x, 3), round(y, 3), int(c)] for x, y, c in ns["svd2d"]],
            "variancia_svd2": ns["variancia_svd"]["2"],
            "variancia_svd100": ns["variancia_svd"]["100"],
            "kmeans": ns["kmeans"], "kmeans_ari": ns["kmeans_ari"], "hierarquico_ari": ns["hierarquico_ari"],
            "nmf": ns["nmf_topicos"], "anomalia": ns["anomalia"], "so_real": ns["anomalia_so_email_real"],
        },
        "torneio": R["torneio"],
        "achados": {
            "lr_antes": R["achados"]["lr_mcc_antes"], "lr_depois": R["achados"]["lr_mcc_depois"],
            "knn_antes": R["achados"]["knn_mcc_antes"], "knn_depois": R["achados"]["knn_mcc_depois"],
            "votacao": R["achados"]["votacao_mcc"],
            "representacoes": [{"rep": k, "lr": v["Reg. Logística"], "knn": v["kNN cosseno"]}
                               for k, v in R["achados"]["tfidf_log"].items()],
        },
        "rodada2": {"comparacoes": R["rodada2"]["comparacoes"]},
        "stacking": {
            "variantes": st["variantes"],
            "variantes_d": {v["nome"]: v for v in st["variantes"]},
            "erros_modelos": st["erros_em_comum"]["modelos"], "erros_matriz": st["erros_em_comum"]["matriz"],
            "tabela_erros": [{"membro": t["familia"], "erros": t["erros"], "so_ele": t["só ele erra"],
                              "resgata": t["acerta quando a maioria (≥4) erra"]} for t in st["tabela_erros"]],
            "pesos": sorted(({"membro": k, "peso": v} for k, v in st["pesos_meta"].items()),
                            key=lambda d: -d["peso"]),
            "confirmacao": st["confirmacao_20_folds"],
            "s7_x_lsvc": st["comparacao"][0],
            "s7_x_s13": comp15["Stacking 13 (todos)"],
            "s7_x_s4": comp15["Stacking 4 (LR+NB+RF+LGBM)"],
            "externo": st["externo_exp02"],
        },
        "ga": {
            "combinacoes": ge["combinacoes"], "avaliacoes": ge["avaliacoes"],
            "curva_ga": ge["curva_ga"], "curva_aleatoria": ge["curva_aleatoria"],
            "mcc_padrao": ge["mcc_busca_padrao"], "mcc_ga": ge["mcc_busca_ga"],
            "mcc_aleatoria": ge["mcc_busca_aleatoria"],
            "ganho_busca": ge["ganho_busca"], "ganho_novas": ge["ganho_sementes_novas"],
            "fracao_otimismo": ge["fracao_otimismo"], "sensibilidade": ge["sensibilidade"],
            "piso": ge["piso"], "confirmacao": ge["confirmacao"],
            "ga_x_padrao": comp_ga[("Stacking 7 do GA (Especialista)", "Stacking 7 padrão")],
            "historico": ge["historico"],
        },
        "corrida": R["corrida"]["resumo"],
        "tabpfn": {
            "versao_nuvem": R["tabpfn"]["versao_nuvem"], "versao_local": R["tabpfn"]["versao_local"],
            "interno": R["tabpfn"]["interno"], "externo": R["tabpfn"]["externo"],
            "interno_d": {r["metodo"]: r for r in R["tabpfn"]["interno"]},
            "externo_d": {r["metodo"]: r for r in R["tabpfn"]["externo"]},
            "local_x_nuvem": R["tabpfn"]["local_x_nuvem"],
        },
        "terceira": {
            "n": tb["n"], "n_email_real": tb["n_email_real"], "n_spam": tb["n_spam"],
            "metodos": tb["metodos"], "metodos_d": {r["metodo"]: r for r in tb["metodos"]},
            "decisao": tb["decisao"], "controle": tb["controle_dentro_da_base"],
            "cobertura": [{"base": k, "fracao": v} for k, v in tb["cobertura_vocabulario"].items()],
            "regra": regra_d10(),
        },
        "finais": R["finais"],
        "simulador": R["simulador_limiar"],
        "contra": {
            "permutacao": ce["permutacao"], "ablacao": ce["ablacao"],
            "n_palavras_enron": len(ce["palavras_enron"]), "palavras_enron": ce["palavras_enron"],
            "curva": ce["curva_aprendizado"], "sensibilidade_custo": ce["sensibilidade_custo"],
            "vencedor_por_custo": ce["vencedor_por_custo"],
            "reais_lixeira": [dict(r, texto=re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", r["texto"])[:240])
                              for r in ce["emails_reais_na_lixeira"]],
            "n_reais_lixeira": ce["n_emails_reais_na_lixeira"], "pesos_palavras": ce["pesos_palavras"],
        },
        "exemplos": R["exemplos_demo"],
    }
    D["der"] = derivados(D)
    return arred(D)


def derivados(D: dict) -> dict:
    """Valores calculados a partir do JSON (razões, diferenças, contagens). Nada digitado."""
    b, st, ga, tp, tb, ce = D["base"], D["stacking"], D["ga"], D["tabpfn"], D["terceira"], D["contra"]
    tor = D["torneio"]
    unicos = [t for t in tor if t["familia"] not in ("combinação", "referência")]
    var = st["variantes_d"]
    piso20 = next(p["dif"] for p in D["protocolo"]["piso"] if p["folds"] == D["protocolo"]["folds"])
    curva = {(c["metodo"], c["n"]): c["mean"] for c in ce["curva"]}
    ns_ = sorted({c["n"] for c in ce["curva"]})
    tab_i, tab_e = tp["interno_d"], tp["externo_d"]
    tm = tb["metodos_d"]
    perm = ce["permutacao"]["mcc_permutado"]
    erros = {t["membro"]: t for t in st["tabela_erros"]}
    pesos = {p["membro"]: p["peso"] for p in st["pesos"]}
    dummy = next(t for t in tor if t["familia"] == "referência")
    return {
        "frac_spam": b["n_spam"] / b["n_emails"],
        "frac_real": b["n_email_real"] / b["n_emails"],
        "frac_teste": D["protocolo"]["teste_n"] / b["n_emails"],
        "n_dev": b["n_emails"] - D["protocolo"]["teste_n"],
        "dummy_mcc": dummy["mcc"],
        "n_torneio": len(tor),
        "n_familias": len(st["pesos"]),
        "melhor_unico_nome": unicos[0]["modelo"], "melhor_unico_mcc": unicos[0]["mcc"],
        "ganho_lr": D["achados"]["lr_depois"] - D["achados"]["lr_antes"],
        "ganho_knn": D["achados"]["knn_depois"] - D["achados"]["knn_antes"],
        "s7_mcc": var["Stacking 7 (1 por família)"]["mcc"],
        "s13_mcc": var["Stacking 13 (todos)"]["mcc"],
        "s13_tempo_extra": var["Stacking 13 (todos)"]["treino_s"] / var["Stacking 7 (1 por família)"]["treino_s"] - 1,
        "votacao13_mcc": var["Média 13 (votação)"]["mcc"],
        "knn_mcc_sozinho": var["kNN cosseno"]["mcc"],
        "knn_peso": pesos["kNN cosseno"], "nb_peso": pesos["Naive Bayes"],
        "knn_so_ele": erros["kNN cosseno"]["so_ele"], "knn_resgata": erros["kNN cosseno"]["resgata"],
        "lr_resgata": erros["Reg. Logística"]["resgata"],
        "jaccard_lr_svm": st["erros_matriz"][0][1],
        "jaccard_min": min(st["erros_matriz"][i][j] for i in range(7) for j in range(i)),
        "piso20": piso20,
        "piso_razao": piso20 / ga["ganho_novas"],
        "piso_1000": next(p["dif"] for p in D["protocolo"]["piso"] if p["folds"] == 1000),
        "ga_menos_aleatoria": ga["mcc_ga"] - ga["mcc_aleatoria"],
        "ga_sensib_max": max(s["amplitude"] for s in ga["sensibilidade"]),
        "tab_mcc": tab_i["TabPFN-3.5-Plus"]["mcc"],
        "tab_custo_int": tab_i["TabPFN-3.5-Plus"]["custo_por_mil"],
        "lsvc_mcc_int": tab_i["LinearSVC otimizado"]["mcc"],
        "s8_mcc_int": tab_i["Stacking 8 padrão (+TabPFN)"]["mcc"],
        "s7_mcc_int": tab_i["Stacking 7 padrão"]["mcc"],
        "tab_custo_ext": tab_e["TabPFN-3.5-Plus"]["custo_por_mil"],
        "tab_lixeira_ext": tab_e["TabPFN-3.5-Plus"]["real_lixeira_custo"],
        "tab_passou_ext": tab_e["TabPFN-3.5-Plus"]["spam_passou_custo"],
        "tab_prauc_ext": tab_e["TabPFN-3.5-Plus"]["pr_auc"],
        "s7_custo_ext": tab_e["Stacking 7 padrão"]["custo_por_mil"],
        "s8_custo_ext": tab_e["Stacking 8 padrão (+TabPFN)"]["custo_por_mil"],
        "lsvc_custo_ext": tab_e["LinearSVC otimizado"]["custo_por_mil"],
        "s8_reducao_ext": 1 - tab_e["Stacking 8 padrão (+TabPFN)"]["custo_por_mil"] / tab_e["Stacking 7 padrão"]["custo_por_mil"],
        "sa_s8_custo": tm["Stacking 8 padrão (+TabPFN)"]["custo_por_mil"],
        "sa_s7_custo": tm["Stacking 7 padrão (Generalista atual)"]["custo_por_mil"],
        "sa_s7_lixeira": tm["Stacking 7 padrão (Generalista atual)"]["real_lixeira_custo"],
        "sa_tab_custo": tm["TabPFN-3.5-Plus sozinho"]["custo_por_mil"],
        "sa_lsvc_custo": tm["LinearSVC padrão"]["custo_por_mil"],
        "sa_lsvc_mcc": tm["LinearSVC padrão"]["mcc_custo"],
        "sa_mcc_min": min(r["mcc_custo"] for r in tb["metodos"]),
        "sa_mcc_max": max(r["mcc_custo"] for r in tb["metodos"]),
        "perm_media": sum(perm) / len(perm),
        "perm_max": max(perm),
        "n_perm": len(perm),
        "curva_n_min": ns_[0], "curva_n_max": ns_[-1],
        "curva_s7_min": curva[("Stacking 7 padrão", ns_[0])], "curva_s7_max": curva[("Stacking 7 padrão", ns_[-1])],
        "curva_knn_min": curva[("kNN cosseno", ns_[0])], "curva_knn_max": curva[("kNN cosseno", ns_[-1])],
        "curva_tab_min": curva[("TabPFN-3.5-Plus", ns_[0])], "curva_tab_max": curva[("TabPFN-3.5-Plus", ns_[-1])],
        "abl_sa_todas": ce["ablacao"][0]["mcc_spamassassin"], "abl_sa_sem": ce["ablacao"][1]["mcc_spamassassin"],
        "abl_caixa_todas": ce["ablacao"][0]["mcc_esta_caixa"], "abl_caixa_sem": ce["ablacao"][1]["mcc_esta_caixa"],
        "ex2_prob": D["exemplos"][2]["etapas"]["stacking7_padrao"]["prob_spam"],
        "ex2_cobertura": D["exemplos"][2]["cobertura"],
    }


# --------------------------------------------------------------------------------------------- formatos
def _num(x: float, casas: int) -> str:
    s = f"{abs(x):.{casas}f}"
    neg = x < 0 and float(s) != 0
    return ("−" if neg else "") + s.replace(".", ",")


def _int(x) -> str:
    v = int(round(x))
    a = f"{abs(v):,}".replace(",", ".") if abs(v) >= 10000 else str(abs(v))
    return ("−" if v < 0 else "") + a


def _sinal(x: float, casas: int) -> str:
    return ("+" if x > 0 and float(f"{x:.{casas}f}") != 0 else "") + _num(x, casas)


FORMATOS = {
    "int": _int,
    "n": lambda x: str(int(x)),
    "mcc": lambda x: _num(x, 3),
    "d1": lambda x: _num(x, 1), "d2": lambda x: _num(x, 2), "d3": lambda x: _num(x, 3), "d4": lambda x: _num(x, 4),
    "pct": lambda x: _num(100 * x, 1) + "%",
    "custo": lambda x: _num(x, 1),
    "s3": lambda x: _sinal(x, 3), "s4": lambda x: _sinal(x, 4),
    "spct": lambda x: _sinal(100 * x, 1) + "%",
    "p": lambda x: "< 0,001" if x < 0.001 else _num(x, 3),
    "txt": str,
}
TOKEN = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|\s*([a-z0-9]+)\s*)?\}\}")


def resolver(D: dict, caminho: str):
    o = D
    for parte in caminho.split("."):
        if isinstance(o, list) and parte.isdigit() and int(parte) < len(o):
            o = o[int(parte)]
        elif isinstance(o, dict) and parte in o:
            o = o[parte]
        else:
            raise KeyError(caminho)
    return o


def substituir(texto: str, D: dict, origem: str) -> str:
    erros = []

    def troca(m):
        caminho, fmt = m.group(1), m.group(2)
        try:
            v = resolver(D, caminho)
        except KeyError:
            erros.append(f"{origem}: caminho inexistente em DADOS: {caminho}")
            return m.group(0)
        if fmt is None:
            fmt = "int" if isinstance(v, int) and not isinstance(v, bool) else "d3" if isinstance(v, float) else "txt"
        if fmt not in FORMATOS:
            erros.append(f"{origem}: formato desconhecido '{fmt}' em {{{{{caminho}|{fmt}}}}}")
            return m.group(0)
        return FORMATOS[fmt](v)

    saida = TOKEN.sub(troca, texto)
    if erros:
        raise SystemExit("ERRO no build:\n  " + "\n  ".join(erros))
    return saida


# --------------------------------------------------------------------------------------------- montagem
def inline(html: str, pasta: Path) -> str:
    def css(m):
        return "<style>\n" + (pasta / m.group(1)).read_text(encoding="utf-8") + "\n</style>"

    def js(m):
        codigo = (pasta / m.group(1)).read_text(encoding="utf-8")
        return "<script>\n" + codigo.replace("</script", "<\\/script") + "\n</script>"

    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', css, html)
    return re.sub(r'<script src="([^"]+)"></script>', js, html)


def montar(template: Path, D: dict, extra: dict | None = None) -> str:
    html = substituir(template.read_text(encoding="utf-8"), D, template.name)
    for k, v in (extra or {}).items():
        html = html.replace(k, v)
    html = inline(html, template.parent)
    dados = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return html.replace("<!--@DADOS@-->", f"<script>const DADOS = {dados};</script>")


# --------------------------------------------------------------------------------------------- verificação
class _Texto(HTMLParser):
    """Coleta o texto visível (fora de <script> e <style>), com a linha de origem."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pilha, self.trechos = [], []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.pilha.append(tag)

    def handle_endtag(self, tag):
        if self.pilha and tag == self.pilha[-1]:
            self.pilha.pop()

    def handle_data(self, data):
        if not self.pilha and data.strip():
            self.trechos.append((self.getpos()[0], data))


# Nomes próprios e identificadores com dígitos (não são resultados).
NOMES = [r"Enron\s?[1-6](?:\s?[-–]\s?[1-6])?", r"Stacking\s?\d+", r"\bS[478]\b", r"TabPFN-3\.5(?:-Plus|-Fast)?",
         r"\bD\d{1,2}\b", r"Projeto\s3", r"Data Science\s2", r"\bIC95\b", r"RA\s46643", r"\b2D\b",
         r"Metsis et al\., 2006", r"χ²"]
RE_NOMES = re.compile("|".join(NOMES))
RE_PROIBIDA = re.compile(r"\bham\b", re.I)
RE_GLOSSARIO = re.compile(r'<script type="application/json" id="glossario-dados">.*?</script>', re.S)


def digitos_soltos(template: Path) -> list[str]:
    p = _Texto()
    p.feed(TOKEN.sub("§", template.read_text(encoding="utf-8")))
    achados = []
    for linha, t in p.trechos:
        resto = RE_NOMES.sub("", t)
        for m in re.finditer(r"\d+", resto):
            ctx = resto[max(0, m.start() - 25): m.end() + 25].replace("\n", " ").strip()
            achados.append(f"{template.name}:{linha}: '{m.group(0)}' em «{ctx}»")
    return achados


def verificar(saidas: dict[str, str]) -> list[str]:
    erros = []
    for nome, html in saidas.items():
        if TOKEN.search(html):
            erros.append(f"{nome}: token não resolvido: {TOKEN.search(html).group(0)}")
        if re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", html) or re.search(r"@import|url\(\s*['\"]?https?:", html):
            erros.append(f"{nome}: recurso externo (CDN, fonte ou imagem remota)")
        fora = RE_PROIBIDA.findall(RE_GLOSSARIO.sub("", html))
        if fora:
            erros.append(f"{nome}: a palavra proibida aparece {len(fora)} vez(es) fora do glossário")
    for tpl in [FONTE / f for f in [*DECKS.values(), "index.html"]]:
        erros += [f"dígito digitado à mão: {a}" for a in digitos_soltos(tpl)]
    return erros


# --------------------------------------------------------------------------------------------- main
def main() -> int:
    R = json.loads(METRICAS.read_text(encoding="utf-8"))
    D = extrair(R)
    saidas, contagem = {}, {}
    for nome, arq in DECKS.items():
        html = montar(FONTE / arq, D)
        contagem[nome] = len(re.findall(r'<section class="slide[ "]', html))
        destino = AQUI / nome / "index.html"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(html, encoding="utf-8")
        saidas[f"{nome}/index.html"] = html
    extra = {"@N_BANCA@": str(contagem["banca"]), "@N_EQUIPE@": str(contagem["equipe"])}
    idx = montar(FONTE / "index.html", D, extra)
    (AQUI / "index.html").write_text(idx, encoding="utf-8")
    saidas["index.html"] = idx

    for nome, html in saidas.items():
        n = f" · {contagem[nome.split('/')[0]]} slides" if "/" in nome else ""
        print(f"gerado apresentacao/{nome}  ({len(html.encode('utf-8')) / 1024:.0f} KB{n})")
    gl = RE_GLOSSARIO.search(saidas["equipe/index.html"])
    print(f"glossário (equipe): a palavra em inglês aparece {len(RE_PROIBIDA.findall(gl.group(0))) if gl else 0} vez(es), só lá")

    erros = verificar(saidas)
    if erros:
        print("\nVERIFICAÇÃO FALHOU:")
        for e in erros:
            print("  -", e)
        return 1
    print("verificação: ok (tokens resolvidos, nada externo, vocabulário, nenhum dígito digitado à mão)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
