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
      Caminho inexistente interrompe o build. Formatos: int, mcc, d1..d4, pct, pct0, custo, s3, s4, spct, p, n, txt.
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

# Corte padrão de probabilidade (predict_proba > 0,5 em spamfilter/motor.py e genetico.py). É a definição do
# "limiar 0,5", não um resultado; o JSON não traz esse campo (sugestão: meta.limiar_padrao).
LIMIAR_PADRAO = 0.5
# Nível de significância usado nas frases "passa de 0,05" (convenção; o JSON não traz: sugestão meta.alfa).
ALFA = 0.05
# Contagens esperadas do pytest, copiadas de COMO_TESTAR.md ("Teste rápido"): não estão no JSON.
# Se o COMO_TESTAR.md mudar, atualize aqui (sugestão: um campo testes.* no JSON).
TESTES = {"clone": "42 passed, 3 skipped", "com_filtros": "45 passed", "fonte": "COMO_TESTAR.md"}


# --------------------------------------------------------------------------------------------- extração
def arred(o, sig: int = 10):
    """Arredonda floats a `sig` algarismos significativos (reduz o tamanho do JSON embutido).
    Só para o JSON embutido: os tokens {{…}} usam o valor cheio. Com poucos algarismos, o arredondamento
    duplo muda a última casa exibida (207,6496 → 207,650 → "207,7")."""
    if isinstance(o, float):
        return float(f"{o:.{sig}g}")
    if isinstance(o, list):
        return [arred(x, sig) for x in o]
    if isinstance(o, dict):
        return {k: arred(v, sig) for k, v in o.items()}
    return o


def regra_d10() -> dict:
    """Os limites da regra de D10 e o nº de reamostragens do bootstrap não estão no JSON: são lidos do
    registro de decisões (seção D10, o texto pré-registrado)."""
    texto = REGISTRO.read_text(encoding="utf-8")
    sec = texto[texto.index("### D10"):texto.index("### D11")]
    p = re.search(r"P\(redução\)\s*≥\s*([0-9]+,[0-9]+)", sec)
    t = re.search(r"não piora o MCC[^.]*?mais de\s*([0-9]+,[0-9]+)", sec)
    b = re.search(r"bootstrap[^.]*?com\s*([0-9.]+)\s*reamostragens", sec)
    if not (p and t and b):
        raise SystemExit("ERRO: não achei os limites da regra D10 em docs/REGISTRO_DECISOES.md")
    return {"p_min": float(p.group(1).replace(",", ".")), "tol_mcc": float(t.group(1).replace(",", ".")),
            "n_boot": int(b.group(1).replace(".", ""))}


def extrair(R: dict) -> dict:
    b, ns, st, ge, tb, ce = (R["base"], R["nao_supervisionado"], R["stacking"], R["genetico"],
                             R["terceira_base"], R["contra_estudo"])
    pr = R["protocolo"]
    comp15 = {c["B"]: c for c in st["comparacoes_15_folds"]}
    comp_ga = {(c["A"], c["B"]): c for c in ge["comparacoes"]}
    comp_ga_custo = {(c["A"], c["B"]): c for c in ge["comparacoes_custo"]}
    GA_, PAD_, LSO_ = "Stacking 7 do GA (Especialista)", "Stacking 7 padrão", "LinearSVC otimizado"
    D = {
        "meta": {k: R["meta"][k] for k in ("custo_fp", "custo_fn", "seed_teste")},
        "limiar_padrao": LIMIAR_PADRAO, "alfa": ALFA, "testes": TESTES,
        "base": {k: v for k, v in b.items() if k not in ("origem", "contagem")},
        "dummy_acuracia": R["metricas_didaticas"]["dummy_acuracia"],
        "palavras": R["palavras"],
        "dominio": R["dominio"],
        "dominio_d": {d["palavra"]: d for d in R["dominio"]},
        "protocolo": {
            "teste_n": pr["teste_trancado"]["n"], "teste_real": pr["teste_trancado"]["email_real"],
            "teste_spam": pr["teste_trancado"]["spam"], "teste_usos": pr["teste_trancado"]["usos"],
            "teste_usos_descricao": pr["teste_trancado"]["usos_descricao"],
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
        "rodada2": {"comparacoes": R["rodada2"]["comparacoes"], "mcc_15": R["rodada2"]["mcc_15_folds"],
                    "chave_mcc": next(k for k in R["rodada2"] if re.fullmatch(r"mcc_\d+_folds", k))},
        "stacking": {
            "variantes": st["variantes"],
            "variantes_d": {v["nome"]: v for v in st["variantes"]},
            "erros_modelos": st["erros_em_comum"]["modelos"], "erros_matriz": st["erros_em_comum"]["matriz"],
            # P(linha erra | coluna erra), matriz assimétrica; e a média sobre os outros seis, por membro
            "p_erra": st["erros_em_comum"]["p_erra_se_outro_erra"], "erra_junto": st["erros_em_comum"]["erra_junto"],
            "tabela_erros": [{"membro": t["familia"], "erros": t["erros"], "so_ele": t["só ele erra"],
                              "resgata": t["acerta quando a maioria (≥4) erra"], "junto": t["P(erra | outro erra)"],
                              "jaccard": t["Jaccard médio"]} for t in st["tabela_erros"]],
            # pesos do meta-modelo: média e desvio padrão entre os 20 folds
            "pesos": sorted(({"membro": k, "peso": v, "dp": st["pesos_meta_dp"][k]} for k, v in st["pesos_meta"].items()),
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
            "ganho_novas_ic95": ge["ganho_sementes_novas_ic95"],
            "sensibilidade": ge["sensibilidade"],
            "piso": ge["piso"], "piso_par": ge["piso_par"], "piso_poder": ge["piso_poder"],
            "confirmacao": ge["confirmacao"],
            "ga_x_padrao": comp_ga[(GA_, PAD_)],
            "s7_x_lsvc_otim": comp_ga[(PAD_, LSO_)], "ga_x_lsvc_otim": comp_ga[(GA_, LSO_)],
            "n_comparacoes": len(ge["comparacoes"]),
            "custo_ga_x_padrao": comp_ga_custo[(GA_, PAD_)],
            "historico": ge["historico"],
        },
        # comparações de custo nos 20 folds entre filtros finais (t corrigido + Holm + bayesiano); chave "A__B"
        "custo20": {f'{c["A"]}__{c["B"]}': c for c in R["comparacoes_custo_20_folds"]},
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
            "nao_filtrar_custo": tb["nao_filtrar_custo"], "sensibilidade_custo": tb["sensibilidade_custo"],
            "membros": tb["membros"], "membros_d": {r["membro"]: r for r in tb["membros"]},
            "conversor": tb["conversor_spam"], "fontes_spam": tb["fontes_spam"],
            "fontes_spam_d": {r["base"]: r for r in tb["fontes_spam"]},
            "semelhanca": tb["semelhanca_spam"], "por_pasta": tb["por_pasta"],
        },
        "finais": R["finais"],
        "simulador": R["simulador_limiar"],
        "contra": {
            "permutacao": ce["permutacao"], "ablacao": ce["ablacao"], "quase_copias": ce["quase_copias"],
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
    D["der"]["mod"] = modelos_um_a_um(R, D)
    return D


# Os 8 modelos explicados em docs/MODELOS.md: chave curta → (nome no torneio, nome no stacking e nas configurações).
MODELOS = {
    "lr": ("Reg. Logística TF-IDF", "Reg. Logística"),
    "lsvc": ("LinearSVC TF-IDF", None),
    "svm": ("SVM RBF SVD300", "SVM RBF"),
    "knn": ("kNN TF-IDF cosseno", "kNN cosseno"),
    "mlp": ("MLP (rede neural)", "MLP"),
    "nb": ("Naive Bayes Complement", "Naive Bayes"),
    "et": ("Extra Trees", "Extra Trees"),
    "lgbm": ("LightGBM", "LightGBM"),
}


def _cfg(txt: str) -> dict:
    """'k=5, w=uniform' → {'k': '5', 'w': 'uniform'} (os valores continuam texto, vindos do JSON)."""
    return dict(p.split("=", 1) for p in txt.split(", ") if "=" in p)


def modelos_um_a_um(R: dict, D: dict) -> dict:
    """MCC sozinho no torneio, peso no meta-modelo (média dos 20 folds ± dp e filtros entregues), quanto erra
    junto com os outros (P(erra | outro erra)) e configurações. O papel no conselho vira frase a partir das
    posições (nada digitado). O Jaccard não entra: ele favorece quem erra muito."""
    tor = {t["modelo"]: t for t in R["torneio"]}
    pesos, dps = R["stacking"]["pesos_meta"], R["stacking"]["pesos_meta_dp"]
    junto = R["stacking"]["erros_em_comum"]["erra_junto"]
    resg = {t["familia"]: t["acerta quando a maioria (≥4) erra"] for t in R["stacking"]["tabela_erros"]}
    ord_peso = sorted(pesos, key=lambda m: -pesos[m])
    ord_junto = sorted(junto, key=lambda m: junto[m])
    ord_resg = sorted(resg, key=lambda m: -resg[m])
    fin = R["finais"]

    def papel(m):
        rp, rj = ord_peso.index(m) + 1, ord_junto.index(m) + 1
        # sem "no conselho" no fim: a ficha já tem o rótulo "No conselho"; "intermediário" evita confusão com "peso médio"
        p = ("o maior peso médio" if rp == 1 else "peso alto" if rp <= 3 else
             "peso intermediário" if rp <= 5 else "quase nenhum peso" if pesos[m] < 0.1 else "peso baixo")
        j = ("o que menos erra junto com os outros" if rj == 1 else "erra pouco junto com os outros" if rj <= 3 else
             "às vezes erra junto com os outros" if rj <= 5 else "o que mais erra junto com os outros" if rj == len(ord_junto)
             else "erra muito junto com os outros")
        return p, j

    out = {}
    for k, (nt, ns) in MODELOS.items():
        t = tor[nt]
        d = {"nome_torneio": nt, "mcc": t["mcc"], "treino_s": t["treino_s"]}
        if ns:
            pp, pj = papel(ns)
            pe = fin["especialista"]["pesos_meta"][ns]
            d.update(membro=ns, peso=pesos[ns], peso_dp=dps[ns], peso_entregue=fin["generalista_sem_chave"]["pesos_meta"][ns],
                     peso_esp=pe, esp_txt="no Especialista, praticamente desligado" if abs(pe) < 0.1 else "",
                     junto=junto[ns], resgata=resg[ns], ordem_peso=ord_peso.index(ns) + 1,
                     ordem_junto=ord_junto.index(ns) + 1, ordem_resgate=ord_resg.index(ns) + 1,
                     papel_peso=pp, papel_junto=pj,
                     cfg_padrao=fin["generalista_sem_chave"]["config"][ns], cfg_esp=fin["especialista"]["config"][ns],
                     cfg_padrao_d=_cfg(fin["generalista_sem_chave"]["config"][ns]), cfg_esp_d=_cfg(fin["especialista"]["config"][ns]))
        else:
            d.update(cfg_padrao=fin["linha_de_base"]["config"]["LinearSVC"], cfg_padrao_d=_cfg(fin["linha_de_base"]["config"]["LinearSVC"]))
        out[k] = d
    # o melhor modelo único do torneio (a linha de base) e a ordem dos MCC
    unicos = [t for t in R["torneio"] if t["familia"] not in ("combinação", "referência")]
    out["melhor_unico"] = unicos[0]["modelo"]
    return out


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
    NAO_F = "Não filtrar (referência)"
    filtros_sa = [r for r in tb["metodos"] if r["metodo"] != NAO_F]
    perm = ce["permutacao"]
    erros = {t["membro"]: t for t in st["tabela_erros"]}
    pesos = {p["membro"]: p["peso"] for p in st["pesos"]}
    dummy = next(t for t in tor if t["familia"] == "referência")
    fin = D["finais"]
    FIN5 = ["especialista", "generalista", "tabpfn", "generalista_sem_chave", "linha_de_base"]
    STK = ["especialista", "generalista", "generalista_sem_chave"]
    # Jaccard médio dos erros de cada membro com os outros seis (fora da diagonal)
    mods, M = st["erros_modelos"], st["erros_matriz"]
    jac = sorted(({"membro": m, "media": sum(M[i][j] for j in range(len(mods)) if j != i) / (len(mods) - 1)}
                  for i, m in enumerate(mods)), key=lambda d: d["media"])
    resg = sorted(({"membro": t["membro"], "n": t["resgata"]} for t in st["tabela_erros"]), key=lambda d: -d["n"])
    c20 = D["custo20"]
    red = [c20[f"{a}__linha_de_base"] for a in STK]
    tcr_ext = [fin[f]["externo"]["tcr"] for f in FIN5]
    tcr_sa = sorted(((fin[f]["spamassassin"]["tcr"], f) for f in FIN5), reverse=True)
    nf_ext = fin["nao_filtrar"]["externo"]["por_base"]
    mb = tb["membros_d"]
    # quem erra junto com os outros: P(erra | outro erra), média sobre os outros seis (crescente)
    junto = sorted(({"membro": m, "p": p} for m, p in st["erra_junto"].items()), key=lambda d: d["p"])
    # o tópico do NMF com mais spam, usado como filtro: VP = n × fs; FP = n − VP; FN = n_spam − VP; VN = resto
    tp_ = max(D["ns"]["nmf"], key=lambda x: x["fracao_spam"])
    vp = tp_["n_emails"] * tp_["fracao_spam"]; fp_ = tp_["n_emails"] - vp; fn_ = b["n_spam"] - vp; vn = b["n_emails"] - vp - fp_ - fn_
    nmf_mcc = (vp * vn - fp_ * fn_) / ((vp + fp_) * (vp + fn_) * (vn + fp_) * (vn + fn_)) ** 0.5
    # no Especialista (filtro entregue), membros que o meta-modelo praticamente desligou (|peso| < 0,1)
    pesp = fin["especialista"]["pesos_meta"]
    desl = [m for m in st["erros_modelos"] if abs(pesp[m]) < 0.1]
    c_lsvc = next(c for c in D["rodada2"]["comparacoes"] if c["B"] == "LinearSVC TF-IDF")
    frb = {k: 1 - v["custo_por_mil"] / 1000 for k, v in nf_ext.items()}   # custo de não filtrar = 1000 × fração de spam
    cus_b = lambda f, b: fin[f]["externo"]["por_base"][b]["custo_por_mil"]
    tab_caro = [b for b in nf_ext if cus_b("tabpfn", b) == max(cus_b(f, b) for f in FIN5)]
    sens = {(r["custo_fp"], r["metodo"]): r["custo_por_mil"] for r in tb["sensibilidade_custo"]}
    sa_dif = {l: sens[(l, "Stacking 7 padrão")] - sens[(l, "Stacking 8 padrão (+TabPFN)")]
              for l in sorted({r["custo_fp"] for r in tb["sensibilidade_custo"]})}
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
        "jaccard_medio": jac,                       # crescente: quem menos erra junto com os outros primeiro
        "jaccard_d": {d["membro"]: d["media"] for d in jac},
        "resgate": resg,                            # decrescente: acertos quando a maioria (≥4) erra
        "resgate_min_top3": min(d["n"] for d in resg[:3]), "resgate_max_top3": max(d["n"] for d in resg[:3]),
        "resgate_min": resg[-1]["n"],
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
        "sa_s7_custo": tm["Stacking 7 padrão"]["custo_por_mil"],
        "sa_s7_lixeira": tm["Stacking 7 padrão"]["real_lixeira_custo"],
        "sa_s8_lixeira": tm["Stacking 8 padrão (+TabPFN)"]["real_lixeira_custo"],
        "sa_tab_lixeira": tm["TabPFN-3.5-Plus sozinho"]["real_lixeira_custo"],
        "sa_lsvc_lixeira": tm["LinearSVC padrão"]["real_lixeira_custo"],
        "sa_tab_custo": tm["TabPFN-3.5-Plus sozinho"]["custo_por_mil"],
        "sa_lsvc_custo": tm["LinearSVC padrão"]["custo_por_mil"],
        "sa_lsvc_mcc": tm["LinearSVC padrão"]["mcc_custo"],
        "sa_nao_filtrar_custo": tm[NAO_F]["custo_por_mil"],
        "sa_acaso_pr_auc": tm[NAO_F]["pr_auc"],                      # PR-AUC de quem não ordena = fração de spam
        "sa_mcc_min": min(r["mcc_custo"] for r in filtros_sa),
        "sa_mcc_max": max(r["mcc_custo"] for r in filtros_sa),
        "sa_prauc_min": min(r["pr_auc"] for r in filtros_sa),
        "sa_prauc_max": max(r["pr_auc"] for r in filtros_sa),
        "sa_frac_real": tb["n_email_real"] / tb["n"],
        # custo de não filtrar = 1000 × fração de spam (λ não entra: nenhum e-mail real vai para a lixeira)
        "frac_real_base": frb,
        "bases_maioria_real_txt": " e ".join(b for b in nf_ext if frb[b] > 0.5),
        "tab_mais_caro_txt": " e ".join(tab_caro) if tab_caro else "nenhuma",
        "tab_mais_caro_custo": max([cus_b("tabpfn", b) for b in tab_caro] or [float("nan")]),
        "frac_real_ext": 1 - fin["nao_filtrar"]["externo"]["custo_por_mil"] / 1000,
        "n_boot_reducao": round(tb["decisao"]["p_reducao"] * tb["regra"]["n_boot"]),
        "ga_geracoes": len(ga["historico"]), "ga_populacao": ga["historico"][0]["avaliacoes"],
        "junto": junto, "junto_d": {d["membro"]: d["p"] for d in junto},
        "junto_top3_txt": f'{junto[0]["membro"]}, {junto[1]["membro"]} e {junto[2]["membro"]}',
        "junto_max": junto[-1],
        "resgate_top3_txt": f'{resg[0]["membro"]}, {resg[1]["membro"]} e {resg[2]["membro"]}',
        "nmf_top": {"palavras": ", ".join(tp_["palavras"][:3]), "frac_do_spam": vp / b["n_spam"],
                    "frac_real": 1 - tp_["fracao_spam"], "mcc": nmf_mcc},
        "esp_desligados_txt": " e ".join(desl) if desl else "nenhum",
        "n_esp_desligados": len(desl),
        "s4_x_lsvc": c_lsvc,
        # nº de folds da rodada 2, tirado do nome do campo no JSON (rodada2.mcc_15_folds)
        "rodada2_folds": int(re.search(r"mcc_(\d+)_folds", D["rodada2"]["chave_mcc"]).group(1)),
        # o torneio: famílias (sem a referência) e o empate no topo (rodada 2, modelos sozinhos)
        "n_familias_torneio": len({t["familia"] for t in tor if t["familia"] != "referência"}),
        "n_modelos_unicos": len(unicos),
        "topo_spread": max(v for k, v in D["rodada2"]["mcc_15"].items() if not k.startswith("Stacking"))
                       - min(v for k, v in D["rodada2"]["mcc_15"].items() if not k.startswith("Stacking")),
        "n_topo": sum(1 for k in D["rodada2"]["mcc_15"] if not k.startswith("Stacking")),
        "lineares_no_topo5": sum(1 for t in unicos[:5] if t["familia"] == "linear"),
        # o TabPFN como 8º membro, nas duas versões do stacking e nas duas bases externas
        "s8esp_custo_ext": tab_e["Stacking 8 especialista (+TabPFN)"]["custo_por_mil"],
        "s7esp_custo_ext": tab_e["Stacking 7 especialista"]["custo_por_mil"],
        "sa_s8esp_custo": tm["Stacking 8 especialista (+TabPFN)"]["custo_por_mil"],
        "sa_s7esp_custo": tm["Stacking 7 especialista"]["custo_por_mil"],
        "s8esp_prauc_int": tab_i["Stacking 8 especialista (+TabPFN)"]["pr_auc"],
        "s8esp_reducao_ext": 1 - tab_e["Stacking 8 especialista (+TabPFN)"]["custo_por_mil"] / tab_e["Stacking 7 especialista"]["custo_por_mil"],
        # regra D10 em todos os λ testados: quanto o Stacking 8 padrão custa a menos que o Stacking 7 padrão
        "sa_s8_s7_dif_min": min(sa_dif.values()), "sa_s8_s7_dif_max": max(sa_dif.values()),
        "n_lambdas": len(sa_dif),
        "conv_pares_frac": tb["conversor"]["enron4_pares"] / tb["conversor"]["enron4_spams"],
        "tcr_ext_min": min(tcr_ext), "tcr_ext_max": max(tcr_ext),
        "tcr_sa_max": tcr_sa[0][0], "tcr_sa_max_rotulo": fin[tcr_sa[0][1]]["rotulo"],
        "tcr_sa_s8": fin["generalista"]["spamassassin"]["tcr"],
        "custo_red_min": min(c["reducao_relativa"] for c in red), "custo_red_max": max(c["reducao_relativa"] for c in red),
        "custo_holm_min": min(c["p_holm"] for c in red),
        "knn_sa_reais": mb["kNN cosseno"]["reais_acima_de_05"], "lgbm_sa_reais": mb["LightGBM"]["reais_acima_de_05"],
        "lr_sa_reais": mb["Reg. Logística"]["reais_acima_de_05"], "mlp_sa_reais": mb["MLP"]["reais_acima_de_05"],
        "knn_sa_peso": mb["kNN cosseno"]["peso_meta"], "lgbm_sa_peso": mb["LightGBM"]["peso_meta"],
        "svm_sa_peso": mb["SVM RBF"]["peso_meta"],
        "perm_media": perm["mcc_permutado_media"],
        "perm_max": perm["mcc_permutado_max"],
        "n_perm": perm["n_permutacoes"],
        "curva_n_min": ns_[0], "curva_n_max": ns_[-1],
        "curva_s7_min": curva[("Stacking 7 padrão", ns_[0])], "curva_s7_max": curva[("Stacking 7 padrão", ns_[-1])],
        "curva_knn_min": curva[("kNN cosseno", ns_[0])], "curva_knn_max": curva[("kNN cosseno", ns_[-1])],
        "curva_tab_min": curva[("TabPFN-3.5-Plus", ns_[0])], "curva_tab_max": curva[("TabPFN-3.5-Plus", ns_[-1])],
        "abl_sa_todas": ce["ablacao"][0]["mcc_spamassassin"], "abl_sa_sem": ce["ablacao"][1]["mcc_spamassassin"],
        "abl_caixa_todas": ce["ablacao"][0]["mcc_esta_caixa"], "abl_caixa_sem": ce["ablacao"][1]["mcc_esta_caixa"],
        "ex2_prob": D["exemplos"][2]["etapas"]["stacking7_padrao"]["prob_spam"],
        "ex2_cobertura": D["exemplos"][2]["cobertura"],
    }


def afirmacoes(D: dict) -> list[str]:
    """Frases dos slides que dependem de uma comparação. Se o JSON mudar e uma delas deixar de valer,
    o build para, em vez de publicar um texto falso. Cada item: (frase, condição)."""
    fin, tb, tp, der = D["finais"], D["terceira"], D["tabpfn"], D["der"]
    F5 = ["especialista", "generalista", "tabpfn", "generalista_sem_chave", "linha_de_base"]
    bases = list(fin["nao_filtrar"]["externo"]["por_base"])
    lix = lambda f, b: fin[f]["externo"]["por_base"][b]["real_lixeira_custo"]
    cus = lambda f, b: fin[f]["externo"]["por_base"][b]["custo_por_mil"]
    tab_e, tm = tp["externo_d"], tb["metodos_d"]
    maioria_real = [b for b in bases if der["frac_real_base"][b] > 0.5]
    checks = [
        ("TabPFN: o que menos manda e-mail real para a lixeira em cada caixa Enron2–6",
         all(lix("tabpfn", b) == min(lix(f, b) for f in F5) for b in bases)),
        ("TabPFN: o que menos manda e-mail real para a lixeira no SpamAssassin",
         fin["tabpfn"]["spamassassin"]["real_lixeira_custo"] == min(fin[f]["spamassassin"]["real_lixeira_custo"] for f in F5)),
        ("TabPFN: o mais barato nas caixas com maioria de e-mails reais e no SpamAssassin",
         all(cus("tabpfn", b) == min(cus(f, b) for f in F5) for b in maioria_real)
         and fin["tabpfn"]["spamassassin"]["custo_por_mil"] == min(fin[f]["spamassassin"]["custo_por_mil"] for f in F5)),
        ("TabPFN: o mais barato na média do Enron2–6",
         fin["tabpfn"]["externo"]["custo_por_mil"] == min(fin[f]["externo"]["custo_por_mil"] for f in F5)),
        ("TabPFN: o que mais deixa passar spam no SpamAssassin (o mais parecido com não filtrar)",
         fin["tabpfn"]["spamassassin"]["spam_passou_custo"] == max(fin[f]["spamassassin"]["spam_passou_custo"] for f in F5)),
        ("TabPFN como 8º membro barateia o stacking nas duas versões e nas duas bases externas",
         tab_e["Stacking 8 padrão (+TabPFN)"]["custo_por_mil"] < tab_e["Stacking 7 padrão"]["custo_por_mil"]
         and tab_e["Stacking 8 especialista (+TabPFN)"]["custo_por_mil"] < tab_e["Stacking 7 especialista"]["custo_por_mil"]
         and tm["Stacking 8 padrão (+TabPFN)"]["custo_por_mil"] < tm["Stacking 7 padrão"]["custo_por_mil"]
         and tm["Stacking 8 especialista (+TabPFN)"]["custo_por_mil"] < tm["Stacking 7 especialista"]["custo_por_mil"]),
        ("D10: o Stacking 8 padrão custa menos que o Stacking 7 padrão em todos os λ testados", der["sa_s8_s7_dif_min"] > 0),
        ("SpamAssassin, λ = 9: nenhum filtro vence não filtrar", all(fin[f]["spamassassin"]["tcr"] < 1 for f in F5)),
        ("Enron2–6 (média): todo filtro compensa", all(fin[f]["externo"]["tcr"] > 1 for f in F5)),
        ("Enron2: os três stackings perdem para não filtrar",
         all(fin[f]["externo"]["por_base"]["Enron2"]["tcr"] < 1 for f in ("especialista", "generalista", "generalista_sem_chave"))),
        ("SpamAssassin: o spam que passa sobe em todos os filtros",
         all(fin[f]["spamassassin"]["spam_passou_custo"] > fin[f]["cv20"]["spam_passou_custo"] for f in F5)),
        ("Especialista: o menor custo nos 20 folds", fin["especialista"]["cv20"]["custo_por_mil"] == min(fin[f]["cv20"]["custo_por_mil"] for f in F5)),
        ("Stacking 8: empata no topo do PR-AUC desta caixa (diferença abaixo de um milésimo)",
         max(r["pr_auc"] for r in tp["interno"]) - fin["generalista"]["cv20"]["pr_auc"] < 1e-3),
        # frases dos slides de modelos
        ("LinearSVC: o melhor modelo único do torneio (a linha de base)", der["mod"]["melhor_unico"] == "LinearSVC TF-IDF"),
        ("kNN: o maior peso médio no meta-modelo", der["mod"]["knn"]["ordem_peso"] == 1),
        ("Naive Bayes: o que mais erra junto com os outros (P(erra | outro erra) máximo) e o de menor peso",
         der["junto_max"]["membro"] == "Naive Bayes" and der["mod"]["nb"]["ordem_peso"] == 7),
        ("Os três que menos erram junto são os mesmos três que mais salvam quando a maioria erra",
         {d["membro"] for d in der["junto"][:3]} == {d["membro"] for d in der["resgate"][:3]}),
        ("Especialista entregue: o meta-modelo praticamente desligou a Reg. Logística e o Naive Bayes",
         der["esp_desligados_txt"] == "Reg. Logística e Naive Bayes"),
        ("Stacking 7 padrão entregue: o MLP tem peso negativo", fin["generalista_sem_chave"]["pesos_meta"]["MLP"] < 0),
        ("NMF: o tópico de propaganda reúne quase todo o spam (acima de nove décimos) e cerca de um terço dele é e-mail real",
         der["nmf_top"]["frac_do_spam"] > 0.9 and 0.28 < der["nmf_top"]["frac_real"] < 0.4),
        ("Torneio: o 1º lugar (Stacking de 4 membros) sem significância contra o LinearSVC (p de Holm acima de 0,05)",
         der["s4_x_lsvc"]["p_holm"] > ALFA and D["torneio"][0]["modelo"] == "Stacking (meta LR)"),
        ("TabPFN sozinho: o mais caro nesta caixa no limiar de custo",
         fin["tabpfn"]["cv20"]["custo_por_mil"] == max(fin[f]["cv20"]["custo_por_mil"] for f in F5)),
        ("“ect” aparece em todos os e-mails", D["dominio_d"]["ect"]["fracao_email_real"] == 1 and D["dominio_d"]["ect"]["fracao_spam"] == 1),
        ("TabPFN sozinho empata com o LinearSVC otimizado no MCC (limiar 0,5): diferença abaixo de 0,002",
         abs(der["tab_mcc"] - der["lsvc_mcc_int"]) < 0.002),
        ("GA: o IC95 do ganho em divisões novas inclui o zero",
         D["ga"]["ganho_novas_ic95"][0] < 0 < D["ga"]["ganho_novas_ic95"][1]),
        ("Torneio: com TF-IDF com log, a Reg. Logística sobe", D["achados"]["lr_depois"] > D["achados"]["lr_antes"]),
        ("Torneio: os lineares lideram (o melhor modelo sozinho é linear e a maioria dos cinco primeiros também)",
         der["lineares_no_topo5"] * 2 > 5 and der["mod"]["melhor_unico"] == "LinearSVC TF-IDF"),
        ("Torneio: o topo está empatado (os melhores sozinhos a menos de um centésimo de MCC entre si)", der["topo_spread"] < 0.01),
        ("Combinações: Stacking 7 e Stacking 13 equivalentes (P equivalentes acima de nove décimos)",
         D["stacking"]["s7_x_s13"]["P_equivalentes"] > 0.9),
        ("Combinações: o Stacking 7 tem o maior MCC entre as combinações",
         max((v for v in D["stacking"]["variantes"] if v["tipo"] == "combinação"), key=lambda v: v["mcc"])["nome"].startswith("Stacking 7")),
    ]
    return [f"afirmação do slide não vale mais no JSON: {frase}" for frase, ok in checks if not ok]


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
    "pct0": lambda x: _num(100 * x, 0) + "%",
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
    dados = json.dumps(arred(D), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
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
         r"\bD\d{1,2}\b", r"Projeto\s3", r"Data Science\s2", r"\bIC95\b", r"RA\s46643", r"RA\s44776", r"\b2D\b",
         r"Metsis et al\., 2006", r"Androutsopoulos et al\., 2000", r"χ²",
         # endereço do repositório, nome do notebook e versão do Python (nomes, não resultados)
         r"github\.com/felipe44776-eseg/filtro-spam-stacking-enron", r"Projeto3_Classificacao_Spam\.ipynb",
         r"Python 3\.11 ou 3\.12"]
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
    falsas = afirmacoes(D)
    if falsas:
        raise SystemExit("ERRO no build:\n  " + "\n  ".join(falsas))
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
