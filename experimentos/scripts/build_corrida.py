"""Experimento 04: corrida entre os melhores Stacking 7 da busca genetica, so com esta caixa de e-mail.
Reaproveita as secoes 1 a 6 do experimento 03 (motor, busca, GA e confirmacao) e o cache dele."""
import nbformat as nbf, sys, runpy
from pathlib import Path

ga = runpy.run_path(str(Path(__file__).with_name("build_ga.py")), run_name="build_ga")
base = ga["C"][1:17]                 # celulas 1..16: imports ate a tabela de sementes (sem teste trancado/externo)
trocas = [('"cache_ga").resolve()', '"../03_ga/cache_ga").resolve()'),          # mesmo cache: nada e retreinado
          ('"progresso_ga.log")', '"progresso_corrida.log")'),
          ('; R.to_csv("ga_confirmacao.csv", sep=";", decimal=",", index=False)', '')]
for c in base:
    for a, b in trocas:
        c.source = c.source.replace(a, b)
assert any("../03_ga/cache_ga" in c.source for c in base), "troca do cache falhou"

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Experimento 04: qual é o melhor Stacking 7 *para esta caixa*?

**Pergunta.** Medindo só nesta caixa de e-mail (Enron1) e com várias sementes aleatórias, qual configuração do Stacking 7 é a melhor?

**Termos.** *E-mail real* = e-mail legítimo. `real_lixeira` = % de e-mails reais marcados como spam. `spam_passou` = % de spam que chegou à caixa de entrada. *Custo por mil* = 9 × e-mails reais na lixeira + 1 × spams que passaram, a cada 1000 e-mails.

**Como evitar o viés de seleção.** Cada etapa que escolhe algo usa divisões que a etapa seguinte não reaproveita:

| Etapa | Divisões | Papel |
|---|---|---|
| Busca (exp. 03) | sementes 11 e 12 (10 folds) | o algoritmo genético gera os candidatos |
| **Corrida** | sementes 201 e 202 (10 folds) | ordena os 10 melhores candidatos + o padrão |
| **Verificação** | sementes 203 e 204 (10 folds) | mede o vencedor da corrida contra o padrão, sem otimismo |

As seções 1 a 6 repetem o experimento 03 a partir do cache dele, sem retreinar nada. A base de teste trancada e a base externa **não** são usadas aqui.
""")
C.extend(base)

md(r"""
## 10. Corrida entre os candidatos
Candidatos: os 10 cromossomos com maior MCC nas divisões da busca + o padrão (+ o escolhido no exp. 03, se ficar fora do top 10).
""")
code(r'''
top = sorted(AVAL.items(), key=lambda kv: -kv[1])[:10]
CAND, vistos = {"padrão": PADRAO}, {PADRAO}
for i, (gen, _) in enumerate(top):
    if gen in vistos: continue
    vistos.add(gen); CAND[f"#{i + 1}" + (" (GA exp. 03)" if gen == melhor_ga else "")] = gen
if melhor_ga not in vistos:
    CAND["GA exp. 03"] = melhor_ga
NOME_GA = next(n for n, g in CAND.items() if g == melhor_ga)
PRECISA_C = sorted({(fam, gen[i]) for gen in CAND.values() for i, fam in enumerate(FAMILIAS)}, key=str)
log(f"corrida: {len(CAND)} candidatos, {len(PRECISA_C)} configurações distintas de modelos de base")
rodar([(c, fam, k) for c in CONF for fam, k in PRECISA_C], "corrida", blocos=4)

def prever_gen(gen, chave, te):
    METODOS["_cand"] = ("stack", gen)
    try:
        return prever("_cand", chave, te)
    finally:
        METODOS.pop("_cand")

lin = []
for nome, gen in CAND.items():
    for c in CONF:
        yte = y[SPLITS[c]["tes"]["te"]]
        lin.append(dict(candidato=nome, split=c, etapa="corrida" if c[1:4] in ("201", "202") else "verificação",
                        n_tr=len(SPLITS[c]["tr"]), n_te=len(yte), **metricas(yte, *prever_gen(gen, c, "te"))))
RC = pd.DataFrame(lin); RC.to_csv("corrida.csv", sep=";", decimal=",", index=False)

def mudancas(gen):
    m = [f"{fam}: {desc(fam, gen[i])}" for i, fam in enumerate(FAMILIAS) if gen[i] != 0]
    if gen[-1] != 0: m.append(f"meta-modelo: C={META_C[gen[-1]]}")
    return "; ".join(m) or "(configuração padrão)"

cor = RC[RC.etapa == "corrida"]
pos = cor.pivot(index="split", columns="candidato", values="mcc").rank(axis=1, ascending=False).mean()
tab_corrida = pd.DataFrame({
    "MCC busca": pd.Series({n: aptidao(g) for n, g in CAND.items()}),
    "MCC corrida": cor.groupby("candidato").mcc.mean(),
    "posição média": pos,
    "MCC limiar custo": cor.groupby("candidato").mcc_custo.mean(),
    "custo por mil": cor.groupby("candidato").custo_por_mil.mean(),
    "% reais na lixeira (custo)": cor.groupby("candidato").real_lixeira_custo.mean() * 100,
}).sort_values("MCC corrida", ascending=False)
VENCEDOR = tab_corrida.index[0]
print(f"vencedor da corrida: {VENCEDOR}\n")
print(tab_corrida.round(4).to_string())
print("\no que cada candidato muda em relação ao padrão:")
for n in tab_corrida.index:
    print(f"  {n:22s} {mudancas(CAND[n])}")
''')

md(r"""
## 11. Verificação: o vencedor da corrida em divisões que não participaram da escolha
`P(melhor)` = probabilidade de a diferença ser positiva, por qualquer margem (t corrigido bayesiano). `P(equiv.)` = probabilidade de a diferença ficar abaixo de 0,01 de MCC.
""")
code(r'''
ver = RC[RC.etapa == "verificação"]
n_trv, n_tev = ver.n_tr.mean(), ver.n_te.mean(); rho = n_tev / n_trv

def p_melhor(d):
    d = np.asarray(d); n = len(d); s = np.sqrt((1 / n + rho) * d.var(ddof=1))
    return float(1 - stats.t.cdf(0, n - 1, loc=d.mean(), scale=s)) if s > 0 else float(d.mean() > 0)

linhas_v = []
for rival in dict.fromkeys(["padrão", NOME_GA, tab_corrida.index[1]]):
    if rival == VENCEDOR: continue
    for met, nome_met, maior_melhor in [("mcc", "MCC", True), ("mcc_custo", "MCC limiar custo", True), ("custo_por_mil", "custo por mil", False)]:
        P = ver.pivot(index="split", columns="candidato", values=met)
        d = (P[VENCEDOR] - P[rival]) * (1 if maior_melhor else -1)    # positivo = vencedor melhor
        _, t, p = t_corrigido(P[VENCEDOR], P[rival], n_trv, n_tev)
        eq = np.nan
        if met != "custo_por_mil":
            try: eq = baycomp.two_on_single(P[VENCEDOR].values, P[rival].values, rope=0.01, runs=2)[1]
            except Exception: pass
        linhas_v.append(dict(contra=rival, metrica=nome_met, vencedor=P[VENCEDOR].mean(), rival=P[rival].mean(),
                             p=p, **{"P(melhor)": p_melhor(d), "P(equiv.)": eq}))
print(f"vencedor da corrida: {VENCEDOR} | verificação: {ver.split.nunique()} folds (sementes 203 e 204)\n")
print(pd.DataFrame(linhas_v).round(4).to_string(index=False))

print(f"\nconfiguração completa de {VENCEDOR}:")
g = CAND[VENCEDOR]
for i, fam in enumerate(FAMILIAS):
    print(f"  {fam:15s} {desc(fam, g[i])}" + ("" if g[i] else "   (= padrão)"))
print(f"  {'meta-modelo':15s} C={META_C[g[-1]]}" + ("" if g[-1] else "   (= padrão)"))
log("fim")
''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1]); print("ok", sys.argv[1], "|", len(C), "células")
