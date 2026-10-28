# %% [markdown]
# # 06. A corrida pelo melhor filtro desta caixa, e o modelo de fundação (TabPFN)
#
# **Corrida (experimento 04).** Os 10 melhores da busca genética + o padrão disputam em divisões novas
# (sementes 201 e 202); o vencedor é **verificado** em outras divisões (203 e 204). Objetivo: o melhor filtro
# *para esta caixa*, com o limiar de custo.
#
# **TabPFN (experimento 05).** O TabPFN-3.5 é um modelo de fundação para tabelas: pré-treinado em milhões de
# tabelas sintéticas, ele classifica uma tabela nova sem ajuste. Testamos sozinho e como 8ª família do stacking
# (Stacking 8), na nuvem (3.5-Plus, 20 folds + base externa) e localmente em CPU (3.5-Fast, 5 divisões).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spamfilter import caminhos, graficos, saidas

C = pd.read_csv(caminhos.EXPERIMENTOS / "04_corrida" / "corrida.csv", sep=";", decimal=",")
corr = C.groupby(["etapa", "candidato"])[["mcc", "mcc_custo", "custo_por_mil"]].mean()
print(corr.round(4).to_string())

# %% [markdown]
# **Leitura.** Todos os candidatos empatam no MCC (platô): o Stacking 7 é pouco sensível a hiperparâmetros.
# O primeiro colocado da corrida **não** se confirmou na verificação (maldição do vencedor de novo). Decisão (D8):
# o filtro **Especialista** é a configuração do GA do experimento 03, definida antes da corrida. Nos 20 folds
# (re-partições dos mesmos e-mails, as mesmas usadas na escolha) ela custa menos que o padrão (números abaixo);
# a vantagem é provável, não comprovada. Na base de teste trancada, as duas empatam (script 09).

# %%
G5 = saidas.ler("05_genetico")["genetico"]
cf = {r["metodo"]: r for r in G5["confirmacao"]}
cc = next(r for r in G5["comparacoes_custo"] if r["A"] == "Stacking 7 do GA (Especialista)" and r["B"] == "Stacking 7 padrão")
print(f"custo por mil nos 20 folds: Especialista {cf['Stacking 7 do GA (Especialista)']['custo_por_mil']:.1f} × padrão "
      f"{cf['Stacking 7 padrão']['custo_por_mil']:.1f} | P(Especialista mais barato) = {cc['P_A_melhor']:.2f} | p (Holm) = {cc['p_holm']:.3f}")

# %%
T = caminhos.EXPERIMENTOS / "05_tabpfn"
T20 = pd.read_csv(T / "tabpfn_20folds.csv", sep=";", decimal=",")
TEX = pd.read_csv(T / "tabpfn_externo.csv", sep=";", decimal=",")
TLN = pd.read_csv(T / "tabpfn_local_x_nuvem.csv", sep=";", decimal=",")
t20 = T20.groupby("metodo")[["mcc", "pr_auc", "mcc_custo", "custo_por_mil"]].mean().sort_values("custo_por_mil")
tex = TEX.groupby("metodo")[["mcc", "pr_auc", "mcc_custo", "real_lixeira_custo", "spam_passou_custo", "custo_por_mil"]].mean().sort_values("custo_por_mil")
tln = TLN.groupby("metodo")[["mcc", "pr_auc", "custo_por_mil"]].mean()
print("== nesta caixa (20 folds) ==\n" + t20.round(4).to_string())
print("\n== caixas novas (Enron2-6) ==\n" + tex.round(4).to_string())
print("\n== local (3.5-Fast, CPU) × nuvem (3.5-Plus), 5 divisões ==\n" + tln.round(4).to_string())

# %%
comum = [m for m in t20.index if m in tex.index]
fig, ax = plt.subplots(figsize=(6.8, 4))
for i, mt in enumerate(sorted(comum, key=lambda k: tex.loc[k, "custo_por_mil"])):
    a, b = t20.loc[mt, "custo_por_mil"], tex.loc[mt, "custo_por_mil"]
    cor = graficos.CATEGORICA[6] if "8" in mt else graficos.CATEGORICA[0] if "Stacking 7" in mt else graficos.CATEGORICA[2] if "TabPFN" in mt else graficos.TINTA_MUDA
    ax.plot([0, 1], [a, b], color=cor, marker="o", ms=7, mec=graficos.SUPERFICIE, mew=1.5)
    ax.text(1.04, b, f"{mt} ({b:.0f})", va="center", fontsize=8.5, color=graficos.TINTA)
ax.set_xticks([0, 1], ["nesta caixa\n(20 folds)", "caixas novas\n(Enron2-6)"]); ax.set_xlim(-0.15, 1.9)
ax.set_ylabel("custo por mil e-mails"); ax.grid(axis="x", visible=False)
ax.set_title("Quem é barato aqui fica caro fora (e vice-versa)")
graficos.mostrar(fig, "06_custo_dentro_fora")

# %% [markdown]
# **Leitura.**
# - **Nesta caixa**, o TabPFN sozinho, sem nenhum ajuste de hiperparâmetro, empata com o melhor modelo único
#   (LinearSVC otimizado) no MCC. No stacking, o Stacking 8 empata com o Stacking 7 no MCC e tem o maior PR-AUC.
# - **Em caixas novas**, ele é o mais conservador com e-mails reais e o filtro mais barato no Enron2-6; dentro do
#   stacking, baixa o custo. Esse achado levou à decisão pré-registrada D10, testada numa terceira base (script 08).
# - **Local × nuvem:** as notas têm correlação alta (valor abaixo); o experimento pode ser refeito sem conta e sem
#   internet (cerca de 3 h em CPU para 5 divisões).

# %%
TC = T / "cache_tabpfn"   # notas de teste das 5 divisões da semente 201: local (3.5-Fast, CPU) e nuvem (3.5-Plus)
corr_ln = float(np.mean([np.corrcoef(np.load(TC / "local" / f"c201_{f}.npz")["zte_te"], np.load(TC / "nuvem" / f"c201_{f}.npz")["zte_te"])[0, 1]
                         for f in range(5)]))
print(f"correlação média das notas local × nuvem: {corr_ln:.4f}")
saidas.gravar("06_corrida_tabpfn", {
    "corrida": dict(resumo=corr.reset_index(), decisao="Especialista = configuração do GA do experimento 03 (#5), definida antes da corrida"),
    "tabpfn": dict(versao_nuvem="TabPFN-3.5-Plus", versao_local="TabPFN-3.5-Fast", interno=t20.reset_index(), externo=tex.reset_index(),
                   local_x_nuvem=dict(tabela=tln.reset_index(), correlacao_notas=corr_ln)),
}, fonte="scripts/06_corrida_e_tabpfn.py")
print("gravado: resultados/06_corrida_tabpfn.json")
