# %% [markdown]
# # Projeto 3: classificação de spam
# **Data Science 2 · ESEG · Prof. Marino Catarino**
# **Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins
#
# Este notebook conta a história do projeto do começo ao fim, com o código principal visível. Ele roda em poucos
# minutos (*Restart & Run All*): as partes pesadas (o torneio de 22 modelos, os milhares de treinos do algoritmo
# genético, o TabPFN na nuvem) foram executadas nos notebooks técnicos `notebooks/01` a `09`, e os resultados
# estão em `resultados/`. **Nenhum número do texto foi digitado à mão**: todos vêm de `resultados/metricas_finais.json`.
#
# **Termos.** *E-mail real* = e-mail legítimo (classe 0). *Spam* = classe 1. *Na lixeira* = e-mail real marcado
# como spam (o erro caro). *Spam que passou* = spam que chegou à caixa de entrada.

# %% [markdown]
# **No Google Colab**, crie uma célula de código acima desta e rode:
# ```
# !git clone https://github.com/felipe44776-eseg/filtro-spam-stacking-enron.git
# %cd filtro-spam-stacking-enron
# !pip install -q -r requirements.txt && pip install -q -e . --no-deps
# ```
# No computador, siga o [INSTALACAO.md](INSTALACAO.md) e não rode nada disso.

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from spamfilter import caminhos, conversao, dados, estatistica, genetico, graficos, metricas, modelos
from spamfilter.motor import Motor

R = json.loads((caminhos.RESULTADOS / "metricas_finais.json").read_text(encoding="utf-8"))
fig_ = lambda nome: display(Image(filename=str(caminhos.RESULTADOS / "figuras" / f"{nome}.png"), width=720))
pct = lambda x: f"{100 * x:.1f}%".replace(".", ",")
num = lambda x, c=3: f"{x:.{c}f}".replace(".", ",")

# %% [markdown]
# ## 1. O problema e a base
# 5172 e-mails, cada um descrito por quantas vezes 3000 palavras aparecem nele. Descobrimos que o CSV é o
# **Enron1** do corpus Enron-Spam (Metsis et al., 2006): e-mails reais da caixa de um funcionário da Enron e spams
# de 2003-2005. E descobrimos **como** as palavras foram contadas: `texto.count(palavra)`, como *substring*,
# diferenciando maiúsculas.

# %%
base = dados.carregar()
b = R["base"]
print(f"{b['n_emails']} e-mails | {b['n_email_real']} reais | {b['n_spam']} spams | {b['n_palavras']} colunas | "
      f"{pct(b['fracao_zeros'])} das células são zero")
print(f"linhas do CSV reproduzidas a partir dos textos originais: {b['linhas_reproduzidas']}/{b['n_emails']}")

# a mesma contagem, aplicada a um texto qualquer:
exemplo = "Subject: cheap meds online\nget your prescription drugs now , no doctor needed !"
v = conversao.vetor(conversao.para_aueb(exemplo), base.vocab)
print({w: int(v[i]) for i, w in enumerate(base.vocab) if v[i] and len(w) >= 3})

# %% [markdown]
# Um detalhe muda a leitura das colunas: como a contagem é por substring, o vocabulário tem as 26 letras soltas
# e "ect" aparece em **100%** dos e-mails, porque está dentro de "subj**ect**".

# %%
fig_("01_palavras_que_separam")

# %% [markdown]
# ## 2. Qualidade dos dados: duplicata indica spam? Não.

# %%
d = dados.resumo_duplicatas(base)
print(f"cópias excedentes: {d['duplicatas_excedentes']}, das quais {pct(d['fracao_duplicatas_email_real'])} são e-mails reais "
      f"(avisos automáticos de modelo fixo)")
dev, teste = dados.indices_teste(base)
print(f"protocolo: cópias idênticas ficam sempre do mesmo lado (StratifiedGroupKFold por grupo). "
      f"Base de teste trancada: {len(teste)} e-mails, sorteada uma vez e guardada.")

# %% [markdown]
# ## 3. Olhando sem rótulo (não supervisionado)

# %%
ns = R["nao_supervisionado"]
fig_("02_svd_2d")
print(f"K-means com 2 grupos: ARI = {num(ns['kmeans_ari'])} (os grupos naturais são assuntos, não 'spam × real')")
for t in ns["nmf_topicos"][:4]:
    print(f"tópico {t['topico']}: {pct(t['fracao_spam'])} de spam | {', '.join(t['palavras'][:6])}")
fig_("02_anomalia")

# %% [markdown]
# Sem rótulo, a melhor pista é "este e-mail não se parece com os meus e-mails reais" (erro de reconstrução). Os
# detectores clássicos (Isolation Forest, One-Class SVM) ficam abaixo do acaso: o spam fica *fora* do espaço dos
# e-mails reais, perto do centro da projeção, e não parece isolado.

# %% [markdown]
# ## 4. Como medir
# Quem nunca marca spam acerta 71%. Por isso usamos o **MCC** (−1 a 1; 0 = acaso), o **PR-AUC** (qualidade da
# ordenação, sem limiar) e o **custo**: perder 1 e-mail real custa o mesmo que receber 9 spams. O limiar de cada
# filtro é escolhido para minimizar esse custo, **só com dados de treino**:

# %%
import inspect
print(inspect.getsource(metricas.limiar_custo))
print(f"acurácia de quem nunca marca spam: {pct(R['metricas_didaticas']['dummy_acuracia'])}")

# %% [markdown]
# ## 5. Torneio de 22 modelos

# %%
fig_("03_torneio")
a = R["achados"]
print(f"O detalhe que virou o jogo, TF-IDF com log: Reg. Logística {num(a['lr_mcc_antes'])} → {num(a['lr_mcc_depois'])}; "
      f"kNN {num(a['knn_mcc_antes'])} → {num(a['knn_mcc_depois'])}")

# %% [markdown]
# ## 6. Ninguém vence sozinho: o Stacking 7
# Um modelo de cada família; um meta-modelo aprende quanto confiar em cada um. As notas que alimentam o
# meta-modelo são *out-of-fold* (nenhum modelo dá nota a um e-mail que usou para treinar). O motor guarda essas
# notas em cache, e por isso dá para recalcular os 20 folds de confirmação aqui, em segundos:

# %%
m = Motor(base, log=lambda *a: None)
CONF = m.adicionar_divisoes([201, 202, 203, 204])
linhas = []
for c in CONF:
    yte = m.y_teste(c)
    linhas.append(dict(metodo="Stacking 7", **metricas.metricas(yte, *m.prever_stacking(modelos.STACKING7_PADRAO, c))))
    linhas.append(dict(metodo="LinearSVC (melhor modelo único)", **metricas.metricas(yte, *m.prever_linearsvc(0, c))))
T = pd.DataFrame(linhas).groupby("metodo")[["mcc", "pr_auc", "custo_por_mil"]].mean()
print(T.round(4).to_string())
fig_("04_erros_em_comum"); fig_("04_pesos_meta")

# %% [markdown]
# O kNN, dos mais fracos sozinho, é o que menos erra junto com os outros, e o meta-modelo lhe dá o maior peso.
# Mais modelos não melhoram: o Stacking com 13 empata com o de 7. **O que importa é a diversidade.**

# %% [markdown]
# ## 7. O algoritmo genético e a maldição do vencedor
# 7.077.888 combinações de hiperparâmetros. Com o cache, cada avaliação é só o meta-modelo: o GA inteiro roda aqui
# em cerca de um minuto e meio.

# %%
BUSCA = m.adicionar_divisoes([11, 12], prefixo="b")
apt = genetico.Aptidao(m, BUSCA)
melhor, hist = genetico.algoritmo_genetico(apt)
print(f"{len(apt.memoria)} avaliações | MCC na busca: padrão {num(apt(modelos.STACKING7_PADRAO), 4)} → GA {num(apt(melhor), 4)}")
g = R["genetico"]
print(f"em divisões novas, o ganho cai de {g['ganho_busca']:+.4f} para {g['ganho_sementes_novas']:+.4f}: "
      f"{pct(g['fracao_otimismo'])} do ganho era otimismo da seleção")
fig_("05_ga_x_aleatoria")

# %% [markdown]
# ## 8. O modelo de fundação: TabPFN
# O TabPFN-3.5 (Prior Labs) foi pré-treinado em milhões de tabelas sintéticas e classifica sem ajuste. Nesta caixa,
# empata com o melhor modelo único e é redundante no stacking; em caixas novas, é o mais conservador.

# %%
fig_("06_custo_dentro_fora")

# %% [markdown]
# ## 9. A grande final: três avaliações, uma regra escrita antes
# A base externa (Enron2-6) sugeriu que o Stacking 8 (o Stacking 7 + TabPFN) generaliza melhor. Para decidir sem
# viés, a regra foi registrada **antes** de rodar numa terceira base nunca usada (SpamAssassin):

# %%
tb = R["terceira_base"]; dcs = tb["decisao"]
print(f"SpamAssassin: {tb['n']} mensagens. Stacking 8 − Stacking 7: {dcs['diferenca_custo']:+.0f} por mil "
      f"(P(redução) = {num(dcs['p_reducao'])}); MCC {dcs['diferenca_mcc_custo']:+.3f}")
print("decisão:", "o Stacking 8 vira o Generalista" if dcs["stacking8_vira_generalista"] else "o Generalista não muda")
fig_("09_filtros_finais")
F = R["finais"]
tab = pd.DataFrame({n: {"MCC (20 folds)": F[n]["cv20"]["mcc"], "custo/mil (20 folds)": F[n]["cv20"]["custo_por_mil"],
                        "Enron2-6": F[n]["externo"]["custo_por_mil"], "SpamAssassin": F[n]["spamassassin"]["custo_por_mil"]}
                    for n in F}).T
print(tab.round(3).to_string())

# %% [markdown]
# ## 10. Contra-estudo: tentando derrubar as conclusões

# %%
ce = R["contra_estudo"]
print(f"rótulos embaralhados: MCC {ce['permutacao']['mcc_real']:.3f} → {np.mean(ce['permutacao']['mcc_permutado']):+.3f} "
      f"(p = {ce['permutacao']['p']:.3f}): o protocolo não vaza")
print(pd.DataFrame(ce["ablacao"]).set_index("variante").round(3).to_string())
fig_("07_curva_aprendizado")
print(f"controle da 3ª base: treinado e testado dentro do SpamAssassin, MCC {num(tb['controle_dentro_da_base']['mcc'])} "
      f"→ a conversão funciona; o que cai é a transferência entre caixas.")
fig_("08_terceira_base")

# %% [markdown]
# ## 11. A demonstração: um e-mail, todas as etapas

# %%
from spamfilter.filtro import FiltroStacking
esp = FiltroStacking.carregar(caminhos.RESULTADOS / "modelos" / "especialista.joblib")
for ex in R["exemplos_demo"]:
    e = esp.explicar(ex["texto"])
    print(f"{ex['rotulo']:24s} cobertura {pct(e['cobertura'])} | P(spam) {pct(e['prob_spam'])} | limiar {pct(e['limiar'])} | "
          f"{'SPAM' if e['spam'] else 'e-mail real'}")

# %% [markdown]
# ## 12. Conclusões
# 1. **Conhecer os dados mudou tudo:** a base é o Enron1, a contagem é por substring e as duplicatas são e-mails
#    reais; o protocolo separa cópias por grupo.
# 2. **A representação vale mais que o algoritmo:** TF-IDF com log levou modelos simples ao topo.
# 3. **Diversidade vence quantidade:** 7 famílias que erram diferente superam qualquer modelo sozinho; 13 não superam 7.
# 4. **Ajuste fino rende pouco e engana muito:** 85% do ganho do algoritmo genético era otimismo; há um piso
#    estatístico abaixo do qual nenhuma diferença é detectável.
# 5. **O filtro aprende a caixa, não o spam:** fora da caixa de origem tudo piora, e os filtros simples e
#    conservadores custam menos. A regra pré-registrada promoveu o Stacking 8 a Generalista; para uso real, o
#    caminho é retreinar com e-mails da caixa de destino.
#
# **Filtros entregues:** Especialista (esta caixa), Generalista (Stacking 8, caixas novas), Stacking 7 padrão
# (caixas novas, sem chave) e LinearSVC (linha de base). Detalhes, decisões e números: `docs/REGISTRO_DECISOES.md`.
