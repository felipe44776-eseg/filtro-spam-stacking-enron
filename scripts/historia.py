# %% [markdown]
# # Projeto 3: classificação de spam
# **Data Science 2 · ESEG · Prof. Marino Catarino**
# **Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins (RA 44776)
#
# Este notebook conta a história do projeto do começo ao fim, com o código principal visível. Ele roda em poucos
# minutos (*Restart & Run All*): as partes pesadas (o torneio de 22 modelos, os milhares de treinos do algoritmo
# genético, o TabPFN na nuvem) foram executadas nos notebooks técnicos `notebooks/01` a `09`, e os resultados
# estão em `resultados/`. **Nenhum resultado foi digitado à mão no texto:** os números aparecem nas saídas das
# células e vêm de `resultados/metricas_finais.json`. No texto ficam só nomes e parâmetros do projeto, como o custo 9:1.
#
# **Termos.** *E-mail real* = e-mail legítimo (classe 0). *Spam* = classe 1. *Na lixeira* = e-mail real marcado
# como spam (o erro caro). *Spam que passou* = spam que chegou à caixa de entrada.

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from spamfilter import caminhos, conversao, dados, estatistica, genetico, graficos, metricas, modelos, publicados
from spamfilter.motor import Motor

R = json.loads((caminhos.RESULTADOS / "metricas_finais.json").read_text(encoding="utf-8"))
F = R["finais"]
fig_ = lambda nome: display(Image(filename=str(caminhos.RESULTADOS / "figuras" / f"{nome}.png"), width=720))
pct = lambda x: f"{100 * x:.1f}%".replace(".", ",")
num = lambda x, c=3: f"{x:.{c}f}".replace(".", ",")
sn = lambda x, c=4: f"{x:+.{c}f}".replace(".", ",")          # com sinal: +0,0063

# %% [markdown]
# ## 1. O problema e a base
# **De onde vêm os e-mails.**
# 1. **A Enron** era uma empresa de energia de Houston (EUA) que faliu em 2001 num escândalo de fraude contábil.
#    Na investigação, a agência reguladora de energia dos EUA (FERC) publicou os e-mails de cerca de 150
#    funcionários: a maior coleção pública de e-mails reais de uma empresa.
# 2. **Metsis, Androutsopoulos e Paliouras (2006)** juntaram os e-mails reais de seis desses funcionários com spams
#    de coleções públicas e montaram o corpus **Enron-Spam** (Enron1 a Enron6). O artigo compara cinco versões do
#    Naive Bayes, retreinando o filtro a cada 100 e-mails, como faria um usuário novo; a versão preferida pelos
#    autores olha só se a palavra aparece ou não. Usamos os dados do artigo, não a análise dele (resumo em
#    `dados/LEIA-ME.md`).
# 3. **O CSV do Kaggle** (o arquivo do enunciado) traz 5172 e-mails e 3000 colunas de contagem, descritos como
#    escolhidos ao acaso.
# 4. **O que descobrimos:** refazendo a contagem a partir dos textos originais da AUEB, o CSV é o **Enron1 inteiro**:
#    e-mails reais da caixa `farmer-d` e spams da coleção GP, recebidos por um dos autores, Paliouras, de 2003 a 2005. E descobrimos **como** as palavras foram
#    contadas: `texto.count(palavra)`, como *substring*, diferenciando maiúsculas. Cada coluna conta um pedaço de
#    texto achado até dentro de outras palavras; não é um *token* (a palavra recortada do texto), e sim uma substring.

# %%
base = dados.carregar()
b = R["base"]
fonte = R["terceira_base"]["fontes_spam"][0]
print(f"{b['n_emails']} e-mails | {b['n_email_real']} reais | {b['n_spam']} spams | {b['n_palavras']} colunas | "
      f"{pct(b['fracao_zeros'])} das células são zero")
print(f"e-mails reais: caixa {fonte['email_real']}, de {fonte['email_real_de']} a {fonte['email_real_ate']} | "
      f"spams: coleção {fonte['spam']}, de {fonte['spam_de']} a {fonte['spam_ate']}")
print(f"linhas do CSV reproduzidas a partir dos textos originais: {b['linhas_reproduzidas']}/{b['n_emails']}")

# a mesma contagem, aplicada a um texto qualquer:
exemplo = "Subject: cheap meds online\nget your prescription drugs now , no doctor needed !"
v = conversao.vetor(conversao.para_aueb(exemplo), base.vocab)
print({w: int(v[i]) for i, w in enumerate(base.vocab) if v[i] and len(w) >= 3})

# %% [markdown]
# Um detalhe muda a leitura das colunas: o vocabulário do autor do CSV tem letras soltas, e a contagem é por substring.
# Então a coluna "e" conta toda letra "e", e "ect" aparece em todos os e-mails, porque está dentro de "subj**ect**".

# %%
j_ect = list(base.vocab).index("ect")
print(f"colunas de uma letra só: {b['colunas_1_letra']} | e-mails com 'ect': {pct((base.X[:, j_ect] > 0).mean())}")
fig_("01_palavras_que_separam")

# %% [markdown]
# Uma palavra da figura não é vocabulário de spam: **paliourg** é o usuário de e-mail de G. Paliouras, um dos
# pesquisadores que montaram a coleção de spam. Ela aparece nos spams por causa da coleta: é um artefato da base,
# e o contra-estudo (seção 10) a retira junto com as palavras da Enron.

# %% [markdown]
# ## 2. Qualidade dos dados: duplicata indica spam? Não.
# **Cópia** aqui quer dizer dois e-mails com a **mesma linha de 3000 contagens**. Como até as letras soltas são colunas,
# os textos precisam ter as mesmas letras; só muda o que o vocabulário não vê: algarismos, pontuação, espaços e a ordem
# dos trechos (o mesmo aviso diário com outro dia vira a mesma linha).

# %%
d = dados.resumo_duplicatas(base)
cp = b["copias"]
print(f"cópias excedentes: {d['duplicatas_excedentes']}, das quais {pct(d['fracao_duplicatas_email_real'])} são e-mails reais "
      f"(avisos automáticos de trabalho)")
print(f"grupos de cópias: {cp['grupos']} ({cp['grupos_pares']} pares; o maior tem {cp['maior_grupo']} e-mails: "
      f"\"{cp['maior']['exemplo_assunto']}\", {cp['maior']['textos_distintos']} textos diferentes)")
print(f"nos textos originais: {cp['grupos_texto_identico']} grupos de texto idêntico; nos outros {cp['grupos_texto_diferente']}, "
      f"mudam só os espaços ({cp['grupos_diferem_so_em_espacos']}), os algarismos e sinais ({cp['grupos_diferem_em_numeros_e_sinais']}) "
      f"ou a ordem dos trechos ({cp['grupos_diferem_na_ordem']})")
dev, teste = dados.indices_teste(base)
print(f"protocolo: as cópias ficam sempre do mesmo lado (StratifiedGroupKFold, grupo = hash da linha). "
      f"Base de teste trancada: {len(teste)} e-mails, sorteada uma vez e guardada.")

# %% [markdown]
# **Folds.** Na validação cruzada, os e-mails são divididos em cinco partes (*folds*): o filtro treina em quatro e
# faz a prova na quinta, girando até cada parte ser a prova uma vez. Todas as cópias de um grupo caem na mesma parte.
#
# O agrupamento só junta cópias (linhas iguais). Quase-cópias (o mesmo aviso com uma ou outra palavra diferente)
# podem cair dos dois lados da divisão; o efeito delas é medido no contra-estudo (seção 10).

# %% [markdown]
# ## 3. Olhando sem rótulo (não supervisionado) e com meio rótulo (semissupervisionado)

# %%
ns = R["nao_supervisionado"]
fig_("02_svd_2d")
print(f"ARI (0 = acaso, 1 = coincide com spam × real): K-means {num(ns['kmeans_ari'])} | hierárquico {num(ns['hierarquico_ari'])}")
for t in ns["nmf_topicos"][:4]:
    print(f"tópico {t['topico']}: {pct(t['fracao_spam'])} de spam | {', '.join(t['palavras'][:6])}")
fig_("02_anomalia")
tp = max(ns["nmf_topicos"], key=lambda t: t["fracao_spam"]); spam_tp = tp["n_emails"] * tp["fracao_spam"]
print(f"tópico com mais spam ({', '.join(tp['palavras'][:3])}): {tp['n_emails']} e-mails, {pct(tp['fracao_spam'])} de spam, "
      f"{pct(spam_tp / b['n_spam'])} de todo o spam da base")
an = ns["anomalia_so_email_real"]
print(f"detecção de novidade (treina só com e-mails reais): erro de reconstrução, PR-AUC {num(an['reconstrucao_pr_auc'])} "
      f"(sortear daria {num(b['n_spam'] / b['n_emails'])})")

# %% [markdown]
# **K-means, agrupamento hierárquico e NMF não usam o rótulo.** Os grupos do K-means e do hierárquico são *assuntos*,
# não "spam × real". O NMF acha um tópico de linguagem de propaganda que reúne quase todo o spam, mas um em cada
# três e-mails dele é real (saída abaixo): há pista sem rótulo, mas não um filtro bom. A detecção de novidade é
# **semissupervisionada**: treina só com e-mails sabidamente reais, sem
# nenhum spam rotulado. Os detectores clássicos (Isolation Forest, One-Class SVM) ficam abaixo do acaso; o erro de
# reconstrução, que mede quanto do e-mail as direções dos e-mails reais não explicam, separa bem.

# %% [markdown]
# ## 4. Como medir
# Quem nunca marca spam já acerta a maioria (saída abaixo); por isso a acurácia engana. Usamos o **MCC** (−1 a 1;
# 0 = acaso), o **PR-AUC** (qualidade da ordenação, sem limiar) e o **custo**: perder 1 e-mail real custa o mesmo
# que receber 9 spams (o cenário λ = 9 de Androutsopoulos et al., 2000). O limiar de cada filtro é escolhido para
# minimizar esse custo, **só com dados de treino**. A referência de todo filtro é **não filtrar**: todo spam passa e
# nenhum e-mail real se perde. O **TCR** é o custo de não filtrar dividido pelo custo do filtro: acima de 1, o filtro
# compensa.

# %%
import inspect
print(inspect.getsource(metricas.limiar_custo))
print(f"acurácia de quem nunca marca spam: {pct(R['metricas_didaticas']['dummy_acuracia'])} | "
      f"custo λ = {R['meta']['custo_fp']}:{R['meta']['custo_fn']}")

# %% [markdown]
# ## 5. Um torneio de modelos supervisionados, e o ranking
# Sem rótulo, os grupos eram assuntos (seção 3): para separar spam, precisamos das respostas. Testamos então muitos
# modelos supervisionados na **mesma prova** (os mesmos e-mails, 5 folds separados por grupo, a mesma régua, o MCC)
# e ordenamos do melhor ao pior. Cada família de modelo raciocina de um jeito (`docs/MODELOS.md`).

# %%
fig_("03_torneio")
rk = pd.DataFrame(R["torneio"]).sort_values("mcc", ascending=False).reset_index(drop=True)
rk.index += 1
print(rk[["modelo", "familia", "mcc", "pr_auc", "treino_s"]].round(4).to_string())
a = R["achados"]
print(f"O detalhe que virou o jogo, TF-IDF com log: Reg. Logística {num(a['lr_mcc_antes'])} → {num(a['lr_mcc_depois'])}; "
      f"kNN {num(a['knn_mcc_antes'])} → {num(a['knn_mcc_depois'])}")

# %% [markdown]
# **O que o ranking mostrou:** os lineares lideram, mas no topo os melhores modelos sozinhos ficam muito perto uns
# dos outros, e o 1º lugar já era uma combinação simples, o "Stacking (meta LR)" com 4 membros (sem diferença
# significativa para o LinearSVC). Daí a próxima pergunta: qual combinação funciona melhor, e por quê?
#
# ## 6. Ninguém vence sozinho: o Stacking 7
# Um modelo de cada família; um meta-modelo aprende quanto confiar em cada um. Os sete, em uma frase cada
# (explicação completa, com analogias, em `docs/MODELOS.md`):
#
# | Modelo | Como decide |
# |---|---|
# | Regressão Logística | dá um peso a cada palavra, soma e transforma a soma numa probabilidade |
# | SVM com kernel RBF | resume as palavras em "temas" (SVD) e traça uma fronteira curva por semelhança com e-mails de referência |
# | kNN | procura os e-mails de treino mais parecidos (cosseno) e copia a maioria |
# | Rede neural (MLP) | uma camada de neurônios aprende combinações de palavras |
# | Naive Bayes | combina a frequência de cada palavra em cada classe, supondo as palavras independentes |
# | Extra Trees | centenas de árvores de decisão com cortes sorteados votam juntas |
# | LightGBM | árvores em sequência, cada uma corrigindo os erros das anteriores |
#
# O **meta-modelo** (uma Regressão Logística) recebe as sete notas e aprende quanto pesar cada uma. As notas que o
# alimentam são *out-of-fold* (nenhum modelo dá nota a um e-mail que usou para treinar). O motor guarda essas notas
# em cache, e por isso dá para recalcular os 20 folds de confirmação aqui, em segundos:

# %%
m = Motor(base, log=lambda *a: None)
CONF = m.adicionar_divisoes([201, 202, 203, 204])
linhas = []
for c in CONF:
    yte = m.y_teste(c)
    linhas.append(dict(metodo="Stacking 7 padrão", **metricas.metricas(yte, *m.prever_stacking(modelos.STACKING7_PADRAO, c))))
    linhas.append(dict(metodo="LinearSVC (melhor modelo único)", **metricas.metricas(yte, *m.prever_linearsvc(0, c))))
T = pd.DataFrame(linhas).groupby("metodo")[["mcc_custo", "pr_auc", "custo_por_mil"]].mean()
print(T.round(4).to_string())
mm = next(r for r in R["genetico"]["comparacoes"] if r["A"] == "Stacking 7 padrão" and r["B"] == "LinearSVC otimizado")
print(f"MCC (limiar 0,5): Stacking 7 − LinearSVC otimizado = {sn(mm['dif'])}, p de Holm {num(mm['p_holm'])} "
      f"({'significativo' if mm['p_holm'] < 0.05 else 'não significativo'} a 5%)")
cmp = {(r["A"], r["B"]): r for r in R["comparacoes_custo_20_folds"]}
for k in ["generalista_sem_chave", "especialista"]:
    r = cmp[(k, "linha_de_base")]
    print(f"custo: {F[k]['rotulo']} × LinearSVC = {sn(r['dif'], 1)} por mil ({pct(r['reducao_relativa'])} menos; IC95 "
          f"{sn(r['ic95_lo'], 1)} a {sn(r['ic95_hi'], 1)}; p de Holm {num(r['p_holm'])}; P(mais barato) {num(r['P_A_melhor'], 2)})")
fig_("04_erros_em_comum"); fig_("04_pesos_meta")
ec = R["stacking"]["erros_em_comum"]
juntos = pd.Series(ec["erra_junto"]).sort_values()
resg = pd.Series({r["familia"]: r["acerta quando a maioria (≥4) erra"] for r in R["stacking"]["tabela_erros"]}).sort_values(ascending=False)
print("quando outro membro erra, com que frequência cada um erra junto (menor = traz mais informação nova):",
      ", ".join(f"{k} {pct(v)}" for k, v in juntos.items()))
print("e-mails que cada um salva quando a maioria (4 ou mais) erra:", ", ".join(f"{k} {v}" for k, v in resg.items()))
print("peso no meta-modelo (média dos 20 folds):", ", ".join(f"{k} {num(v, 2)}" for k, v in sorted(R["stacking"]["pesos_meta"].items(), key=lambda x: -x[1])))
print("peso no Especialista entregue:", ", ".join(f"{k} {num(v, 2)}" for k, v in sorted(F["especialista"]["pesos_meta"].items(), key=lambda x: -x[1])))
eq13 = next(c for c in R["stacking"]["comparacoes_15_folds"] if c["B"] == "Stacking 13 (todos)")
print(f"Stacking 7 × Stacking 13: P(equivalentes) = {num(eq13['P_equivalentes'])}")

# %% [markdown]
# As árvores e o kNN são os que menos erram junto com os outros e os que mais salvam e-mails quando a maioria erra;
# o kNN, fraco sozinho, recebe o maior peso médio. O Naive Bayes erra muito e erra junto: peso quase zero. No
# Especialista, o meta-modelo chega a desligar a Reg. Logística e o Naive Bayes (os pesos dividem crédito entre
# membros correlacionados, então não são uma medida pura de confiança). Treze modelos empatam com sete. Contra o melhor modelo único, o stacking ganha no MCC e custa menos em média; a saída acima diz
# quais dessas diferenças são estatisticamente significativas com estes e-mails.

# %% [markdown]
# ## 7. O algoritmo genético e a maldição do vencedor
# O espaço de hiperparâmetros tem milhões de combinações (saída abaixo). Com o cache, cada avaliação é só o
# meta-modelo: o GA inteiro roda aqui em cerca de um minuto e meio.

# %%
BUSCA = m.adicionar_divisoes([11, 12], prefixo="b")
apt = genetico.Aptidao(m, BUSCA)
melhor, hist = genetico.algoritmo_genetico(apt)
g = R["genetico"]
print(f"{len(apt.memoria)} avaliações, de {g['combinacoes']:,} combinações possíveis".replace(",", "X").replace("X de", ", de").replace("X", "."))
print(f"MCC na busca: padrão {num(apt(modelos.STACKING7_PADRAO), 4)} → GA {num(apt(melhor), 4)} (ganho {sn(g['ganho_busca'])})")
lo, hi = g["ganho_sementes_novas_ic95"]
print(f"em 20 divisões de confirmação: ganho {sn(g['ganho_sementes_novas'])} (IC95 corrigido {sn(lo)} a {sn(hi)})")
fig_("05_ga_x_aleatoria")

# %% [markdown]
# Na busca, o GA melhora o MCC. Nas divisões de confirmação, o ganho encolhe e o intervalo de confiança inclui o
# zero: **o ganho não se confirmou**. O encolhimento é o efeito esperado de escolher o melhor entre centenas de
# candidatos nas mesmas divisões (a "maldição do vencedor"), mas o intervalo é largo demais para dizer quanto do
# ganho era otimismo. A configuração do GA virou o **Especialista**: é o filtro de menor custo nestas divisões, que
# são re-partições dos e-mails usados na escolha.

# %% [markdown]
# ## 8. O modelo de fundação: TabPFN
# O TabPFN-3.5 (Prior Labs) foi pré-treinado em milhões de tabelas sintéticas e classifica uma tabela nova **sem
# nenhum ajuste**. Ele entra de duas formas: sozinho e como 8º membro do stacking (**Stacking 8**).

# %%
tab = {r["metodo"]: r for r in R["tabpfn"]["interno"]}
print(f"nesta caixa, 20 folds: TabPFN sozinho MCC {num(tab['TabPFN-3.5-Plus']['mcc'])} × LinearSVC otimizado "
      f"{num(tab['LinearSVC otimizado']['mcc'])} (afinado) | PR-AUC: Stacking 8 {num(F['generalista']['cv20']['pr_auc'], 4)}, "
      f"Stacking 7 {num(F['generalista_sem_chave']['cv20']['pr_auc'], 4)}")
print(f"fora da caixa (Enron2-6), custo por mil: TabPFN sozinho {num(F['tabpfn']['externo']['custo_por_mil'], 1)} | "
      f"Stacking 8 {num(F['generalista']['externo']['custo_por_mil'], 1)} | Stacking 7 {num(F['generalista_sem_chave']['externo']['custo_por_mil'], 1)} | "
      f"LinearSVC {num(F['linha_de_base']['externo']['custo_por_mil'], 1)}")
print(f"e-mails reais na lixeira no Enron2-6: TabPFN {pct(F['tabpfn']['externo']['real_lixeira_custo'])} × "
      f"Stacking 7 {pct(F['generalista_sem_chave']['externo']['real_lixeira_custo'])}")
fig_("06_custo_dentro_fora")

# %% [markdown]
# Sem nenhum ajuste, o TabPFN empata com o melhor modelo único afinado nesta caixa. Fora dela, é o mais conservador:
# o que menos manda e-mail real para a lixeira. Por isso é o mais barato onde predominam e-mails reais e na média do
# Enron2-6, e caro onde predomina spam. Como 8º membro, deixa o stacking bem mais barato fora da caixa.

# %% [markdown]
# ## 9. A grande final: três avaliações, uma regra escrita antes
# A base externa (Enron2-6) sugeriu que o Stacking 8 generaliza melhor que o Stacking 7. Para decidir sem viés, a
# regra foi registrada **antes** de rodar numa terceira base que nenhuma escolha tinha usado (SpamAssassin):

# %%
tb = R["terceira_base"]; dcs = tb["decisao"]
print(f"SpamAssassin: {tb['n']} mensagens. Stacking 8 − Stacking 7: {sn(dcs['diferenca_custo'], 0)} por mil "
      f"(P(redução) = {num(dcs['p_reducao'])}); MCC no limiar de custo {sn(dcs['diferenca_mcc_custo'], 3)}")
print("decisão (D10):", "o Stacking 8 substitui o Stacking 7 como stacking para caixas novas" if dcs["stacking8_vira_generalista"]
      else "o Stacking 7 continua")
fig_("09_filtros_finais")
tab = pd.DataFrame({F[n]["rotulo"]: {"MCC (limiar de custo, 20 folds)": F[n]["cv20"]["mcc_custo"], "PR-AUC (20 folds)": F[n]["cv20"]["pr_auc"],
                                    "custo/mil (20 folds)": F[n]["cv20"]["custo_por_mil"],
                                    "Enron2-6": F[n]["externo"]["custo_por_mil"], "SpamAssassin": F[n]["spamassassin"]["custo_por_mil"],
                                    "TCR SpamAssassin": F[n]["spamassassin"]["tcr"]} for n in F}).T
print(tab.round(3).to_string())
for onde, chave in [("Enron2-6", "externo"), ("SpamAssassin", "spamassassin")]:
    ref = F["nao_filtrar"][chave]["custo_por_mil"]
    vence = [F[n]["rotulo"] for n in F if n != "nao_filtrar" and F[n][chave]["custo_por_mil"] < ref]
    print(f"{onde}: não filtrar custa {num(ref, 1)}; mais baratos que isso: {', '.join(vence) or 'nenhum filtro'}")

# %% [markdown]
# **Leitura.** Nesta caixa, o Especialista é o mais barato nos 20 folds (no teste trancado, empata com o padrão), e o
# Stacking 8 empata no topo do PR-AUC. Nas cinco
# caixas Enron2-6 somadas, todos os filtros valem a pena, e o TabPFN e o Stacking 8 são os que mais economizam;
# caixa a caixa, nas que têm maioria de e-mails reais (como o Enron2), os stackings chegam a perder para não
# filtrar. No
# SpamAssassin, com λ = 9, nenhum filtro treinado no Enron1 vence "não filtrar": o TabPFN sozinho chega mais perto, e
# o TabPFN também deixa os dois stackings bem mais baratos (a regra D10 trocou o Stacking 7 padrão pelo Stacking 8). **Para uma caixa muito diferente, o caminho é retreinar.**

# %% [markdown]
# ## 10. Contra-estudo: tentando derrubar as conclusões

# %%
ce = R["contra_estudo"]; pm = ce["permutacao"]
print(f"teste de sinal: com os rótulos embaralhados {pm['n_permutacoes']} vezes, o MCC vai de {num(pm['mcc_real'])} para "
      f"{sn(pm['mcc_permutado_media'], 3)} em média (máximo {sn(pm['mcc_permutado_max'], 3)}); p = {num(pm['p'])}, o menor possível. "
      f"O MCC não vem do acaso.")
q = ce["quase_copias"]
print(f"quase-cópias (cosseno ≥ {num(q['limiar_cosseno'], 2)}): {pct(q['folds_fracao'])} dos e-mails de teste nos 20 folds; "
      f"sem elas, o MCC do Stacking 7 vai de {num(q['mcc'], 4)} para {num(q['mcc_sem_quase_copias'], 4)}")
print(pd.DataFrame(ce["ablacao"]).set_index("variante").round(3).to_string())
fig_("07_curva_aprendizado")
venc = pd.DataFrame(ce["vencedor_por_custo"])
print("mais barato para cada λ:\n" + venc.round(1).to_string(index=False))
cv = tb["conversor_spam"]
print(f"controle da 3ª base: treinado e testado dentro do SpamAssassin, MCC {num(tb['controle_dentro_da_base']['mcc'])}: "
      f"a representação funciona lá. Conversor no spam: a decisão quase não muda ({pct(cv['enron4_spam_passou_aueb'])} × "
      f"{pct(cv['enron4_spam_passou_conversor'])} de spam que passou).")
print(pd.DataFrame(tb["fontes_spam"])[["base", "email_real", "spam"]].to_string(index=False))
fig_("08_terceira_base")

# %% [markdown]
# O teste de permutação mostra que há sinal; a proteção contra vazamento é o agrupamento das cópias, e as
# quase-cópias inflam pouco o resultado. O spam de cada caixa vem de uma coleção: caixas com a mesma coleção não são
# independentes no spam, e o spam do SpamAssassin é parente do spam do Enron2 e do Enron5. Fora da caixa, o stacking
# erra mais nos e-mails reais e no spam: ele aprende o que é "normal" na caixa do treino e na coleção de spam dela.

# %% [markdown]
# ## 11. A demonstração: um e-mail, todas as etapas
# O filtro treinado fica na página *Releases* do GitHub; se não estiver em `resultados/modelos/`, é baixado aqui.

# %%
from spamfilter.filtro import FiltroStacking
esp = FiltroStacking.carregar(publicados.garantir("especialista"))
for ex in R["exemplos_demo"]:
    e = esp.explicar(ex["texto"])
    print(f"{ex['rotulo']:24s} cobertura {pct(e['cobertura'])} | P(spam) {pct(e['prob_spam'])} | limiar {pct(e['limiar'])} | "
          f"{'SPAM' if e['spam'] else 'e-mail real'}")

# %% [markdown]
# O exemplo em português é um e-mail real, mas o filtro, treinado em inglês, o manda para a lixeira: com cobertura
# baixa, a decisão não tem sentido.
#
# ## 12. Conclusões
# 1. **Conhecer os dados mudou tudo:** a base é o Enron1, a contagem é por substring e as duplicatas são e-mails
#    reais; o protocolo separa cópias por grupo.
# 2. **A representação vale mais que o algoritmo:** TF-IDF com log levou modelos simples ao topo.
# 3. **Modelos que erram diferente se completam:** um modelo de cada família, combinados por um meta-modelo,
#    superam o melhor modelo único no MCC e custam menos em média; treze modelos não superam sete.
# 4. **O ajuste fino rende pouco:** o ganho do algoritmo genético não se confirmou em divisões novas, e há um piso
#    estatístico abaixo do qual nenhuma diferença é detectável com estes e-mails.
# 5. **O TabPFN é a surpresa:** sem nenhum ajuste, empata com o melhor modelo único afinado; fora da caixa, é o
#    mais conservador com e-mails reais e deixa o stacking bem mais barato.
# 6. **O stacking aprende a caixa:** fora dela, tudo piora. Numa caixa muito diferente, com λ = 9, nenhum filtro
#    treinado aqui vence não filtrar; o caminho é retreinar com e-mails da caixa de destino.
#
# **Filtros:** entregues prontos para uso, o Especialista (Stacking 7 do GA), o Stacking 7 padrão (roda sem chave, usado
# na demo) e o LinearSVC (linha de base); avaliados com as notas gravadas do TabPFN, o Stacking 8 e o TabPFN sozinho. Detalhes, decisões e números: `docs/REGISTRO_DECISOES.md`.
