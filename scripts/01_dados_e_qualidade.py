# %% [markdown]
# # 01. Dados e qualidade
#
# **Pergunta:** de onde vêm os 5172 e-mails, como as palavras foram contadas e o que pode enganar um modelo?
#
# **Termos.** *E-mail real* = e-mail legítimo (classe 0). *Spam* = classe 1. Cada coluna conta quantas vezes uma
# palavra aparece no e-mail.

# %%
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from spamfilter import dados, externo, graficos, saidas
import matplotlib.pyplot as plt

base = dados.carregar()
A = base.X.toarray()
vocab = np.array(base.vocab)
print(f"{base.n} e-mails × {len(vocab)} palavras | e-mails reais: {(base.y == 0).sum()} | spam: {(base.y == 1).sum()} "
      f"({base.y.mean():.1%})")

# %% [markdown]
# ## 1. A acurácia engana
# Um "filtro" que diz *e-mail real* para tudo acerta 71% sem olhar nada. Por isso o projeto mede com MCC, PR-AUC
# e o custo de cada tipo de erro, e não com acurácia.

# %%
dummy_acuracia = float((base.y == 0).mean())
print(f"acurácia de quem nunca marca spam: {dummy_acuracia:.1%}")

# %% [markdown]
# ## 2. Uma matriz quase vazia
# A maioria das 3000 palavras não aparece na maioria dos e-mails.

# %%
fracao_zeros = float((A == 0).mean())
distintas = (A > 0).sum(1)
total = A.sum(1)
por_classe = pd.DataFrame({"palavras distintas (mediana)": pd.Series(distintas).groupby(base.y).median(),
                           "contagem total (mediana)": pd.Series(total).groupby(base.y).median()})
por_classe.index = ["e-mail real", "spam"]
print(f"células com zero: {fracao_zeros:.1%} | palavras distintas por e-mail (mediana): {np.median(distintas):.0f}")
print(por_classe.to_string())

fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.arange(0, 401, 10)
for c in (0, 1):
    ax.hist(distintas[base.y == c], bins=bins, histtype="step", lw=2, color=graficos.CLASSE[c],
            density=True, label=graficos.NOME_CLASSE[c])
ax.set_xlabel("palavras distintas do vocabulário no e-mail"); ax.set_ylabel("densidade")
ax.set_title("Quantas das 3000 palavras cada e-mail usa")
graficos.legenda_classes(ax)
graficos.mostrar(fig, "01_palavras_por_email")

# %% [markdown]
# ## 3. De onde vêm os dados
# O CSV é o **Enron1** do corpus Enron-Spam (Metsis et al., 2006): e-mails reais da caixa de um funcionário da Enron
# (`farmer-d`, 1999 a 2002) e spams coletados em 2003 a 2005. Contando cada palavra como *substring* do texto
# pré-processado da AUEB (`texto.count(palavra)`, diferenciando maiúsculas), reproduzimos **todas** as linhas.

# %%
textos, y_aueb, nomes = externo.ler_tar(1)
X1 = externo.vetorizar(textos, vocab)
linhas_csv = pd.Series([r.astype(np.int64).tobytes() for r in A]).value_counts()
linhas_aueb = pd.Series([r.astype(np.int64).tobytes() for r in X1]).value_counts()
reproduzidas = int(sum(min(n, linhas_aueb.get(k, 0)) for k, n in linhas_csv.items()))
print(f"linhas do CSV reproduzidas a partir dos textos originais: {reproduzidas}/{base.n}")

i = next(j for j, n in enumerate(nomes) if "/spam/" in n)
print("\nexemplo (início de um spam original):\n" + textos[i][:300])
v = np.array(X1[i]); longas = np.array([len(w) >= 3 for w in vocab])
top = [j for j in np.argsort(-v) if longas[j]][:8]
print("\npalavras (3+ letras) mais contadas nesse e-mail:", {str(vocab[j]): int(v[j]) for j in top})

# %% [markdown]
# **Um detalhe que muda a leitura das colunas:** o vocabulário do autor do CSV inclui as 26 letras soltas e 127 pares
# de letras, e a contagem é por *substring*. Então a coluna "e" conta todas as letras "e" do e-mail, e "ect" também
# conta pedaços de "subject", "project" e "direct". As colunas medem mais *tamanho e estilo do texto* do que palavras
# inteiras. Não há coluna com algarismo nem com sinal de pontuação.

# %%
tamanhos = pd.Series([min(len(w), 6) for w in vocab]).value_counts().sort_index()
tamanhos.index = ["1 letra", "2 letras", "3 letras", "4 letras", "5 letras", "6 ou mais"]
print(tamanhos.to_string())
print(f"\ncontagem da letra 'e' nesse spam: {int(v[base.vocab.index('e')])}")

# %% [markdown]
# ## 4. Cópias (linhas de contagem iguais): são e-mails reais, não spam
# Hipótese inicial: "duplicata indica spam" (campanhas repetidas). Os dados dizem o contrário: as cópias são
# sobretudo e-mails reais de modelo fixo (avisos e relatórios automáticos).
#
# **Consequência para o protocolo:** se uma cópia fica no treino e a outra no teste, o modelo "lembra" a resposta.
# Por isso toda divisão treino/teste é feita **por grupo** (cópias sempre do mesmo lado).

# %%
dup = dados.resumo_duplicatas(base)
g = pd.Series(base.g)
tamanho = g.map(g.value_counts())
conflito = pd.DataFrame({"g": base.g, "y": base.y}).groupby("g").y.nunique().gt(1).sum()
regra = ((tamanho > 1).astype(int).to_numpy() == base.y).mean()
excedentes_por_classe = pd.Series(base.y[g.duplicated().to_numpy()]).map(graficos.NOME_CLASSE).value_counts()
print(f"cópias excedentes: {dup['duplicatas_excedentes']} ({dup['fracao_duplicatas_email_real']:.0%} delas são e-mails reais)")
print(excedentes_por_classe.to_string())
print(f"grupos com rótulos diferentes para o mesmo vetor: {int(conflito)}")
print(f"acurácia da regra 'tem cópia => spam': {regra:.1%} (pior que nunca marcar spam: {dummy_acuracia:.1%})")

# %% [markdown]
# **O que conta como cópia.** Duas linhas com as **mesmas 3000 contagens**. Como as 26 letras também são colunas, isso
# só acontece quando os textos têm exatamente as mesmas letras; o que pode mudar são algarismos, pontuação, espaços e
# a ordem dos trechos, que o vocabulário não vê. Com os textos originais da AUEB, separamos os grupos de cópias pelo
# que muda entre os textos.

# %%
chave_aueb = [r.astype(np.int64).tobytes() for r in X1]
membros = defaultdict(list)
for k, t, n, c in zip(chave_aueb, textos, nomes, y_aueb):
    membros[k].append((t, n, int(c)))
grupos_copia = {k: v for k, v in membros.items() if len(v) > 1}
tam = pd.Series({k: len(v) for k, v in grupos_copia.items()})
identico = {k: len({t for t, _, _ in v}) == 1 for k, v in grupos_copia.items()}
# o que muda entre os textos de um grupo: só espaços e quebras de linha; algarismos e sinais (as letras, na mesma
# ordem, são iguais); ou a ordem dos trechos (as mesmas letras em outra ordem)
so_espacos = lambda t: re.sub(r"\s+", " ", t).strip()
so_letras = lambda t: re.sub(r"[^a-z]", "", t.lower())
def diferenca(v):
    ts = {t for t, _, _ in v}
    if len(ts) == 1:
        return "identico"
    if len({so_espacos(t) for t in ts}) == 1:
        return "espacos"
    return "numeros_e_sinais" if len({so_letras(t) for t in ts}) == 1 else "ordem"
dif = {k: diferenca(v) for k, v in grupos_copia.items()}
sem_letras = lambda t: not re.search(r"[a-z]", t.lower().removeprefix("subject:"))
classe_grupo = {k: {c for _, _, c in v} for k, v in grupos_copia.items()}
assunto = lambda t: t.split("\n", 1)[0].removeprefix("Subject:").strip()
k_maior = tam.idxmax()
k_spam = max((k for k in grupos_copia if classe_grupo[k] == {1}), key=lambda k: tam[k])
COPIAS = dict(
    criterio="mesmas 3000 contagens (hash da linha)",
    grupos=int(len(grupos_copia)), linhas_em_grupos=int(tam.sum()), grupos_pares=int((tam == 2).sum()),
    maior_grupo=int(tam.max()),
    grupos_so_email_real=sum(1 for k in grupos_copia if classe_grupo[k] == {0}),
    grupos_so_spam=sum(1 for k in grupos_copia if classe_grupo[k] == {1}),
    grupos_texto_identico=sum(identico.values()), grupos_texto_diferente=sum(not v for v in identico.values()),
    grupos_diferem_so_em_espacos=sum(d == "espacos" for d in dif.values()),
    grupos_diferem_em_numeros_e_sinais=sum(d == "numeros_e_sinais" for d in dif.values()),
    grupos_diferem_na_ordem=sum(d == "ordem" for d in dif.values()),
    maior=dict(n=int(tam[k_maior]), classe="e-mail real" if classe_grupo[k_maior] == {0} else "spam",
               textos_distintos=len({t for t, _, _ in grupos_copia[k_maior]}),
               exemplo_assunto=assunto(grupos_copia[k_maior][0][0]),
               diferenca=dif[k_maior]),
    maior_spam=dict(n=int(tam[k_spam]), textos_distintos=len({t for t, _, _ in grupos_copia[k_spam]}),
                    sem_letras=all(sem_letras(t) for t, _, _ in grupos_copia[k_spam])),
)
print(f"grupos de cópias: {COPIAS['grupos']} ({COPIAS['linhas_em_grupos']} linhas); pares: {COPIAS['grupos_pares']}; "
      f"maior grupo: {COPIAS['maior_grupo']}")
print(f"só e-mails reais: {COPIAS['grupos_so_email_real']} | só spam: {COPIAS['grupos_so_spam']}")
print(f"texto idêntico: {COPIAS['grupos_texto_identico']} | texto diferente, mesma contagem: {COPIAS['grupos_texto_diferente']} "
      f"(só espaços: {COPIAS['grupos_diferem_so_em_espacos']}; algarismos e sinais: {COPIAS['grupos_diferem_em_numeros_e_sinais']}; "
      f"ordem dos trechos: {COPIAS['grupos_diferem_na_ordem']})")
print("maior grupo:", COPIAS["maior"])
print("maior grupo de spam:", COPIAS["maior_spam"])

# %% [markdown]
# ## 5. As palavras que mais separam as classes
# Razão entre a fração de spams e a fração de e-mails reais que contêm a palavra (em log). Positivo = típica de spam.

# %%
p_spam = (A[base.y == 1] > 0).mean(0); p_real = (A[base.y == 0] > 0).mean(0); eps = 1e-3
razao = pd.Series(np.log((p_spam + eps) / (p_real + eps)), index=vocab)
frequente = pd.Series(np.maximum(p_spam, p_real), index=vocab) >= 0.05      # aparece em ≥ 5% de alguma classe
top_spam = razao[frequente].sort_values(ascending=False).head(12)
top_real = razao[frequente].sort_values().head(12)

fig, ax = plt.subplots(figsize=(7, 5.6))
s = pd.concat([top_real.iloc[::-1], top_spam.iloc[::-1]])
cores = [graficos.REAL if v < 0 else graficos.SPAM for v in s.values]
ax.barh(range(len(s)), s.values, height=0.6, color=cores)
ax.set_yticks(range(len(s)), s.index); ax.axvline(0, color=graficos.EIXO, lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("log(fração de spams com a palavra / fração de e-mails reais com a palavra)")
ax.set_title("Palavras que mais separam spam de e-mail real")
graficos.legenda_classes(ax, loc="lower right")
graficos.mostrar(fig, "01_palavras_que_separam")

dominio = ["enron", "ect", "hou", "hpl", "daren", "farmer", "meter", "forwarded"]
tab_dominio = pd.DataFrame({"em e-mails reais": p_real[[base.vocab.index(w) for w in dominio]],
                            "em spams": p_spam[[base.vocab.index(w) for w in dominio]]}, index=dominio)
print("palavras da própria empresa (fração de e-mails que as contêm):")
print((tab_dominio * 100).round(1).to_string())

# %% [markdown]
# **Leitura.** Parte do que separa as classes é o vocabulário *da Enron* ("enron", "ect", "hou", nomes de pessoas):
# um atalho que funciona nesta caixa e não em outras. O contra-estudo (script 07) mede quanto o filtro depende dele.

# %%
saidas.gravar("01_dados", {
    "base": dict(n_emails=base.n, n_email_real=int((base.y == 0).sum()), n_spam=int((base.y == 1).sum()),
                 n_palavras=len(vocab), fracao_zeros=fracao_zeros, palavras_distintas_mediana=float(np.median(distintas)),
                 duplicatas_excedentes=dup["duplicatas_excedentes"], textos_distintos=dup["textos_distintos"],
                 fracao_duplicatas_email_real=dup["fracao_duplicatas_email_real"], grupos_rotulo_conflitante=int(conflito),
                 colunas_1_letra=int(tamanhos.iloc[0]), colunas_2_letras=int(tamanhos.iloc[1]),
                 duplicatas_email_real=int(excedentes_por_classe.get("e-mail real", 0)),
                 duplicatas_spam=int(excedentes_por_classe.get("spam", 0)),
                 acuracia_regra_copia_spam=float(regra), linhas_reproduzidas=reproduzidas, copias=COPIAS,
                 origem="Enron1 (caixa farmer-d + spams GP), corpus Enron-Spam da AUEB",
                 contagem="substring, sensível a maiúsculas, sobre o texto pré-processado da AUEB"),
    "metricas_didaticas": dict(dummy_acuracia=dummy_acuracia),
    "palavras": dict(spam=[dict(palavra=w, log_razao=v) for w, v in top_spam.items()],
                     email_real=[dict(palavra=w, log_razao=v) for w, v in top_real.items()]),
    "dominio": [dict(palavra=w, fracao_email_real=r["em e-mails reais"], fracao_spam=r["em spams"]) for w, r in tab_dominio.iterrows()],
}, fonte="scripts/01_dados_e_qualidade.py")
print("gravado: resultados/01_dados.json")
