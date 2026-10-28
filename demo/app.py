"""Demonstração: cole um e-mail e veja cada etapa, da conversão até a decisão do filtro.

    streamlit run demo/app.py

Usa os filtros salvos por scripts/09_modelos_finais.py (resultados/modelos/*.joblib); se faltarem, baixa da Release
v1.0 do GitHub. Roda sem chave: o Stacking 8 e o TabPFN precisam da nuvem da Prior Labs, então a demo mostra o
Especialista e o Stacking 7 padrão.
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from spamfilter import conversao, publicados  # noqa: E402
from spamfilter.filtro import FiltroStacking  # noqa: E402
from spamfilter.modelos import FAMILIAS  # noqa: E402

AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#898781"
COR_FAMILIA = dict(zip(FAMILIAS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))

st.set_page_config(page_title="Filtro de spam: passo a passo", page_icon="✉️", layout="wide")


@st.cache_resource
def _carregar_filtros():
    # se o download falhar, a exceção sobe e nada fica no cache: a próxima visita tenta de novo
    return {"Especialista (esta caixa)": FiltroStacking.carregar(publicados.garantir("especialista")),
            "Stacking 7 padrão (roda sem chave)": FiltroStacking.carregar(publicados.garantir("stacking7_padrao"))}


def filtros():
    try:
        return _carregar_filtros()
    except Exception:
        return None


@st.cache_data
def resultados():
    arq = RAIZ / "resultados" / "metricas_finais.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}


F = filtros()
R = resultados()
st.title("✉️ Filtro de spam, passo a passo")
st.caption("Projeto 3 · Data Science 2 · ESEG. Cole um e-mail (texto puro ou a mensagem completa, com cabeçalhos) "
           "e acompanhe cada etapa até a decisão.")
if F is None:
    st.error("Os filtros não estão em `resultados/modelos/` e não foi possível baixá-los da Release do GitHub. "
             "Baixe-os da página Releases ou rode: `python scripts/09_modelos_finais.py`")
    st.stop()
if R:
    fin = R["finais"]
    st.info(f"ℹ️ Os filtros foram treinados numa única caixa de e-mail (a de um funcionário da Enron: "
            f"{R['base']['n_email_real']} e-mails reais e {R['base']['n_spam']} spams). Em caixas diferentes eles erram muito mais: "
            f"no SpamAssassin, {fin['especialista']['spamassassin']['real_lixeira_custo']:.0%} dos e-mails reais foram para a lixeira "
            f"com o Especialista e {fin['generalista_sem_chave']['spamassassin']['real_lixeira_custo']:.0%} com o Stacking 7 padrão. "
            "Leia a decisão como um exercício, não como um veredito sobre o seu e-mail.")

ex = {f"{e['rotulo']} — {e['origem']}": e["texto"] for e in R.get("exemplos_demo", [])}
escolha = st.selectbox("Comece com um exemplo (e-mails que o treino nunca viu) ou cole o seu abaixo:", ["(colar meu e-mail)"] + list(ex))
texto = st.text_area("E-mail", value=ex.get(escolha, ""), height=220,
                     placeholder="Subject: ...\n\nTexto do e-mail")
if not texto.strip():
    st.stop()

esp = F["Especialista (esta caixa)"]
vocab = np.array(esp.vocab)

# 1-2. conversão
st.header("1. O texto colado → 2. o formato da base (AUEB)")
aueb = conversao.para_aueb(texto)
c1, c2 = st.columns(2)
c1.text_area("como chegou", texto[:3000], height=220, disabled=True)
c2.text_area("como a base foi construída: só o assunto + corpo, minúsculas, pontuação separada", aueb[:3000], height=220, disabled=True)

# 3. contagem
st.header("3. Contagem das 3000 palavras do vocabulário")
v = conversao.vetor(aueb, vocab)
cob = conversao.cobertura(aueb, vocab)
k1, k2, k3 = st.columns(3)
k1.metric("colunas diferentes de zero", f"{int((v > 0).sum())} de 3000")
k2.metric("cobertura do vocabulário", f"{cob:.0%}")
k3.metric("contagem total", f"{int(v.sum())}")
if cob < 0.6:
    st.warning("⚠️ Cobertura baixa: o filtro não conhece boa parte deste vocabulário (por exemplo, um texto em português "
               "ou de outro assunto). A decisão abaixo é ainda menos confiável. Cobertura alta também não garante acerto: "
               "o que pesa é se o e-mail se parece com a caixa do treino.")
st.caption("A contagem é por *substring* e diferencia maiúsculas: 'ect' também conta dentro de 'subject' e 'project'; "
           "as 26 letras soltas também são colunas.")
nz = np.flatnonzero(v)
tab = pd.DataFrame({"palavra": vocab[nz], "contagem": v[nz], "presença (0/1)": 1}).sort_values("contagem", ascending=False)
st.dataframe(tab[tab.palavra.str.len() >= 3].head(40), hide_index=True, width="stretch", height=260)

# 4. vetor
st.header("4. O vetor: contagens e presença (0/1)")
st.caption("Cada célula é uma das 3000 colunas (50 × 60). Escuro = a palavra aparece.")
grade = pd.DataFrame({"x": np.arange(3000) % 60, "y": np.arange(3000) // 60, "contagem": np.log1p(v), "palavra": vocab, "n": v})
st.altair_chart(alt.Chart(grade).mark_rect().encode(
    x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None),
    color=alt.Color("contagem:Q", scale=alt.Scale(range=["#f0efec", "#0d366b"]), legend=None),
    tooltip=["palavra", "n"]).properties(height=260), width="stretch")

# 5. TF-IDF
st.header("5. TF-IDF com log")
X = sparse.csr_matrix(v.astype(np.float32)[None, :])
t = esp.palavras[0][0].transform(X).toarray()[0]
top = np.argsort(-t)[:15]
st.caption("O log achata as repetições (contar 'free' 10 vezes não é 10 vezes mais evidência) e o IDF dá menos peso "
           "às palavras que aparecem em quase todo e-mail.")
st.dataframe(pd.DataFrame({"palavra": vocab[top], "contagem": v[top], "TF-IDF": t[top].round(3)}), hide_index=True, width="stretch")

# 6-8. os dois filtros
COMO_DECIDE = {
    "Reg. Logística": "dá um peso a cada palavra, soma e transforma a soma numa probabilidade",
    "SVM RBF": "resume as palavras em \"temas\" (SVD) e traça uma fronteira curva; a nota é a distância à fronteira",
    "kNN cosseno": "procura os e-mails de treino mais parecidos e copia a maioria",
    "MLP": "rede neural: uma camada de neurônios aprende combinações de palavras",
    "Naive Bayes": "combina a frequência de cada palavra em spam e em e-mail real, supondo as palavras independentes",
    "Extra Trees": "centenas de árvores de decisão com cortes sorteados votam juntas",
    "LightGBM": "árvores em sequência, cada uma corrigindo os erros das anteriores",
}
with st.expander("O que é cada um dos 7 modelos? (explicação completa em docs/MODELOS.md)"):
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in COMO_DECIDE.items()) +
                "\n\nO **meta-modelo** recebe as 7 notas e aprende, no treino, quanto pesar cada uma. O **limiar de custo** "
                "decide a partir de que probabilidade o e-mail vai para a lixeira.")
for nome, f in F.items():
    e = f.explicar(texto)
    st.header(f"6–8. {nome}")
    a, b = st.columns(2)
    notas = pd.DataFrame({"modelo": FAMILIAS, "nota": [e["notas_membros"][k] for k in FAMILIAS]})
    a.markdown("**6. A nota de cada um dos 7 modelos** (probabilidade de spam; o SVM dá uma distância à fronteira)")
    a.altair_chart(alt.Chart(notas).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y("modelo:N", sort=None, title=None), x=alt.X("nota:Q", title="nota"),
        color=alt.Color("modelo:N", scale=alt.Scale(domain=FAMILIAS, range=list(COR_FAMILIA.values())), legend=None),
        tooltip=["modelo", alt.Tooltip("nota:Q", format=".3f")]).properties(height=240), width="stretch")
    contr = pd.DataFrame({"modelo": FAMILIAS + ["constante"], "contribuição": [e["contribuicoes"][k] for k in FAMILIAS] + [e["intercepto"]]})
    b.markdown("**7. A soma ponderada do meta-modelo** (quanto cada nota empurra para spam ou para e-mail real)")
    b.altair_chart(alt.Chart(contr).mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y("modelo:N", sort=None, title=None), x=alt.X("contribuição:Q"),
        color=alt.condition(alt.datum["contribuição"] > 0, alt.value(LARANJA), alt.value(AZUL)),
        tooltip=["modelo", alt.Tooltip("contribuição:Q", format="+.2f")]).properties(height=240), width="stretch")
    p, lim = e["prob_spam"], e["limiar"]
    st.markdown(f"**8. Probabilidade de spam = {p:.1%}**. Limiar de custo = {lim:.1%} (perder 1 e-mail real custa o mesmo que receber 9 spams).")
    st.progress(min(max(p, 0.0), 1.0))
    if e["spam"]:
        st.error(f"🗑️ **SPAM** — {p:.1%} está acima do limiar de {lim:.1%}")
    else:
        st.success(f"📥 **E-MAIL REAL** — {p:.1%} está abaixo do limiar de {lim:.1%}")

# 9. palavras
st.header("9. As palavras que mais pesaram")
e = esp.explicar(texto)
a, b = st.columns(2)
a.markdown("**empurram para spam**"); a.dataframe(pd.DataFrame(e["empurram_spam"].items(), columns=["palavra", "efeito"]).round(3), hide_index=True)
b.markdown("**empurram para e-mail real**"); b.dataframe(pd.DataFrame(e["empurram_real"].items(), columns=["palavra", "efeito"]).round(3), hide_index=True)
st.caption("Efeito = TF-IDF da palavra × peso numa Regressão Logística treinada nos 5172 e-mails (uma aproximação legível; "
           "o filtro de verdade é o stacking acima).")
