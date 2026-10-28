"""Gera README.md e docs/MODEL_CARD.md a partir de resultados/metricas_finais.json (nenhum resultado digitado à mão).

    python scripts/gerar_documentos.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = json.loads((RAIZ / "resultados" / "metricas_finais.json").read_text(encoding="utf-8"))
REPO_ID = "felipe44776-eseg/filtro-spam-stacking-enron"
REPO = f"https://github.com/{REPO_ID}"
PAGES = "https://felipe44776-eseg.github.io/filtro-spam-stacking-enron"
COLAB = f"https://colab.research.google.com/github/{REPO_ID}/blob/main"
SELO = "https://colab.research.google.com/assets/colab-badge.svg"


def n(x, c=3):
    return f"{x:.{c}f}".replace(".", ",")


def pct(x, c=1):
    return f"{100 * x:.{c}f}%".replace(".", ",")


def sn(x, c=4):
    """Com sinal e vírgula decimal: +0,0063."""
    return f"{x:+.{c}f}".replace(".", ",")


def milhar(x):
    return f"{x:,}".replace(",", ".")


def n0(x, c=3):
    """Como n(), mas sem o sinal de menos em zeros arredondados ("-0,000")."""
    return n(0.0 if abs(x) < 0.5 * 10 ** -c else x, c)


F, B, G, A = R["finais"], R["base"], R["genetico"], R["achados"]
TB, CE, NS, TP = R["terceira_base"], R["contra_estudo"], R["nao_supervisionado"], R["tabpfn"]
dec, PERM, QC, CONV = TB["decisao"], CE["permutacao"], CE["quase_copias"], TB["conversor_spam"]
CMP = {(r["A"], r["B"]): r for r in R["comparacoes_custo_20_folds"]}
FONTES = {f["base"]: f for f in TB["fontes_spam"]}
TT = R["protocolo"]["teste_trancado"]
ORDEM = ["especialista", "generalista", "tabpfn", "generalista_sem_chave", "linha_de_base", "nao_filtrar"]
O_QUE = {"especialista": "Stacking 7 afinado pelo algoritmo genético", "generalista": "Stacking 7 padrão + TabPFN-3.5 como 8º membro",
         "tabpfn": "modelo de fundação para tabelas, sem nenhum ajuste", "generalista_sem_chave": "um modelo de cada família, hiperparâmetros padrão",
         "linha_de_base": "LinearSVC com TF-IDF log", "nao_filtrar": "referência: todo spam passa, nenhum e-mail real se perde"}
BRILHA = {"especialista": "o menor custo nos 20 folds (no teste trancado, empata com o padrão)",
          "generalista": "com o TabPFN, o stacking fica bem mais barato fora da caixa (regra D10)",
          "tabpfn": "o mais conservador fora da caixa: o que menos manda e-mail real para a lixeira",
          "generalista_sem_chave": "roda sem internet e sem chave (usado na demo)",
          "linha_de_base": "simples e rápido", "nao_filtrar": "o piso que todo filtro precisa vencer"}
NEGRITO = {"especialista", "generalista", "tabpfn"}

linhas = []
for k in ORDEM:
    f = F[k]; tt = f["teste_trancado"] or {}
    nome = f"**{f['rotulo']}**" if k in NEGRITO else f["rotulo"]
    pr = "–" if k == "nao_filtrar" else n(f["cv20"]["pr_auc"], 4)
    mcc = "–" if k == "nao_filtrar" else n(f["cv20"]["mcc_custo"])
    linhas.append(f"| {nome} | {O_QUE[k]} | {BRILHA[k]} | {mcc} | {pr} | {n(f['cv20']['custo_por_mil'], 1)} | "
                  f"{tt.get('email_real_lixeira', '–')} / {tt.get('spam_passou_n', '–')} | {n(f['externo']['custo_por_mil'], 1)} | "
                  f"{n(f['spamassassin']['custo_por_mil'], 1)} |")
tabela = "\n".join(linhas)

c7 = CMP[("generalista_sem_chave", "linha_de_base")]; ce = CMP[("especialista", "linha_de_base")]
esp_pad = CMP[("especialista", "generalista_sem_chave")]; s8_s7 = CMP[("generalista", "generalista_sem_chave")]
mcc_s7 = next(r for r in G["comparacoes"] if r["A"] == "Stacking 7 padrão" and r["B"] == "LinearSVC otimizado")
s8_lb = CMP[("generalista", "linha_de_base")]
tab_in = {r["metodo"]: r for r in TP["interno"]}
mais_barato_fora = min((k for k in ORDEM if k != "nao_filtrar"), key=lambda k: F[k]["externo"]["custo_por_mil"])
vence_sa = [F[k]["rotulo"] for k in ORDEM if k != "nao_filtrar" and F[k]["spamassassin"]["custo_por_mil"] < F["nao_filtrar"]["spamassassin"]["custo_por_mil"]]
lo, hi = G["ganho_sementes_novas_ic95"]
venc_lam = [v for v in CE["vencedor_por_custo"] if v["onde"] == "Enron2-6"]
lam_tab = [str(v["custo_fp"]) for v in venc_lam if v["metodo"].startswith("TabPFN")]
lam_outros = [f"λ = {v['custo_fp']}: {v['metodo']}" for v in venc_lam if not v["metodo"].startswith("TabPFN")]
def lista(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " e " + xs[-1]
sa_lam = [r for r in TB["sensibilidade_custo"] if r["metodo"] != "Não filtrar (referência)" and r["custo_fp"] != 9
          and r["custo_por_mil"] < next(x["custo_por_mil"] for x in TB["sensibilidade_custo"] if x["metodo"] == "Não filtrar (referência)" and x["custo_fp"] == r["custo_fp"])]
_por_lam = {}
for r in sa_lam:
    _por_lam.setdefault(r["custo_fp"], []).append(r["metodo"])
frase_sa_lam = ("Com outro peso para o erro, a conta muda: " + "; ".join(
    f"com λ = {k}, {' e '.join(v)} {'vence' if len(v) == 1 else 'vencem'} não filtrar" for k, v in sorted(_por_lam.items())) + ". ") if sa_lam else ""

# caixa a caixa (λ = 9)
BASES = sorted(F["especialista"]["externo"]["por_base"])
FILT = [k for k in ORDEM if k != "nao_filtrar"]
pb = lambda k, b, campo: F[k]["externo"]["por_base"][b][campo]
tab_menos_lixeira = (all(pb("tabpfn", b, "real_lixeira_custo") <= min(pb(k, b, "real_lixeira_custo") for k in FILT) for b in BASES)
                     and F["tabpfn"]["spamassassin"]["real_lixeira_custo"] <= min(F[k]["spamassassin"]["real_lixeira_custo"] for k in FILT))
tab_mais_barato = [b for b in BASES if min(FILT, key=lambda k: pb(k, b, "custo_por_mil")) == "tabpfn"]
if min(FILT, key=lambda k: F[k]["spamassassin"]["custo_por_mil"]) == "tabpfn":
    tab_mais_barato.append("SpamAssassin")
tab_mais_caro = [b for b in BASES if max(FILT, key=lambda k: pb(k, b, "custo_por_mil")) == "tabpfn"]
perde_nf = {b: [F[k]["rotulo"] for k in FILT if pb(k, b, "custo_por_mil") > pb("nao_filtrar", b, "custo_por_mil")] for b in BASES}
perde_txt = "; ".join(f"no {b}, {lista(v)}" for b, v in perde_nf.items() if v)
tcr_txt = lambda onde: ", ".join(f"{F[k]['rotulo']} {n(F[k][onde]['tcr'], 2)}" for k in FILT)
# o TabPFN como 8º membro: o stacking fica mais barato fora da caixa, nas duas versões e nas duas bases?
TEX = {r["metodo"]: r for r in TP["externo"]}; TSA = {r["metodo"]: r for r in TB["metodos"]}
PARES8 = [("Stacking 7 padrão", "Stacking 8 padrão (+TabPFN)"), ("Stacking 7 especialista", "Stacking 8 especialista (+TabPFN)")]
barateia = all(TEX[b8]["custo_por_mil"] < TEX[b7]["custo_por_mil"] and TSA[b8]["custo_por_mil"] < TSA[b7]["custo_por_mil"] for b7, b8 in PARES8)
b8txt = "; ".join(f"{b7.replace('Stacking 7 ', '')}: {n(TEX[b7]['custo_por_mil'], 1)} → {n(TEX[b8]['custo_por_mil'], 1)} no Enron2-6 e "
                  f"{n(TSA[b7]['custo_por_mil'], 1)} → {n(TSA[b8]['custo_por_mil'], 1)} no SpamAssassin" for b7, b8 in PARES8)
LS = TB["sensibilidade_custo"]
_c = lambda metodo, l: next(x["custo_por_mil"] for x in LS if x["metodo"] == metodo and x["custo_fp"] == l)
dif_lam = [_c("Stacking 8 padrão (+TabPFN)", l) - _c("Stacking 7 padrão", l) for l in sorted({x["custo_fp"] for x in LS})]
pr_top = max(FILT, key=lambda k: F[k]["cv20"]["pr_auc"])
pr_empate = max(F[k]["cv20"]["pr_auc"] for k in FILT) - sorted(F[k]["cv20"]["pr_auc"] for k in FILT)[-2] < 0.001
# o tópico do NMF com mais spam, usado como filtro (contas exatas a partir do tamanho e da fração de spam do tópico)
TOP_NMF = max(NS["nmf_topicos"], key=lambda t: t["fracao_spam"])
_vp = TOP_NMF["n_emails"] * TOP_NMF["fracao_spam"]; _fp = TOP_NMF["n_emails"] - _vp; _fn = B["n_spam"] - _vp; _vn = B["n_emails"] - _vp - _fp - _fn
MCC_NMF = (_vp * _vn - _fp * _fn) / ((_vp + _fp) * (_vp + _fn) * (_vn + _fp) * (_vn + _fn)) ** 0.5
NMF_FRAC_SPAM = _vp / B["n_spam"]
CMP_R2 = {c["B"]: c for c in R["rodada2"]["comparacoes"]}
st4_lsvc = CMP_R2["LinearSVC TF-IDF"]

NOTEBOOKS = sorted((RAIZ / "notebooks").glob("[0-9][0-9]_*.ipynb"))
def titulo(nb: Path) -> str:
    fonte = RAIZ / "scripts" / (nb.stem + ".py")
    for l in fonte.read_text(encoding="utf-8").splitlines():
        if l.startswith("# # "):
            return l[4:].split(". ", 1)[-1]
    return nb.stem
lista_nb = "\n".join(f"| [`{nb.name}`](notebooks/{nb.name}) | {titulo(nb)} | [![Abrir no Colab]({SELO})]({COLAB}/notebooks/{nb.name}) |" for nb in NOTEBOOKS)

README = f"""# Filtro de spam com stacking e TabPFN: o que um classificador aprende de uma caixa de e-mail

Projeto 3 de **Data Science 2 (ESEG)**, com o Prof. Marino Catarino.
**Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins (RA 44776).

São {B['n_emails']} e-mails ({B['n_email_real']} e-mails reais e {B['n_spam']} spams), cada um descrito pela contagem de {B['n_palavras']} palavras. O ponto de partida foi um classificador. A investigação foi além: de onde vêm os dados, quais modelos erram de formas diferentes, quanto do ajuste fino é ilusão, o que um modelo de fundação (TabPFN) acrescenta e o que acontece quando o filtro encontra uma caixa de e-mail que nunca viu.

| | |
|---|---|
| 📓 Notebook para o professor | [`Projeto3_Classificacao_Spam.ipynb`](Projeto3_Classificacao_Spam.ipynb): a história completa, com o código principal visível, em poucos minutos · [![Abrir no Colab]({SELO})]({COLAB}/Projeto3_Classificacao_Spam.ipynb) |
| 🧪 Notebooks técnicos | uma etapa por notebook, todos executados e prontos para o Colab ([lista abaixo](#os-notebooks)) |
| 🎞️ Apresentações | [banca]({PAGES}/apresentacao/banca/) (técnica) · [equipe]({PAGES}/apresentacao/equipe/) (didática, com glossário) |
| ✉️ Demonstração | `streamlit run demo/app.py`: cole um e-mail e veja cada etapa até a decisão |
| ✅ Para a banca | [INSTALACAO.md](INSTALACAO.md) · [COMO_TESTAR.md](COMO_TESTAR.md) |
| 🧠 Os modelos, explicados | [docs/MODELOS.md](docs/MODELOS.md): cada modelo em uma frase, uma analogia e como ele decide; o stacking, o algoritmo genético e o TabPFN |
| ⏱️ Testar em 5 minutos | [COMO_TESTAR.md](COMO_TESTAR.md#teste-rápido-5-minutos): pelo Colab, sem instalar nada, ou com 3 comandos no computador |
| 🧭 Decisões e números | [docs/REGISTRO_DECISOES.md](docs/REGISTRO_DECISOES.md) · [docs/METODOLOGIA.md](docs/METODOLOGIA.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/REFERENCIAS.md](docs/REFERENCIAS.md) |

## A história em 8 capítulos

1. **A pergunta e os dados.** Separar spam de e-mail real em {B['n_emails']} e-mails. Descobrimos que o CSV é o Enron1 e que as palavras foram contadas como pedaços de texto.
2. **Primeiro, sem rótulo.** O K-means agrupa os e-mails por assunto (ARI {n(NS['kmeans_ari'], 2)}). O NMF acha um tópico de propaganda com {pct(NMF_FRAC_SPAM, 1)} do spam, mas um em cada três e-mails dele é real (MCC {n(MCC_NMF, 2)} como filtro). Conclusão: há pista sem rótulo, mas, para um filtro bom, precisamos das respostas, ou seja, de aprendizado supervisionado.
3. **Um torneio de {len(R['torneio'])} modelos supervisionados, com um ranking.** Mesma prova para todos, incluindo uma referência que nunca marca spam. Os lineares lideraram (o melhor modelo único foi o {sorted((r for r in R['torneio'] if r['familia'] not in ('combinação', 'referência')), key=lambda r: -r['mcc'])[0]['modelo']}, com MCC {n(sorted((r for r in R['torneio'] if r['familia'] not in ('combinação', 'referência')), key=lambda r: -r['mcc'])[0]['mcc'])}). O detalhe que mudou o ranking foi o TF-IDF com log. Os melhores ficaram muito perto uns dos outros, e o 1º lugar já era uma combinação simples de 4 modelos. Ranking completo: [docs/MODELOS.md](docs/MODELOS.md).
4. **Combinar um de cada família: o stacking.** Os modelos erram em e-mails diferentes, e um meta-modelo aprende a pesar as 7 notas. No limiar 0,5, o Stacking 7 superou o melhor modelo único no MCC (p de Holm {n(mcc_s7['p_holm'])}); no custo, ficou mais barato em média.
5. **Afinar: o algoritmo genético.** Avaliou {G['avaliacoes']} de {milhar(G['combinacoes'])} combinações de ajustes. O ganho da busca não se confirmou em divisões novas.
6. **Um convidado: o TabPFN.** Um modelo de fundação, sem nenhum ajuste, empatou com o melhor modelo único afinado e, como 8º membro, deixou o stacking bem mais barato fora da caixa.
7. **A prova final em três lugares:** esta caixa, outras 5 caixas da Enron e o SpamAssassin, sempre contra a referência "não filtrar". O stacking aprende a caixa e a coleção de spam.
8. **Lições e como testar:** a seguir, e no [COMO_TESTAR.md](COMO_TESTAR.md).

## Os filtros

Todos usam o **limiar de custo**: perder 1 e-mail real custa o mesmo que receber 9 spams (o cenário λ = 9 de Androutsopoulos et al., 2000). O custo é por mil e-mails; quanto menor, melhor. A última linha é a referência que todo filtro precisa vencer: **não filtrar** nada.

| Filtro | O que é | Onde brilha | MCC (20 folds) | PR-AUC (20 folds) | Custo (20 folds) | Teste trancado: reais na lixeira / spams que passaram | Custo Enron2-6 | Custo SpamAssassin |
|---|---|---|---|---|---|---|---|---|
{tabela}

- **20 folds e teste trancado** medem a caixa do treino. Os 20 folds são re-partições dos e-mails de desenvolvimento; o teste trancado ({TT['n']} e-mails) foi separado no começo e só serviu para relatar.
- **Enron2-6** são as caixas de outros 5 funcionários da Enron, com spam de coleções parecidas com a do treino. O custo é a média das 5 caixas. Caixa a caixa, com λ = 9, perdem para não filtrar: {perde_txt or 'nenhum filtro'}.
- **SpamAssassin** é um corpus público de 2002, com e-mails de listas de discussão e pessoais. Decidiu a regra pré-registrada D10 e, na revisão D11, também serviu para decidir o que destacar: não é mais uma base intocada.
- **TCR** (custo de não filtrar ÷ custo do filtro; acima de 1, filtrar compensa). Enron2-6: {tcr_txt('externo')}. SpamAssassin: {tcr_txt('spamassassin')}.
- Nenhuma dessas bases externas entrou no treino.

![Os filtros nas três avaliações](resultados/figuras/09_filtros_finais.png)

## As tecnologias em destaque

- **Stacking 7.** Um modelo de cada família (linear, SVM, kNN, rede neural, Naive Bayes, árvores, boosting), combinados por um meta-modelo treinado nas notas *out-of-fold*.
  - **No MCC, no limiar 0,5**, supera o melhor modelo único afinado (LinearSVC otimizado) por {n(mcc_s7['dif'], 4)}, com p de Holm {n(mcc_s7['p_holm'])} nas {len(G['comparacoes'])} comparações do script 05 (com mais comparações na família de Holm, passa de 0,05) e P(stacking melhor) = {n(mcc_s7['P_A_melhor'], 2)} no teste bayesiano. No limiar de custo, os dois empatam no MCC.
  - **No custo**, comparado à linha de base (LinearSVC padrão), o Especialista custa {pct(ce['reducao_relativa'], 0)} menos, com IC95 de {n(ce['ic95_lo'], 1)} a {n(ce['ic95_hi'], 1)} por mil e P(mais barato) = {n(ce['P_A_melhor'], 2)} no teste bayesiano. O Stacking 8 custa {pct(s8_lb['reducao_relativa'], 0)} menos, e o Stacking 7 padrão, {pct(c7['reducao_relativa'], 0)}. Corrigida pelas {len(CMP)} comparações (Holm), nenhuma dessas diferenças de custo {'deixa de ser' if ce['p_holm'] < 0.05 else 'chega a ser'} significativa a 5% (p = {n(ce['p_holm'])} para o Especialista).
- **Algoritmo genético.** Buscou os hiperparâmetros dos 7 modelos e do meta-modelo: avaliou {G['avaliacoes']} de {milhar(G['combinacoes'])} combinações, graças ao cache das notas. A configuração que ele achou é o **Especialista**, o filtro de menor custo nos 20 folds ({n(F['especialista']['cv20']['custo_por_mil'], 1)} por mil, contra {n(F['generalista_sem_chave']['cv20']['custo_por_mil'], 1)} do padrão; P(mais barato) = {n(esp_pad['P_A_melhor'], 2)}). Os 20 folds são re-partições dos e-mails usados na escolha; no teste trancado, os dois empatam ({F['especialista']['teste_trancado']['custo']} contra {F['generalista_sem_chave']['teste_trancado']['custo']}).
- **TabPFN-3.5** (Prior Labs), um modelo de fundação para tabelas, pré-treinado em milhões de tabelas sintéticas:
  - **sozinho e sem nenhum ajuste**, empata nesta caixa com o melhor modelo único afinado (MCC no limiar 0,5: {n(tab_in['TabPFN-3.5-Plus']['mcc'])} contra {n(tab_in['LinearSVC otimizado']['mcc'])}); no limiar de custo, deixa passar mais spam que os outros aqui;
  - **fora da caixa, é o mais conservador:** {'o que menos manda e-mail real para a lixeira em todas as bases externas (as 5 caixas Enron e o SpamAssassin)' if tab_menos_lixeira else 'um dos que menos mandam e-mail real para a lixeira'}. Por isso é o mais barato onde predominam e-mails reais ({lista(tab_mais_barato)}) e na média do Enron2-6 ({n(F['tabpfn']['externo']['custo_por_mil'], 1)} por mil){'; onde predomina spam, chega a ser o mais caro (' + lista(tab_mais_caro) + ')' if tab_mais_caro else ''}.
- **Stacking 8** (o Stacking 7 + o TabPFN como 8º membro):
  - nesta caixa, {'empata no topo do PR-AUC' if pr_empate else 'tem o maior PR-AUC'} ({n(F['generalista']['cv20']['pr_auc'], 4)});
  - fora dela, {'o TabPFN deixa o stacking mais barato nas duas versões e nas duas bases externas' if barateia else 'o TabPFN deixa o stacking mais barato na maioria das comparações'} ({b8txt});
  - no SpamAssassin, pela regra escrita antes do teste (D10), o Stacking 8 padrão custou menos que o Stacking 7 padrão em todos os λ testados (de {n(-max(dif_lam), 0)} a {n(-min(dif_lam), 0)} a menos por mil; com λ = 9, P(redução) = {n(dec['p_reducao'], 2)}).

## O que descobrimos

1. **O CSV é o Enron1, e as palavras foram contadas como substring.** Reproduzimos {B['linhas_reproduzidas']} de {B['n_emails']} linhas a partir dos textos originais. Há {B['colunas_1_letra']} colunas de uma letra só, e "ect" aparece em todos os e-mails, porque está dentro de "subject".
2. **Duplicata não indica spam.** {pct(B['fracao_duplicatas_email_real'], 0)} das {B['duplicatas_excedentes']} cópias são e-mails reais. Cópias idênticas ficam sempre do mesmo lado da divisão treino/teste.
3. **Sem rótulo, os grupos naturais são assuntos, não "spam × real".** O K-means tem ARI de {n(NS['kmeans_ari'], 2)}. Sem nenhum spam rotulado, treinando só com e-mails reais (semissupervisionado), a melhor pista é "não se parece com meus e-mails reais", o erro de reconstrução, com PR-AUC {n(NS['anomalia_so_email_real']['reconstrucao_pr_auc'], 2)}.
4. **A representação vale mais que o algoritmo.** Com TF-IDF com log, a Reg. Logística sobe de {n(A['lr_mcc_antes'])} para {n(A['lr_mcc_depois'])}, e o kNN, de {n(A['knn_mcc_antes'])} para {n(A['knn_mcc_depois'])}.
5. **Modelos que erram diferente se completam.** O kNN, fraco sozinho, é um dos que menos erram junto com os outros e recebe o maior peso do meta-modelo ({n(R['stacking']['pesos_meta']['kNN cosseno'], 2)}). Treze modelos empatam com sete.
6. **O ajuste fino rende pouco.** Na busca, o algoritmo genético ganhou {sn(G['ganho_busca'])} de MCC. Em divisões novas, o ganho foi {sn(G['ganho_sementes_novas'])}, com IC95 de {sn(lo)} a {sn(hi)}: não se confirmou, como se espera de escolher o melhor entre centenas de candidatos (a maldição do vencedor).
7. **O stacking aprende a caixa e a coleção de spam.** Fora da caixa do treino, todos pioram, nos e-mails reais e no spam. No SpamAssassin, com λ = 9, {'nenhum filtro treinado aqui vence' if not vence_sa else 'só ' + ', '.join(vence_sa) + ' vence'} não filtrar ({n(F['nao_filtrar']['spamassassin']['custo_por_mil'], 1)} por mil), embora um modelo treinado *dentro* dele chegue a MCC {n(TB['controle_dentro_da_base']['mcc'])}. {frase_sa_lam}**Para uso real, o caminho é retreinar com e-mails da caixa de destino.**
8. **O MCC não vem do acaso.** Com os rótulos embaralhados {PERM['n_permutacoes']} vezes, o MCC vai de {n(PERM['mcc_real'])} para {n0(PERM['mcc_permutado_media'])} (p = {n(PERM['p'])}, o menor possível). Quase-cópias (cosseno ≥ {n(QC['limiar_cosseno'], 2)}), que o agrupamento não separa, estão em {pct(QC['folds_fracao'], 0)} dos e-mails de teste dos 20 folds; no subconjunto sem elas, o MCC do Stacking 7 é {n(QC['mcc_sem_quase_copias'], 4)}, contra {n(QC['mcc'], 4)} no total.

## Limites (leia antes de usar)

- **Uma caixa só.** Os filtros aprendem o que é "normal" na caixa `farmer-d`. Em caixas muito diferentes, erram muito mais (item 7).
- **λ = 9 é uma escolha.** Com outro peso para o erro, os custos mudam. No Enron2-6, o mais barato é o TabPFN sozinho com λ = {lista(lam_tab)}{'; ' + '; '.join(lam_outros) if lam_outros else ''} (`notebooks/07_contra_estudo.ipynb`).
- **Spam de coleções aparentadas.** O spam do Enron1 e o do Enron4 vêm da mesma coleção ({FONTES['Enron1']['spam']}); o do SpamAssassin entrou na coleção do Enron2 e do Enron5 ({FONTES['Enron2']['spam']}). As bases externas não são independentes no spam.
- **O teste trancado foi usado {TT['usos']} vezes**, sempre para relatar, nunca para escolher. A decisão D8 citou como critério de desempate o fato de o candidato já ter sido avaliado nele.
- **O Stacking 8 e o TabPFN foram avaliados, não entregues prontos.** Precisam da API da Prior Labs (chave pessoal) e foram medidos com as notas gravadas em `resultados/cache_tabpfn/` e `experimentos/05_tabpfn/`; não há inferência pronta para um e-mail novo. A demo e os filtros salvos (Especialista, Stacking 7 padrão, LinearSVC) rodam sem chave.
- **Colab.** O ambiente do Colab (Python 3.12, `requirements-colab.txt`) foi reproduzido localmente numa cópia limpa do repositório: os testes passam, e o notebook-história e o da terceira base rodam até o fim (o filtro é baixado da Release). O Colab em si não foi executado por nós de ponta a ponta.

## Os notebooks

Todos rodam no **Google Colab**: a primeira célula baixa o projeto, instala os pacotes e reinicia o ambiente uma vez. Depois, use *Executar tudo* de novo. No computador, siga o [INSTALACAO.md](INSTALACAO.md).

| Notebook | Etapa | Colab |
|---|---|---|
| [`Projeto3_Classificacao_Spam.ipynb`](Projeto3_Classificacao_Spam.ipynb) | a história completa (para o professor) | [![Abrir no Colab]({SELO})]({COLAB}/Projeto3_Classificacao_Spam.ipynb) |
{lista_nb}

## Como rodar no computador

```powershell
pip install -r requirements.txt; pip install -e . --no-deps   # Python 3.11 ou 3.12
pytest                                                       # testes rápidos, menos de 1 min
jupyter notebook Projeto3_Classificacao_Spam.ipynb           # ou: streamlit run demo/app.py
```
Passo a passo para Windows, macOS/Linux e Colab: [INSTALACAO.md](INSTALACAO.md). Os filtros treinados ficam em *Releases* e são baixados sozinhos pelo notebook e pela demo; também podem ser refeitos com `python scripts/09_modelos_finais.py`.

## Estrutura

```text
├── Projeto3_Classificacao_Spam.ipynb   notebook-história (para o professor)
├── notebooks/        01 a 09: um notebook técnico por etapa, gerado dos scripts e executado
├── scripts/          a fonte dos notebooks (formato # %%) e os geradores (notebooks, documentos)
├── spamfilter/       o pacote: dados, conversão, modelos, métricas, estatística, motor com cache, genético, filtro final
├── tests/            pytest (rápidos) e pytest -m lento (regressão completa contra o registro)
├── resultados/       metricas_finais.json (fonte única dos números), figuras, CSVs, caches
├── apresentacao/     decks HTML (banca e equipe), gerados por build.py a partir dos resultados
├── demo/             app "cole um e-mail" (Streamlit)
├── dados/            CSV do enunciado, Enron1-6 da AUEB, SpamAssassin e os e-mails brutos (ver dados/LEIA-ME.md)
├── experimentos/     os experimentos originais 01 a 05, com caches e notebooks, como foram executados
└── docs/             registro de decisões, metodologia, ficha do modelo, referências, contrato dos resultados
```

## Dados, licenças e citação

- **CSV:** Kaggle, *Email Spam Classification Dataset CSV* (`balaka18`), com uso acadêmico autorizado.
- **Enron-Spam:** V. Metsis, I. Androutsopoulos e G. Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", CEAS 2006. Fonte: http://www2.aueb.gr/users/ion/data/enron-spam/
- **SpamAssassin public corpus:** https://spamassassin.apache.org/old/publiccorpus/ (cópia em `dados/spamassassin/`).
- **TabPFN-3.5** (Prior Labs): uso acadêmico, sem fins comerciais. A chave fica só na variável `TABPFN_TOKEN`.
- **Código:** licença MIT ([LICENSE](LICENSE)). Bibliografia completa: [docs/REFERENCIAS.md](docs/REFERENCIAS.md).

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json` ({R['meta']['gerado_em'][:10]}).*
"""

esp, s8, tpf = F["especialista"], F["generalista"], F["tabpfn"]
COLS = [("especialista", "Especialista"), ("generalista", "Stacking 8"), ("tabpfn", "TabPFN sozinho"), ("nao_filtrar", "Não filtrar")]
def linha(rot, f):
    return f"| {rot} | " + " | ".join(f(F[k]) for k, _ in COLS) + " |"
def mc(t):
    m = t.get("matriz_confusao") if t else None
    return f"{m['spam_na_lixeira']} / {m['real_na_caixa']} / {m['real_na_lixeira']} / {m['spam_na_caixa']}" if m else "–"
tt_ = lambda k, campo, c=3: n(F[k]["teste_trancado"][campo], c) if F[k]["teste_trancado"] else "–"

MODEL_CARD = f"""# Ficha do modelo (model card)

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json`.*

## Para que serve
Classificar e-mails em **spam** ou **e-mail real**. É um projeto acadêmico de Data Science 2 (ESEG), não um produto: os filtros servem para estudar como um classificador se comporta dentro e fora da caixa de e-mail em que foi treinado.

| Filtro | Uso pretendido |
|---|---|
| Especialista (Stacking 7 do GA) | a caixa do treino (Enron1, caixa `farmer-d`) |
| Stacking 8 (Stacking 7 + TabPFN) | fora da caixa, com o TabPFN o stacking fica bem mais barato (regra D10); precisa da API do TabPFN (chave). Avaliado com notas gravadas; não há inferência pronta para um e-mail novo |
| TabPFN-3.5 sozinho | o mais conservador fora da caixa (o que menos manda e-mail real para a lixeira); precisa da API do TabPFN (chave). Avaliado, sem inferência pronta |
| Stacking 7 padrão | roda sem internet nem chave (usado na demo) |
| LinearSVC | linha de base simples |

## Dados de treino
- Origem: {B['n_emails']} e-mails do Enron1, com {B['n_email_real']} e-mails reais da caixa {FONTES['Enron1']['email_real']} ({FONTES['Enron1']['email_real_de']} a {FONTES['Enron1']['email_real_ate']}) e {B['n_spam']} spams da coleção {FONTES['Enron1']['spam']} ({FONTES['Enron1']['spam_de']} a {FONTES['Enron1']['spam_ate']}).
- Representação: contagem por substring de {B['n_palavras']} palavras, sobre o texto pré-processado da AUEB.
- O texto vem em inglês, num contexto corporativo do setor de energia dos EUA.

## Desempenho (limiar de custo: perder 1 e-mail real = 9 spams recebidos)

| Avaliação | Especialista | Stacking 8 | TabPFN sozinho | Não filtrar |
|---|---|---|---|---|
{linha("Esta caixa, 20 folds: MCC", lambda f: n(f['cv20']['mcc_custo']))}
{linha("Esta caixa, 20 folds: custo por mil", lambda f: n(f['cv20']['custo_por_mil'], 1))}
{linha("Esta caixa, 20 folds: e-mails reais na lixeira", lambda f: pct(f['cv20']['real_lixeira_custo']))}
{linha("Esta caixa, 20 folds: spam que passou", lambda f: pct(f['cv20']['spam_passou_custo']))}
{linha("Teste trancado: precisão (spam)", lambda f: n(f['teste_trancado']['precisao_custo']) if f['teste_trancado'] else '–')}
{linha("Teste trancado: revocação (spam)", lambda f: n(f['teste_trancado']['revocacao_custo']) if f['teste_trancado'] else '–')}
{linha("Teste trancado: F1 (spam)", lambda f: n(f['teste_trancado']['f1_custo']) if f['teste_trancado'] else '–')}
{linha("Teste trancado: matriz de confusão (VP / VN / FP / FN)", lambda f: mc(f['teste_trancado']))}
{linha("Enron2-6 (outras 5 caixas): custo por mil", lambda f: n(f['externo']['custo_por_mil'], 1))}
{linha("Enron2-6: e-mails reais na lixeira", lambda f: pct(f['externo']['real_lixeira_custo']))}
{linha("SpamAssassin: custo por mil", lambda f: n(f['spamassassin']['custo_por_mil'], 1))}
{linha("SpamAssassin: e-mails reais na lixeira", lambda f: pct(f['spamassassin']['real_lixeira_custo']))}
{linha("Esta caixa, 20 folds: TCR (acima de 1 = compensa filtrar)", lambda f: n(f['cv20']['tcr'], 1))}
{linha("Enron2-6: TCR", lambda f: n(f['externo']['tcr'], 2))}
{linha("SpamAssassin: TCR", lambda f: n(f['spamassassin']['tcr'], 2))}

VP = spam na lixeira, VN = e-mail real na caixa, FP = e-mail real na lixeira, FN = spam que passou. O teste trancado tem {TT['email_real']} e-mails reais e {TT['spam']} spams.

## Limitações (leia antes de usar)
1. **O filtro aprende a caixa.** Fora da caixa do treino, o desempenho cai muito. No SpamAssassin, com λ = 9, {'nenhum filtro treinado aqui custa' if not vence_sa else 'só ' + ', '.join(vence_sa) + ' custa'} menos que não filtrar; um modelo treinado dentro do SpamAssassin chega a MCC {n(TB['controle_dentro_da_base']['mcc'])}. **Para usar de verdade, retreine com e-mails da caixa de destino.**
2. **O peso do erro importa.** O vencedor fora da caixa muda com λ.
3. **Só inglês.** Num e-mail em português, a cobertura do vocabulário cai muito, e a decisão perde o sentido. A demo avisa quando a cobertura é baixa.
4. **Época.** Os spams são de {FONTES['Enron1']['spam_de'][:4]} a {FONTES['Enron1']['spam_ate'][:4]}. Spams atuais (outros produtos, imagens, links encurtados) não estão representados.
5. **Vocabulário da empresa.** Palavras como "enron", "hpl" e nomes de colegas ajudam nesta caixa. Retirá-las quase não muda o resultado aqui.

## Considerações éticas
- Os e-mails da Enron tornaram-se públicos na investigação da FERC (EUA). Os do SpamAssassin foram publicados para pesquisa. O projeto identifica a caixa de origem (`farmer-d`), como o próprio corpus Enron-Spam faz, e não acrescenta dados pessoais.
- Um filtro de spam errando para o lado do spam esconde mensagens legítimas. Por isso o limiar foi escolhido para que esse erro custe 9 vezes mais.
- O TabPFN é usado em pesquisa acadêmica, sem fins comerciais.
"""

# ---------------------------------------------------------------------------------------------------------
# docs/MODELOS.md: cada modelo explicado em linguagem simples (texto fixo; números do JSON)
# ---------------------------------------------------------------------------------------------------------
TOR = {r["modelo"]: r for r in R["torneio"]}
PESO = R["stacking"]["pesos_meta"]
EC = R["stacking"]["erros_em_comum"]
JUNTO = EC["erra_junto"]                                   # P(erra | outro membro erra), média sobre os outros 6
ORD_JUNTO = sorted(JUNTO, key=JUNTO.get)
PESO_DP = R["stacking"]["pesos_meta_dp"]
RESGATE = {r["familia"]: r["acerta quando a maioria (≥4) erra"] for r in R["stacking"]["tabela_erros"]}
ORD_RESGATE = sorted(RESGATE, key=RESGATE.get, reverse=True)
PESO_E, PESO_P = F["especialista"]["pesos_meta"], F["generalista_sem_chave"]["pesos_meta"]
DESLIGADOS = [f for f, w in PESO_E.items() if abs(w) < 0.1]
TOP_NMF = max(NS["nmf_topicos"], key=lambda t: t["fracao_spam"])
_vp = TOP_NMF["n_emails"] * TOP_NMF["fracao_spam"]; _fp = TOP_NMF["n_emails"] - _vp; _fn = B["n_spam"] - _vp; _vn = B["n_emails"] - _vp - _fp - _fn
MCC_NMF = (_vp * _vn - _fp * _fn) / ((_vp + _fp) * (_vp + _fn) * (_vn + _fp) * (_vn + _fn)) ** 0.5
NMF_FRAC_SPAM = _vp / B["n_spam"]
CMP_R2 = {c["B"]: c for c in R["rodada2"]["comparacoes"]}
st4_lsvc = CMP_R2["LinearSVC TF-IDF"]
CFG_E, CFG_P = F["especialista"]["config"], F["generalista_sem_chave"]["config"]

FICHAS = [
    ("Regressão Logística", "Reg. Logística", "Reg. Logística TF-IDF",
     "Dá um peso a cada palavra, soma os pesos das palavras do e-mail e transforma a soma numa probabilidade de spam.",
     "Uma balança: as palavras típicas de spam pesam num prato, as típicas de e-mail real no outro.",
     ["Cada palavra recebe um peso positivo (puxa para spam) ou negativo (puxa para e-mail real), aprendido no treino.",
      "A soma passa por uma curva em S (a função logística), que a converte numa nota entre 0 e 1.",
      "O parâmetro C controla quanto os pesos podem crescer (C maior = menos freio); `cw=balanced` compensa o spam ser minoria."],
     "TF-IDF com log", "rápida e fácil de explicar: dá para ler quais palavras pesam.",
     "só traça fronteiras retas: não percebe combinações de palavras."),
    ("LinearSVC (a linha de base)", None, "LinearSVC TF-IDF",
     "Também traça uma fronteira reta entre spam e e-mail real, mas escolhe a que deixa a maior folga entre os dois lados.",
     "A rua mais larga possível entre dois bairros: quanto mais larga, menos chance de errar o lado.",
     ["Procura o hiperplano (a \"reta\" em 3000 dimensões) com a maior margem até os e-mails mais próximos de cada classe.",
      "A nota é a distância à fronteira, não uma probabilidade.",
      "Foi o melhor modelo único do torneio e virou a **linha de base**: o filtro simples que o stacking precisa superar."],
     "TF-IDF com log", "rápido e muito forte em texto.", "fronteira reta, como a Regressão Logística."),
    ("SVM com kernel RBF", "SVM RBF", "SVM RBF SVD300",
     "Resume as 3000 colunas em algumas centenas de \"temas\" e traça uma fronteira curva, comparando cada e-mail com e-mails de referência.",
     "Um mapa de bairros com qualquer formato: o e-mail é classificado pelo bairro onde cai.",
     ["Primeiro, o SVD (decomposição em valores singulares) comprime as 3000 colunas em 300 ou 500 direções que resumem os assuntos.",
      "Depois, o kernel RBF mede a semelhança do e-mail com os vetores de suporte, os e-mails de treino mais próximos da fronteira.",
      "Isso permite fronteiras curvas (a curvatura vem do kernel); o C troca folga por acertos no treino: C maior = fronteira mais colada ao treino."],
     "TF-IDF com log + SVD", "capta padrões que uma reta não capta.", "mais lento; a nota é uma distância, não uma probabilidade."),
    ("kNN (vizinhos mais próximos)", "kNN cosseno", "kNN TF-IDF cosseno",
     "Procura no treino os e-mails mais parecidos com o novo e copia a decisão da maioria deles.",
     "Perguntar aos vizinhos: se os 5 e-mails mais parecidos eram spam, este provavelmente também é.",
     ["A semelhança é o cosseno entre os vetores TF-IDF: dois e-mails com as mesmas palavras, nas mesmas proporções, apontam para o mesmo lado.",
      "k é o número de vizinhos consultados (5 no padrão, 3 no Especialista).",
      "Não aprende pesos: guarda o treino inteiro e compara."],
     "TF-IDF com log", "reconhece e-mails quase iguais a algum do treino (avisos automáticos, spam em série).",
     "decora a caixa: fora dela, e-mails reais de outro assunto não têm vizinhos parecidos e são confundidos com spam."),
    ("Rede neural (MLP)", "MLP", "MLP (rede neural)",
     "Uma rede neural pequena: uma camada de neurônios aprende combinações de palavras, e uma camada final junta tudo numa nota.",
     "Uma equipe de detetives, cada um atento a um padrão diferente, e um chefe que soma as pistas.",
     ["Uma camada oculta com 128 neurônios (64 no Especialista); cada neurônio é uma soma ponderada das palavras seguida de um corte não linear.",
      "O treino ajusta os pesos aos poucos e para quando o desempenho numa parte separada do treino deixa de melhorar (*early stopping*)."],
     "TF-IDF com log", "capta combinações de palavras.", "é uma caixa-preta e varia com a semente."),
    ("Naive Bayes (Complement)", "Naive Bayes", "Naive Bayes Complement",
     "Conta, no treino, com que frequência cada palavra aparece em cada classe e combina essas chances para dar a nota.",
     "Um detetive que olha cada pista isoladamente e multiplica as chances, sem pensar em como as pistas se relacionam.",
     ["\"Ingênuo\" porque supõe que as palavras são independentes dentro de cada classe, o que é falso, mas funciona razoavelmente.",
      "Com duas classes, a versão Complement equivale ao Naive Bayes multinomial sem a fração de spam da base como ponto de partida: o spam, que é minoria, não começa em desvantagem.",
      "O α suaviza as contagens, para uma palavra rara não zerar a conta."],
     "contagens", "o mais rápido de todos.",
     "fraco sozinho, erra muito e erra junto com os outros: no stacking, recebe peso quase zero."),
    ("Extra Trees (árvores extremamente aleatórias)", "Extra Trees", "Extra Trees",
     "Centenas de árvores de decisão, cada uma com perguntas sorteadas, votam juntas.",
     "Um júri de centenas de pessoas: cada uma olhou pistas diferentes, e a maioria decide.",
     ["Cada árvore faz perguntas do tipo \"a palavra X aparece mais de N vezes?\" até chegar a uma folha (spam ou e-mail real).",
      "Nas Extra Trees, cada árvore vê todos os e-mails, mas os pontos de corte e as palavras candidatas são sorteados: as árvores ficam diferentes entre si, e a média delas erra menos que cada uma.",
      "São 400 árvores no padrão e 800 no Especialista."],
     "contagens", "não precisa de TF-IDF e capta regras do tipo \"se... e se...\".", "decora detalhes da caixa do treino."),
    ("LightGBM (boosting de árvores)", "LightGBM", "LightGBM",
     "Também usa árvores, mas em sequência: cada árvore nova corrige um pouco o erro que ainda sobra.",
     "Um aluno que revisa a prova várias vezes e, a cada revisão, corrige um pouco cada resposta, mais onde errou mais.",
     ["São 400 árvores pequenas; cada uma é treinada para reduzir o erro que restou em todos os e-mails, com mais força onde o erro é maior (*gradient boosting*).",
      "A taxa de aprendizado (0,05) controla o tamanho de cada correção; o número de folhas, o tamanho de cada árvore."],
     "contagens", "muito forte em tabelas.", "aqui ficou atrás dos lineares: o texto favorece modelos lineares depois do TF-IDF com log."),
]


def ficha(nome, fam, tor, frase, analogia, como, entrada, forte, fraco):
    t = TOR.get(tor, {})
    linhas = [f"### {nome}", "", f"**Em uma frase.** {frase}", "", f"**Analogia.** {analogia}", "", "**Como decide:**"]
    linhas += [f"- {c}" for c in como]
    linhas += ["", f"- **Entrada:** {entrada}."]
    if fam:
        linhas.append(f"- **Configuração:** padrão `{CFG_P[fam]}`; Especialista `{CFG_E[fam]}`.")
    if t:
        linhas.append(f"- **Sozinho, no torneio (5 folds):** MCC {n(t['mcc'])}, treino em {n(t['treino_s'], 1)} s.")
    if fam:
        pos = ORD_JUNTO.index(fam) + 1
        linhas.append(f"- **No stacking:** quando outro membro erra, ele erra junto em {pct(JUNTO[fam], 0)} das vezes ({pos}º menor entre os 7; "
                      f"quanto menor, mais informação nova ele traz) e salva {RESGATE[fam]} e-mails que a maioria errou. Peso no meta-modelo: "
                      f"{n(PESO[fam], 2)} ± {n(PESO_DP[fam], 2)} na média dos 20 folds; {n(PESO_P[fam], 2)} no Stacking 7 padrão entregue e "
                      f"{n(PESO_E[fam], 2)} no Especialista.")
    linhas += [f"- **Ponto forte:** {forte}", f"- **Ponto fraco:** {fraco}", ""]
    return "\n".join(linhas)


st7 = next(v for v in R["stacking"]["variantes"] if v["nome"].startswith("Stacking 7"))
st13 = next(v for v in R["stacking"]["variantes"] if v["nome"] == "Stacking 13 (todos)")
MODELOS_MD = f"""# Os modelos, explicados

*Gerado por `scripts/gerar_documentos.py`: o texto explica cada modelo; os números vêm de `resultados/metricas_finais.json`.*

## Antes dos modelos: o que eles recebem

- **A base.** Cada e-mail é uma linha com {B['n_palavras']} contagens: quantas vezes cada palavra do vocabulário aparece no texto, contada como pedaço de texto (\
- **TF-IDF com log.** Para os modelos lineares, o SVM, o kNN e a rede neural, as contagens viram pesos. O log achata as repetições (contar "free" 10 vezes não é 10 vezes mais evidência), e o IDF dá menos peso às palavras que aparecem em quase todo e-mail. Esse detalhe valeu mais que trocar de modelo: a Regressão Logística foi de MCC {n(A['lr_mcc_antes'])} para {n(A['lr_mcc_depois'])}.
- **A nota e o limiar.** Todo modelo devolve uma nota de spam. O **limiar de custo** decide a partir de que nota o e-mail vai para a lixeira: ele é escolhido no treino para minimizar 9 × (e-mails reais na lixeira) + 1 × (spams que passaram).

## O torneio: o ranking dos {len(R['torneio'])} modelos supervisionados

Todos fizeram a mesma prova: os mesmos {B['n_emails']} e-mails, a mesma validação cruzada (5 folds, com cópias idênticas sempre do mesmo lado) e a mesma régua (MCC, de −1 a 1; 0 = acaso). O "Dummy" é a referência: nunca marca spam.

| # | Modelo | Família | MCC | PR-AUC | Treino (s) |
|---|---|---|---|---|---|
{chr(10).join(f"| {i} | {r['modelo']} | {r['familia']} | {n(r['mcc'])} | {n(r['pr_auc'])} | {n(r['treino_s'], 1)} |" for i, r in enumerate(sorted(R['torneio'], key=lambda r: -r['mcc']), 1))}

**O que o ranking mostrou:**
- os modelos lineares lideram, depois do TF-IDF com log;
- no topo, os melhores modelos sozinhos ficam muito perto uns dos outros: nenhum domina com folga;
- o 1º lugar já era uma combinação simples, o "Stacking (meta LR)", com 4 membros (Reg. Logística, Naive Bayes, Random Forest e LightGBM). Na rodada de 15 folds, ele ficou à frente do LinearSVC, mas sem significância (p de Holm {n(st4_lsvc["p_holm"])}).

Daí a ideia seguinte: testar qual combinação funciona melhor e entender por quê. A resposta veio no stacking, abaixo: modelos de famílias diferentes erram em e-mails diferentes.

## Os 7 modelos do stacking (um por família) e a linha de base

{chr(10).join(ficha(*f) for f in FICHAS)}
## O stacking: como os 7 viram um filtro só

**Em uma frase.** Os 7 modelos dão uma nota a cada e-mail, e um oitavo modelo, o **meta-modelo** (uma Regressão Logística), aprende quanto confiar em cada nota.

**Analogia.** Um conselho de 7 especialistas e um presidente que aprendeu, pelo histórico, quanto ouvir cada um.

**Como funciona:**
1. Cada modelo é treinado várias vezes, cada vez sem uma parte do treino, e dá nota justamente à parte que não viu. São as notas *out-of-fold*: ninguém dá nota a um e-mail que usou para aprender.
2. O meta-modelo aprende com essas notas honestas qual combinação acerta mais. Os pesos dele (média dos 20 folds, Stacking 7 padrão): {", ".join(f"{k} {n(v, 2)}" for k, v in sorted(PESO.items(), key=lambda x: -x[1]))}. Os filtros entregues, treinados nos {B["n_emails"]} e-mails, têm pesos próprios: no Especialista, o meta-modelo praticamente desligou {lista(DESLIGADOS) if DESLIGADOS else "nenhum membro"} (peso perto de zero).
3. Para um e-mail novo, os 7 dão nota, o meta-modelo combina e o limiar de custo decide.

**Por que funciona.** Os modelos erram em e-mails diferentes. Os que menos erram junto com os outros ({lista(ORD_JUNTO[:3])}) são também os que mais salvam e-mails quando a maioria erra ({lista(ORD_RESGATE[:3])}); o meta-modelo dá os maiores pesos médios a {lista([f for f in sorted(PESO, key=PESO.get, reverse=True)[:3] if f in ORD_JUNTO[:3]])}. O {ORD_JUNTO[-1]} é o contrário: erra muito e erra junto (em {pct(JUNTO[ORD_JUNTO[-1]], 0)} das vezes em que outro erra), por isso recebe peso quase zero. Na validação cruzada do experimento 02 (15 folds), o Stacking 7 chega a MCC {n(st7['mcc'])}; com 13 modelos, {n(st13['mcc'])}: mais modelos parecidos não ajudam.

**Por que estes representantes?** Um de cada família de raciocínio (linear, kernel, vizinhos, rede neural, probabilístico, comitê de árvores aleatórias e *boosting*), não necessariamente o melhor de cada família no ranking: o melhor linear sozinho foi o LinearSVC. O projeto não registrou um critério formal para cada escolha. O que os dados mostram é que ela não muda o resultado: o Stacking 13, que inclui também o LinearSVC, o Ridge, o SGD, o NBSVM, o XGBoost e o Random Forest, empata com o Stacking 7.

**Os nomes do projeto:**
- **Stacking 7 padrão:** os 7 com ajustes padrão. Roda sem chave; é o da demo.
- **Especialista:** os 7 com os ajustes achados pelo algoritmo genético. Tem o menor custo nesta caixa nos 20 folds.
- **Stacking 8:** o Stacking 7 mais o TabPFN como 8º membro. Fora da caixa, fica bem mais barato que o Stacking 7.

## O algoritmo genético: como os ajustes foram escolhidos

**Em uma frase.** Uma busca inspirada na evolução: combinações de ajustes competem, as melhores se cruzam e sofrem pequenas mutações, por {len(G['historico'])} gerações.

**Analogia.** Criar cavalos de corrida: cruzar os mais rápidos e, de vez em quando, deixar surgir uma variação.

**Como funciona:**
- Cada indivíduo é uma combinação de 8 genes: o ajuste de cada um dos 7 modelos e o C do meta-modelo. São {milhar(G['combinacoes'])} combinações possíveis.
- População de 40, torneio de 3, cruzamento uniforme, mutação de 15% por gene e os 2 melhores passam direto. Ao todo, {G['avaliacoes']} combinações foram avaliadas.
- **Truque que torna a busca viável:** cada ajuste de cada modelo é treinado uma vez por divisão e a nota fica guardada (o *cache* do motor); avaliar um indivíduo é só treinar o meta-modelo.
- **Resultado:** na busca, o MCC subiu {sn(G['ganho_busca'])}; em divisões novas, {sn(G['ganho_sementes_novas'])}, com intervalo de confiança que inclui o zero. A lição é a "maldição do vencedor": escolher o melhor entre centenas de candidatos nas mesmas divisões infla o resultado.

## O TabPFN: o modelo de fundação

**Em uma frase.** Uma rede neural do tipo *transformer*, da Prior Labs, pré-treinada em tabelas sintéticas geradas por computador, que classifica uma tabela nova sem ser treinada nela.

**Analogia.** Um aluno que nunca estudou esta matéria, mas já resolveu milhões de provas parecidas e reconhece o padrão na hora.

**Como funciona:**
- Recebe, de uma vez, a tabela de treino (os e-mails e os rótulos) e os e-mails a classificar, e devolve as notas sem ajustar os próprios pesos: é o aprendizado em contexto (*in-context learning*). Por dentro, faz a média de alguns estimadores sobre versões embaralhadas da tabela.
- Ele tem hiperparâmetros, mas nós não ajustamos nenhum: usamos o padrão e as contagens cruas, sem TF-IDF.
- Versões: TabPFN-3.5-Plus, pela API na nuvem (exige chave), e TabPFN-3.5-Fast, no computador (cerca de 3 h em CPU para 5 divisões). As notas das duas têm correlação de {n(TP['local_x_nuvem']['correlacao_notas'])}.

**O que ele fez de melhor:**
- Sozinho, nesta caixa, empata com o melhor modelo único afinado: MCC {n(tab_in['TabPFN-3.5-Plus']['mcc'])} contra {n(tab_in['LinearSVC otimizado']['mcc'])}.
- Fora da caixa, é o mais conservador: o que menos manda e-mail real para a lixeira em todas as bases externas.
- Como 8º membro, deixa o stacking bem mais barato fora da caixa (o Stacking 8).

## Os modelos sem rótulo (não supervisionado e semissupervisionado)

- **SVD em 2 dimensões:** comprime as {B['n_palavras']} colunas num mapa, para ver se spam e e-mail real se separam sozinhos.
- **K-means e agrupamento hierárquico:** formam grupos sem saber o rótulo. O ARI compara os grupos com spam × e-mail real (0 = acaso; {n(NS['kmeans_ari'], 2)} no K-means): os grupos que eles acham são assuntos.
- **NMF (tópicos):** acha assuntos na caixa. Um tópico de linguagem de propaganda ("{', '.join(TOP_NMF['palavras'][:3])}") reúne {pct(NMF_FRAC_SPAM, 1)} de todo o spam, mas {pct(1 - TOP_NMF['fracao_spam'], 0)} dos e-mails dele são reais: usado como filtro, daria MCC {n(MCC_NMF, 2)}. Sem rótulo há uma pista forte, mas é preciso o rótulo para chegar ao nível dos supervisionados.
- **Detecção de novidade (semissupervisionada):** treina só com e-mails sabidamente reais e marca como spam o que "não se parece" com eles. A medida que funciona é o erro de reconstrução (PR-AUC {n(NS['anomalia_so_email_real']['reconstrucao_pr_auc'], 2)}).
"""

(RAIZ / "README.md").write_text(README, encoding="utf-8")
(RAIZ / "docs" / "MODEL_CARD.md").write_text(MODEL_CARD, encoding="utf-8")
(RAIZ / "docs" / "MODELOS.md").write_text(MODELOS_MD, encoding="utf-8")
print("gerados: README.md, docs/MODEL_CARD.md, docs/MODELOS.md")
