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
| 🧭 Decisões e números | [docs/REGISTRO_DECISOES.md](docs/REGISTRO_DECISOES.md) · [docs/METODOLOGIA.md](docs/METODOLOGIA.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/REFERENCIAS.md](docs/REFERENCIAS.md) |

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

(RAIZ / "README.md").write_text(README, encoding="utf-8")
(RAIZ / "docs" / "MODEL_CARD.md").write_text(MODEL_CARD, encoding="utf-8")
print("gerados: README.md, docs/MODEL_CARD.md")
