"""Gera README.md e docs/MODEL_CARD.md a partir de resultados/metricas_finais.json (nenhum número digitado à mão).

    python scripts/gerar_documentos.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = json.loads((RAIZ / "resultados" / "metricas_finais.json").read_text(encoding="utf-8"))
REPO = "https://github.com/felipe44776-eseg/filtro-spam-stacking-enron"
PAGES = "https://felipe44776-eseg.github.io/filtro-spam-stacking-enron"


def n(x, c=3):
    return f"{x:.{c}f}".replace(".", ",")


def pct(x, c=1):
    return f"{100 * x:.{c}f}%".replace(".", ",")


def milhar(x):
    return f"{x:,}".replace(",", ".")


def n0(x, c=3):
    """Como n(), mas sem o sinal de menos em zeros arredondados ("-0,000")."""
    return n(0.0 if abs(x) < 0.5 * 10 ** -c else x, c)


F, B, G, A = R["finais"], R["base"], R["genetico"], R["achados"]
TB, CE, NS = R["terceira_base"], R["contra_estudo"], R["nao_supervisionado"]
dec = TB["decisao"]
rot = {"especialista": "**Especialista** (esta caixa)", "generalista": "**Generalista** (caixas novas)",
       "generalista_sem_chave": "Stacking 7 padrão (caixas novas, sem chave)", "linha_de_base": "LinearSVC (linha de base)"}
cfg = {"especialista": "Stacking 7 afinado pelo algoritmo genético", "generalista": "Stacking 7 padrão + TabPFN-3.5 (Stacking 8)",
       "generalista_sem_chave": "Stacking 7 padrão (roda offline; usado na demo)", "linha_de_base": "LinearSVC com TF-IDF log"}
linhas = []
for k in ["especialista", "generalista", "generalista_sem_chave", "linha_de_base"]:
    f = F[k]; tt = f["teste_trancado"] or {}
    linhas.append(f"| {rot[k]} | {cfg[k]} | {n(f['cv20']['mcc'])} | {n(f['cv20']['custo_por_mil'], 1)} | "
                  f"{tt.get('email_real_lixeira', '–')} / {tt.get('spam_passou_n', '–')} | {n(f['externo']['custo_por_mil'], 1)} | "
                  f"{n(f['spamassassin']['custo_por_mil'], 1)} |")
tabela = "\n".join(linhas)
perm = CE["permutacao"]

README = f"""# Filtro de spam com stacking: o que um classificador aprende de uma caixa de e-mail

Projeto 3 de **Data Science 2 (ESEG)**, com o Prof. Marino Catarino.
**Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins.

São {B['n_emails']} e-mails ({B['n_email_real']} e-mails reais e {B['n_spam']} spams), cada um descrito pela contagem de {B['n_palavras']} palavras. O ponto de partida foi um classificador. A investigação foi além: de onde vêm os dados, quais modelos erram de formas diferentes, quanto do ajuste fino é ilusão e o que acontece quando o filtro encontra uma caixa de e-mail que nunca viu.

| | |
|---|---|
| 📓 Notebook para o professor | [`Projeto3_Classificacao_Spam.ipynb`](Projeto3_Classificacao_Spam.ipynb): a história completa, com o código principal visível, em poucos minutos |
| 🎞️ Apresentações | [banca]({PAGES}/apresentacao/banca/) (técnica) · [equipe]({PAGES}/apresentacao/equipe/) (didática, com glossário) |
| ✉️ Demonstração | `streamlit run demo/app.py`: cole um e-mail e veja cada etapa até a decisão |
| ✅ Para a banca | [INSTALACAO.md](INSTALACAO.md) · [COMO_TESTAR.md](COMO_TESTAR.md) |
| 🧭 Decisões e números | [docs/REGISTRO_DECISOES.md](docs/REGISTRO_DECISOES.md) · [docs/METODOLOGIA.md](docs/METODOLOGIA.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |

## Os filtros

Todos usam o **limiar de custo**: perder 1 e-mail real custa o mesmo que receber 9 spams. O custo é por mil e-mails; quanto menor, melhor.

| Filtro | O que é | MCC (20 folds) | Custo (20 folds) | Teste trancado: reais na lixeira / spams que passaram | Custo Enron2-6 | Custo SpamAssassin |
|---|---|---|---|---|---|---|
{tabela}

As colunas "20 folds" e "teste trancado" medem a mesma caixa de e-mail do treino. "Enron2-6" são caixas de outras 5 pessoas, e "SpamAssassin" é outra empresa e outra época. Nenhuma das duas foi usada no treino.

![Os filtros nas três avaliações](resultados/figuras/09_filtros_finais.png)

## O que descobrimos

1. **O CSV é o Enron1, e as palavras foram contadas como substring.** Reproduzimos {B['linhas_reproduzidas']} de {B['n_emails']} linhas a partir dos textos originais. Há {B['colunas_1_letra']} colunas de uma letra só, e "ect" aparece em todos os e-mails, porque está dentro de "subject".
2. **Duplicata não indica spam.** {pct(B['fracao_duplicatas_email_real'], 0)} das {B['duplicatas_excedentes']} cópias são e-mails reais. Cópias idênticas ficam sempre do mesmo lado da divisão treino/teste.
3. **Sem rótulo, os grupos naturais são assuntos, não "spam × real".** O K-means tem ARI de {n(NS['kmeans_ari'], 2)}. A melhor pista sem rótulo é "não se parece com meus e-mails reais", o erro de reconstrução, com PR-AUC {n(NS['anomalia_so_email_real']['reconstrucao_pr_auc'], 2)}.
4. **A representação vale mais que o algoritmo.** Com TF-IDF com log, a Reg. Logística sobe de {n(A['lr_mcc_antes'])} para {n(A['lr_mcc_depois'])}, e o kNN, de {n(A['knn_mcc_antes'])} para {n(A['knn_mcc_depois'])}.
5. **Diversidade vence quantidade.** O Stacking 7 tem um modelo por família. O kNN, fraco sozinho, recebe o maior peso do meta-modelo ({n(R['stacking']['pesos_meta']['kNN cosseno'], 2)}), e 13 modelos não superam 7.
6. **O ajuste fino engana.** O algoritmo genético explorou {milhar(G['combinacoes'])} combinações. Em divisões novas, {pct(G['fracao_otimismo'], 0)} do ganho medido na busca desapareceu (a maldição do vencedor).
7. **O filtro aprende a caixa, não o spam.** Numa base nunca usada (SpamAssassin), todos os filtros caem, embora o mesmo pipeline treinado *dentro* dela chegue a MCC {n(TB['controle_dentro_da_base']['mcc'])}. A regra registrada antes do teste promoveu o **Stacking 8 (com o TabPFN)** a Generalista: ele custou {n(abs(dec['diferenca_custo']), 0)} a menos por mil, com P(redução) = {n(dec['p_reducao'], 2)}. Para uso real, o caminho é retreinar com e-mails da caixa de destino.
8. **O protocolo não vaza.** Com os rótulos embaralhados, o MCC vai de {n(perm['mcc_real'])} para {n0(sum(perm['mcc_permutado']) / len(perm['mcc_permutado']))} (p = {n(perm['p'])}).

## Como rodar em 3 comandos

```powershell
pip install -r requirements.txt; pip install -e . --no-deps   # Python 3.11
pytest                                                       # 31 testes rápidos, menos de 1 min
jupyter notebook Projeto3_Classificacao_Spam.ipynb           # ou: streamlit run demo/app.py
```
Passo a passo para Windows, macOS/Linux e Colab: [INSTALACAO.md](INSTALACAO.md). Os filtros treinados estão em *Releases*, ou podem ser refeitos com `python scripts/09_modelos_finais.py`.

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
├── dados/            CSV do enunciado, Enron1-6 da AUEB e contagens do SpamAssassin (ver dados/LEIA-ME.md)
├── experimentos/     os experimentos originais 01 a 05, com caches e notebooks, como foram executados
└── docs/             registro de decisões, metodologia, ficha do modelo, contrato dos resultados
```

## Dados, licenças e citação

- **CSV:** Kaggle, *Email Spam Classification Dataset CSV* (`balaka18`). A licença não pôde ser verificada (a página exige login); o uso é acadêmico.
- **Enron-Spam:** V. Metsis, I. Androutsopoulos e G. Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", CEAS 2006. Fonte: http://www2.aueb.gr/users/ion/data/enron-spam/
- **SpamAssassin public corpus:** https://spamassassin.apache.org/old/publiccorpus/. O copyright do texto é dos remetentes, por isso o repositório guarda só as contagens. Para baixar: `spamfilter.externo.baixar_spamassassin()`.
- **TabPFN-3.5** (Prior Labs): licença não comercial, que permite pesquisa e avaliação. A chave fica só na variável `TABPFN_TOKEN`.
- **Código:** licença MIT ([LICENSE](LICENSE)).

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json` ({R['meta']['gerado_em'][:10]}).*
"""

esp, gen = F["especialista"], F["generalista"]
MODEL_CARD = f"""# Ficha do modelo (model card)

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json`.*

## Para que serve
Classificar e-mails em **spam** ou **e-mail real**. É um projeto acadêmico de Data Science 2 (ESEG), não um produto: os filtros servem para estudar como um classificador se comporta dentro e fora da caixa de e-mail em que foi treinado.

| Filtro | Uso pretendido |
|---|---|
| Especialista | a caixa do treino (Enron1, caixa `farmer-d`) |
| Generalista (Stacking 8) | caixas novas; precisa do TabPFN (nuvem com chave, ou CPU) |
| Stacking 7 padrão | caixas novas, sem internet nem chave (usado na demo) |
| LinearSVC | linha de base simples |

## Dados de treino
- Origem: {B['n_emails']} e-mails do Enron1, com {B['n_email_real']} e-mails reais de um funcionário da Enron (1999 a 2002) e {B['n_spam']} spams (2003 a 2005).
- Representação: contagem por substring de {B['n_palavras']} palavras, sobre o texto pré-processado da AUEB.
- O texto vem em inglês, num contexto corporativo do setor de energia dos EUA.

## Desempenho (limiar de custo: perder 1 e-mail real = 9 spams recebidos)

| Avaliação | Especialista | Generalista (S8) |
|---|---|---|
| Esta caixa, 20 folds: MCC | {n(esp['cv20']['mcc'])} | {n(gen['cv20']['mcc'])} |
| Esta caixa, 20 folds: e-mails reais na lixeira | {pct(esp['cv20']['real_lixeira_custo'])} | {pct(gen['cv20']['real_lixeira_custo'])} |
| Esta caixa, 20 folds: spam que passou | {pct(esp['cv20']['spam_passou_custo'])} | {pct(gen['cv20']['spam_passou_custo'])} |
| Enron2-6 (outras 5 caixas): MCC | {n(esp['externo']['mcc_custo'])} | {n(gen['externo']['mcc_custo'])} |
| Enron2-6: e-mails reais na lixeira | {pct(esp['externo']['real_lixeira_custo'])} | {pct(gen['externo']['real_lixeira_custo'])} |
| SpamAssassin (outra empresa, 2002): MCC | {n(esp['spamassassin']['mcc_custo'])} | {n(gen['spamassassin']['mcc_custo'])} |
| SpamAssassin: e-mails reais na lixeira | {pct(esp['spamassassin']['real_lixeira_custo'])} | {pct(gen['spamassassin']['real_lixeira_custo'])} |

## Limitações (leia antes de usar)
1. **O filtro aprende a caixa, não "o spam".** Fora da caixa do treino, o desempenho cai muito: no SpamAssassin, o MCC fica abaixo de 0,6 em todos os filtros. O mesmo pipeline treinado dentro do SpamAssassin chega a {n(TB['controle_dentro_da_base']['mcc'])}. **Para usar de verdade, retreine com e-mails da caixa de destino.**
2. **Só inglês.** Num e-mail em português, a cobertura do vocabulário cai para cerca de 15%, e a decisão perde o sentido. A demo avisa quando a cobertura é baixa.
3. **Época.** Os spams são de 2003 a 2005. Spams atuais (outros produtos, imagens, links encurtados) não estão representados.
4. **Vocabulário da empresa.** Palavras como "enron", "hpl" e nomes de colegas ajudam nesta caixa. Retirá-las quase não muda o resultado aqui, mas melhora um pouco em caixas muito diferentes.

## Considerações éticas
- Os e-mails da Enron tornaram-se públicos na investigação da FERC (EUA). Os do SpamAssassin foram publicados para pesquisa.
- Um filtro de spam errando para o lado do spam esconde mensagens legítimas. Por isso o limiar foi escolhido para que esse erro custe 9 vezes mais.
- O TabPFN é usado sob licença não comercial (pesquisa e avaliação).
"""

(RAIZ / "README.md").write_text(README, encoding="utf-8")
(RAIZ / "docs" / "MODEL_CARD.md").write_text(MODEL_CARD, encoding="utf-8")
print("gerados: README.md, docs/MODEL_CARD.md")
