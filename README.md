# Filtro de spam com stacking e TabPFN: o que um classificador aprende de uma caixa de e-mail

Projeto 3 de **Data Science 2 (ESEG)**, com o Prof. Marino Catarino.
**Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins (RA 44776).

São 5172 e-mails (3672 e-mails reais e 1500 spams), cada um descrito pela contagem de 3000 palavras. O ponto de partida foi um classificador. A investigação foi além: de onde vêm os dados, quais modelos erram de formas diferentes, quanto do ajuste fino é ilusão, o que um modelo de fundação (TabPFN) acrescenta e o que acontece quando o filtro encontra uma caixa de e-mail que nunca viu.

| | |
|---|---|
| 📓 Notebook para o professor | [`Projeto3_Classificacao_Spam.ipynb`](Projeto3_Classificacao_Spam.ipynb): a história completa, com o código principal visível, em poucos minutos · [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/Projeto3_Classificacao_Spam.ipynb) |
| 🧪 Notebooks técnicos | uma etapa por notebook, todos executados e prontos para o Colab ([lista abaixo](#os-notebooks)) |
| 🎞️ Apresentações | [banca](https://felipe44776-eseg.github.io/filtro-spam-stacking-enron/apresentacao/banca/) (técnica) · [equipe](https://felipe44776-eseg.github.io/filtro-spam-stacking-enron/apresentacao/equipe/) (didática, com glossário) |
| ✉️ Demonstração | `streamlit run demo/app.py`: cole um e-mail e veja cada etapa até a decisão |
| ✅ Para a banca | [INSTALACAO.md](INSTALACAO.md) · [COMO_TESTAR.md](COMO_TESTAR.md) |
| 🧭 Decisões e números | [docs/REGISTRO_DECISOES.md](docs/REGISTRO_DECISOES.md) · [docs/METODOLOGIA.md](docs/METODOLOGIA.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/REFERENCIAS.md](docs/REFERENCIAS.md) |

## Os filtros

Todos usam o **limiar de custo**: perder 1 e-mail real custa o mesmo que receber 9 spams (o cenário λ = 9 de Androutsopoulos et al., 2000). O custo é por mil e-mails; quanto menor, melhor. A última linha é a referência que todo filtro precisa vencer: **não filtrar** nada.

| Filtro | O que é | Onde brilha | MCC (20 folds) | PR-AUC (20 folds) | Custo (20 folds) | Teste trancado: reais na lixeira / spams que passaram | Custo Enron2-6 | Custo SpamAssassin |
|---|---|---|---|---|---|---|---|---|
| **Especialista (S7 do GA)** | Stacking 7 afinado pelo algoritmo genético | o menor custo nos 20 folds (no teste trancado, empata com o padrão) | 0,962 | 0,9976 | 36,6 | 1 / 10 | 207,6 | 2416,5 |
| **Stacking 8 (+TabPFN)** | Stacking 7 padrão + TabPFN-3.5 como 8º membro | com o TabPFN, o stacking fica bem mais barato fora da caixa (regra D10) | 0,956 | 0,9978 | 38,2 | 2 / 13 | 138,5 | 1119,9 |
| **TabPFN-3.5 sozinho** | modelo de fundação para tabelas, sem nenhum ajuste | o mais conservador fora da caixa: o que menos manda e-mail real para a lixeira | 0,922 | 0,9965 | 55,6 | 4 / 27 | 126,0 | 503,5 |
| Stacking 7 padrão | um modelo de cada família, hiperparâmetros padrão | roda sem internet e sem chave (usado na demo) | 0,955 | 0,9973 | 41,5 | 0 / 17 | 177,5 | 1977,2 |
| LinearSVC | LinearSVC com TF-IDF log | simples e rápido | 0,939 | 0,9952 | 52,0 | 3 / 23 | 156,9 | 528,4 |
| não filtrar | referência: todo spam passa, nenhum e-mail real se perde | o piso que todo filtro precisa vencer | – | – | 290,1 | 0 / 300 | 546,4 | 313,6 |

- **20 folds e teste trancado** medem a caixa do treino. Os 20 folds são re-partições dos e-mails de desenvolvimento; o teste trancado (1035 e-mails) foi separado no começo e só serviu para relatar.
- **Enron2-6** são as caixas de outros 5 funcionários da Enron, com spam de coleções parecidas com a do treino. O custo é a média das 5 caixas. Caixa a caixa, com λ = 9, perdem para não filtrar: no Enron2, Especialista (S7 do GA), Stacking 8 (+TabPFN) e Stacking 7 padrão; no Enron3, Especialista (S7 do GA).
- **SpamAssassin** é um corpus público de 2002, com e-mails de listas de discussão e pessoais. Decidiu a regra pré-registrada D10 e, na revisão D11, também serviu para decidir o que destacar: não é mais uma base intocada.
- **TCR** (custo de não filtrar ÷ custo do filtro; acima de 1, filtrar compensa). Enron2-6: Especialista (S7 do GA) 2,63, Stacking 8 (+TabPFN) 3,95, TabPFN-3.5 sozinho 4,34, Stacking 7 padrão 3,08, LinearSVC 3,48. SpamAssassin: Especialista (S7 do GA) 0,13, Stacking 8 (+TabPFN) 0,28, TabPFN-3.5 sozinho 0,62, Stacking 7 padrão 0,16, LinearSVC 0,59.
- Nenhuma dessas bases externas entrou no treino.

![Os filtros nas três avaliações](resultados/figuras/09_filtros_finais.png)

## As tecnologias em destaque

- **Stacking 7.** Um modelo de cada família (linear, SVM, kNN, rede neural, Naive Bayes, árvores, boosting), combinados por um meta-modelo treinado nas notas *out-of-fold*.
  - **No MCC, no limiar 0,5**, supera o melhor modelo único afinado (LinearSVC otimizado) por 0,0107, com p de Holm 0,042 nas 3 comparações do script 05 (com mais comparações na família de Holm, passa de 0,05) e P(stacking melhor) = 0,57 no teste bayesiano. No limiar de custo, os dois empatam no MCC.
  - **No custo**, comparado à linha de base (LinearSVC padrão), o Especialista custa 30% menos, com IC95 de -30,1 a -0,7 por mil e P(mais barato) = 0,95 no teste bayesiano. O Stacking 8 custa 27% menos, e o Stacking 7 padrão, 20%. Corrigida pelas 6 comparações (Holm), nenhuma dessas diferenças de custo chega a ser significativa a 5% (p = 0,246 para o Especialista).
- **Algoritmo genético.** Buscou os hiperparâmetros dos 7 modelos e do meta-modelo: avaliou 857 de 7.077.888 combinações, graças ao cache das notas. A configuração que ele achou é o **Especialista**, o filtro de menor custo nos 20 folds (36,6 por mil, contra 41,5 do padrão; P(mais barato) = 0,64). Os 20 folds são re-partições dos e-mails usados na escolha; no teste trancado, os dois empatam (19 contra 17).
- **TabPFN-3.5** (Prior Labs), um modelo de fundação para tabelas, pré-treinado em milhões de tabelas sintéticas:
  - **sozinho e sem nenhum ajuste**, empata nesta caixa com o melhor modelo único afinado (MCC no limiar 0,5: 0,962 contra 0,963); no limiar de custo, deixa passar mais spam que os outros aqui;
  - **fora da caixa, é o mais conservador:** o que menos manda e-mail real para a lixeira em todas as bases externas (as 5 caixas Enron e o SpamAssassin). Por isso é o mais barato onde predominam e-mails reais (Enron2, Enron3 e SpamAssassin) e na média do Enron2-6 (126,0 por mil); onde predomina spam, chega a ser o mais caro (Enron5).
- **Stacking 8** (o Stacking 7 + o TabPFN como 8º membro):
  - nesta caixa, empata no topo do PR-AUC (0,9978);
  - fora dela, o TabPFN deixa o stacking mais barato nas duas versões e nas duas bases externas (padrão: 177,5 → 138,5 no Enron2-6 e 1977,2 → 1119,9 no SpamAssassin; especialista: 207,6 → 130,2 no Enron2-6 e 2416,5 → 997,4 no SpamAssassin);
  - no SpamAssassin, pela regra escrita antes do teste (D10), o Stacking 8 padrão custou menos que o Stacking 7 padrão em todos os λ testados (de 42 a 857 a menos por mil; com λ = 9, P(redução) = 1,00).

## O que descobrimos

1. **O CSV é o Enron1, e as palavras foram contadas como substring.** Reproduzimos 5172 de 5172 linhas a partir dos textos originais. Há 26 colunas de uma letra só, e "ect" aparece em todos os e-mails, porque está dentro de "subject".
2. **Duplicata não indica spam.** 93% das 541 cópias são e-mails reais. Cópias idênticas ficam sempre do mesmo lado da divisão treino/teste.
3. **Sem rótulo, os grupos naturais são assuntos, não "spam × real".** O K-means tem ARI de -0,07. Sem nenhum spam rotulado, treinando só com e-mails reais (semissupervisionado), a melhor pista é "não se parece com meus e-mails reais", o erro de reconstrução, com PR-AUC 0,80.
4. **A representação vale mais que o algoritmo.** Com TF-IDF com log, a Reg. Logística sobe de 0,923 para 0,961, e o kNN, de 0,746 para 0,940.
5. **Modelos que erram diferente se completam.** O kNN, fraco sozinho, é um dos que menos erram junto com os outros e recebe o maior peso do meta-modelo (1,67). Treze modelos empatam com sete.
6. **O ajuste fino rende pouco.** Na busca, o algoritmo genético ganhou +0,0063 de MCC. Em divisões novas, o ganho foi +0,0010, com IC95 de -0,0049 a +0,0068: não se confirmou, como se espera de escolher o melhor entre centenas de candidatos (a maldição do vencedor).
7. **O stacking aprende a caixa e a coleção de spam.** Fora da caixa do treino, todos pioram, nos e-mails reais e no spam. No SpamAssassin, com λ = 9, nenhum filtro treinado aqui vence não filtrar (313,6 por mil), embora um modelo treinado *dentro* dele chegue a MCC 0,968. Com outro peso para o erro, a conta muda: com λ = 1, LinearSVC padrão e TabPFN-3.5-Plus sozinho vencem não filtrar. **Para uso real, o caminho é retreinar com e-mails da caixa de destino.**
8. **O MCC não vem do acaso.** Com os rótulos embaralhados 999 vezes, o MCC vai de 0,961 para 0,001 (p = 0,001, o menor possível). Quase-cópias (cosseno ≥ 0,95), que o agrupamento não separa, estão em 14% dos e-mails de teste dos 20 folds; no subconjunto sem elas, o MCC do Stacking 7 é 0,9703, contra 0,9734 no total.

## Limites (leia antes de usar)

- **Uma caixa só.** Os filtros aprendem o que é "normal" na caixa `farmer-d`. Em caixas muito diferentes, erram muito mais (item 7).
- **λ = 9 é uma escolha.** Com outro peso para o erro, os custos mudam. No Enron2-6, o mais barato é o TabPFN sozinho com λ = 3, 9 e 20; λ = 1: Stacking 7 especialista (`notebooks/07_contra_estudo.ipynb`).
- **Spam de coleções aparentadas.** O spam do Enron1 e o do Enron4 vêm da mesma coleção (GP); o do SpamAssassin entrou na coleção do Enron2 e do Enron5 (SpamAssassin + HoneyPot). As bases externas não são independentes no spam.
- **O teste trancado foi usado 4 vezes**, sempre para relatar, nunca para escolher. A decisão D8 citou como critério de desempate o fato de o candidato já ter sido avaliado nele.
- **O Stacking 8 e o TabPFN foram avaliados, não entregues prontos.** Precisam da API da Prior Labs (chave pessoal) e foram medidos com as notas gravadas em `resultados/cache_tabpfn/` e `experimentos/05_tabpfn/`; não há inferência pronta para um e-mail novo. A demo e os filtros salvos (Especialista, Stacking 7 padrão, LinearSVC) rodam sem chave.
- **Colab.** O ambiente do Colab (Python 3.12, `requirements-colab.txt`) foi reproduzido localmente numa cópia limpa do repositório: os testes passam, e o notebook-história e o da terceira base rodam até o fim (o filtro é baixado da Release). O Colab em si não foi executado por nós de ponta a ponta.

## Os notebooks

Todos rodam no **Google Colab**: a primeira célula baixa o projeto, instala os pacotes e reinicia o ambiente uma vez. Depois, use *Executar tudo* de novo. No computador, siga o [INSTALACAO.md](INSTALACAO.md).

| Notebook | Etapa | Colab |
|---|---|---|
| [`Projeto3_Classificacao_Spam.ipynb`](Projeto3_Classificacao_Spam.ipynb) | a história completa (para o professor) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/Projeto3_Classificacao_Spam.ipynb) |
| [`01_dados_e_qualidade.ipynb`](notebooks/01_dados_e_qualidade.ipynb) | Dados e qualidade | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/01_dados_e_qualidade.ipynb) |
| [`02_exploracao_nao_supervisionada.ipynb`](notebooks/02_exploracao_nao_supervisionada.ipynb) | Olhando sem rótulo (aprendizado não supervisionado) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/02_exploracao_nao_supervisionada.ipynb) |
| [`03_modelos_independentes.ipynb`](notebooks/03_modelos_independentes.ipynb) | Torneio de 22 modelos independentes | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/03_modelos_independentes.ipynb) |
| [`04_stacking.ipynb`](notebooks/04_stacking.ipynb) | Ninguém vence sozinho: o stacking | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/04_stacking.ipynb) |
| [`05_algoritmo_genetico.ipynb`](notebooks/05_algoritmo_genetico.ipynb) | O algoritmo genético e a maldição do vencedor | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/05_algoritmo_genetico.ipynb) |
| [`06_corrida_e_tabpfn.ipynb`](notebooks/06_corrida_e_tabpfn.ipynb) | A corrida pelo melhor filtro desta caixa, e o modelo de fundação (TabPFN) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/06_corrida_e_tabpfn.ipynb) |
| [`07_contra_estudo.ipynb`](notebooks/07_contra_estudo.ipynb) | Contra-estudo: tentando derrubar as próprias conclusões | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/07_contra_estudo.ipynb) |
| [`08_terceira_base.ipynb`](notebooks/08_terceira_base.ipynb) | Terceira base intocada: SpamAssassin (decisão D10, pré-registrada) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/08_terceira_base.ipynb) |
| [`09_modelos_finais.ipynb`](notebooks/09_modelos_finais.ipynb) | A grande final: os filtros entregues | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe44776-eseg/filtro-spam-stacking-enron/blob/main/notebooks/09_modelos_finais.ipynb) |

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

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json` (2026-10-28).*
