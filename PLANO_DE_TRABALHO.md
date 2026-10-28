# Plano de trabalho: do experimento ao repositório público e à apresentação

Criado em 2026-10-27. Os resultados que este plano usa estão em `docs/REGISTRO_DECISOES.md` (decisões D1 a D9).

## Onde estamos

| Etapa | Situação |
|---|---|
| Análise da base (Enron1, contagem por substring, duplicatas) | feita, nos scripts soltos de `experimentos/scripts/` |
| Torneio de 22 modelos independentes | feito (`experimentos/01_benchmark`) |
| Stacking ampliado e escolha do Stacking 7 | feito (`02_stacking`) |
| Algoritmo genético e confirmação com sementes novas | feito (`03_ga`) |
| Corrida pelo melhor filtro para esta caixa | feita (`04_corrida`) |
| TabPFN-3.5 sozinho e no stacking (nuvem e local) | feito (`05_tabpfn`, D9) |
| Bloco **não supervisionado** (visto em aula) | **falta** |
| Interpretação (palavras e erros), teste de permutação, curva de aprendizado, sensibilidade ao custo | **falta** |
| Código organizado, notebooks finais, README, instalação, apresentação, GitHub | **falta** |

### Os dois filtros finais

| | **Especialista** (esta caixa) | **Generalista** (outras caixas) |
|---|---|---|
| Configuração | Stacking 7 do algoritmo genético (#5, exp. 03) | Stacking 7 padrão |
| Limiar | de custo (1 e-mail real perdido = 9 spams) | de custo |
| Custo por mil e-mails, 20 folds desta caixa | **36,6** | 41,5 |
| Teste trancado: e-mails reais na lixeira / spams que passaram | 1 / 10 (custo 19) | **0 / 17 (custo 17)** |
| Custo por mil e-mails no Enron2–6 | 207,6 | **177,5** |

O LinearSVC é a linha de base simples. Em caixas novas ele custa menos ainda (135,8), mas tem MCC menor (0,829 contra 0,891). Isso entra como limitação no contra-estudo.

## Regras que evitam retrabalho

1. **Uma única fonte para o código.** O código fica no pacote `spamfilter/`. Os scripts usam o pacote, os notebooks são **gerados** dos scripts e nunca editados à mão, e a apresentação lê os resultados de arquivos.
2. **Uma única fonte para os números.** Todo número que aparece no README, no notebook ou nos slides sai de `resultados/*.json` ou `resultados/*.csv`, gerados pelos scripts. Nenhum número é digitado à mão; se um número mudar, basta regenerar.
3. **Congelar os números antes de escrever a história.** A narrativa e os slides só começam depois da Fase 4.
4. **A base de teste trancada já foi usada 4 vezes, todas para relatar** (a última em D11). Ela não serve para escolher nada.
5. **Regressão numérica.** Ao mover o código para o pacote, os números do REGISTRO precisam se repetir (tolerância de ±0,001). Assim, a refatoração não muda as conclusões.
6. **Vocabulário.** Sempre "e-mail real", nunca "ham" (exceto uma vez, no glossário). Os CSVs usam `;` como separador e `,` como decimal.

## Decisões (respondidas em 2026-10-27)

| # | Decisão | Resposta |
|---|---|---|
| **P1** | GitHub | Repositório **público** na conta `felipe44776-eseg`. **Autores:** Phelipe Torres Pamponet da França (RA 46643) e Felipe Marins (RA 44776). Professor: Marino Catarino. Disciplina: Data Science 2, ESEG |
| **P2** | Dados | **Vão todos para o repositório** (uso acadêmico): o CSV, o Enron1–6 vetorizado, os originais da AUEB e os caches. Onde não puder, entra o link. Exceção: `experimentos/referencias/*.txt` (texto integral de artigos, com direito autoral) vira bibliografia com links |
| **P3** | Apresentações | **Duas**, ambas em slides HTML com estilos e animações em JavaScript: **(a) banca/professor**, técnica e impactante; **(b) equipe**, com linguagem simples, todas as siglas explicadas e o passo a passo do que fizemos. O GitHub ganha também um **passo a passo de como testar** (`COMO_TESTAR.md`) |
| **P4** | Notebook do professor | Um notebook-história (≤ 10 min, código principal visível) e os notebooks técnicos 01 a 08 |
| **P5** | Capa e público | Sem nome de equipe. Professor Marino Catarino. A duração não foi definida: planejar 15 a 20 min, com slides de reserva |
| **P6** | TabPFN | Entra. ✅ Testada a versão mais nova (3.5), sozinha e como 8ª família do stacking: nuvem (3.5-Plus) e réplica local (3.5-Fast). Resultado em D9 |
| **P7** | Demonstração | **Sim:** colar um e-mail em texto puro e ver cada etapa da conversão até a decisão (T8.4). Experimento opcional com a caixa ESEG na Fase 11 |
| **P8** | Nome do repositório | `filtro-spam-stacking-enron` (sugestão, aceita salvo objeção) |

**A confirmar:**
- O RA do Felipe. O e-mail sugere 44776.
- Se o e-mail dele aparece no repositório. Por padrão, não.
- ~~Se vale criar conta na Prior Labs~~: criada em 2026-10-27 (`felipe-44776-eseg`), licenças aceitas.

---

## Fase 0: preservar o que já existe ✅ (feita em 2026-10-27)

- **T0.1** Experimentos copiados da pasta temporária para `experimentos/` (notebooks, CSVs, caches, índices do teste trancado, Enron vetorizado).
- **T0.2** Arquivos originais da AUEB salvos em `dados/brutos_aueb/` (16,5 MB; estavam em `experimentos/dados_externos/` até a Fase 3). O endereço antigo da AUEB já deu 404 uma vez.
- **T0.3** Registro de decisões D1 a D8 escrito; contexto do projeto em `CLAUDE.md`.

## Fase 1: decisões e preparação (depende de P1 a P8)

- **T1.1** ✅ P1 a P8 respondidas. Faltam as confirmações listadas acima.
- **T1.2** ✅ (2026-10-27) `dados/LEIA-ME.md` escrito.
  - A AUEB não declara licença: pede só a citação de Metsis et al. (2006), e os e-mails são públicos (investigação da FERC).
  - **A licença do CSV no Kaggle não foi verificada**, porque a página exige navegador. Falta conferir em https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv.

  Texto original da tarefa: registrar a origem e a licença dos dados em `dados/LEIA-ME.md`: a página do CSV no Kaggle e a do Enron-Spam na AUEB, com a citação de Metsis et al. (2006). Os dados entram no repositório (P2); a licença serve para citar corretamente e para saber se algum arquivo precisa virar só link.
- **T1.3** ✅ (2026-10-27) `requirements.txt` (ambiente principal) e `requirements-tabpfn-nuvem.txt` (ambiente separado do cliente na nuvem) estão travados.
  - As dependências foram resolvidas com `pip install --dry-run --ignore-installed`, sem conflito.
  - A instalação real numa venv nova fica para a T6.3, o teste em ambiente limpo.

  Texto original da tarefa: travar o ambiente: `requirements.txt` com as versões exatas usadas (Python 3.11, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.2.0, baycomp 1.0.3, jupytext, nbclient, matplotlib e outras). *Pronto quando* uma venv nova instalar sem erro.

## Fase 2: arquitetura do repositório (antes de mover qualquer código)

- **T2.1** 🟡 Esqueleto criado em 2026-10-27:
  - `.gitignore` (deixa de fora `.venv*`, `CLAUDE.md`, `.env`, `experimentos/referencias/`, o PDF do enunciado e `minha_caixa/`);
  - `LICENSE` (MIT, com os dois autores);
  - `pyproject.toml`;
  - `spamfilter/__init__.py`;
  - `dados/LEIA-ME.md`;
  - `docs/`;
  - `README.md` provisório.

  As pastas `resultados/`, `scripts/`, `notebooks/`, `apresentacao/`, `demo/` e `tests/` nascem nas fases que as preenchem. O CSV só muda para `dados/` na Fase 3, junto com os caminhos. Estrutura-alvo:

```text
filtro-spam-stacking-enron/
├── README.md                      resumo, resultados, como rodar em 3 comandos, autores
├── INSTALACAO.md                  passo a passo: Windows, macOS/Linux, Google Colab
├── COMO_TESTAR.md                 passo a passo para a banca: testes, notebooks, demo, conferir números
├── LICENSE                        MIT para o código; dados conforme T1.2
├── requirements.txt · pyproject.toml   pacote instalável (pip install -e .)
├── dados/                         LEIA-ME de origem e licença; CSV ou script de download
├── spamfilter/                    ÚNICA fonte do código
│   ├── dados.py          carga, grupos de duplicatas, base de teste trancada (SEED_TESTE=123)
│   ├── representacao.py  TF-IDF com log, NBFeatures (NBSVM)
│   ├── modelos.py        catálogo dos 22 modelos + STACKING7_PADRAO + STACKING7_ESPECIALISTA
│   ├── metricas.py       MCC, PR-AUC, e-mails reais na lixeira, spam que passou, custo, limiar de custo
│   ├── estatistica.py    t corrigido, Holm, bayesiano, McNemar, bootstrap por grupo, piso de detecção
│   ├── motor.py          notas out-of-fold em cache (o que tornou o algoritmo genético viável)
│   ├── genetico.py       algoritmo genético + busca aleatória de controle
│   ├── externo.py        download da AUEB, vetorização por substring, remoção de cópias
│   └── graficos.py       estilo único dos gráficos
├── scripts/                       uma etapa por arquivo, em formato jupytext (# %%)
│   ├── 01_dados_e_qualidade.py           ├── 05_algoritmo_genetico.py
│   ├── 02_exploracao_nao_supervisionada.py ├── 06_corrida_do_filtro.py
│   ├── 03_modelos_independentes.py       ├── 07_contra_estudo.py
│   └── 04_stacking.py                    └── 08_modelos_finais.py
├── notebooks/                     gerados dos scripts e executados com as saídas
├── Projeto3_Classificacao_Spam.ipynb  notebook-história para o professor
├── resultados/                    CSV, figuras, metricas_finais.json, modelos .joblib (fonte dos números)
├── apresentacao/                  slides HTML (GitHub Pages)
│   ├── banca/index.html           técnica e impactante
│   ├── equipe/index.html          didática, com siglas explicadas e passo a passo
│   └── comum/                     estilo, animações e dados (gerados de resultados/)
├── demo/                          app: cole um e-mail e veja cada etapa até a decisão
├── docs/                          REGISTRO_DECISOES.md · METODOLOGIA.md · MODEL_CARD.md
└── tests/                         testes rápidos (ver T3.7)
```

- **T2.2** ✅ (2026-10-27) `docs/CONTRATO_RESULTADOS.md`, versão 1: unidades, regras de arredondamento, fonte de cada seção e o esquema completo.

  Texto original da tarefa: definir o contrato de `resultados/metricas_finais.json`: quais chaves o README, o notebook e o deck vão ler. *Pronto quando* o esquema estiver escrito. Isso evita refazer a apresentação quando um número mudar.

## Fase 3: mover o código para o pacote, com teste de regressão ✅ (2026-10-28)

- `spamfilter/` criado com os módulos `caminhos`, `dados`, `representacao`, `modelos`, `metricas`, `estatistica`, `motor`, `genetico` e `externo`. O `graficos.py` nasce na Fase 4, com a primeira figura.
- Os dados mudaram para `dados/`: `emails.csv`, que é o `emails (1).csv` renomeado, `enron_vetores.npz` e `brutos_aueb/`. Os scripts antigos já procuram nos caminhos novos.
- `pytest`: **27 testes rápidos em cerca de 20 s**. `pytest -m lento`: **9 testes de regressão em 2 min 53 s** (log em `experimentos/regressao_lenta.log`). **Todos os números da tabela abaixo se repetiram**, com estes destaques:
  - o algoritmo genético refeito do zero dá 857 avaliações, 0,9795, e chega exatamente à configuração do Especialista;
  - SVM, MLP, Extra Trees e LightGBM retreinados repetem as notas do cache;
  - o Enron1 contado de novo a partir dos `.tar.gz` é idêntico ao CSV.
- O cache antigo (`experimentos/03_ga/cache_ga`) é lido pelo `spamfilter.motor` sem conversão.

Ordem: das peças de base para as compostas. Cada peça só avança depois de reproduzir o número já registrado.

| Tarefa | Origem atual | Precisa reproduzir |
|---|---|---|
| **T3.1** `dados.py` | `build_stack.py` (carga e grupos) | a mesma `indices_teste.npz` (1035 e-mails: 735 reais, 300 spams) |
| **T3.2** `representacao.py` + `modelos.py` | `build_bench.py`, `build_ga.py` | Reg. Logística com TF-IDF log: MCC 0,961 (5 folds, semente 42) |
| **T3.3** `metricas.py` + `estatistica.py` | `build_ga.py`, `metricas_demo.py` | p de Holm 0,166 (Stacking 4 × LinearSVC); piso de detecção ≈ 0,0086 com 20 folds |
| **T3.4** `motor.py` | `build_ga.py` (motor com cache) | Stacking 7 padrão: MCC 0,9734 nos 20 folds de confirmação |
| **T3.5** `genetico.py` | `build_ga.py` | 857 avaliações; melhor da busca 0,9795 |
| **T3.6** `externo.py` | `externo.py`, `build_stack.py` | Enron1 vetorizado = CSV (5172/5172); Stacking 7 padrão com MCC externo 0,872 |
| **T3.7** `tests/` | — | contagem por substring 300/300; nenhum grupo dos dois lados; métricas em casos conhecidos; os números acima |

*Pronto quando* `pytest` passar e os caches antigos puderem ser reaproveitados ou regenerados.

## Andamento das Fases 4 a 8 (2026-10-28)

O usuário pediu tudo de uma vez: testes, documentação, apresentações e GitHub. A validação do roteiro (T5.2) foi absorvida nesse pedido.

**Fase 4 ✅:** scripts `scripts/01` a `09` no formato `# %%`, usando o pacote; saídas em `resultados/`.
- **T4.2:** não supervisionado.
- **T4.3 e T4.4:** interpretação e contra-estudo: permutação, ablação, curva de aprendizado com o TabPFN e sensibilidade a λ.
- **D10:** 3ª base, o SpamAssassin, com regra pré-registrada. O Stacking 8 virou o stacking para caixas novas.
- **D11:** revisão crítica. O rótulo "Generalista" foi aposentado; "não filtrar" e o TCR entraram nas comparações; o TabPFN sozinho entrou entre os filtros finais.
- **T4.6:** os filtros finais e o `metricas_finais.json`.
- **Conversor bruto → AUEB** (`spamfilter.conversao`): 90,5% de reprodução exata.

**Fase 5:** o roteiro é o próprio `scripts/historia.py` (12 seções).

**Fase 6:** `scripts/gerar_notebooks.py` gera os 9 notebooks técnicos e o `Projeto3_Classificacao_Spam.ipynb`.

**Fase 7:** documentação.
- `README.md` e `docs/MODEL_CARD.md` são gerados por `scripts/gerar_documentos.py`.
- `INSTALACAO.md`, `COMO_TESTAR.md` e `docs/METODOLOGIA.md`.
- O `REGISTRO` foi para `docs/`.

**Fase 8:**
- decks em `apresentacao/`, construídos pelo agente deck-html, com `build.py` lendo o JSON;
- demo em `demo/app.py` (Streamlit).

## Fase 4: completar as análises que faltam (alimentam a história)

| Tarefa | O que entrega | Capítulo da história |
|---|---|---|
| **T4.1** EDA final | proporção de classes, esparsidade (94% de zeros), palavras que mais separam spam de e-mail real, duplicatas por classe; figuras | 1 e 2 |
| **T4.2** **Bloco não supervisionado** | SVD em 2D colorido pelo rótulo; K-means e agrupamento hierárquico comparados ao rótulo (ARI); tópicos com NMF; Isolation Forest e One-Class SVM treinados **só com e-mails reais** (spam como anomalia), comparados ao supervisionado | 3 |
| **T4.3** Interpretação | pesos das palavras (Reg. Logística), peso de cada família no meta-modelo, **e-mails reais que foram para a lixeira**, com o texto original da AUEB | 6 e 10 |
| **T4.4** Contra-estudo restante | teste de permutação (rótulos embaralhados → MCC ≈ 0); ablação das palavras da Enron ("enron", "ect", "hou", nomes); curva de aprendizado; **sensibilidade ao custo** (λ = 1, 3, 9 e 20: o vencedor muda?) | 10 |
| **T4.5** ✅ TabPFN-3.5 sozinho | feito em `experimentos/05_tabpfn` (D9): 20 folds e base externa pela nuvem (3.5-Plus, aceita as 3000 colunas sem redução); réplica local em CPU (3.5-Fast, 5 divisões, correlação 0,993 com a nuvem) | 5 (estratégia nova da pesquisa) |
| **T4.5b** ✅ TabPFN como 8ª família do stacking | feito (D9): redundante nesta caixa (P(equivalentes) ≥ 0,999); em caixas novas baixa o custo. Adotá-lo no Generalista exige uma 3ª base intocada (SpamAssassin, opcional) | 6, 7 e 10 |
| **T4.6** Modelos finais | Especialista e Generalista treinados nos 5172 e-mails; `.joblib` + limiares + `metricas_finais.json` | 9 |

*Pronto quando* `metricas_finais.json` estiver completo. **A partir daqui os números estão congelados.**

## Fase 5: roteiro da história (antes de notebook e slides)

- **T5.1** Escrever `docs/ROTEIRO.md`. Para cada capítulo: a mensagem em uma frase, a evidência (com a chave do JSON), a figura e o script de origem.

| # | Capítulo | Mensagem central |
|---|---|---|
| 1 | O problema e a base | São 5172 e-mails da Enron. Descobrimos de onde vieram e como as palavras foram contadas |
| 2 | Qualidade dos dados | "Duplicata indica spam?" Não: 93% das duplicatas são e-mails reais. Por isso o protocolo separa por grupo |
| 3 | Olhando sem rótulo | Os e-mails se separam sozinhos? (não supervisionado) |
| 4 | Como medir | A acurácia engana (71% sem fazer nada). Por isso MCC, PR-AUC e o custo de perder um e-mail real |
| 5 | Torneio de 22 modelos | O detalhe que virou o jogo: TF-IDF com log (Reg. Logística de 0,922 para 0,961) |
| 6 | Ninguém vence sozinho | Modelos diferentes erram diferente. O kNN, fraco sozinho, é o reforço mais valioso do time |
| 7 | Mais não é melhor | 13 modelos não superam 7 |
| 8 | O algoritmo genético | 7 milhões de combinações; o ganho não se confirmou em divisões novas; a maldição do vencedor; o piso estatístico |
| 9 | A grande final | Especialista (esta caixa), Stacking 8 e TabPFN (fora da caixa), contra a referência "não filtrar" |
| 10 | Contra-estudo e limites | Em caixas novas o e-mail real muda (cada pessoa tem seu vocabulário); o LinearSVC sai mais barato |
| 11 | Conclusões | O que aprendemos e o que faríamos depois |

- **T5.2** Validar o roteiro com você **antes** da Fase 6. É o ponto mais barato para mudar a narrativa.

## Fase 6: notebooks

- **T6.1** Cada script da Fase 3 vira `notebooks/0N_*.ipynb` via jupytext e é executado com as saídas salvas. Por padrão usa o cache (rápido); com `EXECUCAO_COMPLETA=True`, recalcula tudo (cerca de 2,5 h de CPU).
- **T6.2** `Projeto3_Classificacao_Spam.ipynb`: segue o roteiro, mostra o código principal, lê o restante de `resultados/`, roda em ≤ 10 min (Restart & Run All) e é compatível com o Colab.
- **T6.3** Teste em ambiente limpo: venv nova, seguir o INSTALACAO.md e rodar tudo. Medir os tempos. *Pronto quando* rodar do zero sem intervenção.

## Fase 7: documentação

- **T7.1** `README.md`: o problema em 3 linhas, a tabela dos dois filtros, uma figura, "como rodar em 3 comandos", estrutura, créditos e citação dos dados (Metsis et al., 2006).
- **T7.2** `INSTALACAO.md` para a banca:
  - Windows (PowerShell), macOS/Linux e Colab;
  - tempo esperado de cada etapa;
  - problemas comuns: Python da Microsoft Store, pouca RAM (fechar o navegador; a execução se ajusta sozinha), antivírus lento com venv.
- **T7.3** `docs/METODOLOGIA.md` (protocolo, métricas, testes estatísticos) e `docs/MODEL_CARD.md` (uso pretendido, limites em caixas novas, viés da base de 1999–2005).
- **T7.4** Mover `REGISTRO_DECISOES.md` para `docs/`, com links.
- **T7.5** `COMO_TESTAR.md`, o roteiro da banca em passos numerados, cada um com o resultado esperado:
  1. instalar;
  2. `pytest` (cerca de 1 min);
  3. abrir o notebook-história e rodar tudo;
  4. conferir 3 números-chave contra o README;
  5. abrir os slides;
  6. rodar a demo e colar os 3 e-mails de exemplo (spam, e-mail real e um em português);
  7. opcional: reexecutar um experimento completo.

## Fase 8: apresentação interativa

- **T8.1** Base comum dos dois decks (`apresentacao/comum/`):
  - estilo visual único, slides de 1280×720 e impressão em PDF;
  - animações em JavaScript puro, com gráficos em SVG desenhados e animados por código, sem bibliotecas externas: cada deck é um arquivo que abre offline;
  - os dados vêm do `metricas_finais.json`, embutidos no *build* por um script, sem números digitados à mão.
- **T8.2** **Deck da banca** (12 a 15 slides + reserva), seguindo o roteiro:
  - abertura com o problema;
  - animação do caminho de um e-mail pelo filtro;
  - torneio de 22 modelos em ranking animado;
  - mapa de "quem erra o quê" (por que o stacking funciona);
  - algoritmo genético × busca aleatória em curva animada;
  - simulador do limiar: arrastar o peso do custo e ver e-mails reais na lixeira × spams que passaram, nos dois filtros;
  - a grande final entre os dois filtros;
  - contra-estudo e conclusões.
  Notas do apresentador e cronômetro.
- **T8.3** **Deck da equipe**, didático: o mesmo arco, com linguagem simples, analogias e "o que fizemos em cada passo".
  - Cada sigla é explicada na primeira aparição e reunida num **glossário navegável**: MCC, PR-AUC, ROC-AUC, AUC, TF-IDF, SVD/LSA, PCA, χ², SVM, RBF, kNN, MLP, NB, NBSVM, LR, RF, ET, LightGBM, XGBoost, CV, OOF, GA, ARI, NMF, IC95, ROPE, VP/VN/FP/FN, CSV, JSON, HTML.
  - Linha do tempo clicável: cada etapa abre o que testamos, o que deu certo ou errado e o que decidimos.
- **T8.4** **Demonstração "cole um e-mail"**, um app local (Streamlit) que mostra cada etapa com animação:
  1. o texto colado;
  2. a conversão para o formato da AUEB (minúsculas, pontuação separada por espaços, linha `Subject:`);
  3. a contagem das 3000 palavras do vocabulário, destacando no texto as que foram encontradas, e a **cobertura**: quanto do texto o vocabulário conhece;
  4. o vetor binário de presença (0/1) ao lado do vetor de contagens;
  5. o TF-IDF com log;
  6. a nota de cada um dos 7 modelos;
  7. a soma ponderada do meta-modelo e a probabilidade;
  8. o limiar de custo e a decisão, no **Especialista** e no **Generalista**;
  9. as palavras que mais empurraram para spam ou para e-mail real.

  O app avisa quando a cobertura é baixa (por exemplo, um texto em português): "o filtro não conhece este vocabulário". Os slides trazem a mesma sequência animada para 3 exemplos pré-calculados em Python, com os números exatos. Publicar no Hugging Face Spaces (link sem instalação para a banca) exige uma conta sua: opcional.
- **T8.5** Ensaio cronometrado dos dois decks; ajuste do tempo por slide; PDF de reserva.

## Fase 9: publicação

- **T9.1** Pré-checagem: comando `tenant` no diretório (identidade ESEG); conta gh `felipe44776-eseg`.
- **T9.2** Conteúdo do repositório:
  - **entram** os dados e os caches (cerca de 80 MB no total, maior arquivo com 14 MB, dentro dos limites do GitHub), para a banca reproduzir sem esperar horas;
  - **ficam fora** (`.gitignore`): `.venv/`, `experimentos/referencias/` (direito autoral), `minha_caixa/` (e-mails pessoais, Fase 11) e `CLAUDE.md` (contexto local).
- **T9.3** Varredura antes do push: segredos, caminhos locais (`C:\Users\...`), tamanho dos arquivos.
- **T9.4** Criar o repositório, enviar, ligar o GitHub Pages (deck) e criar uma *release* com os modelos e os caches.
- **T9.5** **Teste da banca:** clonar numa pasta limpa e seguir só o INSTALACAO.md. *Pronto quando* funcionar sem nenhuma ajuda.

## Fase 10: entrega

- **T10.1** Checklist final: links do repositório, do deck e do notebook; PDF de reserva do deck; versões travadas; ensaio feito.

---

## Fase 12: vocabulário para leigos nos decks (pedido de 2026-10-28)

Revisão do usuário depois dos decks em forma de história: o texto ainda supõe que o leitor conhece os termos. Cada tarefa vale para os dois decks (banca e equipe); o que for conteúdo de referência vai também para `docs/`.

- [x] **V1 · Enron.** Explicar que empresa era a Enron, por que os e-mails dela são públicos e de quem é a caixa usada. *Feito:* slide "De onde vêm os e-mails" (banca `fontes`, equipe `e-fontes`), glossário e `dados/LEIA-ME.md`, seção 0.
- [x] **V2 · De onde vieram os dados.** O caminho das fontes: o CSV do Kaggle → a descoberta de que é o Enron1 da AUEB → o Enron2–6 e o SpamAssassin como bases externas, com os endereços. *Feito:* mesma trilha em quatro passos; endereços em `dados/LEIA-ME.md`.
- [x] **V3 · Metsis et al. (2006).** Confirmar que o CSV vem desse artigo e resumir o que ele fez e concluiu (num slide do apêndice e em `docs/`). *Feito:* sim, o CSV é o Enron1 do artigo, mas a análise do artigo é outra (cinco versões do Naive Bayes, retreino a cada 100 e-mails). Resumo no apêndice da banca (`metsis`), na equipe (`e-metsis`) e em `dados/LEIA-ME.md`, seção 3b.
- [x] **V4 · Folds.** Explicar o que é um fold na primeira vez que a palavra aparece (validação cruzada, partes, rodadas). *Feito:* grade das cinco rodadas no slide das cópias, dica no glossário e texto na história.
- [x] **V5 · Token.** Dizer que "pedaço de texto" também se chama *token* (e *substring*). *Feito*, com a ressalva de que aqui a contagem é por substring.
- [x] **V6 · As cópias.** Na primeira menção aos e-mails iguais, explicar os detalhes: o que conta como cópia, quantas são, de que tipo, por que importam e o que fizemos. *Feito:* slide "As cópias" no corpo dos dois decks; medidas novas no `scripts/01` (`base.copias`). Achado: cópia é a mesma linha de contagens, não o mesmo texto (D12).
- [x] **V7 · Notas explicativas ao passar o mouse.** Toda métrica, método e sigla (SVD, K-means, NMF, MCC, PR-AUC, TF-IDF, TCR, λ…) com uma explicação para leigo que aparece ao passar o mouse; uma fonte única de glossário para os dois decks; glossário também na banca (tecla G e apêndice para o PDF). *Feito:* `apresentacao/fonte/comum/glossario.json` (120 termos) → dicas nos dois decks, glossário na banca e `docs/GLOSSARIO.md`. O "Como medir" subiu para o corpo da banca.
- [x] **V8 · Animações dos modelos.** Cada animação deixa explícito o que é cada elemento (o que é um ponto, quais são os dois lados, o que é a linha) e cada modelo ganha uma analogia completa, com os dois lados nomeados. *Feito:* as 11 animações do `modelos.js` com legenda de cores, eixos em palavras reais, os dois lados escritos no desenho, passo a passo numerado e o "e-mail novo" decidido no fim; cartões com analogia de duas a três frases e a linha "Na figura:".
- [x] **V9 · Fechamento.** Regenerar decks e documentos, verificar (build, geometria, testes), revisão crítica independente e publicação. *Feito:* a revisão crítica achou dois erros de conteúdo (as letras soltas também são colunas; os spams não são da caixa do funcionário) e outros dez menores, todos corrigidos (D12, item 4).

## Fase 11 (opcional): a caixa ESEG do Felipe

Feita só depois da Fase 8, porque reaproveita a demo e o pacote. Tudo roda **localmente**, e nada entra no repositório.

- **T11.1** Exportar o Gmail da ESEG pelo Google Takeout (formato mbox, incluindo a pasta Spam). Isso depende de a administração da ESEG permitir. Alternativa garantida: salvar algumas dezenas de e-mails como `.txt`.
- **T11.2** Aplicar o filtro treinado no Enron e medir. A expectativa é de desempenho ruim: o texto é em português e o vocabulário é outro. Isso vira um bom slide de "mudança de domínio".
- **T11.3** Opcional: treinar o mesmo pipeline com um vocabulário tirado da própria caixa. A limitação: só há cerca de 30 dias de spam.

## Por que esta ordem

| Se inverter... | Retrabalho |
|---|---|
| Slides antes de congelar os números (Fase 4) | Refazer gráficos e tabelas a cada análise nova |
| Notebooks antes do pacote (Fase 3) | Corrigir o mesmo erro em 8 notebooks |
| Roteiro depois dos notebooks | Reordenar e reescrever notebooks prontos |
| Publicar antes do teste em ambiente limpo | A banca encontra o erro de instalação antes de nós |
| Não supervisionado por último | O capítulo 3 entra remendado na história |

## Tempo de máquina estimado

| Etapa | CPU |
|---|---|
| Reexecutar 01 a 04 do zero | cerca de 2 h (11 + 22 + 61 + 17 min) |
| Análises novas da Fase 4 | cerca de 30 a 45 min |
| Notebooks com cache | poucos minutos cada |
| Notebook-história | ≤ 10 min |
