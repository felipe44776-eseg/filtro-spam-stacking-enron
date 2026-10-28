# Dados: origem, licença e como foram gerados

## 1. `emails.csv`: a base do projeto
| Item | Valor |
|---|---|
| Fonte | Kaggle, *Email Spam Classification Dataset CSV* (`balaka18`): https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv |
| Arquivo | `dados/emails.csv`: é o arquivo do enunciado, `emails (1).csv`, só renomeado (MD5 `6605665DC3443690DA7973BA4DE35FC6`) |
| Formato | 5172 linhas (e-mails) × 3002 colunas: `Email No.`, 3000 contagens de palavras e `Prediction` (1 = spam, 0 = e-mail real) |
| Licença | **não verificada.** A página do Kaggle exige navegador para exibir a licença. Conferir lá antes de publicar e registrar aqui |

**O que descobrimos sobre ela** (verificado nos experimentos; ver `docs/REGISTRO_DECISOES.md`):
- O CSV é o **Enron1** do corpus Enron-Spam. São 3672 e-mails reais da caixa `farmer-d` (de 10/12/1999 a 11/01/2002) e 1500 spams da coleção `GP` (de 18/12/2003 a 06/09/2005), conforme o `Summary.txt` original.
- Cada célula é `texto.count(palavra)`: a contagem de **substrings**, sensível a maiúsculas, sobre o texto pré-processado da AUEB. A reprodução bate em 5172 de 5172 e-mails (`experimentos/scripts/reproduz.py`, `experimentos/scripts/externo.py`).
- Há 541 cópias excedentes, 93% delas de e-mails reais. Os experimentos as mantêm na base e separam treino e teste por grupo.

## 2. Enron-Spam (AUEB): a base externa e os originais
| Item | Valor |
|---|---|
| Fonte | Athens University of Economics and Business: http://www2.aueb.gr/users/ion/data/enron-spam/ (o HTTPS tem certificado inválido; usar HTTP) |
| Arquivos | `dados/brutos_aueb/enron1.tar.gz` a `enron6.tar.gz` (16,5 MB, versão *preprocessed*), cópia preservada porque o endereço antigo `nlp.cs.aueb.gr` já respondeu 404 |
| Vetorizado | `dados/enron_vetores.npz`: Enron1–6 contados com o mesmo vocabulário de 3000 palavras e o mesmo método do CSV. Para refazer: `spamfilter.externo.gerar_vetores()` (alguns minutos em CPU) |
| Licença | **sem licença explícita.** O `readme.txt` original só diz onde os dados estão disponíveis e pede a citação abaixo. Os e-mails da Enron tornaram-se públicos durante a investigação da FERC (EUA). Uso acadêmico, com citação |

Formato do texto pré-processado: a primeira linha é `Subject: ...`, o restante vem em minúsculas, e a pontuação é separada por espaços. A demonstração do projeto reproduz esse formato antes de contar as palavras.

## 2b. SpamAssassin public corpus: a 3ª base, nunca usada para escolher (decisão D10)
| Item | Valor |
|---|---|
| Fonte | https://spamassassin.apache.org/old/publiccorpus/, conjunto 20030228 (easy_ham, easy_ham_2, hard_ham, spam, spam_2): 6047 mensagens, cerca de 31% de spam |
| No repositório | **só as contagens**: `dados/spamassassin_vetores.npz` (X com 6047 × 3000, y e nomes). Os textos têm copyright dos remetentes, conforme o `readme.html` do corpus |
| Para refazer | `python -c "from spamfilter import externo; externo.gerar_vetores_spamassassin()"` baixa os 5 arquivos (cerca de 7 MB) para `dados/spamassassin/` e conta de novo |
| Conversão | os e-mails brutos passam ao formato da AUEB por `spamfilter.conversao` antes da contagem |

## 2c. E-mails brutos da AUEB (validação do conversor)
`dados/brutos_aueb_raw/` (fora do repositório) guarda:
- `farmer-d.tar.gz`, de http://www2.aueb.gr/users/ion/data/enron-spam/raw/ham/farmer-d.tar.gz;
- `GP.tar.gz`, de .../raw/spam/GP.tar.gz.

São as mensagens originais, com cabeçalhos, que deram origem ao Enron1. Com elas, o conversor foi validado: 90,5% dos 3669 e-mails reais da farmer-d viram exatamente uma linha do CSV (`tests/test_conversao.py`, teste lento).

## 3. Como citar
> V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", *Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006)*, Mountain View, CA, USA, 2006.

## 4. Modelo de terceiros usado nos experimentos
O TabPFN-3.5 (Prior Labs) foi usado sob a licença não comercial *TABPFN-3-5 License v1.0*, que permite "avaliação e pesquisa": https://docs.priorlabs.ai/models. Para rodar, é preciso ter conta própria, aceitar a licença e criar uma chave em https://platform.priorlabs.ai. A chave vai na variável de ambiente `TABPFN_TOKEN` e **nunca** em arquivo.
