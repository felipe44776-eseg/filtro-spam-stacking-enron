# Dados: origem, licença e como foram gerados

## 1. `emails.csv`: a base do projeto
| Item | Valor |
|---|---|
| Fonte | Kaggle, *Email Spam Classification Dataset CSV* (`balaka18`): https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv |
| Arquivo | `dados/emails.csv`: é o arquivo do enunciado, `emails (1).csv`, só renomeado (MD5 `6605665DC3443690DA7973BA4DE35FC6`) |
| Formato | 5172 linhas (e-mails) × 3002 colunas: `Email No.`, 3000 contagens de palavras e `Prediction` (1 = spam, 0 = e-mail real) |
| Licença | uso acadêmico autorizado pela página do dataset no Kaggle |

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

## 2b. SpamAssassin public corpus: a 3ª base (decidiu a regra pré-registrada D10; na revisão D11, serviu também para decidir o que destacar)
| Item | Valor |
|---|---|
| Fonte | https://spamassassin.apache.org/old/publiccorpus/, conjunto 20030228 (easy_ham, easy_ham_2, hard_ham, spam, spam_2): 6047 mensagens, cerca de 31% de spam |
| No repositório | os 5 arquivos originais em `dados/spamassassin/` (cerca de 7 MB, com o `readme.html` do corpus) e as contagens em `dados/spamassassin_vetores.npz` (X com 6047 × 3000, y e nomes). Os textos foram publicados pelo projeto SpamAssassin para pesquisa; o copyright de cada mensagem é do remetente |
| Para refazer | `python -c "from spamfilter import externo; externo.gerar_vetores_spamassassin()"` conta de novo a partir de `dados/spamassassin/` (e baixa os arquivos, se faltarem) |
| Conversão | os e-mails brutos passam ao formato da AUEB por `spamfilter.conversao` antes da contagem |

## 2c. E-mails brutos da AUEB (validação do conversor)
`dados/brutos_aueb_raw/` guarda:
- `farmer-d.tar.gz`, de http://www2.aueb.gr/users/ion/data/enron-spam/raw/ham/farmer-d.tar.gz;
- `GP.tar.gz`, de .../raw/spam/GP.tar.gz.

São as mensagens originais, com cabeçalhos, que deram origem ao Enron1 (a `GP` também ao Enron4). Com elas, o conversor foi validado nos dois lados:
- **e-mails reais:** 90,5% dos 3669 e-mails da farmer-d viram exatamente uma linha do CSV (`tests/test_conversao.py`, teste lento);
- **spam:** a reprodução exata é menor, mas, nos spams do Enron4 com par próximo na coleção bruta (cerca de 90%), a decisão quase não muda entre a versão da AUEB e a convertida (`scripts/08_terceira_base.py`, seção 6; números em `resultados/08_terceira_base.json`).

## 2d. De onde vem o spam de cada caixa
Cada caixa do Enron-Spam junta os e-mails reais de um funcionário com spam de uma coleção, conforme o `Summary.txt` de cada `.tar.gz`: o Enron1 e o Enron4 usam a coleção `GP`; o Enron2 e o Enron5, `SpamAssassin + HoneyPot`; o Enron3 e o Enron6, `BG`. Por isso o spam do Enron4 é parente do spam do treino, e o spam do SpamAssassin, do spam do Enron2 e do Enron5. A tabela gerada está em `resultados/08_terceira_base.json` (`fontes_spam`).

## 3. Como citar
> V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", *Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006)*, Mountain View, CA, USA, 2006.

## 4. Modelo de terceiros usado nos experimentos
O TabPFN-3.5 (Prior Labs) foi usado em pesquisa acadêmica, sem fins comerciais: a versão 3.5-Fast localmente, sob a licença *TABPFN-3-5 License v1.0* dos pesos abertos (https://docs.priorlabs.ai/models), e a versão 3.5-Plus pela API da Prior Labs, sob os termos de uso da plataforma aceitos na criação da conta. Para rodar, é preciso ter conta própria, aceitar a licença e criar uma chave em https://platform.priorlabs.ai. A chave vai na variável de ambiente `TABPFN_TOKEN` e **nunca** em arquivo.
