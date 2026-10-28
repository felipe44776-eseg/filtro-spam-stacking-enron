# Dados: origem, licença e como foram gerados

## 0. O caminho das fontes, em quatro passos
1. **A Enron.** Empresa de energia de Houston (EUA) que faliu em dezembro de 2001 num escândalo de fraude contábil. A FERC, a agência reguladora de energia dos EUA, que investigava a manipulação do mercado de energia do oeste do país, publicou os e-mails da empresa em 2003. A versão organizada para pesquisa por Klimt e Yang (2004) tem cerca de meio milhão de e-mails de uns 150 funcionários, a maioria da diretoria: é a maior coleção pública de e-mails reais de uma empresa.
2. **O Enron-Spam.** Metsis, Androutsopoulos e Paliouras (2006) juntaram os e-mails reais de seis desses funcionários com spams de quatro fontes (três coleções públicas e os spams recebidos pelo próprio Paliouras) e montaram seis caixas, Enron1 a Enron6, publicadas pela AUEB (seção 2). O resumo do artigo está na seção 3b.
3. **O CSV do Kaggle.** O arquivo do enunciado. A página o descreve como 5172 e-mails escolhidos ao acaso, com as 3000 palavras mais comuns como colunas (seção 1).
4. **O que descobrimos.** Refazendo a contagem a partir dos textos da AUEB, o CSV é o Enron1 inteiro, linha por linha (5172 de 5172): não é uma amostra dele. O sorteio que existe é anterior: os autores sortearam os 1500 spams do Enron1 de uma coleção maior (a GP bruta tem 13.719). As outras cinco caixas e o SpamAssassin (seção 2b) viraram a prova fora da caixa.

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
- Cada coluna é um termo do vocabulário (em geral uma palavra, mas também 26 letras soltas e 127 pares de letras), contado como pedaço de texto: não como *token* (a palavra recortada do texto), e sim como substring, achada até dentro de outras palavras. "ect" aparece em todos os e-mails porque está dentro de "Subject", e a coluna "e" conta toda letra "e". Não há coluna com algarismo nem com sinal.
- **Cópias.** "Cópia" quer dizer dois e-mails com a **mesma linha de 3000 contagens**, não o mesmo texto. Como até as letras soltas são colunas, os textos de um grupo têm as mesmas letras; só muda o que o vocabulário não vê. São 541 linhas repetidas (além da primeira de cada grupo) em 286 grupos; 240 grupos são pares e o maior tem 31 e-mails: um mês inteiro do aviso diário "hpl nom for …", em que muda o dia e o mês não. Nos textos originais, 120 grupos têm texto idêntico; nos outros 166, mudam só os espaços (93), os algarismos e sinais (67) ou a ordem dos trechos (6). 93% das cópias são e-mails reais, e nenhum grupo mistura as classes. Os experimentos mantêm as cópias e separam treino e teste por grupo (`resultados/01_dados.json`, `base.copias`).

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

## 3b. Resumo do artigo de Metsis et al. (2006)
**Objetivo.** Comparar cinco versões do Naive Bayes usadas em filtros de spam (Bernoulli multivariado, multinomial com contagens, multinomial com presença ou ausência, Gauss multivariado e Flexible Bayes) e publicar bases de teste mais realistas que as anteriores.

**As bases (o Enron-Spam).** Seis caixas. Cada uma junta os e-mails reais de um funcionário da Enron (farmer-d, kaminski-v, kitchen-l, williams-w3, beck-s e lokay-m, na versão limpa de Bekkerman) com spams de uma coleção:
- **GP:** os spams recebidos pelo próprio Paliouras de dezembro de 2003 a setembro de 2005. É a única coleção que não veio de armadilha, e a única em que as cópias foram mantidas, por fazerem parte do fluxo normal de spam.
- **BG:** a coleção de Bruce Guenter, de agosto de 2004 a julho de 2005.
- **SH:** SpamAssassin + HoneyPot, de maio de 2001 a julho de 2005.

Nas coleções de armadilha, as cópias foram tiradas por uma heurística de linhas em comum. Três caixas têm cerca de três e-mails reais para cada spam (Enron1 a 3); nas outras três, a proporção se inverte. Cada caixa tem entre cinco e seis mil e-mails.

**A limpeza.** Ficaram só o assunto e o corpo, sem cabeçalhos nem HTML. Saíram os e-mails enviados pelo dono da caixa e os spams em alfabetos não latinos, que seriam fáceis demais.

**A prova.** Os e-mails ficam em ordem de chegada dentro de cada classe, e os spams são intercalados ao acaso entre os e-mails reais. O filtro é retreinado a cada 100 e-mails e testado nos 100 seguintes, como um usuário novo. As palavras são escolhidas por ganho de informação a cada treino (500, 1000 ou 3000) e precisam aparecer em pelo menos cinco e-mails.

**As réguas.** A revocação do spam (quanto do spam é pego) e a dos e-mails reais (quanto deles é entregue), em curvas ROC limitadas a até 20% de e-mails reais perdidos. Os autores argumentam que a AUC comum não serve para filtros de spam: ela é dominada por regiões que não interessam na prática.

**O que concluíram.**
- As melhores versões foram o Naive Bayes multinomial com presença ou ausência das palavras e o Flexible Bayes, por margens pequenas e nem sempre significativas. Os autores tendem a preferir o primeiro, mais rápido e com uma troca mais suave entre um erro e outro, mas dizem que mais experimentos seriam necessários. O Bernoulli multivariado pareceu a pior.
- 3000 palavras deram os melhores resultados, mas a diferença para 500 ou 1000 foi pequena.
- Algumas caixas são bem mais fáceis que outras (o Enron4 é a mais fácil; o Enron1, das difíceis), sem explicação clara pela proporção ou pela coleção de spam.
- No Enron1, com 3000 palavras e limiar 0,5, a versão preferida pegou 96,0% do spam e entregou 95,25% dos e-mails reais (Tabelas 4 e 5).

**O que o artigo diz e pesa neste projeto.**
- Em cada caixa, o spam é de outra época que os e-mails reais (o artigo cita isso para explicar por que não dá para ordenar os e-mails pela data). Os autores defendem a mistura: como quase nenhum spam é pessoal, juntar os e-mails reais de uma pessoa com spam recebido por outras dá bases razoáveis.
- Os e-mails reais do SpamAssassin vêm de muitas pessoas: o artigo considera a coleção inadequada para avaliar um filtro pessoal.
- Os mesmos autores, em trabalhos anteriores, usaram λ = 1, 9 e 999. O λ = 9 deste projeto vem de Androutsopoulos et al. (2000).

**A nossa leitura (não está no artigo).** A diferença de época e de pessoa entre o spam e os e-mails reais é uma pista a mais que um filtro pode aprender, além do conteúdo. E foi no SpamAssassin que todos os nossos filtros mais caíram; o aviso do artigo é uma causa possível, ao lado da conversão do formato bruto, da cobertura do vocabulário e do parentesco do spam dele com o do Enron2 e do Enron5.

**O que usamos e o que não usamos.** Usamos os dados (o Enron1 já contado no CSV, as outras caixas e os textos originais). Não usamos o protocolo: aqui a prova é a validação cruzada por grupo, as réguas são MCC, PR-AUC e custo, e o torneio tem 22 concorrentes. Por isso os números do projeto não são comparáveis diretamente com os do artigo. O vocabulário de 3000 colunas também não é o do artigo, que escolhia as palavras de novo a cada treino: ele veio pronto do autor do CSV.

## 4. Modelo de terceiros usado nos experimentos
O TabPFN-3.5 (Prior Labs) foi usado em pesquisa acadêmica, sem fins comerciais: a versão 3.5-Fast localmente, sob a licença *TABPFN-3-5 License v1.0* dos pesos abertos (https://docs.priorlabs.ai/models), e a versão 3.5-Plus pela API da Prior Labs, sob os termos de uso da plataforma aceitos na criação da conta. Para rodar, é preciso ter conta própria, aceitar a licença e criar uma chave em https://platform.priorlabs.ai. A chave vai na variável de ambiente `TABPFN_TOKEN` e **nunca** em arquivo.
