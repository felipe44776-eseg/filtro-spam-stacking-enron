# Metodologia

Como cada número do projeto foi produzido. As decisões e os números estão em [REGISTRO_DECISOES.md](REGISTRO_DECISOES.md), e o código, no pacote `spamfilter/`.

## 1. Dados e representação
- **Base:** `dados/emails.csv` tem 5172 e-mails × 3000 colunas. Cada coluna é `texto.count(palavra)` sobre o texto pré-processado da AUEB: contagem de *substring*, sensível a maiúsculas. A reprodução a partir dos textos originais bate em 5172/5172 (`spamfilter.externo`, `tests/`).
- **Conversão de um e-mail qualquer** (`spamfilter.conversao`). As regras foram deduzidas comparando os e-mails brutos com os pré-processados:
  1. do cabeçalho, só o assunto fica;
  2. o que está entre `<` e `>` e entre `&` e `;` é apagado;
  3. o texto vai para minúsculas;
  4. letras, dígitos e cada sinal viram unidades separadas por espaço;
  5. "0" e "1" colados a letras viram "o" e "l".

  Com essas regras, 90,5% dos e-mails brutos da caixa farmer-d reproduzem exatamente uma linha do CSV.
- **TF-IDF com log** (`sublinear_tf=True`) para os modelos lineares, kernel, vizinhos e rede neural. As árvores e o Naive Bayes usam as contagens.

## 2. Protocolo contra vazamento
- **Grupos:** e-mails com o mesmo vetor de contagens formam um grupo (hash da linha). Toda divisão treino/teste usa `StratifiedGroupKFold`, de modo que cópias ficam sempre do mesmo lado.
- **Base de teste trancada:** 20% dos e-mails (1035), sorteados uma vez (`SEED_TESTE = 123`). Nenhuma escolha olha para ela. Foi usada 3 vezes, todas registradas: no experimento 02, no experimento 03 e no relato final do Stacking 8, depois da decisão D10.
- **Notas out-of-fold:** o meta-modelo do stacking só vê notas que cada modelo deu a e-mails fora do seu próprio treino, com uma CV interna por grupo dentro de cada treino.
- **Busca × confirmação:** o algoritmo genético e a corrida escolhem em um conjunto de divisões (sementes 11 e 12; 201 e 202) e confirmam em outro (sementes 201 a 204; 203 e 204).
- **Pré-registro:** a decisão sobre a terceira base (D10) foi escrita antes de rodar qualquer modelo nela.

## 3. Métricas
| Métrica | O que mede | Por quê |
|---|---|---|
| MCC | correlação entre previsão e verdade (−1 a 1; 0 = acaso) | não se ilude com o desbalanceamento: quem nunca marca spam tem acurácia de 71% e MCC 0 |
| PR-AUC | qualidade da ordenação dos e-mails, sem escolher limiar | separa "o modelo ordena bem" de "o limiar está bem posto" |
| E-mails reais na lixeira | fração de e-mails reais marcados como spam | o erro caro |
| Spam que passou | fração de spams que chegaram à caixa de entrada | o erro barato |
| Custo por mil | `1000 × (9 × FP + 1 × FN) / n` | uma nota única que pesa os dois erros |

**Limiar de custo:** escolhido para minimizar o custo nas notas *out-of-fold* do treino (`spamfilter.metricas.limiar_custo`). Nunca é escolhido no teste. A sensibilidade a λ = 1, 3, 9 e 20 está no contra-estudo.

## 4. Estatística
- **t corrigido de Nadeau e Bengio (2003):** os folds compartilham dados de treino, e o t comum subestima a variância.
- **Holm:** corrige o p quando há várias comparações.
- **Bayesiano correlacionado (Benavoli et al., 2017; `baycomp`):** devolve P(A melhor), P(equivalentes) e P(B melhor), com região de equivalência (ROPE) de ±0,01 de MCC.
- **McNemar:** compara dois modelos nos mesmos e-mails de teste.
- **Bootstrap por grupo:** intervalos de confiança sorteando textos distintos (2000 reamostragens).
- **Piso de detecção:** cerca de 2 erros-padrão corrigidos. Com 20 folds, diferenças de MCC abaixo de ~0,0086 não são detectáveis, e mais folds quase não ajudam (~0,0079 com 1000).

## 5. Modelos
- **Torneio:** 22 modelos de 9 famílias, com os mesmos 5 folds. Os 5 melhores vão para 3 × 5 folds.
- **Stacking 7:** um representante por família (Reg. Logística, SVM RBF sobre SVD, kNN cosseno, MLP, Naive Bayes Complement, Extra Trees e LightGBM) e um meta-modelo de Regressão Logística sobre as 7 notas padronizadas.
- **Stacking 8:** o Stacking 7 mais o TabPFN-3.5 como 8º membro.
- **Algoritmo genético** (`spamfilter.genetico`):
  - cromossomo de 8 genes, com 7.077.888 combinações;
  - população de 40 e 30 gerações;
  - torneio de 3, cruzamento uniforme, mutação de 15% e elitismo de 2;
  - controle por busca aleatória com o mesmo orçamento.

  O motor com cache (`spamfilter.motor`) treina cada configuração uma vez por divisão, e avaliar um cromossomo é só o meta-modelo.

## 6. Avaliações
| Avaliação | O que mede |
|---|---|
| 20 folds (sementes 201 a 204), nos 80% de desenvolvimento | desempenho nesta caixa |
| Base de teste trancada (20%) | confirmação nesta caixa, sem escolha |
| Enron2-6 (cerca de 28 mil e-mails de outras 5 caixas; cópias do CSV removidas) | outras caixas, mesma empresa |
| SpamAssassin (6046 mensagens, 2002) | outra empresa e outra época; base nunca usada para escolher |

## 7. Reprodutibilidade
- Versões travadas em `requirements.txt`; Python 3.11.
- Sementes fixas: 42 nos modelos, 123 no teste trancado.
- `pytest` confere, a partir dos caches, os números principais. `pytest -m lento` refaz:
  - o algoritmo genético;
  - o retreino;
  - a base externa;
  - a vetorização.
- Os notebooks são gerados dos scripts e nunca editados à mão. Os números de README, notebook e apresentações vêm de `resultados/metricas_finais.json`.
