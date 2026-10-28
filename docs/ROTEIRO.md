# Roteiro da história

O mesmo arco está no notebook-história (`scripts/historia.py` → `Projeto3_Classificacao_Spam.ipynb`) e nos dois decks (`apresentacao/`). Cada capítulo tem uma mensagem, a evidência (a chave em `resultados/metricas_finais.json`), a figura e o script de origem.

| # | Capítulo | Mensagem em uma frase | Evidência (chave) | Figura | Script |
|---|---|---|---|---|---|
| 1 | O problema e a base | O CSV é o Enron1, e as palavras foram contadas como substring | `base.linhas_reproduzidas`, `base.colunas_1_letra`, `dominio` | `01_palavras_que_separam` | 01 |
| 2 | Qualidade dos dados | Duplicata não indica spam; cópias ficam do mesmo lado | `base.fracao_duplicatas_email_real` | — | 01 |
| 3 | Olhando sem rótulo | Os grupos naturais são assuntos; "não parecer com meus e-mails" já separa bem | `nao_supervisionado.*` | `02_svd_2d`, `02_kmeans`, `02_anomalia` | 02 |
| 4 | Como medir | A acurácia engana; MCC, PR-AUC e custo 9:1 | `metricas_didaticas.dummy_acuracia` | — | 01 |
| 5 | Torneio de 22 modelos | TF-IDF com log virou o jogo | `torneio`, `achados` | `03_torneio` | 03 |
| 6 | Ninguém vence sozinho | Modelos que erram diferente se completam; o kNN é o reforço | `stacking.pesos_meta`, `stacking.erros_em_comum` | `04_erros_em_comum`, `04_pesos_meta` | 04 |
| 7 | Mais não é melhor | 13 modelos empatam com 7 | `stacking.comparacoes_15_folds` | — | 04 |
| 8 | O algoritmo genético | o ganho do GA não se confirmou em divisões novas; há um piso estatístico | `genetico.*` | `05_ga_x_aleatoria` | 05 |
| 9 | Corrida e TabPFN | Platô nesta caixa; o TabPFN é conservador fora dela | `corrida`, `tabpfn` | `06_custo_dentro_fora` | 06 |
| 10 | A 3ª base e a regra escrita antes | O Stacking 8 vira o stacking para caixas novas; nenhum filtro vence "não filtrar" no SpamAssassin; o stacking aprende a caixa | `terceira_base.decisao`, `terceira_base.controle_dentro_da_base` | `08_terceira_base` | 08 |
| 11 | Contra-estudo | Não vaza; não depende de poucas palavras; λ muda o vencedor fora da caixa | `contra_estudo.*` | `07_curva_aprendizado` | 07 |
| 12 | A grande final e a demo | Quatro filtros, cada um para um uso | `finais`, `simulador_limiar`, `exemplos_demo` | `09_filtros_finais` | 09 |
