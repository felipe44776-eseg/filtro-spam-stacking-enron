# Os modelos, explicados

*Gerado por `scripts/gerar_documentos.py`: o texto explica cada modelo; os números vêm de `resultados/metricas_finais.json`.*

## Antes dos modelos: o que eles recebem

- **A base.** Cada e-mail é uma linha com 3000 contagens: quantas vezes cada palavra do vocabulário aparece no texto, contada como pedaço de texto (- **TF-IDF com log.** Para os modelos lineares, o SVM, o kNN e a rede neural, as contagens viram pesos. O log achata as repetições (contar "free" 10 vezes não é 10 vezes mais evidência), e o IDF dá menos peso às palavras que aparecem em quase todo e-mail. Esse detalhe valeu mais que trocar de modelo: a Regressão Logística foi de MCC 0,923 para 0,961.
- **A nota e o limiar.** Todo modelo devolve uma nota de spam. O **limiar de custo** decide a partir de que nota o e-mail vai para a lixeira: ele é escolhido no treino para minimizar 9 × (e-mails reais na lixeira) + 1 × (spams que passaram).

## O torneio: o ranking dos 22 modelos supervisionados

Todos fizeram a mesma prova: os mesmos 5172 e-mails, a mesma validação cruzada (5 folds, com cópias idênticas sempre do mesmo lado) e a mesma régua (MCC, de −1 a 1; 0 = acaso). O "Dummy" é a referência: nunca marca spam.

| # | Modelo | Família | MCC | PR-AUC | Treino (s) |
|---|---|---|---|---|---|
| 1 | Stacking (meta LR) | combinação | 0,970 | 0,997 | 13,2 |
| 2 | LinearSVC TF-IDF | linear | 0,963 | 0,996 | 0,1 |
| 3 | SVM RBF SVD300 | kernel | 0,962 | 0,996 | 1,6 |
| 4 | SGD (huber) TF-IDF | linear | 0,962 | 0,993 | 0,0 |
| 5 | Reg. Logística TF-IDF | linear | 0,961 | 0,996 | 0,1 |
| 6 | Ridge TF-IDF | linear | 0,960 | 0,996 | 0,0 |
| 7 | MLP (rede neural) | rede neural | 0,960 | 0,994 | 1,9 |
| 8 | Votação (LR+CNB+RF+LGBM) | combinação | 0,959 | 0,996 | 2,6 |
| 9 | NBSVM | linear | 0,957 | 0,992 | 0,0 |
| 10 | HistGradientBoosting | boosting | 0,955 | 0,995 | 42,0 |
| 11 | Extra Trees | bagging | 0,955 | 0,993 | 4,1 |
| 12 | LightGBM | boosting | 0,950 | 0,995 | 2,2 |
| 13 | CatBoost | boosting | 0,949 | 0,994 | 11,3 |
| 14 | kNN TF-IDF cosseno | vizinhos | 0,940 | 0,984 | 0,0 |
| 15 | XGBoost | boosting | 0,936 | 0,993 | 2,2 |
| 16 | Random Forest | bagging | 0,929 | 0,991 | 2,9 |
| 17 | kNN SVD100 | vizinhos | 0,917 | 0,978 | 0,4 |
| 18 | Naive Bayes Complement | probabilística | 0,876 | 0,890 | 0,0 |
| 19 | Naive Bayes Multinomial | probabilística | 0,866 | 0,891 | 0,0 |
| 20 | Árvore de Decisão | árvore | 0,853 | 0,846 | 0,5 |
| 21 | Naive Bayes Bernoulli | probabilística | 0,683 | 0,805 | 0,0 |
| 22 | Dummy (maioria) | referência | 0,000 | 0,290 | 0,0 |

**O que o ranking mostrou:**
- os modelos lineares lideram, depois do TF-IDF com log;
- no topo, os melhores modelos sozinhos ficam muito perto uns dos outros: nenhum domina com folga;
- o 1º lugar já era uma combinação simples, o "Stacking (meta LR)", com 4 membros (Reg. Logística, Naive Bayes, Random Forest e LightGBM). Na rodada de 15 folds, ele ficou à frente do LinearSVC, mas sem significância (p de Holm 0,166).

Daí a ideia seguinte: testar qual combinação funciona melhor e entender por quê. A resposta veio no stacking, abaixo: modelos de famílias diferentes erram em e-mails diferentes.

## Os 7 modelos do stacking (um por família) e a linha de base

### Regressão Logística

**Em uma frase.** Dá um peso a cada palavra, soma os pesos das palavras do e-mail e transforma a soma numa probabilidade de spam.

**Analogia.** Uma balança: as palavras típicas de spam pesam num prato, as típicas de e-mail real no outro.

**Como decide:**
- Cada palavra recebe um peso positivo (puxa para spam) ou negativo (puxa para e-mail real), aprendido no treino.
- A soma passa por uma curva em S (a função logística), que a converte numa nota entre 0 e 1.
- O parâmetro C controla quanto os pesos podem crescer (C maior = menos freio); `cw=balanced` compensa o spam ser minoria.

- **Entrada:** TF-IDF com log.
- **Configuração:** padrão `C=10, cw=balanced`; Especialista `C=1, cw=balanced`.
- **Sozinho, no torneio (5 folds):** MCC 0,961, treino em 0,1 s.
- **No stacking:** quando outro membro erra, ele erra junto em 47% das vezes (6º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 31 e-mails que a maioria errou. Peso no meta-modelo: 0,78 ± 0,43 na média dos 20 folds; 0,94 no Stacking 7 padrão entregue e 0,00 no Especialista.
- **Ponto forte:** rápida e fácil de explicar: dá para ler quais palavras pesam.
- **Ponto fraco:** só traça fronteiras retas: não percebe combinações de palavras.

### LinearSVC (a linha de base)

**Em uma frase.** Também traça uma fronteira reta entre spam e e-mail real, mas escolhe a que deixa a maior folga entre os dois lados.

**Analogia.** A rua mais larga possível entre dois bairros: quanto mais larga, menos chance de errar o lado.

**Como decide:**
- Procura o hiperplano (a "reta" em 3000 dimensões) com a maior margem até os e-mails mais próximos de cada classe.
- A nota é a distância à fronteira, não uma probabilidade.
- Foi o melhor modelo único do torneio e virou a **linha de base**: o filtro simples que o stacking precisa superar.

- **Entrada:** TF-IDF com log.
- **Sozinho, no torneio (5 folds):** MCC 0,963, treino em 0,1 s.
- **Ponto forte:** rápido e muito forte em texto.
- **Ponto fraco:** fronteira reta, como a Regressão Logística.

### SVM com kernel RBF

**Em uma frase.** Resume as 3000 colunas em algumas centenas de "temas" e traça uma fronteira curva, comparando cada e-mail com e-mails de referência.

**Analogia.** Um mapa de bairros com qualquer formato: o e-mail é classificado pelo bairro onde cai.

**Como decide:**
- Primeiro, o SVD (decomposição em valores singulares) comprime as 3000 colunas em 300 ou 500 direções que resumem os assuntos.
- Depois, o kernel RBF mede a semelhança do e-mail com os vetores de suporte, os e-mails de treino mais próximos da fronteira.
- Isso permite fronteiras curvas (a curvatura vem do kernel); o C troca folga por acertos no treino: C maior = fronteira mais colada ao treino.

- **Entrada:** TF-IDF com log + SVD.
- **Configuração:** padrão `k=300, C=10`; Especialista `k=500, C=30`.
- **Sozinho, no torneio (5 folds):** MCC 0,962, treino em 1,6 s.
- **No stacking:** quando outro membro erra, ele erra junto em 47% das vezes (5º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 31 e-mails que a maioria errou. Peso no meta-modelo: 1,09 ± 0,32 na média dos 20 folds; 1,56 no Stacking 7 padrão entregue e 2,75 no Especialista.
- **Ponto forte:** capta padrões que uma reta não capta.
- **Ponto fraco:** mais lento; a nota é uma distância, não uma probabilidade.

### kNN (vizinhos mais próximos)

**Em uma frase.** Procura no treino os e-mails mais parecidos com o novo e copia a decisão da maioria deles.

**Analogia.** Perguntar aos vizinhos: se os 5 e-mails mais parecidos eram spam, este provavelmente também é.

**Como decide:**
- A semelhança é o cosseno entre os vetores TF-IDF: dois e-mails com as mesmas palavras, nas mesmas proporções, apontam para o mesmo lado.
- k é o número de vizinhos consultados (5 no padrão, 3 no Especialista).
- Não aprende pesos: guarda o treino inteiro e compara.

- **Entrada:** TF-IDF com log.
- **Configuração:** padrão `k=5, w=uniform`; Especialista `k=3, w=uniform`.
- **Sozinho, no torneio (5 folds):** MCC 0,940, treino em 0,0 s.
- **No stacking:** quando outro membro erra, ele erra junto em 40% das vezes (2º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 95 e-mails que a maioria errou. Peso no meta-modelo: 1,67 ± 0,21 na média dos 20 folds; 1,69 no Stacking 7 padrão entregue e 1,54 no Especialista.
- **Ponto forte:** reconhece e-mails quase iguais a algum do treino (avisos automáticos, spam em série).
- **Ponto fraco:** decora a caixa: fora dela, e-mails reais de outro assunto não têm vizinhos parecidos e são confundidos com spam.

### Rede neural (MLP)

**Em uma frase.** Uma rede neural pequena: uma camada de neurônios aprende combinações de palavras, e uma camada final junta tudo numa nota.

**Analogia.** Uma equipe de detetives, cada um atento a um padrão diferente, e um chefe que soma as pistas.

**Como decide:**
- Uma camada oculta com 128 neurônios (64 no Especialista); cada neurônio é uma soma ponderada das palavras seguida de um corte não linear.
- O treino ajusta os pesos aos poucos e para quando o desempenho numa parte separada do treino deixa de melhorar (*early stopping*).

- **Entrada:** TF-IDF com log.
- **Configuração:** padrão `h=(128,), a=0.0001`; Especialista `h=(64,), a=0.0001`.
- **Sozinho, no torneio (5 folds):** MCC 0,960, treino em 1,9 s.
- **No stacking:** quando outro membro erra, ele erra junto em 41% das vezes (4º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 61 e-mails que a maioria errou. Peso no meta-modelo: 0,67 ± 0,32 na média dos 20 folds; -0,11 no Stacking 7 padrão entregue e 0,14 no Especialista.
- **Ponto forte:** capta combinações de palavras.
- **Ponto fraco:** é uma caixa-preta e varia com a semente.

### Naive Bayes (Complement)

**Em uma frase.** Conta, no treino, com que frequência cada palavra aparece em cada classe e combina essas chances para dar a nota.

**Analogia.** Um detetive que olha cada pista isoladamente e multiplica as chances, sem pensar em como as pistas se relacionam.

**Como decide:**
- "Ingênuo" porque supõe que as palavras são independentes dentro de cada classe, o que é falso, mas funciona razoavelmente.
- Com duas classes, a versão Complement equivale ao Naive Bayes multinomial sem a fração de spam da base como ponto de partida: o spam, que é minoria, não começa em desvantagem.
- O α suaviza as contagens, para uma palavra rara não zerar a conta.

- **Entrada:** contagens.
- **Configuração:** padrão `a=1, norm=False, usa_tf=False`; Especialista `a=0.1, norm=True, usa_tf=False`.
- **Sozinho, no torneio (5 folds):** MCC 0,876, treino em 0,0 s.
- **No stacking:** quando outro membro erra, ele erra junto em 58% das vezes (7º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 50 e-mails que a maioria errou. Peso no meta-modelo: 0,03 ± 0,13 na média dos 20 folds; 0,09 no Stacking 7 padrão entregue e -0,05 no Especialista.
- **Ponto forte:** o mais rápido de todos.
- **Ponto fraco:** fraco sozinho, erra muito e erra junto com os outros: no stacking, recebe peso quase zero.

### Extra Trees (árvores extremamente aleatórias)

**Em uma frase.** Centenas de árvores de decisão, cada uma com perguntas sorteadas, votam juntas.

**Analogia.** Um júri de centenas de pessoas: cada uma olhou pistas diferentes, e a maioria decide.

**Como decide:**
- Cada árvore faz perguntas do tipo "a palavra X aparece mais de N vezes?" até chegar a uma folha (spam ou e-mail real).
- Nas Extra Trees, cada árvore vê todos os e-mails, mas os pontos de corte e as palavras candidatas são sorteados: as árvores ficam diferentes entre si, e a média delas erra menos que cada uma.
- São 400 árvores no padrão e 800 no Especialista.

- **Entrada:** contagens.
- **Configuração:** padrão `n=400, mf=sqrt, msl=1, cw=balanced`; Especialista `n=800, mf=sqrt, msl=1, cw=balanced`.
- **Sozinho, no torneio (5 folds):** MCC 0,955, treino em 4,1 s.
- **No stacking:** quando outro membro erra, ele erra junto em 39% das vezes (1º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 94 e-mails que a maioria errou. Peso no meta-modelo: 1,36 ± 0,28 na média dos 20 folds; 0,92 no Stacking 7 padrão entregue e 1,00 no Especialista.
- **Ponto forte:** não precisa de TF-IDF e capta regras do tipo "se... e se...".
- **Ponto fraco:** decora detalhes da caixa do treino.

### LightGBM (boosting de árvores)

**Em uma frase.** Também usa árvores, mas em sequência: cada árvore nova corrige um pouco o erro que ainda sobra.

**Analogia.** Um aluno que revisa a prova várias vezes e, a cada revisão, corrige um pouco cada resposta, mais onde errou mais.

**Como decide:**
- São 400 árvores pequenas; cada uma é treinada para reduzir o erro que restou em todos os e-mails, com mais força onde o erro é maior (*gradient boosting*).
- A taxa de aprendizado (0,05) controla o tamanho de cada correção; o número de folhas, o tamanho de cada árvore.

- **Entrada:** contagens.
- **Configuração:** padrão `n=400, eta=0.05, folhas=31`; Especialista `n=400, eta=0.05, folhas=63`.
- **Sozinho, no torneio (5 folds):** MCC 0,950, treino em 2,2 s.
- **No stacking:** quando outro membro erra, ele erra junto em 41% das vezes (3º menor entre os 7; quanto menor, mais informação nova ele traz) e salva 97 e-mails que a maioria errou. Peso no meta-modelo: 0,74 ± 0,13 na média dos 20 folds; 1,24 no Stacking 7 padrão entregue e 1,23 no Especialista.
- **Ponto forte:** muito forte em tabelas.
- **Ponto fraco:** aqui ficou atrás dos lineares: o texto favorece modelos lineares depois do TF-IDF com log.

## O stacking: como os 7 viram um filtro só

**Em uma frase.** Os 7 modelos dão uma nota a cada e-mail, e um oitavo modelo, o **meta-modelo** (uma Regressão Logística), aprende quanto confiar em cada nota.

**Analogia.** Um conselho de 7 especialistas e um presidente que aprendeu, pelo histórico, quanto ouvir cada um.

**Como funciona:**
1. Cada modelo é treinado várias vezes, cada vez sem uma parte do treino, e dá nota justamente à parte que não viu. São as notas *out-of-fold*: ninguém dá nota a um e-mail que usou para aprender.
2. O meta-modelo aprende com essas notas honestas qual combinação acerta mais. Os pesos dele (média dos 20 folds, Stacking 7 padrão): kNN cosseno 1,67, Extra Trees 1,36, SVM RBF 1,09, Reg. Logística 0,78, LightGBM 0,74, MLP 0,67, Naive Bayes 0,03. Os filtros entregues, treinados nos 5172 e-mails, têm pesos próprios: no Especialista, o meta-modelo praticamente desligou Reg. Logística e Naive Bayes (peso perto de zero).
3. Para um e-mail novo, os 7 dão nota, o meta-modelo combina e o limiar de custo decide.

**Por que funciona.** Os modelos erram em e-mails diferentes. Os que menos erram junto com os outros (Extra Trees, kNN cosseno e LightGBM) são também os que mais salvam e-mails quando a maioria erra (LightGBM, kNN cosseno e Extra Trees); o meta-modelo dá os maiores pesos médios a kNN cosseno e Extra Trees. O Naive Bayes é o contrário: erra muito e erra junto (em 58% das vezes em que outro erra), por isso recebe peso quase zero. Na validação cruzada do experimento 02 (15 folds), o Stacking 7 chega a MCC 0,975; com 13 modelos, 0,974: mais modelos parecidos não ajudam.

**Por que estes representantes?** Um de cada família de raciocínio (linear, kernel, vizinhos, rede neural, probabilístico, comitê de árvores aleatórias e *boosting*), não necessariamente o melhor de cada família no ranking: o melhor linear sozinho foi o LinearSVC. O projeto não registrou um critério formal para cada escolha. O que os dados mostram é que ela não muda o resultado: o Stacking 13, que inclui também o LinearSVC, o Ridge, o SGD, o NBSVM, o XGBoost e o Random Forest, empata com o Stacking 7.

**Os nomes do projeto:**
- **Stacking 7 padrão:** os 7 com ajustes padrão. Roda sem chave; é o da demo.
- **Especialista:** os 7 com os ajustes achados pelo algoritmo genético. Tem o menor custo nesta caixa nos 20 folds.
- **Stacking 8:** o Stacking 7 mais o TabPFN como 8º membro. Fora da caixa, fica bem mais barato que o Stacking 7.

## O algoritmo genético: como os ajustes foram escolhidos

**Em uma frase.** Uma busca inspirada na evolução: combinações de ajustes competem, as melhores se cruzam e sofrem pequenas mutações, por 30 gerações.

**Analogia.** Criar cavalos de corrida: cruzar os mais rápidos e, de vez em quando, deixar surgir uma variação.

**Como funciona:**
- Cada indivíduo é uma combinação de 8 genes: o ajuste de cada um dos 7 modelos e o C do meta-modelo. São 7.077.888 combinações possíveis.
- População de 40, torneio de 3, cruzamento uniforme, mutação de 15% por gene e os 2 melhores passam direto. Ao todo, 857 combinações foram avaliadas.
- **Truque que torna a busca viável:** cada ajuste de cada modelo é treinado uma vez por divisão e a nota fica guardada (o *cache* do motor); avaliar um indivíduo é só treinar o meta-modelo.
- **Resultado:** na busca, o MCC subiu +0,0063; em divisões novas, +0,0010, com intervalo de confiança que inclui o zero. A lição é a "maldição do vencedor": escolher o melhor entre centenas de candidatos nas mesmas divisões infla o resultado.

## O TabPFN: o modelo de fundação

**Em uma frase.** Uma rede neural do tipo *transformer*, da Prior Labs, pré-treinada em tabelas sintéticas geradas por computador, que classifica uma tabela nova sem ser treinada nela.

**Analogia.** Um aluno que nunca estudou esta matéria, mas já resolveu milhões de provas parecidas e reconhece o padrão na hora.

**Como funciona:**
- Recebe, de uma vez, a tabela de treino (os e-mails e os rótulos) e os e-mails a classificar, e devolve as notas sem ajustar os próprios pesos: é o aprendizado em contexto (*in-context learning*). Por dentro, faz a média de alguns estimadores sobre versões embaralhadas da tabela.
- Ele tem hiperparâmetros, mas nós não ajustamos nenhum: usamos o padrão e as contagens cruas, sem TF-IDF.
- Versões: TabPFN-3.5-Plus, pela API na nuvem (exige chave), e TabPFN-3.5-Fast, no computador (cerca de 3 h em CPU para 5 divisões). As notas das duas têm correlação de 0,993.

**O que ele fez de melhor:**
- Sozinho, nesta caixa, empata com o melhor modelo único afinado: MCC 0,962 contra 0,963.
- Fora da caixa, é o mais conservador: o que menos manda e-mail real para a lixeira em todas as bases externas.
- Como 8º membro, deixa o stacking bem mais barato fora da caixa (o Stacking 8).

## Os modelos sem rótulo (não supervisionado e semissupervisionado)

- **SVD em 2 dimensões:** comprime as 3000 colunas num mapa, para ver se spam e e-mail real se separam sozinhos.
- **K-means e agrupamento hierárquico:** formam grupos sem saber o rótulo. O ARI compara os grupos com spam × e-mail real (0 = acaso; -0,07 no K-means): os grupos que eles acham são assuntos.
- **NMF (tópicos):** acha assuntos na caixa. Um tópico de linguagem de propaganda ("our, you, your") reúne 96,5% de todo o spam, mas 35% dos e-mails dele são reais: usado como filtro, daria MCC 0,69. Sem rótulo há uma pista forte, mas é preciso o rótulo para chegar ao nível dos supervisionados.
- **Detecção de novidade (semissupervisionada):** treina só com e-mails sabidamente reais e marca como spam o que "não se parece" com eles. A medida que funciona é o erro de reconstrução (PR-AUC 0,80).
