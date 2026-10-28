# Registro de decisões: Projeto 3, classificação de spam

Os números vêm dos notebooks desta pasta. Atualizado em 2026-10-27.

> **Etapa 1 concluída (2026-10-27): escolha da família de modelos.**
> - O Stacking 7 (um modelo por família) supera o LinearSVC e as outras combinações dentro da mesma caixa de e-mail (D5).
> - A otimização genética não muda esse resultado (D6).
> - Em caixas novas, o custo depende da proporção de e-mails reais (D7).
>
> **Etapa 2 concluída (2026-10-27): o melhor Stacking 7 para esta caixa (D8).**
> - O Stacking 7 está num platô: nenhuma configuração se destaca no MCC.
> - O núcleo encontrado pelo algoritmo genético é provavelmente cerca de 12% mais barato que o padrão quando se usa o limiar de custo (P ≈ 0,8), mas isso não está comprovado.

**Termos usados**
- *E-mail real*: e-mail legítimo, que não é spam (no inglês técnico, "ham").
- *Real na lixeira*: e-mail real marcado como spam (falso positivo).
- *Spam que passou*: spam que chegou à caixa de entrada (falso negativo).

## 1. Fatos verificados sobre a base

| Fato | Evidência |
|---|---|
| 5172 e-mails × 3000 contagens de palavras; 29% são spam | `dados/emails.csv` (o `emails (1).csv` do enunciado, renomeado) |
| A base é o **Enron1** (Metsis et al., 2006): 3672 e-mails reais da caixa *farmer-d* (1999-12 a 2002-01) e 1500 spams (2003-12 a 2005-09) | `scripts/externo.py`: o Enron1 baixado da AUEB coincide com o CSV |
| A contagem é por **substring**, sensível a maiúsculas (`texto.count(palavra)`), sobre o texto pré-processado da AUEB | `scripts/reproduz.py`: reproduz 5172 de 5172 linhas |
| Há 541 cópias excedentes (827 linhas em 286 grupos), e **93% delas são e-mails reais** (modelos de texto repetidos) | `scripts/dups_svd.py` |

## 2. Decisões e por quê

### D1. Duplicatas: separar por grupo, não apagar
A hipótese "duplicata indica spam" foi **refutada**: há 7,1% de spam entre as linhas duplicadas, contra 33,2% entre as únicas. As cópias idênticas ficam sempre do mesmo lado na divisão entre treino e teste (`StratifiedGroupKFold` com o hash da linha); se caírem dos dois lados, a nota de teste fica inflada.

### D2. Representação: TF-IDF com escala logarítmica (`sublinear_tf=True`)
A Regressão Logística subiu de MCC 0,922 para 0,961, e o kNN de cerca de 0,75 para 0,94. A mudança inverteu o ranking: os modelos lineares passaram à frente do boosting. A explicação provável é que a contagem por substring infla as frequências, e o log amortece esse efeito.

### D3. Métricas: MCC, PR-AUC e contagem de erros (não acurácia)
- **A acurácia engana nesta base:** um modelo que diz "tudo é e-mail real" já acerta 71%.
- **MCC** mede a decisão final; vale 0 para o chute e 1 para o acerto perfeito.
- **PR-AUC** mede a ordenação, sem depender do limiar. O piso dele é 0,29, a proporção de spam.
- **"Real na lixeira" e "spam que passou"** traduzem os erros em e-mails.

### D4. Torneio de 22 modelos (`01_benchmark/`)
Validação cruzada 5 vezes, separada por grupo. Primeiros colocados:

| Modelo | MCC |
|---|---|
| Stacking 4 | 0,970 |
| LinearSVC | 0,963 |
| SVM RBF | 0,962 |
| SGD | 0,962 |
| Regressão Logística | 0,961 |

Na rodada 2 (15 folds), o Stacking 4 só venceu de forma significativa a Regressão Logística. Contra o LinearSVC deu empate: p de Holm = 0,17 e 60% de chance de equivalência no teste bayesiano.

### D5. **O Stacking 7 é o candidato porque reúne modelos de famílias distintas, que erram em e-mails distintos**
Os sete modelos, um por família:

| Família | Modelo |
|---|---|
| Linear | Regressão Logística |
| Kernel | SVM RBF |
| Vizinhos | kNN |
| Rede neural | MLP |
| Probabilística | Naive Bayes |
| *Bagging* de árvores | Extra Trees |
| *Boosting* | LightGBM |

Evidências (`02_stacking/` e `scripts/stacking_demo.py`):
1. **Os erros pouco se sobrepõem.** Dos 5172 e-mails, 258 são errados por só um entre quatro modelos; são esses que a combinação recupera. A Regressão Logística e o Random Forest erram juntos em só 42 dos 84 erros da Regressão Logística.
2. **O modelo mais útil na combinação não é o melhor sozinho.** O kNN tem MCC 0,931 isolado, mas recebe o maior peso no meta-modelo. Quando a penalidade L1 descarta os modelos inúteis, ele sobrevive em 100% dos folds.
3. **Modelos parecidos se anulam.** No Stacking 13, a Regressão Logística recebe peso negativo, porque o LinearSVC, o Ridge e o SGD dizem a mesma coisa.
4. **Quantidade não é diversidade.** O Stacking 13 (MCC 0,974) não supera o Stacking 7 (0,975) e custa 50% mais tempo de treino. A votação simples (0,969) e o método de Caruana (0,965) ficam atrás, porque não aprendem a pesar cada modelo.

Resultados do Stacking 7 contra o LinearSVC, o melhor modelo único:

| | Stacking 7 | LinearSVC |
|---|---|---|
| MCC na validação cruzada (80% dos dados, 15 folds) | **0,975** | 0,960 |
| Chance de o Stacking 7 ser melhor (bayesiano) | **77%** | 0,1% |
| p de Holm (teste t corrigido) | 0,17 (empate formal) | |
| MCC na base de teste trancada (1035 e-mails) | **0,972** (IC95 0,956–0,986) | 0,963 (IC95 0,944–0,980) |
| Reais na lixeira / spams que passaram (teste trancado) | **8** / 4 | 13 / 3 |
| MCC médio na base externa (Enron2–6, cerca de 28 mil e-mails) | **0,872** | 0,821 |
| % de e-mails reais na lixeira na base externa | **8,8%** | 16,1% |
| PR-AUC médio na base externa | 0,983 | 0,979 |

**Ressalva:** na base externa, o PR-AUC é quase igual nos dois modelos, então parte da vantagem vem do ponto de corte e não de uma separação melhor. O LinearSVC usa `class_weight="balanced"`, que o torna mais propenso a marcar spam. Para uma comparação justa, os dois precisam usar a mesma política de limiar: custo 9 para cada e-mail real perdido contra custo 1 para cada spam recebido.

### D6. Otimização genética: não compensa; o Stacking 7 continua com a configuração padrão (`03_ga/`)

**Como foi feita a busca**
- 62 configurações candidatas, de 6 a 8 por família, mais o C do meta-modelo: 7,1 milhões de combinações possíveis.
- 10 divisões de busca. O algoritmo genético rodou com população de 40 por 30 gerações e fez 857 avaliações únicas.

**Resultados**

| | Padrão | Algoritmo genético | Busca aleatória (mesmo orçamento) |
|---|---|---|---|
| MCC nas divisões da busca | 0,9732 | 0,9795 | 0,9784 |
| MCC em 20 folds com sementes novas | 0,9734 | 0,9744 | |

- **85% do ganho era otimismo de seleção.** Na busca o ganho foi de +0,006; em divisões novas caiu para +0,001. O teste bayesiano dá 99,7% de chance de os dois serem equivalentes (diferença menor que 0,01).
- **O modelo quase não depende dos hiperparâmetros.** Trocar a configuração de uma família muda o MCC em 0,006 no máximo; o kNN é o mais sensível.
- **Na base externa, a configuração do algoritmo genético piora o custo:** 207,6 contra 177,5 por mil e-mails. As opções escolhidas regularizam menos (SVM com C=30, kNN com k=3, meta-modelo com C=10) e se ajustam mais à caixa farmer-d.
- **Mais sementes não resolvem.** As divisões reaproveitam os mesmos 4137 e-mails, então a menor diferença detectável fica perto de 0,009 de MCC com 20 folds e de 0,008 com 1000 folds. O ganho do algoritmo genético (0,001) está 8 vezes abaixo disso.

### D7. Stacking 7 × LinearSVC: o vencedor depende de onde e de como se mede

| | Stacking 7 padrão | LinearSVC otimizado |
|---|---|---|
| MCC, 20 folds novos | **0,973** | 0,963 |
| Custo por mil e-mails, 20 folds (limiar de custo) | **41,5** | 50,0 |
| Teste trancado, limiar de custo: reais na lixeira / spams que passaram | **0 / 17** | 3 / 21 |
| MCC externo, limiar de custo | **0,891** | 0,829 |
| **Custo por mil e-mails externo (média Enron2–6)** | 177,5 | **135,8** |
| Custo Enron2 / Enron3 (cerca de 26% de spam, como o Enron1) | 361 / 235 | **209 / 105** |
| Custo Enron4 / Enron5 / Enron6 (cerca de 73% de spam) | **69 / 136 / 86** | 96 / 174 / 96 |

Notas sobre a tabela:
- MCC dos 20 folds: p de Holm = 0,055; teste bayesiano dá 57% de chance de o stacking ser melhor e 0% de o LinearSVC ser.
- Teste trancado: McNemar com p = 0,09 no padrão e p = 0,001 na versão do algoritmo genético.
- Custo por mil e-mails = 9 × (e-mails reais na lixeira) + 1 × (spams que passaram), sempre com o limiar de custo escolhido só no treino.

**Leitura**
- Na mesma caixa de e-mail, o Stacking 7 vence em tudo.
- Em caixas novas, ele continua melhor no MCC, mas, nas caixas com maioria de e-mails reais (Enron2 e Enron3), marca mais e-mails reais como spam. Com o custo 9, o LinearSVC fica mais barato nelas.
- Isso bate com a validação externa original: o spam generaliza entre caixas, mas o e-mail real não, porque cada pessoa tem o próprio vocabulário. O stacking aprende melhor o vocabulário da caixa farmer-d.

**Decisão**
- O modelo do projeto é o **Stacking 7 padrão**.
- O LinearSVC é a alternativa robusta.
- A limitação em caixas novas entra como conclusão do contra-estudo.

### D8. Melhor Stacking 7 *para esta caixa* (`04_corrida/`)

**Desenho.** Corrida entre os 10 melhores cromossomos da busca e a configuração padrão, em 20 folds com sementes novas:
- a corrida usa as sementes 201 e 202;
- a verificação usa as sementes 203 e 204, que não participaram da escolha.

**Os 10 candidatos convergiram para o mesmo núcleo:**

| Modelo | Configuração |
|---|---|
| Regressão Logística | C=1 |
| SVM RBF | 500 componentes, C=30 |
| kNN | k=3 |
| MLP | uma camada de 64 neurônios |
| Naive Bayes | α=0,1, normalizado |
| Extra Trees | 800 árvores |
| Meta-modelo | C=10 |

Eles diferem quase só na configuração do LightGBM, que não faz diferença.

**Maldição do vencedor.** O primeiro colocado da corrida (#2) não se confirmou na verificação: MCC 0,9713 contra 0,9733 do padrão, com P(equivalentes) = 99%.

**Os 20 folds somados, cada candidato contra o padrão:**

| Métrica | Padrão | 10 candidatos afinados | P(candidato melhor) |
|---|---|---|---|
| MCC (limiar padrão) | 0,9734 | 0,9732 a 0,9751 | 0,48 a 0,71: **empate** |
| MCC (limiar de custo) | 0,9554 | 0,957 a 0,964 | 0,56 a 0,78 |
| Custo por mil e-mails (limiar de custo) | 41,5 | **36,4 a 38,4** (todos os 10 mais baratos) | 0,69 a 0,84 |

**Teste trancado** (avaliado no exp. 03), com limiar de custo:
- GA do exp. 03: 1 e-mail real na lixeira e 10 spams que passaram (custo 19).
- Padrão: 0 e 17 (custo 17).
- Resultado: empate.

**Decisão: o melhor modelo para esta caixa é o Stacking 7 com a configuração do GA do exp. 03 (#5), usando o limiar de custo.** Motivos:
- Ele estava definido antes da corrida.
- Pertence ao núcleo.
- Tem um dos menores custos (36,6; P = 0,82).
- É o único afinado que já passou pelo teste trancado.

Limites da decisão:
- A vantagem sobre o padrão é provável, mas não comprovada.
- Para outras caixas de e-mail continua valendo o padrão: o afinado custa mais fora daqui (D6 e D7).

### D9. TabPFN, modelo de fundação para tabelas: forte sozinho, redundante no stacking (`05_tabpfn/`)

**Versões.** TabPFN-3.5-Plus na nuvem (`tabpfn-client` 0.6.1) e TabPFN-3.5-Fast local (`tabpfn` 9.2), ambos lançados em setembro de 2026. As licenças foram aceitas na conta `felipe-44776-eseg` (uso não comercial, para pesquisa).

**Protocolo.** As mesmas 20 divisões e os mesmos treinos internos dos experimentos 03 e 04, exportados por índice (`splits_tabpfn.npz`). O TabPFN recebe as contagens cruas, sem TF-IDF e sem ajuste.

**Custo.** Cerca de 1,3 milhão de créditos dos 20 milhões da cota mensal: 10 mil por chamada, cerca de 25 s cada.

| Método (20 folds) | MCC | Custo por mil e-mails (limiar de custo) |
|---|---|---|
| TabPFN-3.5-Plus sozinho | 0,9625 | 55,6 |
| LinearSVC otimizado | 0,9627 | 50,0 |
| Stacking 7 padrão | 0,9734 | 41,5 |
| Stacking 8 padrão (+TabPFN) | 0,9727 | 38,2 |
| Stacking 7 especialista | 0,9744 | **36,6** |
| Stacking 8 especialista (+TabPFN) | **0,9747** | 37,9 |

- **Sozinho, ele empata com o melhor modelo único** (P(equivalentes) = 0,87 contra o LinearSVC) e fica abaixo do Stacking 7 (P(stacking melhor) = 0,97). Com o limiar de custo, é o mais caro, porque deixa passar 10% do spam.
- **Como 8ª família, é redundante.** O MCC do Stacking 8 é equivalente ao do Stacking 7 (P(equivalentes) ≥ 0,999 nas duas versões).
  - O meta-modelo dá a ele um peso alto: 1,6 a 2,0, ao lado do kNN e do SVM.
  - Mas esse peso **substitui** o de outros membros (no especialista, a Regressão Logística fica com peso −0,86).
  - Seus erros se sobrepõem mais aos do Extra Trees (Jaccard 0,43), do SVM (0,39) e da Regressão Logística (0,36), e menos aos do kNN (0,19) e do Naive Bayes (0,17).
- **Leitura para a história:** a diversidade que importa já estava nos 7 modelos; nem o modelo de fundação mais recente a amplia. Os dois filtros finais (D8) não mudam.
**Base externa** (Enron2–6, cerca de 28 mil e-mails, treino nos 5172, limiar de custo): **em caixas novas, o TabPFN deixa de ser redundante.**

| Método | MCC | PR-AUC | Custo por mil | Reais na lixeira | Spam que passou |
|---|---|---|---|---|---|
| TabPFN-3.5-Plus sozinho | 0,803 | **0,987** | **126,0** | **0,9%** | 16,6% |
| Stacking 8 especialista (+TabPFN) | 0,873 | 0,985 | **130,2** | 1,9% | 8,5% |
| LinearSVC otimizado | 0,829 | 0,979 | 135,8 | 1,5% | 12,9% |
| Stacking 8 padrão (+TabPFN) | 0,882 | 0,986 | 138,5 | 2,2% | 7,5% |
| Stacking 7 padrão (Generalista em D8) | 0,891 | 0,983 | 177,5 | 3,2% | 5,4% |
| Stacking 7 especialista | **0,894** | 0,983 | 207,6 | 4,0% | 4,4% |

**Leitura**
- Em caixas novas, o TabPFN é o que **melhor ordena** os e-mails (PR-AUC 0,987) e o mais conservador com e-mails reais. Isso é coerente com um modelo pré-treinado em tabelas sintéticas, que não decora o vocabulário da caixa farmer-d.
- No time, ele **baixa o custo externo do stacking**: 22% no padrão (177,5 → 138,5) e 37% no especialista (207,6 → 130,2). O MCC externo cai um pouco: 0,891 → 0,882 no padrão e 0,894 → 0,873 no especialista.
- **Dentro desta caixa**, o resultado continua empatado (ver acima).

**Cuidado metodológico.** Este achado veio de olhar a base externa. Se ele for usado para **trocar o Generalista**, a base externa deixa de ser um teste imparcial. Para adotar o Stacking 8 como Generalista de forma limpa, é preciso uma terceira base nunca usada, por exemplo o corpus público do SpamAssassin, vetorizado com o mesmo vocabulário. Até lá, o achado entra como **resultado do contra-estudo**, e os filtros de D8 continuam valendo.

**Observação para a demo.** O Stacking 8 depende do TabPFN para classificar cada e-mail: a API na nuvem (internet + chave) ou o modelo local em CPU, que é lento.

**Réplica local × nuvem** (executada de 27 para 28/10/2026, das 23h14 às 02h25; `tabpfn_local_x_nuvem.csv`, seção 11 do notebook). O TabPFN-3.5-Fast rodou em CPU, na `.venv`, sem chave e sem internet, nas mesmas 5 divisões da semente 201. Cada divisão levou cerca de 38 min (5 ajustes internos mais 1 final); na previsão final, foram cerca de 0,55 s por e-mail, em lote, com 3310 e-mails de contexto.

| Método (5 divisões, semente 201) | MCC | PR-AUC | Custo por mil |
|---|---|---|---|
| TabPFN-3.5-Plus (nuvem) | 0,9613 | 0,9963 | 64,1 |
| TabPFN-3.5-Fast (local) | 0,9565 | 0,9961 | 55,4 |
| Stacking 7 padrão | 0,9720 | 0,9969 | 46,9 |
| Stacking 8 padrão + 3.5-Plus (nuvem) | 0,9714 | 0,9975 | 47,1 |
| Stacking 8 padrão + 3.5-Fast (local) | **0,9725** | 0,9969 | **42,8** |

- **As duas versões dão praticamente as mesmas notas:** a correlação média nos e-mails de teste é **0,993**.
- **Sozinha,** a versão local fica 0,005 abaixo no MCC. Isso está dentro do ruído de 5 divisões (a menor diferença detectável é cerca de 0,011).
- **No stacking, a versão não muda nada:** as diferenças são menores que 0,001 no MCC. A conclusão de D9 (redundante dentro desta caixa) vale para as duas.
- **Para a banca:** o experimento do TabPFN pode ser reproduzido **sem conta, sem chave e sem internet**, só com a versão local. O preço é o tempo: cerca de 3 h em CPU para 5 divisões, contra cerca de 25 s por chamada na nuvem. Os resultados da nuvem continuam sendo a referência (20 folds e base externa).
- **Para a demo:** classificar um único e-mail com o TabPFN local leva alguns segundos de CPU, o que é viável. Mesmo assim, a demo usa os filtros de D8, que não dependem do TabPFN.

### D10. Terceira base intocada: o Stacking 8 vira o Generalista? (pré-registro)

**Registrado em 2026-10-28, antes de rodar qualquer modelo nesta base.**

**Por que esta base.** Em D9, o Stacking 8 (com o TabPFN) baixou o custo nas caixas Enron2–6. Esse achado, porém, veio de olhar a base externa. Para decidir de forma limpa, é preciso uma base que nenhuma escolha usou.

**Base.**
- O que é: o *SpamAssassin public corpus*, conjunto 20030228 (easy_ham, easy_ham_2, hard_ham, spam e spam_2), com 6047 mensagens e cerca de 31% de spam.
- Diferenças em relação ao Enron: outras pessoas, outra época (2002) e listas de discussão técnicas.
- Conversão: os e-mails brutos passam ao formato da AUEB por `spamfilter.conversao`, que reproduz exatamente 90,5% dos e-mails brutos da caixa farmer-d. Depois são contados com o vocabulário do CSV.
- Licença: o copyright do texto continua com os remetentes. Por isso o repositório público guarda só as contagens e o link.

**Protocolo.**
- Treino: os 5172 e-mails, sem nenhum ajuste.
- Membros do stacking: as mesmas notas *out-of-fold* da base externa (D9); o TabPFN é o 3.5-Plus da nuvem.
- Candidatos:
  - Stacking 7 padrão (o Generalista atual);
  - Stacking 8 padrão;
  - Stacking 7 especialista;
  - Stacking 8 especialista;
  - TabPFN sozinho;
  - LinearSVC padrão.

**Métrica primária.** O custo por mil e-mails no limiar de custo (λ = 9), o mesmo critério de D8.

**Regra de decisão.** O Stacking 8 padrão substitui o Stacking 7 padrão como Generalista se as duas condições valerem:
- ele reduz o custo com P(redução) ≥ 0,90, num bootstrap por grupo com 2000 reamostragens;
- ele não piora o MCC no limiar de custo em mais de 0,01.

Caso contrário, o Generalista fica como está. O Especialista não é decidido por esta base: ele é o filtro da caixa farmer-d.

**Consequência prática, registrada antes.** O Stacking 8 precisa do TabPFN para classificar cada e-mail: na nuvem, com chave e internet, ou no CPU, onde é lento. Se ele vencer:
- ele vira o Generalista recomendado para caixas novas;
- a demo continua com o Stacking 7, que roda sem chave e sem internet.

**Resultado** (2026-10-28; `scripts/08_terceira_base.py`, `resultados/08_terceira_base.json`). Base: 6046 mensagens (4150 e-mails reais e 1896 spams); 1 cópia do CSV foi removida.

| Método (treino no Enron1, limiar de custo) | MCC | PR-AUC | Reais na lixeira | Spam que passou | Custo por mil |
|---|---|---|---|---|---|
| TabPFN-3.5-Plus sozinho | 0,357 | 0,635 | 4,6% | 69,9% | **503,5** |
| LinearSVC padrão | **0,553** | **0,770** | 6,3% | 44,6% | 528,4 |
| Stacking 8 especialista | 0,387 | 0,639 | 13,6% | 50,3% | 997,4 |
| Stacking 8 padrão | 0,437 | 0,668 | 16,0% | 41,0% | 1119,9 |
| Stacking 7 padrão (Generalista até aqui) | 0,415 | 0,684 | 30,7% | 24,8% | 1977,2 |
| Stacking 7 especialista | 0,350 | 0,652 | 37,9% | 24,4% | 2416,5 |

**Aplicação da regra.**
- Custo: o Stacking 8 padrão custa 857 a menos por mil (IC95 de −929 a −789; P(redução) = 1,00).
- MCC: +0,022.
- McNemar: p < 10⁻²³.
- **Decisão: o Stacking 8 padrão passa a ser o Generalista.** O Especialista não muda. A demo segue com o Stacking 7 (sem chave).

**O que a base mostrou além da regra.** Estes pontos são observação, não decisão: não eram parte do pré-registro e não mudam os filtros.
1. **Todos os filtros treinados no Enron1 vão mal numa caixa muito diferente.** O MCC fica entre 0,27 e 0,55.
2. **A conversão está correta.** No controle, o mesmo pipeline treinado e testado **dentro** do SpamAssassin chega a MCC 0,968 e PR-AUC 0,998.
3. **A queda acompanha a cobertura do vocabulário.** O vocabulário conhece 81,6% das palavras dos e-mails do Enron1, 77,2% no Enron3 e 65,4% no SpamAssassin.
4. **O filtro aprende a caixa, não "o spam".** O stacking, sobretudo o kNN e as árvores, decora o que é normal *naquela* caixa. Numa caixa de listas técnicas, ele manda 31% dos e-mails reais para a lixeira.
5. **Os filtros simples e conservadores custam menos fora da caixa de origem.** O TabPFN sozinho e o LinearSVC repetem o padrão do Enron2–6: TabPFN com 126,0 e LinearSVC com 135,8, contra 177,5 do Stacking 7 padrão. São duas bases externas independentes.
6. **Recomendação para uso real:** retreinar com e-mails da caixa de destino.

## 3. Hipóteses descartadas
- A duplicata indica spam (refutada em D1).
- O boosting seria a melhor família. Isso era efeito da representação sem log (D2).
- A votação simples melhoraria o resultado. No experimento 01, ela fez 0,959, menos que a Regressão Logística sozinha (0,961): dá o mesmo peso a um modelo fraco.
- Usar mais modelos no stacking (D5).
- A otimização genética melhoraria o Stacking 7 de forma real. O ganho em divisões novas foi de +0,001, e o modelo ficou pior na base externa (D6).
- Ajustar o C do LinearSVC mudaria o resultado. A chance de equivalência ao padrão é de 98,7%.
- O primeiro colocado da corrida seria o melhor modelo. Ele não se confirmou na verificação (D8).
- O TabPFN-3.5 melhoraria o stacking como 8ª família. Ele é redundante: P(equivalentes) ≥ 0,999 (D9).

## 4. Base de teste trancada
- **Composição:** 20% dos dados, estratificados por classe e separados por grupo; `SEED_TESTE=123`; 1035 e-mails (735 reais e 300 spams).
- **Índices:** `indices_teste.npz`.
- **Uso:** foi avaliada duas vezes: no experimento 02 (Stacking 4, Stacking 7, LinearSVC e Regressão Logística) e no experimento 03 (Stacking 7 padrão, Stacking 7 do algoritmo genético e LinearSVC padrão e otimizado). Cada nova avaliação nela deve ser registrada aqui, e nenhuma escolha pode ser feita olhando para ela.
- **3º uso (2026-10-28, `scripts/09_modelos_finais.py`).** O Stacking 8 padrão foi avaliado nela **depois** da decisão D10, só para completar a tabela dos filtros finais. Nada foi escolhido com base nesse número. O TabPFN foi treinado nos 80% de desenvolvimento: 5 treinos internos para as notas *out-of-fold* e 1 final.
  - Resultado do Stacking 8 padrão, no limiar de custo: 2 e-mails reais na lixeira e 13 spams que passaram (custo 31).
  - Os números registrados antes se repetem:
    - Especialista: 1 e 10 (custo 19);
    - Stacking 7 padrão: 0 e 17 (custo 17);
    - LinearSVC padrão: 3 e 23 (custo 50).

## 5. Experimentos

As pastas ficam em `experimentos/`; os scripts citados, em `experimentos/scripts/`. A versão organizada de cada etapa, usando o pacote `spamfilter`, está em `scripts/01` a `09` e nos notebooks gerados em `notebooks/`.

| Pasta | O que tem | Como reproduzir |
|---|---|---|
| `01_benchmark/` | Torneio de 22 modelos e estatística dos 5 melhores | `scripts/build_bench.py`, depois `scripts/run_nb.py` (cerca de 11 min) |
| `02_stacking/` | 13 modelos de base, 6 combinações, base de teste trancada e validação externa | `scripts/build_stack.py` (cerca de 22 min; com o cache `cache_stacking/`, cerca de 1,5 min) |
| `03_ga/` | Otimização genética do Stacking 7, confirmação com sementes novas, limiar de custo, teste trancado e base externa | `scripts/build_ga.py` (cerca de 61 min; com o cache `cache_ga/`, poucos minutos). Antes, um teste rápido com `GA_SMOKE=1` |
| `04_corrida/` | Corrida entre os 10 melhores da busca + o padrão; verificação do vencedor; somente esta caixa | `scripts/build_corrida.py` (cerca de 17 min; reaproveita `03_ga/cache_ga/`) e `scripts/corrida_20folds.py` |
| `05_tabpfn/` | TabPFN-3.5 sozinho e como 8ª família (nuvem: 20 folds e base externa; local: semente 201) | `scripts/exportar_splits_tabpfn.py`, depois `scripts/tabpfn_notas.py --modo nuvem` no ambiente `.venv-tabpfn-nuvem` e `--modo local` na `.venv`; por fim, `scripts/build_tabpfn.py` (análise em 35 s). Exige `TABPFN_TOKEN` e as licenças aceitas |
| `../dados/` | Enron1–6 vetorizados com o mesmo vocabulário e o mesmo método de contagem (`enron_vetores.npz`) e os originais (`brutos_aueb/`). Movidos de `experimentos/dados_externos/` em 2026-10-28; lá ficou só o diagnóstico inicial (`enron_vetores_resultados.csv`) | `spamfilter.externo.gerar_vetores()` ou `scripts/externo.py` (origem: `http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enronN.tar.gz`) |

Os notebooks 01 e 02 foram executados na pasta temporária da sessão. Para reexecutá-los aqui, gere de novo o notebook a partir de `scripts/build_*.py` (os caminhos já estão ajustados para esta pasta).
