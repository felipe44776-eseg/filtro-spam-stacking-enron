# Glossário

Cada termo, sigla e métrica do projeto explicados para quem não é da área. É o mesmo texto das notas que aparecem ao passar o mouse nos slides. Os números vêm de `resultados/metricas_finais.json`.

[A base e as fontes](#a-base-e-as-fontes) · [Preparar o texto](#preparar-o-texto) · [Sem rótulo](#sem-rótulo) · [Métricas](#métricas) · [Validação e estatística](#validação-e-estatística) · [Modelos](#modelos) · [Combinação e busca](#combinação-e-busca) · [Termos do projeto](#termos-do-projeto) · [Arquivos e ferramentas](#arquivos-e-ferramentas)

## A base e as fontes

**armadilha de spam** (spam trap, honeypot). Endereços de e-mail publicados na internet só para atrair spam: ninguém de verdade escreve para eles. As coleções SpamAssassin, HoneyPot e BG (de Bruce Guenter) foram juntadas assim e repetem muito o mesmo spam; por isso os autores do Enron-Spam tiraram as cópias delas.

**AUEB** (Universidade de Economia e Negócios de Atenas). Universidade grega de um dos autores do Enron-Spam. O site dela guarda os arquivos originais, em texto bruto e pré-processado. Usamos o pré-processado para refazer a contagem do CSV.

**corpus** (coleção de textos). Uma coleção de textos montada para pesquisa. Aqui, coleções de e-mails com a resposta certa (spam ou e-mail real) de cada um.

**CSV** (valores separados por vírgula). Tabela em texto puro. A base é um CSV com uma linha por e-mail e uma coluna por pedaço de texto contado.

**Enron** (empresa americana de energia). Empresa de energia de Houston, nos Estados Unidos, que faliu em dezembro de 2001 depois de um escândalo de fraude contábil. A agência reguladora de energia dos EUA (FERC), que investigava a manipulação do mercado de energia, publicou os e-mails da empresa em 2003. A versão organizada para pesquisa (Klimt e Yang, 2004) tem cerca de meio milhão de e-mails de uns 150 funcionários, a maioria da diretoria: a maior coleção pública de e-mails reais de uma empresa.

**Enron-Spam** (o corpus de Metsis et al., 2006). Seis caixas montadas por pesquisadores da Grécia (Enron1 a Enron6). Cada uma junta os e-mails reais de um funcionário da Enron com spams de uma coleção de spam; uma delas, a GP, é a do próprio Paliouras, um dos autores. Três caixas têm cerca de três e-mails reais para cada spam; nas outras três, é o contrário.

**Enron1** (a base do projeto). A primeira caixa do Enron-Spam: 3672 e-mails reais da caixa farmer-d e 1500 spams da coleção GP, sorteados pelos autores de uma coleção maior. É exatamente o CSV do projeto: refizemos a contagem a partir dos textos originais e ela bateu em 5172 de 5172 linhas.

**Enron2–6** (as outras cinco caixas). As outras cinco caixas do Enron-Spam, com e-mails reais de outros cinco funcionários da Enron e spams de três coleções. Serviram de prova fora da caixa do treino: o filtro nunca viu esses e-mails.

**farmer-d** (a caixa de e-mail usada no projeto). Nome da pasta de e-mails de Daren Farmer, funcionário da Enron. Os 3672 e-mails reais da base são da caixa dele (os autores tiraram os que ele mesmo enviou), de dezembro de 1999 a janeiro de 2002, e tratam sobretudo de entregas de gás natural. Por isso “daren”, “hpl”, “mmbtu” e “nomination” estão entre as palavras que mais indicam e-mail real.

**FERC** (Comissão Federal Reguladora de Energia dos EUA). A agência do governo americano que regula o setor de energia. Ao investigar a manipulação do mercado de energia do oeste dos EUA, publicou em 2003 os e-mails da Enron.

**GP** (coleção de spam de Georgios Paliouras). Os spams que Paliouras, um dos autores do Enron-Spam, recebeu de dezembro de 2003 a setembro de 2005. É a única coleção que não veio de armadilhas e a única em que os autores mantiveram as cópias, por fazerem parte do fluxo normal de spam. É o spam do Enron1 e do Enron4.

**Kaggle** (site de dados e competições). Onde o CSV do projeto foi publicado. A página o descreve como e-mails escolhidos ao acaso, com as palavras mais comuns como colunas. Descobrimos que são os 5172 e-mails do Enron1 inteiro.

**Metsis et al. (2006)** (o artigo que criou a base). Artigo de Metsis, Androutsopoulos e Paliouras na conferência CEAS 2006. Criou e publicou o Enron-Spam e comparou cinco versões do Naive Bayes, retreinando o filtro a cada cem e-mails, como faria um usuário novo. As melhores foram o Flexible Bayes e o Naive Bayes multinomial que olha só se a palavra aparece ou não; os autores preferiram o segundo. Usamos os dados do artigo, não a análise dele: o resumo está no apêndice.

**SpamAssassin** (corpus público de e-mails). Coleção pública de e-mails de listas de discussão e de pessoas que os doaram, com spam, da mesma época da Enron. Foi a terceira base, a mais diferente: decidiu a regra pré-registrada D10. O próprio artigo de Metsis avisa que os e-mails reais dela vêm de muitas pessoas, ao contrário de uma caixa pessoal.

**vocabulário** (as colunas da base). Os 3000 pedaços de texto escolhidos pelo autor do CSV como colunas: palavras, siglas e até as letras soltas. Uma palavra fora do vocabulário não ganha coluna própria, mas as letras dela entram nas colunas de letras soltas. Não há coluna de algarismo nem de pontuação.

## Preparar o texto

**cosseno** (semelhança entre dois e-mails). Mede quanto dois e-mails usam as mesmas palavras nas mesmas proporções, sem ligar para o tamanho: um quer dizer as mesmas palavras nas mesmas proporções; zero, nenhuma palavra em comum. É a régua de semelhança do kNN e das quase-cópias.

**PCA** (análise de componentes principais). Encontra as direções em que os dados mais variam. Em tabelas de palavras, quase todas zeradas, o SVD faz o mesmo papel sem precisar centralizar os dados.

**pré-processado** (o texto já limpo pela AUEB). Os autores do Enron-Spam deixaram só o assunto e o corpo de cada e-mail: tiraram os cabeçalhos e as marcas de HTML, passaram tudo para minúsculas e separaram a pontuação com espaços. A contagem do CSV foi feita sobre esse texto.

**substring** (pedaço de texto, achado até dentro de outras palavras). Trecho contínuo dentro de um texto maior. A base conta substrings: cada coluna diz quantas vezes o pedaço aparece no texto, até dentro de outras palavras. Por isso “ect” aparece em todos os e-mails (está dentro de “Subject”) e a coluna “e” conta toda letra “e”.

**SVD/LSA** (decomposição em valores singulares / análise semântica latente). Resume as milhares de colunas em poucas direções, cada uma juntando palavras que costumam aparecer juntas (como “temas”). Com duas direções, dá para desenhar a base num mapa; com centenas, alimenta o SVM.

**TF-IDF** (frequência do termo × inverso da frequência nos documentos). Um jeito de dar peso a cada palavra: o peso sobe quando a palavra aparece muito neste e-mail e desce quando aparece em quase todos os e-mails, porque aí ela não ajuda a distinguir nada. Usado nos modelos lineares, no SVM, no kNN e na rede neural.

**TF-IDF com log** (a contagem achatada). Antes do TF-IDF, a contagem passa por um logaritmo: contar “free” dez vezes pesa só cerca do dobro de contar duas, e não cinco vezes mais. Foi o detalhe que mais melhorou a Regressão Logística e o kNN no torneio.

**token** (a palavra recortada do texto). O pedaço em que um programa recorta o texto antes de contar: em geral, uma palavra ou um sinal de pontuação, sem sobreposição. Nesta base a contagem não é de tokens, e sim de substrings: o pedaço é achado até dentro de outras palavras (“ect” dentro de “Subject”).

**vetor** (a lista de números de um e-mail). Os 3000 números que representam um e-mail para o computador: quantas vezes cada pedaço do vocabulário aparece nele. Quase todos são zero: 94,4% das células da base.

**χ²** (qui-quadrado). Teste que mede se uma palavra aparece mais numa classe do que o acaso explicaria. Serve para escolher palavras.

## Sem rótulo

**agrupamento hierárquico** (grupos dentro de grupos). Junta os dois e-mails mais parecidos, depois os grupos mais parecidos, e assim por diante, como uma árvore genealógica. Também não separou spam de e-mail real.

**ARI** (índice de Rand ajustado). Mede se os grupos encontrados sem as respostas coincidem com as classes. Zero é o acaso e um é coincidência total; fica negativo quando a coincidência é pior que o acaso (o K-means deu −0,070).

**erro de reconstrução** (quanto o e-mail foge do conhecido). Resume o e-mail com os temas (SVD) dos e-mails reais e mede quanto se perde. Se perde muito, o e-mail não se parece com os reais. Foi a melhor pista de spam sem rótulos de spam.

**Isolation Forest** (floresta de isolamento). Procura o que é fácil de isolar com poucos cortes ao acaso: o raro e diferente. Treinada só com e-mails reais, devia achar o spam como “estranho”; não achou.

**K-means** (agrupamento por k médias). Separa os e-mails em k grupos sem olhar as respostas: cada e-mail vai para o grupo cujo centro é o mais parecido com ele, e os centros se ajustam até parar de mudar. Aqui, os grupos saíram por assunto, não por spam.

**NMF** (fatoração em matrizes não negativas). Quebra a base em tópicos: cada tópico é um conjunto de palavras que costumam aparecer juntas, e cada e-mail é uma mistura de tópicos. Aqui, um tópico de propaganda reuniu quase todo o spam.

**não supervisionado** (aprendizado sem as respostas). O computador procura padrões sem saber o que é spam: mapas, grupos e tópicos. Aqui, os grupos saíram por assunto, e um tópico de propaganda deu uma pista forte de spam, mas não um filtro bom.

**One-Class SVM** (SVM de uma classe). Aprende a fronteira em volta dos e-mails reais conhecidos e marca como suspeito o que cai fora dela.

**rótulo** (a resposta certa). A resposta certa de cada e-mail: spam ou e-mail real. O aprendizado supervisionado usa os rótulos para aprender; o não supervisionado não os usa.

**semissupervisionado** (aprendizado com parte das respostas). Usa só parte das respostas: aqui, treinar só com e-mails reais conhecidos e marcar como suspeito o que foge deles. Não é “sem rótulo”.

**supervisionado** (aprendizado com as respostas). O modelo aprende com exemplos já respondidos (este e-mail é spam, aquele é real) e depois decide sozinho em e-mails novos. Foi o caminho do torneio.

## Métricas

**acurácia** (fração de acertos). A porcentagem de e-mails classificados certo. Engana quando uma classe é bem maior: quem nunca marca spam acerta 71,0% nesta base. Por isso não a usamos para decidir.

**AUC** (área sob a curva). Nome genérico da área sob uma curva de desempenho (ROC ou precisão e revocação). Resume o modelo em todos os limiares de uma vez.

**custo por mil** (o preço dos erros). Quanto os erros custam a cada mil e-mails: cada e-mail real na lixeira vale λ, cada spam que passou vale um. Menor é melhor. Fórmula: mil × (λ × reais na lixeira + spams que passaram) ÷ número de e-mails.

**e-mail real na lixeira** (falso positivo (FP)). Um e-mail legítimo que o filtro mandou para a lixeira. É o erro caro: pode ser um contrato ou um aviso importante. Pesa λ vezes mais que um spam que passou.

**limiar** (ponto de corte). A partir de que nota (chance de spam) o e-mail vai para a lixeira. O limiar de custo é o que dá o menor custo, escolhido só com os dados de treino; o limiar padrão é 0,5.

**MCC** (coeficiente de correlação de Matthews). Uma nota de menos um a um que mede quanto o filtro acerta além do acaso, usando os quatro resultados de uma vez (VP, VN, FP, FN). Zero é chute; um é perfeito. Ao contrário da acurácia, não se ilude com a classe maior. Depende do limiar: o de custo ou o padrão, 0,5.

**não filtrar** (a referência mais simples). Deixar tudo passar: nenhum e-mail real se perde e todo spam chega. O custo por mil dela é mil vezes a fração de spam. Todo filtro precisa custar menos que isso para valer a pena.

**PR-AUC** (área sob a curva de precisão e revocação). Ordene os e-mails do mais suspeito ao menos suspeito: a PR-AUC mede se os spams ficam no topo da lista, sem escolher limiar. Um é perfeito; quem ordena ao acaso tira a fração de spam da base (0,290).

**precisão e revocação** (precision e recall). Precisão: dos e-mails mandados para a lixeira, quantos eram spam. Revocação: dos spams, quantos foram para a lixeira. O F1 é a média harmônica das duas: só é alto se as duas forem altas.

**ROC-AUC** (área sob a curva ROC). A curva ROC mostra, para cada limiar, quanto do spam é pego contra quanto dos e-mails reais vai para a lixeira. A área sob ela é a chance de um spam sorteado receber nota maior que um e-mail real sorteado. Meio é o chute. Com classes desiguais, a PR-AUC é mais exigente.

**spam que passou** (falso negativo (FN)). Um spam que chegou à caixa de entrada. Incomoda, mas custa pouco: pesa um, contra λ do e-mail real na lixeira.

**TCR** (total cost ratio, razão de custo total). Custo de não filtrar dividido pelo custo do filtro (Androutsopoulos et al., 2000). Acima de um, filtrar compensa; abaixo de um, deixar tudo passar sai mais barato. Quando junta várias caixas ou folds, é o custo médio de não filtrar dividido pelo custo médio do filtro.

**VP/VN/FP/FN** (verdadeiro positivo, verdadeiro negativo, falso positivo, falso negativo). Os quatro resultados possíveis. Positivo é “o filtro disse spam”. VP: spam pego. VN: e-mail real que passou. FP: e-mail real na lixeira (o erro caro). FN: spam que passou.

**λ** (lambda, o peso do erro caro). Quantos spams recebidos equivalem a um e-mail real perdido. Usamos λ = 9, um dos cenários de Androutsopoulos et al. (2000), e testamos outros valores.

## Validação e estatística

**ablação** (tirar uma parte e medir de novo). Retira uma parte da entrada (aqui, as palavras da Enron, como nomes e siglas) e mede de novo, para ver quanto o filtro dependia dela.

**bootstrap** (reamostragem com reposição). Sorteia de novo, com reposição, os e-mails (ou os grupos de cópias) para ver quanto um resultado varia. Deu o IC95 e o P(redução) da regra D10.

**curva de aprendizado** (nota × tamanho do treino). Mostra como a nota muda quando o modelo treina com mais e-mails. Se ainda sobe no fim, mais dados ajudariam.

**CV** (validação cruzada). O jeito de medir um modelo sem gastar e-mails novos: divide os dados em folds, treina em todos menos um, testa no que ficou de fora, e gira. Cada e-mail é testado uma vez por sorteio, por um modelo que não o viu.

**cópia** (e-mails com a mesma linha de contagens). Dois e-mails com as mesmas 3000 contagens. Como até as letras soltas são colunas, os textos precisam ter as mesmas letras; só muda o que o vocabulário não vê: algarismos, pontuação, espaços e a ordem dos trechos. “Cópia excedente” é cada repetição além da primeira. As cópias ficam sempre do mesmo lado (treino ou teste), para o modelo não “lembrar” a resposta.

**desvio padrão** (quanto os valores variam). Mede quanto os valores se espalham em torno da média: pequeno, eles ficam perto dela; grande, variam muito de um fold para outro.

**divisões novas** (sorteios que a busca não viu). Folds sorteados de novo, depois da busca, para conferir se um ganho se repete. Se o ganho some, ele era sorte da escolha.

**fold** (uma das partes da validação cruzada). Os e-mails são divididos em partes (folds). Com cinco folds, o filtro treina em quatro partes e faz a prova na quinta; repete cinco vezes, uma para cada parte, e a nota é a média. Os 20 folds de confirmação do projeto são quatro sorteios diferentes dessa divisão em cinco: 20 provas.

**hash** (impressão digital de uma linha). Um código curto calculado a partir da linha de contagens: linhas iguais dão o mesmo hash. É assim que as cópias são encontradas e mantidas juntas.

**IC95** (intervalo de confiança de noventa e cinco por cento). A faixa de valores plausíveis para uma diferença. Se a faixa inclui o zero, a diferença pode não existir.

**maldição do vencedor** (o otimismo de quem escolhe o melhor). Quando se escolhe o melhor entre centenas de candidatos, a nota do escolhido sai inflada por sorte; em dados novos, ela cai. Foi o que aconteceu com o algoritmo genético.

**McNemar** (teste de McNemar). Compara dois filtros nos mesmos e-mails, contando onde só um deles acerta.

**OOF** (out-of-fold, fora da dobra). Nota que um modelo dá a um e-mail que ele não viu no treino. O presidente do stacking só aprende com notas assim, para não premiar quem decorou.

**p de Holm** (valor-p corrigido por Holm). Quando se fazem várias comparações, alguma sai “significativa” por sorte. A correção de Holm aumenta os valores-p para compensar.

**P(equivalentes)** (chances do teste bayesiano e do bootstrap). Chances calculadas a partir dos folds ou de reamostragens. P(equivalentes): a chance de a diferença entre dois modelos ser pequena demais para importar (dentro da ROPE). P(mais barato) e P(redução): a chance de um filtro custar menos que o outro.

**piso de detecção** (a menor diferença visível). A menor diferença de MCC que 20 folds conseguem enxergar com 80% de poder. Diferenças menores podem existir, mas os testes não as distinguem do acaso.

**poder** (poder estatístico). A chance de um teste enxergar uma diferença que existe de verdade.

**quase-cópia** (e-mail quase idêntico a outro). E-mail com cosseno de pelo menos 0,95 com algum e-mail do treino, sem ser cópia (o mesmo aviso com uma ou outra palavra diferente). O agrupamento só junta cópias, então as quase-cópias podem cair dos dois lados e inflar um pouco a nota; medimos quanto.

**ROPE** (região de equivalência prática). Faixa de diferença pequena demais para importar na prática. Aqui, um centésimo de MCC ou três de custo por mil.

**StratifiedGroupKFold** (validação cruzada estratificada por grupo). A divisão em folds usada aqui. “Estratificada”: cada parte fica com quase a mesma fração de spam. “Por grupo”: todas as cópias de um e-mail caem na mesma parte.

**t corrigido de Nadeau-Bengio** (teste t para validação cruzada). Teste que compara dois modelos nos mesmos folds. Como os treinos dos folds se repetem em boa parte, ele aumenta a incerteza (Nadeau e Bengio, 2003); sem a correção, diferenças pequenas pareceriam reais.

**teste bayesiano correlacionado** (comparação bayesiana nos mesmos folds). Compara dois modelos nos mesmos folds e dá três chances: de A ser melhor, de B ser melhor e de serem equivalentes, isto é, diferença dentro da ROPE (Benavoli et al., 2017).

**teste de permutação** (embaralhar as respostas). Embaralha as respostas e refaz a avaliação muitas vezes. Se a nota verdadeira fica muito acima das embaralhadas, ela não vem do acaso. Não prova que não houve vazamento.

**teste trancado** (o envelope lacrado). Os 1035 e-mails separados no início, por grupo de cópias, e guardados para a prova final. Foram abertos 4 vezes para relatar um resultado; numa decisão (D8), ter passado por ele serviu de desempate.

**valor-p** (p). A chance de ver uma diferença assim, ou maior, só por acaso, se na verdade não houvesse diferença nenhuma. Abaixo de 0,05, costuma-se dizer que a diferença é significativa.

**vazamento** (quando o teste dá a resposta). Quando informação do teste escapa para o treino, por exemplo uma cópia de cada lado. O modelo parece melhor do que é, como um aluno que viu a prova antes.

## Modelos

**aprendizado em contexto** (in-context learning). O modelo recebe, de uma vez, os exemplos respondidos e os casos novos, e responde sem treinar. É assim que o TabPFN funciona.

**bagging** (muitos modelos em paralelo). Treina muitas árvores, cada uma num sorteio diferente dos e-mails, e faz a média. A Random Forest é desse tipo; as Extra Trees não (cada árvore vê todos os e-mails).

**boosting** (muitos modelos em fila). Árvores em fila: cada nova corrige um pouco o erro que sobra, mais onde ele é maior. LightGBM, XGBoost e CatBoost são desse tipo.

**CatBoost** (boosting de árvores (Yandex)). Mais uma implementação de boosting de árvores, que entrou no torneio.

**Dummy** (o filtro que não faz nada). O “modelo” que nunca marca spam. É o piso do torneio: MCC zero. Todo modelo precisa ser melhor que ele.

**Extra Trees** (ET, árvores extremamente aleatórias). Centenas de árvores em que os pontos de corte e as palavras candidatas são sorteados; a resposta é a média dos votos. Cada árvore vê todos os e-mails (não é bagging). Mais variada e mais rápida que a Random Forest.

**família** (modelos que raciocinam do mesmo jeito). Grupo de modelos com o mesmo jeito de decidir: lineares (pesam palavras), vizinhos, kernel, rede neural, probabilística e dois tipos de árvores. O conselho tem um representante de cada uma das sete famílias.

**hiperparâmetro** (botão de ajuste). Um botão que se escolhe antes de treinar: quantos vizinhos o kNN consulta, quantas árvores, quanto o modelo pode se ajustar ao treino. O algoritmo genético procurou a melhor combinação de botões.

**kernel** (régua de semelhança do SVM). A função que o SVM usa para comparar dois e-mails. Trocar o kernel muda o formato possível da fronteira: reta (linear) ou curva (RBF).

**kNN** (k vizinhos mais próximos). Para um e-mail novo, procura os k e-mails de treino mais parecidos (pelo cosseno) e copia a resposta da maioria. Fraco sozinho, mas erra em lugares diferentes dos outros: no conselho, é o reforço mais valioso.

**LightGBM** (boosting de árvores (Microsoft)). Árvores em fila: cada nova árvore corrige um pouco o erro que ainda sobra, com mais força onde ele é maior; a soma das correções é a nota. Rápido em tabelas grandes.

**LinearSVC** (SVM linear). SVM com fronteira reta, rápido em textos com milhares de colunas. Foi, por pouco, o melhor modelo sozinho do torneio, num topo empatado, e é a linha de base do projeto.

**linha de base** (a referência simples). O modelo simples que os outros precisam superar para valer a complicação. Aqui, o LinearSVC com TF-IDF com log e limiar de custo.

**MLP** (perceptron de múltiplas camadas, uma rede neural). Rede neural pequena: uma camada de “neurônios” aprende combinações de palavras e uma saída junta tudo numa nota de spam.

**modelo de fundação** (modelo pré-treinado de uso geral). Modelo treinado antes, em muitos dados, que se usa direto num problema novo, sem ajuste. O TabPFN é um modelo de fundação para tabelas.

**Naive Bayes** (NB). Calcula quanto cada palavra é mais comum no spam ou no e-mail real e combina essas chances como se uma palavra não tivesse nada a ver com a outra (por isso “ingênuo”). A versão usada aqui (Complement) aprende com o contrário de cada classe. Rápido, mas fraco nesta base.

**NBSVM** (Naive Bayes com SVM). Usa as razões de chance do Naive Bayes para reescalar as palavras e treina um modelo linear sobre elas (Wang e Manning, 2012). Clássico em textos curtos.

**Prior Labs** (a empresa do TabPFN). Empresa alemã que criou o TabPFN. A versão 3.5-Plus roda na nuvem dela e exige uma chave de acesso; a 3.5-Fast roda no computador.

**Random Forest** (RF, floresta aleatória). Muitas árvores de decisão, cada uma treinada num sorteio diferente dos e-mails; a cada corte, só um sorteio das palavras é considerado; a resposta é a média delas (bagging).

**RBF** (função de base radial). Um jeito de medir semelhança pela distância entre dois e-mails. Com ele, o SVM consegue traçar fronteiras curvas, de qualquer formato.

**Regressão Logística** (LR). Dá um peso a cada palavra (positivo puxa para spam, negativo para e-mail real), soma tudo e transforma a soma numa chance de spam, de zero a um. Simples e rápida. Também é o presidente (meta-modelo) do stacking.

**regularização** (o freio contra decorar). Um freio que impede o modelo de decorar os e-mails de treino. Na Regressão Logística e no SVM, é o botão C: C pequeno freia mais.

**SVM** (máquina de vetores de suporte). Traça a fronteira com a maior folga possível entre os e-mails reais e os spams do treino. Os e-mails da beira da fronteira (os “vetores de suporte”) decidem onde ela passa. A nota é a distância até a fronteira, não uma probabilidade.

**TabPFN** (rede pré-treinada para tabelas (Prior Labs)). Modelo de fundação treinado em milhões de tabelas inventadas por computador (Hollmann et al., 2025, sobre a versão 2; usamos a 3.5). Classifica sem treinar nesta base: recebe os exemplos respondidos como contexto e responde de uma vez. Tem botões de ajuste, que não mexemos. Foi usado sozinho e como oitavo membro do Stacking 8.

**XGBoost** (boosting de árvores). Outra implementação de boosting de árvores, muito usada em competições de dados.

**árvore de decisão** (perguntas de sim ou não). Uma sequência de perguntas de sim ou não (“o e-mail tem a palavra X mais de duas vezes?”) que termina numa resposta: spam ou e-mail real.

## Combinação e busca

**Caruana** (seleção de ensembles (Caruana et al., 2004)). Monta a combinação escolhendo, passo a passo, o modelo que mais melhora o conjunto, podendo repetir modelos. Foi uma das combinações comparadas com o stacking.

**diversidade** (errar em e-mails diferentes). Modelos que erram em e-mails diferentes se completam: onde um erra, outro acerta. É o que faz uma combinação valer a pena; juntar modelos parecidos não ajuda.

**erra junto** (erros em comum). Quando outro membro do conselho erra um e-mail, a chance de este errar o mesmo e-mail. Quanto menor, mais informação nova o membro traz para o conselho.

**Especialista** (o filtro desta caixa). O Stacking 7 com os ajustes escolhidos pelo algoritmo genético. Foi o de menor custo nos folds desta caixa; no teste trancado, empatou com o Stacking 7 padrão. Fora da caixa, custa mais.

**GA** (algoritmo genético). Busca inspirada na evolução: cada candidato é uma combinação de ajustes; os melhores se cruzam, os filhos sofrem pequenas mutações, e isso se repete por gerações. Testou 857 das 7.077.888 combinações possíveis.

**geração** (uma rodada do algoritmo genético). A população é o grupo de candidatos de uma rodada. Em cada geração, os melhores se cruzam e sofrem mutações, e os filhos formam a geração seguinte.

**Jaccard** (índice de Jaccard). Fração de erros em comum entre dois modelos: erros que os dois cometem, divididos pelos erros de qualquer um deles. Favorece quem erra muito; por isso, para saber quem erra junto, usamos a chance de um errar quando o outro erra.

**meta-modelo** (o presidente do conselho). Recebe as notas dos membros do stacking e decide. Aqui é uma Regressão Logística; os pesos dela mostram quanto cada nota conta, mas as notas se parecem entre si, então peso não é só confiança.

**stacking** (empilhamento, o conselho). Combinação de modelos: cada membro dá uma nota ao e-mail e um meta-modelo (o presidente) aprende quanto confiar em cada um. Funciona quando os membros erram em e-mails diferentes.

**Stacking 7 e Stacking 8** (os conselhos do projeto). O Stacking 7 combina um modelo de cada uma das sete famílias. O Stacking 8 é o Stacking 7 padrão com o TabPFN como oitavo membro. O Stacking 13 usa treze modelos e empata com o de sete.

## Termos do projeto

**decisões D1 a D12** (o registro de decisões). As decisões do projeto, numeradas na ordem em que foram tomadas, com o porquê e os números, em docs/REGISTRO_DECISOES.md. As mais citadas: D8, o melhor filtro para esta caixa; D10, a regra escrita antes de testar no SpamAssassin; D11, a revisão crítica; D12, o vocabulário para leigos e a definição de cópia.

**e-mail real** (e-mail legítimo). O e-mail que não é spam. É o termo que usamos em todo o projeto.

**fora da caixa** (e-mails de outras pessoas). Testar o filtro em e-mails de outras caixas (Enron2–6 e SpamAssassin), que ele nunca viu e que vêm de outras pessoas e outras coleções de spam. É onde todos os filtros pioram.

**ham** (nome em inglês para e-mail real). Termo da literatura em inglês para o e-mail legítimo, em oposição a spam. Neste projeto escrevemos sempre “e-mail real”.

## Arquivos e ferramentas

**API** (interface de programação). O jeito de um programa usar um serviço de outro pela internet. O TabPFN-3.5-Plus roda na nuvem da Prior Labs e é chamado por API, com uma chave pessoal.

**Colab** (Google Colab). Serviço do Google que roda notebooks no navegador, sem instalar nada. Na primeira execução, o notebook instala as bibliotecas e reinicia uma vez.

**CPU** (o processador do computador). O processador comum do computador, sem placa de vídeo. A réplica local do TabPFN rodou na CPU.

**HTML** (linguagem de marcação de hipertexto). A linguagem das páginas web. Estes slides são um único arquivo HTML que abre sem internet.

**JSON** (notação de objetos JavaScript). Formato de texto para dados estruturados. Os números do projeto saem de um único JSON de resultados; os fatos de fora (anos, números de artigos) ficam num bloco à parte, com a fonte.

**notebook** (caderno Jupyter). Documento que mistura texto, código e resultados, e roda célula por célula. O do professor é o Projeto3_Classificacao_Spam.ipynb.

**pytest** (testes automáticos). Programa que roda os testes automáticos do código: confere métricas em casos conhecidos, a separação das cópias e os números do relatório.

**Release** (página de downloads do GitHub). Onde ficam os filtros já treinados, para baixar. O notebook e a demo baixam sozinhos o que faltar, conferindo a impressão digital de cada arquivo.

**Streamlit** (página web a partir de Python). Biblioteca que transforma um programa em Python numa página web. É a base da demo: cola-se um e-mail e vê-se cada etapa até a decisão.

---
*Gerado por `apresentacao/build.py` a partir de `apresentacao/fonte/comum/glossario.json`.*
