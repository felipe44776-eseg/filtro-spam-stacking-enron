# Ficha do modelo (model card)

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json`.*

## Para que serve
Classificar e-mails em **spam** ou **e-mail real**. É um projeto acadêmico de Data Science 2 (ESEG), não um produto: os filtros servem para estudar como um classificador se comporta dentro e fora da caixa de e-mail em que foi treinado.

| Filtro | Uso pretendido |
|---|---|
| Especialista (Stacking 7 do GA) | a caixa do treino (Enron1, caixa `farmer-d`) |
| Stacking 8 (Stacking 7 + TabPFN) | fora da caixa, com o TabPFN o stacking fica bem mais barato (regra D10); precisa da API do TabPFN (chave). Avaliado com notas gravadas; não há inferência pronta para um e-mail novo |
| TabPFN-3.5 sozinho | o mais conservador fora da caixa (o que menos manda e-mail real para a lixeira); precisa da API do TabPFN (chave). Avaliado, sem inferência pronta |
| Stacking 7 padrão | roda sem internet nem chave (usado na demo) |
| LinearSVC | linha de base simples |

## Dados de treino
- Origem: 5172 e-mails do Enron1, com 3672 e-mails reais da caixa farmer-d (1999-12-10 a 2002-01-11) e 1500 spams da coleção GP (2003-12-18 a 2005-09-06).
- Representação: contagem por substring de 3000 palavras, sobre o texto pré-processado da AUEB.
- O texto vem em inglês, num contexto corporativo do setor de energia dos EUA.

## Desempenho (limiar de custo: perder 1 e-mail real = 9 spams recebidos)

| Avaliação | Especialista | Stacking 8 | TabPFN sozinho | Não filtrar |
|---|---|---|---|---|
| Esta caixa, 20 folds: MCC | 0,962 | 0,956 | 0,922 | 0,000 |
| Esta caixa, 20 folds: custo por mil | 36,6 | 38,2 | 55,6 | 290,1 |
| Esta caixa, 20 folds: e-mails reais na lixeira | 0,4% | 0,4% | 0,4% | 0,0% |
| Esta caixa, 20 folds: spam que passou | 4,6% | 5,3% | 10,0% | 100,0% |
| Teste trancado: precisão (spam) | 0,997 | 0,993 | 0,986 | 0,000 |
| Teste trancado: revocação (spam) | 0,967 | 0,957 | 0,910 | 0,000 |
| Teste trancado: F1 (spam) | 0,981 | 0,975 | 0,946 | 0,000 |
| Teste trancado: matriz de confusão (VP / VN / FP / FN) | 290 / 734 / 1 / 10 | 287 / 733 / 2 / 13 | 273 / 731 / 4 / 27 | 0 / 735 / 0 / 300 |
| Enron2-6 (outras 5 caixas): custo por mil | 207,6 | 138,5 | 126,0 | 546,4 |
| Enron2-6: e-mails reais na lixeira | 4,0% | 2,2% | 0,9% | 0,0% |
| SpamAssassin: custo por mil | 2416,5 | 1119,9 | 503,5 | 313,6 |
| SpamAssassin: e-mails reais na lixeira | 37,9% | 16,0% | 4,6% | 0,0% |
| Esta caixa, 20 folds: TCR (acima de 1 = compensa filtrar) | 7,9 | 7,6 | 5,2 | 1,0 |
| Enron2-6: TCR | 2,63 | 3,95 | 4,34 | 1,00 |
| SpamAssassin: TCR | 0,13 | 0,28 | 0,62 | 1,00 |

VP = spam na lixeira, VN = e-mail real na caixa, FP = e-mail real na lixeira, FN = spam que passou. O teste trancado tem 735 e-mails reais e 300 spams.

## Limitações (leia antes de usar)
1. **O filtro aprende a caixa.** Fora da caixa do treino, o desempenho cai muito. No SpamAssassin, com λ = 9, nenhum filtro treinado aqui custa menos que não filtrar; um modelo treinado dentro do SpamAssassin chega a MCC 0,968. **Para usar de verdade, retreine com e-mails da caixa de destino.**
2. **O peso do erro importa.** O vencedor fora da caixa muda com λ.
3. **Só inglês.** Num e-mail em português, a cobertura do vocabulário cai muito, e a decisão perde o sentido. A demo avisa quando a cobertura é baixa.
4. **Época.** Os spams são de 2003 a 2005. Spams atuais (outros produtos, imagens, links encurtados) não estão representados.
5. **Vocabulário da empresa.** Palavras como "enron", "hpl" e nomes de colegas ajudam nesta caixa. Retirá-las quase não muda o resultado aqui.

## Considerações éticas
- Os e-mails da Enron tornaram-se públicos na investigação da FERC (EUA). Os do SpamAssassin foram publicados para pesquisa. O projeto identifica a caixa de origem (`farmer-d`), como o próprio corpus Enron-Spam faz, e não acrescenta dados pessoais.
- Um filtro de spam errando para o lado do spam esconde mensagens legítimas. Por isso o limiar foi escolhido para que esse erro custe 9 vezes mais.
- O TabPFN é usado em pesquisa acadêmica, sem fins comerciais.
