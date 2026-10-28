# Ficha do modelo (model card)

*Gerado por `scripts/gerar_documentos.py` a partir de `resultados/metricas_finais.json`.*

## Para que serve
Classificar e-mails em **spam** ou **e-mail real**. É um projeto acadêmico de Data Science 2 (ESEG), não um produto: os filtros servem para estudar como um classificador se comporta dentro e fora da caixa de e-mail em que foi treinado.

| Filtro | Uso pretendido |
|---|---|
| Especialista | a caixa do treino (Enron1, caixa `farmer-d`) |
| Generalista (Stacking 8) | caixas novas; precisa do TabPFN (nuvem com chave, ou CPU) |
| Stacking 7 padrão | caixas novas, sem internet nem chave (usado na demo) |
| LinearSVC | linha de base simples |

## Dados de treino
- Origem: 5172 e-mails do Enron1, com 3672 e-mails reais de um funcionário da Enron (1999 a 2002) e 1500 spams (2003 a 2005).
- Representação: contagem por substring de 3000 palavras, sobre o texto pré-processado da AUEB.
- O texto vem em inglês, num contexto corporativo do setor de energia dos EUA.

## Desempenho (limiar de custo: perder 1 e-mail real = 9 spams recebidos)

| Avaliação | Especialista | Generalista (S8) |
|---|---|---|
| Esta caixa, 20 folds: MCC | 0,974 | 0,973 |
| Esta caixa, 20 folds: e-mails reais na lixeira | 0,4% | 0,4% |
| Esta caixa, 20 folds: spam que passou | 4,6% | 5,3% |
| Enron2-6 (outras 5 caixas): MCC | 0,894 | 0,882 |
| Enron2-6: e-mails reais na lixeira | 4,0% | 2,2% |
| SpamAssassin (outra empresa, 2002): MCC | 0,350 | 0,437 |
| SpamAssassin: e-mails reais na lixeira | 37,9% | 16,0% |

## Limitações (leia antes de usar)
1. **O filtro aprende a caixa, não "o spam".** Fora da caixa do treino, o desempenho cai muito: no SpamAssassin, o MCC fica abaixo de 0,6 em todos os filtros. O mesmo pipeline treinado dentro do SpamAssassin chega a 0,968. **Para usar de verdade, retreine com e-mails da caixa de destino.**
2. **Só inglês.** Num e-mail em português, a cobertura do vocabulário cai para cerca de 15%, e a decisão perde o sentido. A demo avisa quando a cobertura é baixa.
3. **Época.** Os spams são de 2003 a 2005. Spams atuais (outros produtos, imagens, links encurtados) não estão representados.
4. **Vocabulário da empresa.** Palavras como "enron", "hpl" e nomes de colegas ajudam nesta caixa. Retirá-las quase não muda o resultado aqui, mas melhora um pouco em caixas muito diferentes.

## Considerações éticas
- Os e-mails da Enron tornaram-se públicos na investigação da FERC (EUA). Os do SpamAssassin foram publicados para pesquisa.
- Um filtro de spam errando para o lado do spam esconde mensagens legítimas. Por isso o limiar foi escolhido para que esse erro custe 9 vezes mais.
- O TabPFN é usado sob licença não comercial (pesquisa e avaliação).
