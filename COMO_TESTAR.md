# Como testar (roteiro para a banca)

## Teste rápido (5 minutos)

| Quero... | Faça | Deve aparecer |
|---|---|---|
| ver tudo funcionando sem instalar nada | no README, clique em **Abrir no Colab** ao lado do `Projeto3_Classificacao_Spam.ipynb` e use **Ambiente de execução → Executar tudo**; quando o Colab reiniciar, **Executar tudo** de novo | o notebook roda até o fim; na seção 7, o algoritmo genético roda ao vivo; na seção 11, o filtro classifica três e-mails de exemplo |
| testar no meu computador | `pip install -r requirements.txt`, `pip install -e . --no-deps` e `pytest` (Python 3.11 ou 3.12; detalhes no [INSTALACAO.md](INSTALACAO.md)) | `42 passed, 3 skipped` (ou `45 passed`, se os filtros treinados já estiverem baixados) |
| colar um e-mail e ver a decisão | `streamlit run demo/app.py` | as 9 etapas, do texto colado à decisão dos dois filtros |
| entender os modelos antes de testar | leia [docs/MODELOS.md](docs/MODELOS.md) | cada modelo em uma frase, uma analogia e como decide |

## Roteiro completo

Sete passos, do mais rápido ao mais completo. Cada um diz o que fazer e o que deve aparecer. Antes, siga o [INSTALACAO.md](INSTALACAO.md). Quem não quiser instalar nada pode começar pelo passo 3, no Google Colab.

## 1. Os testes automáticos (cerca de 30 s)

```powershell
pytest
```
**Esperado:** `45 passed, 11 deselected` com os filtros treinados em `resultados/modelos/` (veja o passo 3); sem eles, `42 passed, 3 skipped, 11 deselected`. Os testes conferem:
- que a contagem por substring reproduz o CSV;
- que nenhum e-mail repetido fica dos dois lados de uma divisão;
- as métricas e os testes estatísticos em casos calculados à mão (custo, limiar com empates, TCR, piso de detecção, McNemar, intervalo corrigido);
- os números dos 20 folds de confirmação, com tolerância de 1e-5 (o cálculo a partir do cache é determinístico);
- que o filtro salvo explica a mesma decisão que toma, e que o arquivo local é o publicado na Release.

## 2. A regressão completa (cerca de 3 min)

```powershell
pytest -m lento
```
**Esperado:** `11 passed` com os filtros de *Releases* em `resultados/modelos/`; sem eles, `10 passed, 1 skipped` (o teste da demo). Os testes refazem algumas das contas mais pesadas e conferem os números contra o registro de decisões:
- o algoritmo genético inteiro: 857 avaliações, chegando à configuração do Especialista (tolerância de 1e-5);
- o retreino das famílias pesadas (±0,001 no MCC, onde há sorteio);
- a base externa;
- a vetorização do Enron1 a partir dos textos originais;
- o conversor contra os e-mails brutos da caixa farmer-d (`dados/brutos_aueb_raw/`);
- a demo de ponta a ponta, sem abrir o navegador: o spam de exemplo precisa ir para a lixeira e o e-mail real, para a caixa de entrada, **nos dois filtros**.

## 3. O notebook-história (cerca de 3 min)

Abra `Projeto3_Classificacao_Spam.ipynb` e use **Kernel → Restart & Run All** (no computador: `jupyter notebook Projeto3_Classificacao_Spam.ipynb`).

**No Google Colab:** abra pelo botão *Abrir no Colab* do README e use *Ambiente de execução → Executar tudo*. Na primeira vez, a célula de preparo baixa o projeto, instala os pacotes e reinicia o ambiente; quando o Colab reconectar, use *Executar tudo* de novo.

**Esperado:** roda até o fim sem erro. O algoritmo genético roda ao vivo na seção 7, em cerca de 1,5 min. Na seção 11, o filtro Especialista é baixado da Release v1.0 se não estiver em `resultados/modelos/` (precisa de internet; o sha256 é conferido).

## 4. Conferir três números-chave contra o README

| Número | Onde aparece no README | Onde conferir |
|---|---|---|
| Custo do Especialista nos 20 folds | tabela "Os filtros" | seção 9 do notebook-história, ou `resultados/metricas_finais.json` → `finais.especialista.cv20.custo_por_mil` |
| Ganho do algoritmo genético em divisões novas, com o IC95 | "O que descobrimos", item 6 | seção 7 do notebook-história, ou `genetico.ganho_sementes_novas` e `genetico.ganho_sementes_novas_ic95` |
| Decisão da 3ª base (SpamAssassin) e a comparação com não filtrar | "As tecnologias em destaque" e "O que descobrimos", item 7 | seção 9 do notebook-história, ou `resultados/08_terceira_base.json` |

O README é gerado a partir de `resultados/metricas_finais.json` por `scripts/gerar_documentos.py`. Nenhum resultado foi digitado à mão.

## 5. As apresentações

Abra no navegador:
- `apresentacao/banca/index.html`, a apresentação técnica;
- `apresentacao/equipe/index.html`, a didática, com glossário.

Navegação: setas ← →, `F` para tela cheia e `N` para as notas do apresentador. Também estão publicadas no GitHub Pages (link no README).

## 6. A demonstração: cole um e-mail

```powershell
streamlit run demo/app.py
```
O navegador abre sozinho (na primeira vez, os filtros são baixados da Release se faltarem). Escolha um dos três exemplos, ou cole um e-mail seu, em texto puro ou com cabeçalhos. A tela mostra as 9 etapas:
1. o texto colado;
2. o formato da base;
3. a contagem e a cobertura do vocabulário;
4. o vetor binário;
5. o TF-IDF;
6. as notas dos 7 modelos;
7. a soma do meta-modelo;
8. a decisão nos dois filtros;
9. as palavras que pesaram.

**Esperado:** o spam de exemplo vai para a lixeira, e o e-mail real de exemplo vai para a caixa de entrada. O e-mail em português mostra o aviso de cobertura baixa, porque o filtro foi treinado em inglês. Um aviso fixo lembra que, fora da caixa do treino, o filtro erra muito mais.

## 7. Opcional: refazer um experimento do zero

| O quê | Comando | Tempo |
|---|---|---|
| Uma etapa da história, a partir dos caches | `python scripts/05_algoritmo_genetico.py` | cerca de 2 min |
| Todos os notebooks técnicos, com as saídas | `python scripts/gerar_notebooks.py` | cerca de 10 min |
| O torneio de 22 modelos do zero | apague `resultados/03_torneio_rodada*.csv` e rode `scripts/03_modelos_independentes.py` com a variável `TORNEIO_DO_ZERO=1` (sem ela, o script reaproveita o resultado histórico de `experimentos/01_benchmark/`) | cerca de 15 min |
| As notas do motor do zero | gere o notebook com `python experimentos/scripts/build_ga.py experimentos/03_ga/novo.ipynb` e execute-o com `python experimentos/scripts/run_nb.py experimentos/03_ga/novo.ipynb`. Para treinar tudo de novo, mova antes a pasta `experimentos/03_ga/cache_ga/`: sem ela, os testes rápidos falham até o cache ser refeito | cerca de 1 h de CPU |
