# Como testar (roteiro para a banca)

Sete passos, do mais rápido ao mais completo. Cada um diz o que fazer e o que deve aparecer. Antes, siga o [INSTALACAO.md](INSTALACAO.md).

## 1. Os testes automáticos (cerca de 20 s)

```powershell
pytest
```
**Esperado:** `31 passed, 10 deselected`. Os testes conferem:
- que a contagem por substring reproduz o CSV;
- que nenhum e-mail repetido fica dos dois lados de uma divisão;
- as métricas em casos conhecidos;
- os números dos 20 folds de confirmação.

## 2. A regressão completa (cerca de 3 min)

```powershell
pytest -m lento
```
**Esperado:** `10 passed, 1 skipped`, com os filtros de *Releases* em `resultados/modelos/`; sem eles, `9 passed, 2 skipped`. São pulados os testes que dependem de arquivos fora do repositório:
- o do conversor contra os e-mails brutos da farmer-d (veja `dados/LEIA-ME.md`);
- o da demo, sem os filtros treinados.

Os testes refazem algumas das contas mais pesadas e conferem os números contra o registro de decisões, com tolerância de ±0,001 no MCC:
- o algoritmo genético inteiro: 857 avaliações, chegando à configuração do Especialista;
- o retreino das famílias pesadas;
- a base externa;
- a vetorização do Enron1 a partir dos textos originais;
- a demo de ponta a ponta, sem abrir o navegador: o spam de exemplo precisa ir para a lixeira e o e-mail real, para a caixa de entrada.

## 3. O notebook-história (cerca de 3 min)

Abra `Projeto3_Classificacao_Spam.ipynb` e use **Kernel → Restart & Run All**.

**Esperado:** roda até o fim sem erro. O algoritmo genético roda ao vivo na seção 7, em cerca de 1,5 min.

## 4. Conferir três números-chave contra o README

| Número | Onde aparece no README | Onde conferir |
|---|---|---|
| MCC do Especialista nos 20 folds | tabela "Os filtros" | seção 9 do notebook-história, ou `resultados/metricas_finais.json` → `finais.especialista.cv20.mcc` |
| Fração do ganho do algoritmo genético que era otimismo | "O que descobrimos" | seção 7 do notebook-história |
| Decisão da 3ª base (SpamAssassin) | "O que descobrimos" | seção 9 do notebook-história, ou `resultados/08_terceira_base.json` |

O README é gerado a partir de `resultados/metricas_finais.json` por `scripts/gerar_documentos.py`. Nenhum número foi digitado à mão.

## 5. As apresentações

Abra no navegador:
- `apresentacao/banca/index.html`, a apresentação técnica;
- `apresentacao/equipe/index.html`, a didática, com glossário.

Navegação: setas ← →, `F` para tela cheia e `N` para as notas do apresentador. Também estão publicadas no GitHub Pages (link no README).

## 6. A demonstração: cole um e-mail

```powershell
streamlit run demo/app.py
```
O navegador abre sozinho. Escolha um dos três exemplos, ou cole um e-mail seu, em texto puro ou com cabeçalhos. A tela mostra as 9 etapas:
1. o texto colado;
2. o formato da base;
3. a contagem e a cobertura do vocabulário;
4. o vetor binário;
5. o TF-IDF;
6. as notas dos 7 modelos;
7. a soma do meta-modelo;
8. a decisão nos dois filtros;
9. as palavras que pesaram.

**Esperado:** o spam de exemplo vai para a lixeira, e o e-mail real de exemplo vai para a caixa de entrada. O e-mail em português mostra o aviso de cobertura baixa, porque o filtro foi treinado em inglês.

## 7. Opcional: refazer um experimento do zero

| O quê | Comando | Tempo |
|---|---|---|
| Uma etapa da história, a partir dos caches | `python scripts/05_algoritmo_genetico.py` | cerca de 2 min |
| Todos os notebooks técnicos, com as saídas | `python scripts/gerar_notebooks.py` | cerca de 10 min |
| O torneio de 22 modelos do zero | apague `resultados/03_torneio_rodada*.csv` e rode `scripts/03_modelos_independentes.py` | cerca de 15 min |
| As notas do motor do zero | apague `experimentos/03_ga/cache_ga/` e rode `experimentos/scripts/build_ga.py` | cerca de 1 h de CPU |
