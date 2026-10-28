"""Onde ficam os arquivos do projeto. Todos os módulos usam estes caminhos, nunca um caminho digitado."""
import os
from pathlib import Path

# A raiz é a pasta do repositório (onde está o pyproject.toml). Pode ser trocada pela variável SPAMFILTER_RAIZ.
RAIZ = Path(os.environ.get("SPAMFILTER_RAIZ", Path(__file__).resolve().parents[1]))

DADOS = RAIZ / "dados"
CSV = DADOS / "emails.csv"                      # o arquivo do enunciado ("emails (1).csv"), sem alteração
ENRON_VETORES = DADOS / "enron_vetores.npz"     # Enron1-6 contados com o vocabulário do CSV
BRUTOS_AUEB = DADOS / "brutos_aueb"             # enron1..6.tar.gz, texto pré-processado da AUEB

EXPERIMENTOS = RAIZ / "experimentos"
INDICES_TESTE = EXPERIMENTOS / "indices_teste.npz"     # base de teste trancada, gravada no experimento 02
CACHE_MOTOR = EXPERIMENTOS / "03_ga" / "cache_ga"      # notas out-of-fold dos experimentos 03 a 05

RESULTADOS = RAIZ / "resultados"
