"""Objetos caros compartilhados pelos testes (carregados uma vez por sessão).

Os testes só LEEM o cache do motor (experimentos/03_ga/cache_ga). Quem retreina grava numa pasta temporária.
"""
import pytest

from spamfilter import dados
from spamfilter.motor import Motor

# Números de referência: docs/REGISTRO_DECISOES.md e os CSVs de cada experimento.
TOL_MCC = 0.001


@pytest.fixture(scope="session")
def base():
    return dados.carregar()


@pytest.fixture(scope="session")
def motor(base):
    m = Motor(base, log=lambda *a: None)
    m.adicionar_divisoes([201, 202, 203, 204])          # 20 folds de confirmação (experimento 03)
    return m
