"""T3.3: métricas em casos conhecidos."""
import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from spamfilter import metricas


def test_mcc_rapido_igual_ao_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
        assert metricas.mcc_np(a, b) == pytest.approx(matthews_corrcoef(a, b), abs=1e-12)


def test_custo_por_mil():
    # 1 e-mail real na lixeira (custa 9) + 1 spam que passou (custa 1), em 4 e-mails
    assert metricas.custo_por_mil([0, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(2500)


def test_limiar_de_custo_casos_conhecidos():
    # Cortar logo após a nota 0,9 custa 1 (um spam passa); qualquer corte abaixo põe um e-mail real na lixeira (9).
    assert metricas.limiar_custo([0.9, 0.8, 0.3, 0.2], [1, 0, 1, 0]) == pytest.approx(0.85)
    # Quando marcar qualquer coisa como spam custa mais que deixar tudo passar: limiar infinito.
    assert metricas.limiar_custo([0.9, 0.1], [0, 1]) == np.inf


def test_limiar_de_custo_nao_corta_dentro_de_um_empate():
    """Três notas 0,9 empatadas (2 spams e 1 e-mail real): a regra "nota > limiar" não separa empates. Marcar o
    bloco custa 9 (o e-mail real) e não marcar custa 2 (os dois spams): o certo é não marcar nada."""
    s, y = np.array([0.9, 0.9, 0.9, 0.2]), np.array([1, 1, 0, 0])
    lim = metricas.limiar_custo(s, y)
    assert lim == np.inf
    assert metricas.custo_por_mil(y, (s > lim).astype(int)) == pytest.approx(500)


def test_nao_filtrar_e_tcr():
    y = np.array([0, 0, 0, 1, 1])
    assert metricas.custo_nao_filtrar(y) == pytest.approx(400)              # 2 spams passam em 5 e-mails
    assert metricas.tcr(y, np.zeros(5, int)) == pytest.approx(1.0)          # não filtrar empata consigo mesmo
    assert metricas.tcr(y, np.array([1, 0, 0, 1, 1])) == pytest.approx(400 / 1800)   # 1 e-mail real perdido: pior que não filtrar


def test_limiar_de_custo_e_conservador():
    """Com o custo 9:1, o limiar fica acima do 0,5 numa nota bem calibrada."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    s = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.25, 5000), 0, 1)
    assert metricas.limiar_custo(s, y) > 0.5


def test_metricas_chaves_e_taxas():
    y = np.array([0, 0, 0, 1, 1]); yh = np.array([0, 1, 0, 1, 0]); yc = np.array([0, 0, 0, 1, 0])
    m = metricas.metricas(y, yh.astype(float), yh, yc)
    assert m["real_lixeira"] == pytest.approx(1 / 3) and m["spam_passou"] == pytest.approx(1 / 2)
    assert m["real_lixeira_custo"] == 0 and m["custo_por_mil"] == pytest.approx(1000 * 1 / 5)
    assert m["precisao_custo"] == 1 and m["revocacao_custo"] == pytest.approx(1 / 2) and m["f1_custo"] == pytest.approx(2 / 3)
    assert m["tcr"] == pytest.approx(400 / 200)
