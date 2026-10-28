"""T3.1: carga, duplicatas e base de teste trancada."""
import numpy as np

from spamfilter import caminhos, dados


def test_formato_da_base(base):
    assert base.X.shape == (5172, 3000)
    assert len(base.vocab) == 3000
    assert int((base.y == 0).sum()) == 3672 and int((base.y == 1).sum()) == 1500


def test_duplicatas_sao_em_maioria_emails_reais(base):
    r = dados.resumo_duplicatas(base)
    assert r["duplicatas_excedentes"] == 541
    assert abs(r["fracao_duplicatas_email_real"] - 0.93) < 0.01


def test_base_trancada_recalculada_igual_a_gravada(base):
    dev, teste = dados.indices_teste(base)
    gravado = np.load(caminhos.INDICES_TESTE)
    assert np.array_equal(dev, gravado["dev"]) and np.array_equal(teste, gravado["teste"])
    assert len(teste) == 1035
    assert int((base.y[teste] == 0).sum()) == 735 and int((base.y[teste] == 1).sum()) == 300


def test_nenhum_grupo_dos_dois_lados(base):
    dev, teste = dados.indices_teste(base)
    assert not set(base.g[dev]) & set(base.g[teste])
    y, g = base.y[dev], base.g[dev]
    for s in (201, 202, 203, 204):
        for tr, te in dados.divisoes(y, g, s):
            assert not set(g[tr]) & set(g[te])
