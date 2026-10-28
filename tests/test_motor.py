"""T3.4: o motor reproduz os números da confirmação (20 folds) a partir do cache, e retreinar dá as mesmas notas."""
import numpy as np
import pandas as pd
import pytest

from spamfilter import metricas, modelos
from spamfilter.motor import Motor

from conftest import TOL_CACHE

CONF = [f"c{s}_{f}" for s in (201, 202, 203, 204) for f in range(5)]

# experimentos/03_ga/ga_confirmacao.csv (média de 20 folds): (MCC no limiar 0,5, custo por mil no limiar de custo).
# O cálculo a partir do cache é determinístico: a tolerância (TOL_CACHE) fica muito abaixo da diferença entre o
# Especialista e o padrão (0,00097), que é justamente o que o teste precisa enxergar.
ESPERADO = {
    "Stacking 7 padrão": (modelos.STACKING7_PADRAO, 0.973398, 41.513035),
    "Stacking 7 especialista": (modelos.STACKING7_ESPECIALISTA, 0.974363, 36.617861),
}
ESPERADO_LSVC = {"LinearSVC padrão": (modelos.LINEARSVC_PADRAO, 0.960036, 52.029336),
                 "LinearSVC otimizado": (modelos.LINEARSVC_OTIMIZADO, 0.962669, 50.034027)}


def _media(motor, prever, cfg):
    linhas = []
    for c in CONF:
        linhas.append(metricas.metricas(motor.y_teste(c), *prever(cfg, c)))
    return pd.DataFrame(linhas).mean()


@pytest.mark.parametrize("nome", list(ESPERADO))
def test_stacking_20_folds(motor, nome):
    cfg, mcc, custo = ESPERADO[nome]
    m = _media(motor, motor.prever_stacking, cfg)
    assert m.mcc == pytest.approx(mcc, abs=TOL_CACHE)
    assert m.custo_por_mil == pytest.approx(custo, abs=1e-4)


@pytest.mark.parametrize("nome", list(ESPERADO_LSVC))
def test_linearsvc_20_folds(motor, nome):
    k, mcc, custo = ESPERADO_LSVC[nome]
    m = _media(motor, motor.prever_linearsvc, k)
    assert m.mcc == pytest.approx(mcc, abs=TOL_CACHE)
    assert m.custo_por_mil == pytest.approx(custo, abs=1e-4)


def test_configuracoes_dos_filtros_finais():
    esp = modelos.descrever_stacking(modelos.STACKING7_ESPECIALISTA)
    assert esp["Reg. Logística"] == "C=1, cw=balanced" and esp["SVM RBF"] == "k=500, C=30"
    assert esp["kNN cosseno"] == "k=3, w=uniform" and esp["MLP"] == "h=(64,), a=0.0001"
    assert esp["Naive Bayes"] == "a=0.1, norm=True, usa_tf=False" and esp["Extra Trees"].startswith("n=800")
    assert esp["LightGBM"] == "n=400, eta=0.05, folhas=63" and esp["meta-modelo"] == "C=10.0"
    assert modelos.descrever("LinearSVC", modelos.LINEARSVC_OTIMIZADO) == "C=1, cw=None"
    assert int(np.prod(modelos.CARD)) == 7_077_888


@pytest.mark.parametrize("familia", ["Reg. Logística", "kNN cosseno", "Naive Bayes"])
def test_retreinar_reproduz_o_cache(base, motor, familia, tmp_path):
    """Treina de novo a configuração padrão numa divisão e compara com as notas gravadas no experimento 03."""
    novo = Motor(base, cache=tmp_path, log=lambda *a: None)
    novo.adicionar_divisoes([201])
    novo.rodar([("c201_0", familia, 0)], blocos=1)
    a_tr, a_te = novo.carregar("c201_0", familia, 0)
    b_tr, b_te = motor.carregar("c201_0", familia, 0)
    assert np.allclose(a_tr, b_tr, atol=1e-6) and np.allclose(a_te, b_te, atol=1e-6)
