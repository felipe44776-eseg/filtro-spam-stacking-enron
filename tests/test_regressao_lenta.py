"""Regressão completa (pytest -m lento): retreina modelos, refaz o algoritmo genético e a base externa.
Cada teste confere um número do registro de decisões (tolerância de ±0,001 no MCC)."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from spamfilter import caminhos, dados, externo, genetico, metricas, modelos
from spamfilter.motor import Motor

from conftest import TOL_MCC

pytestmark = pytest.mark.lento


def test_reg_logistica_tfidf_log_5_folds(base):
    """T3.2: Reg. Logística com TF-IDF log, 5 folds da semente 42 nos 5172 e-mails: MCC 0,9612 (experimento 01)."""
    criar = modelos.torneio()["Reg. Logistica TF-IDF"]
    mcc = [metricas.metricas_simples(base.y[te], criar().fit(base.X[tr], base.y[tr]).predict(base.X[te]),
                                     np.zeros(len(te)) + 0.5)["mcc"]
           for tr, te in dados.divisoes(base.y, base.g, 42)]
    gravado = pd.read_csv(caminhos.EXPERIMENTOS / "01_benchmark" / "bench_rodada1.csv", sep=";", decimal=",")
    assert np.mean(mcc) == pytest.approx(0.9612, abs=TOL_MCC)
    assert np.mean(mcc) == pytest.approx(gravado[gravado.modelo == "Reg. Logistica TF-IDF"].mcc.mean(), abs=1e-6)


@pytest.mark.parametrize("familia", ["SVM RBF", "MLP", "Extra Trees", "LightGBM"])
def test_retreinar_familias_pesadas(base, motor, familia, tmp_path):
    """T3.4: as famílias com sorteio também repetem as notas do cache (mesma semente)."""
    novo = Motor(base, cache=tmp_path, log=lambda *a: None)
    novo.adicionar_divisoes([201])
    novo.rodar([("c201_0", familia, 0)], blocos=1)
    a_tr, a_te = novo.carregar("c201_0", familia, 0)
    b_tr, b_te = motor.carregar("c201_0", familia, 0)
    assert np.corrcoef(a_te, b_te)[0, 1] > 0.999 and np.corrcoef(a_tr, b_tr)[0, 1] > 0.999
    y = motor.y_teste("c201_0"); lim = 0.5
    assert metricas.mcc_np(y, a_te > lim) == pytest.approx(metricas.mcc_np(y, b_te > lim), abs=0.005)


def test_algoritmo_genetico_reproduz_a_busca(base):
    """T3.5: 857 avaliações; MCC na busca 0,9732 (padrão), 0,9795 (GA), 0,9784 (busca aleatória);
    o melhor do GA é a configuração do Especialista."""
    m = Motor(base, log=lambda *a: None)
    chaves = m.adicionar_divisoes([11, 12], prefixo="b")
    apt = genetico.Aptidao(m, chaves)
    melhor, hist = genetico.algoritmo_genetico(apt)
    n = len(apt.memoria)
    aleatoria = genetico.busca_aleatoria(apt, n)
    assert n == 857
    assert apt(modelos.STACKING7_PADRAO) == pytest.approx(0.9732, abs=TOL_MCC)
    assert apt(melhor) == pytest.approx(0.9795, abs=TOL_MCC)
    assert max(aleatoria) == pytest.approx(0.9784, abs=TOL_MCC)
    assert tuple(melhor) == modelos.STACKING7_ESPECIALISTA
    mcc_lsvc = [apt.mcc_linearsvc(k) for k in range(len(modelos.LINEARSVC))]
    assert int(np.argmax(mcc_lsvc)) == modelos.LINEARSVC_OTIMIZADO


def test_base_externa(base):
    """T3.6: Enron2-6 sem cópias; Stacking 7 padrão com MCC externo 0,8717 e custo 177,5 por mil (experimento 03)."""
    Xext, yext, removidas = externo.carregar_externo(base)
    assert sum(len(v) for v in yext.values()) > 27000
    m = Motor(base, log=lambda *a: None)
    m.adicionar_externo(Xext)
    R = pd.DataFrame([metricas.metricas(yext[b], *m.prever_stacking(modelos.STACKING7_PADRAO, "externo", b)) for b in Xext])
    assert R.mcc.mean() == pytest.approx(0.8717, abs=TOL_MCC)
    assert R.custo_por_mil.mean() == pytest.approx(177.45, abs=0.1)


def test_vetores_gravados_do_enron1_sao_o_csv(base):
    """T3.6: o Enron1 do arquivo enron_vetores.npz é o CSV (mesmo multiconjunto de linhas e de rótulos)."""
    z = np.load(caminhos.ENRON_VETORES)
    csv = Counter(r.astype(np.int64).tobytes() for r in base.X.toarray())
    npz = Counter(r.astype(np.int64).tobytes() for r in z["X1"])
    assert csv == npz and sorted(base.y) == sorted(z["y1"])


def test_vetorizar_enron1_do_zero_reproduz_o_csv(base):
    """T3.6: contar o Enron1 original de novo, pelo pacote, reproduz as 5172 linhas do CSV."""
    textos, y, _ = externo.ler_tar(1)
    X1 = externo.vetorizar(textos, base.vocab)
    csv = Counter(r.astype(np.int64).tobytes() for r in base.X.toarray())
    assert Counter(r.astype(np.int64).tobytes() for r in X1) == csv
    assert sorted(y) == sorted(base.y)
