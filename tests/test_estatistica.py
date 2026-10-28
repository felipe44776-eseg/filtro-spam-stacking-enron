"""T3.3: testes estatísticos, conferidos contra os resultados gravados dos experimentos 01 e 03."""
import numpy as np
import pandas as pd
import pytest

from spamfilter import caminhos, estatistica


def test_holm_caso_conhecido():
    assert np.allclose(estatistica.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_do_torneio_reproduz_0166():
    """Experimento 01: o stacking empata com o LinearSVC depois da correção de Holm (p = 0,166)."""
    d = caminhos.EXPERIMENTOS / "01_benchmark"
    r1 = pd.read_csv(d / "bench_rodada1.csv", sep=";", decimal=",")
    r2 = pd.read_csv(d / "bench_rodada2.csv", sep=";", decimal=",")
    top = r1.groupby("modelo").mcc.mean().sort_values(ascending=False).index[:5]
    R = pd.concat([r1, r2]); R = R[R.modelo.isin(top)]
    P = R.pivot_table(index=["seed", "fold"], columns="modelo", values="mcc")
    melhor = P.mean().idxmax()
    assert melhor == "Stacking (meta LR)"
    pares = [(melhor, o) for o in P.columns.drop(melhor)]
    C = estatistica.comparar(P, pares, R.n_tr.mean(), R.n_te.mean(), runs=3).set_index("B")
    assert C.loc["LinearSVC TF-IDF", "p_holm"] == pytest.approx(0.166, abs=0.001)


def test_piso_de_deteccao_com_20_folds():
    """Experimento 03: com 20 folds, só diferenças de MCC acima de ~0,0086 são detectáveis."""
    R = pd.read_csv(caminhos.EXPERIMENTOS / "03_ga" / "ga_confirmacao.csv", sep=";", decimal=",")
    P = R.pivot(index="split", columns="metodo", values="mcc")
    dp = (P["Stacking 7 otimizado (GA)"] - P["LinearSVC otimizado"]).std(ddof=1)
    razao = R.n_te.mean() / R.n_tr.mean()
    assert estatistica.menor_diferenca_detectavel(dp, 20, razao) == pytest.approx(0.0086, abs=1e-4)
    assert estatistica.menor_diferenca_detectavel(dp, 1000, razao) == pytest.approx(0.0079, abs=0.0002)


def test_mcnemar_conta_so_os_desacordos():
    a = np.array([1, 1, 1, 0, 1, 1]).astype(bool); b = np.array([1, 0, 0, 1, 1, 1]).astype(bool)
    na, nb, p = estatistica.mcnemar(a, b)
    assert (na, nb) == (2, 1) and 0 < p <= 1


def test_t_corrigido_e_mais_conservador_que_o_t_comum():
    rng = np.random.default_rng(0); a = 0.97 + rng.normal(0, 0.005, 20); b = a - 0.003 + rng.normal(0, 0.003, 20)
    _, _, p_corr = estatistica.t_corrigido(a, b, 3300, 830)
    _, _, p_comum = estatistica.t_corrigido(a, b, 1e12, 1)          # razão ~0: o t comum
    assert p_corr > p_comum


def test_bootstrap_por_grupo_e_deterministico():
    y = np.array([0, 0, 1, 1, 0, 1]); g = np.array([1, 1, 2, 3, 4, 5]); prev = {"m": np.array([0, 0, 1, 0, 0, 1])}
    a = estatistica.bootstrap_grupo(y, prev, g, n=50, seed=3); b = estatistica.bootstrap_grupo(y, prev, g, n=50, seed=3)
    assert a.equals(b) and a.shape == (50, 1)
