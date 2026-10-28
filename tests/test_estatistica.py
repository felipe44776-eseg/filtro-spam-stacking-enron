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


def test_piso_de_deteccao_caso_calculado_a_mao():
    """dp = 0,01, 20 folds, teste/treino = 0,25: EP = 0,01 × √(1/20 + 0,25) = 0,005477.
    Quantis t com 19 graus de liberdade: t(0,975) = 2,09302 e t(0,80) = 0,86095.
    80% de poder: (2,09302 + 0,86095) × EP = 0,016180; 50% de poder: 2,09302 × EP = 0,011464."""
    assert estatistica.menor_diferenca_detectavel(0.01, 20, 0.25) == pytest.approx(0.016180, abs=1e-6)
    assert estatistica.menor_diferenca_detectavel(0.01, 20, 0.25, poder=0.5) == pytest.approx(0.011464, abs=1e-6)
    # o piso: com infinitos folds, o t vira z (1,95996 + 0,84162 = 2,80158) e sobra √0,25 = 0,5 EP
    assert estatistica.menor_diferenca_detectavel(0.01, 10**9, 0.25) == pytest.approx(2.80158 * 0.005, abs=1e-5)


def test_mcnemar_conta_so_os_desacordos():
    a = np.array([1, 1, 1, 0, 1, 1]).astype(bool); b = np.array([1, 0, 0, 1, 1, 1]).astype(bool)
    na, nb, p = estatistica.mcnemar(a, b)
    assert (na, nb) == (2, 1)
    assert p == pytest.approx(1.0)                          # binomial exato bilateral de 1 em 3: p = 1


def test_mcnemar_p_exato_caso_conhecido():
    """10 desacordos, todos a favor de A: p bilateral = 2 × 0,5^10 = 0,001953."""
    a = np.r_[np.ones(10, bool), np.ones(5, bool)]; b = np.r_[np.zeros(10, bool), np.ones(5, bool)]
    assert estatistica.mcnemar(a, b)[2] == pytest.approx(2 * 0.5 ** 10, rel=1e-9)


def test_t_corrigido_com_variancia_zero():
    assert estatistica.t_corrigido([0.9, 0.95, 0.97], [0.9, 0.95, 0.97], 80, 20)[2] == 1.0   # métodos idênticos
    assert estatistica.t_corrigido([0.91, 0.96, 0.98], [0.9, 0.95, 0.97], 80, 20)[2] == 0.0


def test_ic_corrigido_contem_a_media_e_bate_com_o_t():
    rng = np.random.default_rng(1); a = 0.97 + rng.normal(0, 0.004, 20); b = a - 0.002 + rng.normal(0, 0.003, 20)
    d, lo, hi = estatistica.ic_corrigido(a, b, 3300, 830)
    _, _, p = estatistica.t_corrigido(a, b, 3300, 830)
    assert lo < d < hi
    assert (lo > 0 or hi < 0) == (p < 0.05)                  # o IC de 95% exclui o zero exatamente quando p < 0,05


def test_t_corrigido_e_mais_conservador_que_o_t_comum():
    rng = np.random.default_rng(0); a = 0.97 + rng.normal(0, 0.005, 20); b = a - 0.003 + rng.normal(0, 0.003, 20)
    _, _, p_corr = estatistica.t_corrigido(a, b, 3300, 830)
    _, _, p_comum = estatistica.t_corrigido(a, b, 1e12, 1)          # razão ~0: o t comum
    assert p_corr > p_comum


def test_bootstrap_por_grupo_e_deterministico():
    y = np.array([0, 0, 1, 1, 0, 1]); g = np.array([1, 1, 2, 3, 4, 5]); prev = {"m": np.array([0, 0, 1, 0, 0, 1])}
    a = estatistica.bootstrap_grupo(y, prev, g, n=50, seed=3); b = estatistica.bootstrap_grupo(y, prev, g, n=50, seed=3)
    assert a.equals(b) and a.shape == (50, 1)
