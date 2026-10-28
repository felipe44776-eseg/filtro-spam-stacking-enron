"""Testes para decidir se uma diferença entre modelos é real.

- t corrigido (Nadeau e Bengio, 2003): os treinos dos folds se sobrepõem, então o t comum subestima a variância.
- Holm: corrige o p quando há várias comparações.
- Bayesiano correlacionado (Benavoli et al., 2017, pacote baycomp): P(A melhor), P(equivalentes), P(B melhor),
  com equivalência = |diferença de MCC| < rope.
- McNemar: compara dois modelos nos mesmos e-mails de teste (só os desacordos importam).
- Bootstrap por grupo: intervalo de confiança sorteando textos distintos (cópias entram juntas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metricas import mcc_np


def t_corrigido(a, b, n_tr: float, n_te: float) -> tuple[float, float, float]:
    """(diferença média, t, p bilateral) entre as métricas por fold de A e B."""
    d = np.asarray(a, float) - np.asarray(b, float); J = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), J - 1))


def holm(p) -> np.ndarray:
    """p ajustado por Holm, na ordem original."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    aj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    saida = np.empty(m); saida[o] = aj
    return saida


def bayes(a, b, rope: float = 0.01, runs: int = 1) -> tuple[float, float, float]:
    """(P(A melhor), P(equivalentes), P(B melhor)). runs = número de repetições da CV (sementes)."""
    import baycomp
    return tuple(float(v) for v in baycomp.two_on_single(np.asarray(a), np.asarray(b), rope=rope, runs=runs))


def comparar(P: pd.DataFrame, pares, n_tr: float, n_te: float, rope: float = 0.01, runs: int = 1) -> pd.DataFrame:
    """Tabela de comparações. P: linhas = folds, colunas = métodos. pares: lista de (A, B)."""
    linhas = []
    for a, b in pares:
        d, t, p = t_corrigido(P[a], P[b], n_tr, n_te)
        try:
            pa, pe, pb = bayes(P[a].values, P[b].values, rope=rope, runs=runs)
        except Exception:   # baycomp falha se as diferenças forem todas iguais
            pa = pe = pb = np.nan
        linhas.append(dict(A=a, B=b, dif=d, t=t, p=p, P_A_melhor=pa, P_equivalentes=pe, P_B_melhor=pb))
    R = pd.DataFrame(linhas)
    R["p_holm"] = holm(R.p.values)
    return R.sort_values("p").reset_index(drop=True)


def mcnemar(acerto_a, acerto_b) -> tuple[int, int, float]:
    """(só A acerta, só B acerta, p exato binomial)."""
    acerto_a, acerto_b = np.asarray(acerto_a, bool), np.asarray(acerto_b, bool)
    na, nb = int((acerto_a & ~acerto_b).sum()), int((~acerto_a & acerto_b).sum())
    return na, nb, (float(stats.binomtest(min(na, nb), na + nb, 0.5).pvalue) if na + nb else 1.0)


def bootstrap_grupo(y, previsoes: dict, grupos, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """MCC de cada modelo em n reamostragens de textos distintos. previsoes: nome -> decisão 0/1."""
    _, inv = np.unique(grupos, return_inverse=True)
    membros = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    rng = np.random.default_rng(seed); yb = np.asarray(y).astype(bool)
    saida = {nome: [] for nome in previsoes}
    for _ in range(n):
        sel = np.concatenate([membros[k] for k in rng.integers(0, len(membros), len(membros))])
        for nome, yh in previsoes.items():
            saida[nome].append(mcc_np(yb[sel], np.asarray(yh)[sel].astype(bool)))
    return pd.DataFrame(saida)


def menor_diferenca_detectavel(dp_dif: float, folds: int, razao_teste_treino: float) -> float:
    """≈ 2 erros-padrão corrigidos. Tem um piso: mais folds não ajudam, porque os folds reusam os mesmos e-mails."""
    return float(2 * dp_dif * np.sqrt(1 / folds + razao_teste_treino))
