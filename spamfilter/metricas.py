"""Métricas do projeto e o limiar de custo.

Vocabulário: "e-mail real" = e-mail legítimo (classe 0); spam = classe 1.
- real_lixeira: fração de e-mails reais marcados como spam (falso positivo, o erro caro).
- spam_passou: fração de spams que chegaram à caixa de entrada (falso negativo).
- custo_por_mil = 1000 × (CUSTO_FP × FP + CUSTO_FN × FN) / n.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

CUSTO_FP, CUSTO_FN = 9, 1   # perder 1 e-mail real custa o mesmo que receber 9 spams


def mcc_np(a, b) -> float:
    """MCC rápido para vetores booleanos (usado milhares de vezes no bootstrap e no algoritmo genético)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    tp = np.sum(a & b); tn = np.sum(~a & ~b); fp = np.sum(~a & b); fn = np.sum(a & ~b)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else 0.0


def contagem_erros(y, yh) -> tuple[int, int]:
    """(e-mails reais na lixeira, spams que passaram)."""
    y, yh = np.asarray(y), np.asarray(yh)
    return int(((yh == 1) & (y == 0)).sum()), int(((yh == 0) & (y == 1)).sum())


def custo_por_mil(y, yh, custo_fp: float = CUSTO_FP, custo_fn: float = CUSTO_FN) -> float:
    fp, fn = contagem_erros(y, yh)
    return 1000 * (custo_fp * fp + custo_fn * fn) / len(y)


def limiar_custo(s, y, custo_fp: float = CUSTO_FP, custo_fn: float = CUSTO_FN) -> float:
    """Limiar que minimiza custo_fp × FP + custo_fn × FN nas notas s (out-of-fold do treino; nunca do teste).

    Ordena as notas da maior para a menor e testa cortar depois de cada uma. Devolve o ponto médio entre
    duas notas vizinhas; +inf quando é mais barato não marcar nada como spam.
    """
    s, y = np.asarray(s, float), np.asarray(y)
    o = np.argsort(-s); ss, yy = s[o], y[o]
    custo = custo_fp * np.cumsum(yy == 0) + custo_fn * ((yy == 1).sum() - np.cumsum(yy == 1))
    i = int(np.argmin(custo))
    if custo_fn * (yy == 1).sum() <= custo[i]:
        return np.inf
    return (ss[i] + ss[i + 1]) / 2 if i + 1 < len(ss) else ss[i] - 1e-9


def metricas(y, s, yh, yc, custo_fp: float = CUSTO_FP, custo_fn: float = CUSTO_FN) -> dict:
    """Métricas no limiar padrão (yh) e no limiar de custo (yc). s = nota contínua (para o PR-AUC)."""
    y, yh, yc = np.asarray(y), np.asarray(yh), np.asarray(yc)
    return dict(mcc=matthews_corrcoef(y, yh), pr_auc=average_precision_score(y, s),
                real_lixeira=(yh[y == 0] == 1).mean(), spam_passou=(yh[y == 1] == 0).mean(),
                mcc_custo=matthews_corrcoef(y, yc), real_lixeira_custo=(yc[y == 0] == 1).mean(),
                spam_passou_custo=(yc[y == 1] == 0).mean(), custo_por_mil=custo_por_mil(y, yc, custo_fp, custo_fn))


def metricas_simples(y, yh, s) -> dict:
    """Métricas de um único limiar (torneio e experimento 02)."""
    y, yh = np.asarray(y), np.asarray(yh)
    return dict(mcc=matthews_corrcoef(y, yh), pr_auc=average_precision_score(y, s), f1=f1_score(y, yh),
                real_lixeira=(yh[y == 0] == 1).mean(), spam_passou=(yh[y == 1] == 0).mean())
