"""Algoritmo genético para escolher a configuração do Stacking 7, com a busca aleatória como controle.

Cromossomo: 8 genes = índice da configuração de cada uma das 7 famílias + índice do C do meta-modelo
(7.077.888 combinações). Aptidão: MCC médio nas divisões de busca, calculado só com o meta-modelo sobre as
notas em cache (motor.py). Operadores: torneio de 3, cruzamento uniforme, mutação por gene e elitismo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metricas import mcc_np
from .modelos import CARD, FAMILIAS, META_C, STACKING7_PADRAO, TODOS, meta


class Aptidao:
    """MCC médio de um cromossomo nas divisões de busca, com memória (cada cromossomo é avaliado uma vez).
    A ordem de inserção em `memoria` é a ordem das avaliações (usada nas curvas do GA e no orçamento)."""

    def __init__(self, motor, chaves):
        self.chaves = list(chaves)
        self.Z = {c: {fam: [motor.carregar(c, fam, k) for k in range(len(TODOS[fam]))] for fam in TODOS} for c in self.chaves}
        self.Y = {c: (motor.y_treino(c), motor.y_teste(c)) for c in self.chaves}
        self.memoria: dict[tuple, float] = {}

    def __call__(self, gen) -> float:
        gen = tuple(int(v) for v in gen)
        if gen not in self.memoria:
            v = []
            for c in self.chaves:
                Ztr = np.column_stack([self.Z[c][fam][gen[i]][0] for i, fam in enumerate(FAMILIAS)])
                Zte = np.column_stack([self.Z[c][fam][gen[i]][1] for i, fam in enumerate(FAMILIAS)])
                p = meta(META_C[gen[-1]]).fit(Ztr, self.Y[c][0]).predict_proba(Zte)[:, 1]
                v.append(mcc_np(self.Y[c][1], p > 0.5))
            self.memoria[gen] = float(np.mean(v))
        return self.memoria[gen]

    def mcc_linearsvc(self, k: int) -> float:
        """MCC médio do LinearSVC (configuração k) nas mesmas divisões, para afiná-lo também."""
        return float(np.mean([mcc_np(self.Y[c][1], self.Z[c]["LinearSVC"][k][1] > 0) for c in self.chaves]))


def algoritmo_genetico(aptidao, card=CARD, n_pop: int = 40, n_ger: int = 30, p_cruz: float = 0.9,
                       p_mut: float = 0.15, elite: int = 2, seed: int = 42, padrao=STACKING7_PADRAO):
    """Devolve (melhor cromossomo, histórico por geração). A população inicial inclui o padrão."""
    rng = np.random.default_rng(seed)
    pop = [tuple(padrao)] + [tuple(int(rng.integers(c)) for c in card) for _ in range(n_pop - 1)]
    hist = []
    for ger in range(n_ger):
        apt = np.array([aptidao(i) for i in pop]); ordem = np.argsort(-apt)
        hist.append(dict(geracao=ger, melhor=apt[ordem[0]], media=apt.mean(), avaliacoes=len(aptidao.memoria)))

        def torneio():
            c = rng.choice(len(pop), 3, replace=False); return pop[c[np.argmax(apt[c])]]

        nova = [pop[i] for i in ordem[:elite]]
        while len(nova) < n_pop:
            a, b = torneio(), torneio()
            filho = tuple(x if m else z for x, z, m in zip(a, b, rng.random(len(card)) < 0.5)) if rng.random() < p_cruz else a
            nova.append(tuple(int(rng.integers(c)) if rng.random() < p_mut else x for x, c in zip(filho, card)))
        pop = nova
    apt = np.array([aptidao(i) for i in pop])
    return pop[int(np.argmax(apt))], pd.DataFrame(hist)


def busca_aleatoria(aptidao, n: int, card=CARD, seed: int = 43) -> list[float]:
    """Controle com o mesmo orçamento: n cromossomos sorteados. Devolve a aptidão de cada um, na ordem."""
    rng = np.random.default_rng(seed)
    return [aptidao(tuple(int(rng.integers(c)) for c in card)) for _ in range(n)]


def sensibilidade(aptidao, melhor, card=CARD) -> pd.DataFrame:
    """Partindo do melhor, troca um gene por vez: mostra quais famílias importam."""
    linhas = []
    for i, nome in enumerate(FAMILIAS + ["meta-modelo (C)"]):
        v = [aptidao(melhor[:i] + (k,) + melhor[i + 1:]) for k in range(card[i])]
        linhas.append(dict(gene=nome, pior=min(v), melhor=max(v), amplitude=max(v) - min(v)))
    return pd.DataFrame(linhas).sort_values("amplitude", ascending=False).reset_index(drop=True)
