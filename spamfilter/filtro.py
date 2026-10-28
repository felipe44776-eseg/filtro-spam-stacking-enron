"""O filtro pronto para uso: os 7 membros treinados nos 5172 e-mails, o meta-modelo e o limiar de custo.

    from spamfilter.filtro import FiltroStacking
    f = FiltroStacking.carregar("resultados/modelos/especialista.joblib")
    f.explicar("Subject: cheap meds online ...")     # cada etapa, da conversão à decisão

O meta-modelo é treinado nas notas out-of-fold dos membros (as mesmas do motor, divisão "externo"), e o limiar
de custo sai de uma validação cruzada do meta-modelo, só no treino: o mesmo procedimento avaliado nos 20 folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import conversao
from .dados import SEED, Base
from .metricas import CUSTO_FN, CUSTO_FP, limiar_custo
from .modelos import FAMILIAS, META_C, construir, descrever_stacking, meta, nota
from .representacao import tfidf


@dataclass
class FiltroStacking:
    nome: str
    cfg: tuple
    vocab: tuple = ()
    membros: dict = field(default_factory=dict)
    meta_modelo: object = None
    limiar: float = 0.5
    descricao: dict = field(default_factory=dict)
    palavras: object = None            # (vocab, pesos) de uma Reg. Logística, só para explicar a decisão

    # ----- treino -----------------------------------------------------------------------------------------
    @classmethod
    def treinar(cls, nome: str, cfg, base: Base, ztr: np.ndarray, n_jobs: int = 4, log=print) -> "FiltroStacking":
        """ztr: notas out-of-fold dos 7 membros nos 5172 e-mails (colunas na ordem de FAMILIAS)."""
        f = cls(nome=nome, cfg=tuple(cfg), vocab=tuple(base.vocab), descricao=descrever_stacking(cfg))
        for i, fam in enumerate(FAMILIAS):
            log(f"{nome}: treinando {fam} ({f.descricao[fam]})")
            f.membros[fam] = construir(fam, cfg[i], n_jobs).fit(base.X, base.y)
        m = meta(META_C[cfg[-1]])
        oof = cross_val_predict(m, ztr, base.y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        f.meta_modelo = m.fit(ztr, base.y)
        f.limiar = float(limiar_custo(oof, base.y, CUSTO_FP, CUSTO_FN))
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        lr = make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")).fit(base.X, base.y)
        f.palavras = (lr, np.asarray(lr[-1].coef_).ravel())
        return f

    def salvar(self, caminho) -> Path:
        caminho = Path(caminho); caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho, compress=3)
        return caminho

    @staticmethod
    def carregar(caminho) -> "FiltroStacking":
        return joblib.load(caminho)

    # ----- uso --------------------------------------------------------------------------------------------
    def notas(self, X) -> np.ndarray:
        return np.column_stack([nota(self.membros[f], X) for f in FAMILIAS])

    def probabilidade(self, X) -> np.ndarray:
        return self.meta_modelo.predict_proba(self.notas(X))[:, 1]

    def decidir(self, X) -> np.ndarray:
        """1 = spam (limiar de custo)."""
        return (self.probabilidade(X) > self.limiar).astype(int)

    def vetorizar(self, texto: str) -> tuple[str, np.ndarray]:
        aueb = conversao.para_aueb(texto)
        return aueb, conversao.vetor(aueb, self.vocab)

    def explicar(self, texto: str, n_palavras: int = 8) -> dict:
        """Todas as etapas para um e-mail: conversão, contagem, notas dos membros, soma do meta-modelo, decisão."""
        from scipy import sparse
        aueb, v = self.vetorizar(texto)
        X = sparse.csr_matrix(v.astype(np.float32)[None, :])
        z = self.notas(X)[0]
        esc, lr = self.meta_modelo[0], self.meta_modelo[-1]
        zpad = (z - esc.mean_) / esc.scale_
        contrib = zpad * lr.coef_.ravel()
        logit = float(lr.intercept_[0] + contrib.sum()); p = float(1 / (1 + np.exp(-logit)))
        lrw, pesos = self.palavras
        t = lrw[0].transform(X).toarray()[0]
        efeito = t * pesos
        ordem = np.argsort(efeito)
        vocab = np.array(self.vocab)
        return dict(
            texto_aueb=aueb, cobertura=conversao.cobertura(aueb, self.vocab),
            palavras_encontradas={str(vocab[j]): int(v[j]) for j in np.argsort(-v)[: 3 * n_palavras] if v[j] > 0 and len(vocab[j]) >= 3},
            colunas_nao_zero=int((v > 0).sum()), vetor=v, presenca=(v > 0).astype(int),
            notas_membros=dict(zip(FAMILIAS, map(float, z))), contribuicoes=dict(zip(FAMILIAS, map(float, contrib))),
            intercepto=float(lr.intercept_[0]), logit=logit, prob_spam=p, limiar=self.limiar, spam=bool(p > self.limiar),
            empurram_spam={str(vocab[j]): float(efeito[j]) for j in ordem[::-1][:n_palavras] if efeito[j] > 0},
            empurram_real={str(vocab[j]): float(efeito[j]) for j in ordem[:n_palavras] if efeito[j] < 0},
        )
