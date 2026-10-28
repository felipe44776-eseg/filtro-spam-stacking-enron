"""Motor do stacking com cache: o que tornou o algoritmo genético viável.

Ideia: para cada divisão (treino/teste) e cada configuração de cada família, o modelo é treinado UMA vez.
Guardamos (a) a nota out-of-fold no treino, de uma CV interna por grupo igual para todas as configurações, e
(b) a nota em cada base de teste, do modelo refeito no treino inteiro. Depois disso, avaliar qualquer
combinação do Stacking 7 é só treinar o meta-modelo sobre 7 colunas: milissegundos.

Arquivo de cache: {chave}__f{FID}__k{k}.npz com ztr, zte_{nome_do_teste} e t (segundos de treino).
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from . import caminhos
from .dados import SEED, Base, carregar, divisoes, indices_teste
from .metricas import limiar_custo
from .modelos import FAMILIAS, FID, META_C, TODOS, construir, meta, nota

# famílias mais lentas primeiro, para equilibrar os blocos paralelos
_LENTO = {"Extra Trees": 0, "LightGBM": 1, "MLP": 2, "SVM RBF": 3}


def _treinar(destino: Path, familia: str, k: int, Xtr, ytr, gtr, tes: dict, si: int, n_jobs: int) -> None:
    """Nota out-of-fold no treino + nota em cada teste. Escrita atômica (arquivo _tmp renomeado no fim)."""
    if destino.exists():
        return
    t0 = time.time(); ztr = np.zeros(Xtr.shape[0])
    for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=si).split(Xtr, ytr, gtr):
        m = construir(familia, k, n_jobs).fit(Xtr[itr], ytr[itr]); ztr[ite] = nota(m, Xtr[ite])
    m = construir(familia, k, n_jobs).fit(Xtr, ytr)
    tmp = destino.with_name(destino.stem + "_tmp.npz")
    np.savez(tmp, ztr=ztr, t=time.time() - t0, **{f"zte_{n}": nota(m, A) for n, A in tes.items()})
    os.replace(tmp, destino)


def _estender_um(destino: Path, origem: Path, familia: str, k: int, Xtr, ytr, tes: dict, n_jobs: int) -> None:
    """Reaproveita a nota out-of-fold de `origem` (mesmo treino) e só treina o modelo final para os testes novos."""
    if destino.exists():
        return
    ztr = np.load(origem)["ztr"]; t0 = time.time()
    m = construir(familia, k, n_jobs).fit(Xtr, ytr)
    tmp = destino.with_name(destino.stem + "_tmp.npz")
    np.savez(tmp, ztr=ztr, t=time.time() - t0, **{f"zte_{n}": nota(m, A) for n, A in tes.items()})
    os.replace(tmp, destino)


class Motor:
    """Divisões + cache. Bases: "dev" (80% de desenvolvimento) e "all" (os 5172 e-mails).
    `cache` recebe os arquivos novos; `leitura_extra` são outros caches só lidos (por exemplo, o do experimento 03)."""

    def __init__(self, base: Base | None = None, cache: Path | str | None = None, log=print, leitura_extra=()):
        self.base = base or carregar()
        self.dev, self.teste = indices_teste(self.base)
        b = self.base
        self.bases = {"dev": (b.X[self.dev], b.y[self.dev], b.g[self.dev]), "all": (b.X, b.y, b.g)}
        self.cache = Path(cache or caminhos.CACHE_MOTOR); self.cache.mkdir(parents=True, exist_ok=True)
        self.caches = [self.cache] + [Path(c) for c in leitura_extra]
        self.splits: dict[str, dict] = {}
        self.log = log

    # ----- divisões ------------------------------------------------------------------------------------
    def adicionar_divisoes(self, sementes, prefixo: str = "c", n_folds: int = 5) -> list[str]:
        """5 folds por semente nos 80% de desenvolvimento. Chave {prefixo}{semente}_{fold}; CV interna = semente*100+fold."""
        _, y, g = self.bases["dev"]; chaves = []
        for s in sementes:
            for f, (tr, te) in enumerate(divisoes(y, g, s, n_folds)):
                c = f"{prefixo}{s}_{f}"
                self.splits[c] = dict(base="dev", tr=tr, tes={"te": te}, si=s * 100 + f); chaves.append(c)
        return chaves

    def adicionar_teste_trancado(self) -> str:
        """Treino nos 80%, teste nos 1035 e-mails trancados. ATENÇÃO: a base trancada já foi usada 4 vezes, todas para relatar (docs/REGISTRO_DECISOES.md, seção 4)."""
        self.splits["teste"] = dict(base="all", tr=self.dev, tes={"teste": self.teste}, si=SEED)
        return "teste"

    def adicionar_externo(self, Xext: dict) -> str:
        """Treino nos 5172 e-mails, teste em cada base externa (dict nome -> matriz)."""
        self.splits["externo"] = dict(base="all", tr=np.arange(self.base.n), tes=dict(Xext), si=SEED)
        return "externo"

    def y_treino(self, chave: str) -> np.ndarray:
        sp = self.splits[chave]; return self.bases[sp["base"]][1][sp["tr"]]

    def y_teste(self, chave: str, te: str = "te", yext: dict | None = None) -> np.ndarray:
        sp = self.splits[chave]; v = sp["tes"][te]
        if isinstance(v, np.ndarray) and v.ndim == 1:
            return self.bases[sp["base"]][1][v]
        if yext is None:
            raise ValueError("base externa: passe yext")
        return yext[te]

    # ----- cache ---------------------------------------------------------------------------------------
    def arquivo(self, chave: str, familia: str, k: int) -> Path:
        """Onde está (ou onde será gravado) o arquivo: o primeiro cache que já o tem; senão, o cache de gravação."""
        nome = f"{chave}__f{FID[familia]}__k{k}.npz"
        return next((c / nome for c in self.caches if (c / nome).exists()), self.cache / nome)

    def _args(self, chave, familia, k, n_jobs):
        sp = self.splits[chave]; Xb, yb, gb = self.bases[sp["base"]]; tr = sp["tr"]
        tes = {n: (Xb[v] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for n, v in sp["tes"].items()}
        return self.arquivo(chave, familia, k), familia, k, Xb[tr], yb[tr], gb[tr], tes, sp["si"], n_jobs

    def rodar(self, tarefas, rotulo: str = "motor", blocos: int = 8) -> None:
        """Treina o que falta no cache. A cada bloco escolhe 1 a 4 processos pela RAM livre."""
        import psutil
        from joblib import Parallel, delayed, parallel_config
        pend = sorted([t for t in tarefas if not self.arquivo(*t).exists()], key=lambda t: _LENTO.get(t[1], 9))
        self.log(f"{rotulo}: {len(tarefas)} tarefas, {len(pend)} pendentes")
        t_ini = time.time()
        for q in range(blocos):
            bloco = pend[q::blocos]
            if not bloco:
                continue
            livre = psutil.virtual_memory().available / 2**30
            nw = 4 if livre > 5 else 3 if livre > 3.5 else 2 if livre > 2.5 else 1
            nj = 4 if nw == 1 else 2; t0 = time.time()
            if nw == 1:
                for t in bloco:
                    _treinar(*self._args(*t, nj))
            else:
                with parallel_config(backend="loky", inner_max_num_threads=nj):
                    Parallel(n_jobs=nw)(delayed(_treinar)(*self._args(*t, nj)) for t in bloco)
            gc.collect()
            self.log(f"{rotulo}: bloco {q + 1}/{blocos} | {len(bloco)} tarefas | {nw} em paralelo | RAM livre {livre:.1f} GB | "
                     f"{time.time() - t0:.0f}s (total {time.time() - t_ini:.0f}s)")

    def estender(self, origem: str, nova: str, tes: dict, pares, n_jobs: int = 2, processos: int = 3) -> str:
        """Nova chave com o mesmo treino de `origem` e testes novos. Copia a nota out-of-fold de `origem` e treina
        só o modelo final de cada (família, k) em `pares`: 6 vezes mais barato que rodar a divisão de novo."""
        from joblib import Parallel, delayed, parallel_config
        sp = self.splits[origem]
        self.splits[nova] = dict(base=sp["base"], tr=sp["tr"], tes=dict(tes), si=sp["si"])
        Xb, yb, _ = self.bases[sp["base"]]; Xtr, ytr = Xb[sp["tr"]], yb[sp["tr"]]
        pend = [(f, k) for f, k in pares if not self.arquivo(nova, f, k).exists()]
        self.log(f"estender {origem} -> {nova}: {len(pares)} configurações, {len(pend)} pendentes")
        t0 = time.time()
        with parallel_config(backend="loky", inner_max_num_threads=n_jobs):
            Parallel(n_jobs=processos)(delayed(_estender_um)(self.cache / self.arquivo(nova, f, k).name, self.arquivo(origem, f, k),
                                                             f, k, Xtr, ytr, tes, n_jobs) for f, k in pend)
        self.log(f"estender {nova}: {time.time() - t0:.0f}s")
        return nova

    def carregar(self, chave: str, familia: str, k: int, te: str = "te") -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.arquivo(chave, familia, k)); return z["ztr"], z[f"zte_{te}"]

    def tempo(self, chave: str, familia: str, k: int) -> float:
        return float(np.load(self.arquivo(chave, familia, k))["t"])

    # ----- previsões -----------------------------------------------------------------------------------
    def notas_stacking(self, cfg, chave: str, te: str = "te") -> tuple[np.ndarray, np.ndarray]:
        """(Ztr, Zte): as 7 notas lado a lado (colunas na ordem de FAMILIAS)."""
        par = [self.carregar(chave, fam, cfg[i], te) for i, fam in enumerate(FAMILIAS)]
        return np.column_stack([a for a, _ in par]), np.column_stack([b for _, b in par])

    def prever_stacking(self, cfg, chave: str, te: str = "te", extra=None, custo_fp: float = 9, custo_fn: float = 1):
        """(probabilidade, decisão no limiar 0,5, decisão no limiar de custo). O limiar de custo sai de uma CV
        do meta-modelo só no treino. `extra` = (nota_treino, nota_teste) de um 8º membro (ex.: TabPFN)."""
        ytr = self.y_treino(chave); Ztr, Zte = self.notas_stacking(cfg, chave, te)
        if extra is not None:
            Ztr, Zte = np.column_stack([Ztr, extra[0]]), np.column_stack([Zte, extra[1]])
        m = meta(META_C[cfg[-1]])
        oof = cross_val_predict(m, Ztr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        p = m.fit(Ztr, ytr).predict_proba(Zte)[:, 1]
        return p, (p > 0.5).astype(int), (p > limiar_custo(oof, ytr, custo_fp, custo_fn)).astype(int)

    def prever_linearsvc(self, k: int, chave: str, te: str = "te", custo_fp: float = 9, custo_fn: float = 1):
        """(nota, decisão no limiar 0, decisão no limiar de custo)."""
        ytr = self.y_treino(chave); ztr, zte = self.carregar(chave, "LinearSVC", k, te)
        return zte, (zte > 0).astype(int), (zte > limiar_custo(ztr, ytr, custo_fp, custo_fn)).astype(int)

    def pesos_meta(self, cfg, chave: str) -> dict:
        """Peso de cada família no meta-modelo (notas padronizadas: os pesos são comparáveis)."""
        Ztr, _ = self.notas_stacking(cfg, chave)
        m = meta(META_C[cfg[-1]]).fit(Ztr, self.y_treino(chave))
        return dict(zip(FAMILIAS, m[-1].coef_.ravel())) | {"intercepto": float(m[-1].intercept_[0])}

    def tarefas_stacking(self, chaves, cfgs, linearsvc=()) -> list[tuple]:
        """Lista (chave, família, k) que cobre as configurações pedidas."""
        precisa = {(fam, cfg[i]) for cfg in cfgs for i, fam in enumerate(FAMILIAS)} | {("LinearSVC", k) for k in linearsvc}
        return [(c, fam, k) for c in chaves for fam, k in sorted(precisa, key=str)]


__all__ = ["Motor", "TODOS"]
