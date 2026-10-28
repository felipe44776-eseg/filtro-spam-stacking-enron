"""Gravação dos resultados (fonte única dos números de README, notebook, decks e demo).

Cada script grava `resultados/<nome>.json`; o 08_modelos_finais.py junta tudo em `metricas_finais.json`
(esquema em docs/CONTRATO_RESULTADOS.md). Taxas são frações de 0 a 1, sem arredondamento.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import caminhos


def _converter(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, Path):
        return o.relative_to(caminhos.RAIZ).as_posix() if o.is_absolute() else o.as_posix()
    raise TypeError(f"não sei gravar {type(o)}")


def gravar(nome: str, dados: dict, fonte: str) -> Path:
    """Grava resultados/<nome>.json com o campo `fonte` (script de origem)."""
    caminhos.RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = caminhos.RESULTADOS / f"{nome}.json"
    destino.write_text(json.dumps(dados | {"fonte": fonte}, default=_converter, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


def ler(nome: str) -> dict:
    return json.loads((caminhos.RESULTADOS / f"{nome}.json").read_text(encoding="utf-8"))


def gravar_csv(nome: str, df: pd.DataFrame) -> Path:
    """CSV no padrão do projeto: separador ; e decimal ,."""
    caminhos.RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = caminhos.RESULTADOS / f"{nome}.csv"
    df.to_csv(destino, sep=";", decimal=",", index=False)
    return destino
