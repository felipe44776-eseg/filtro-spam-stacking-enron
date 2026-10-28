"""Estilo único das figuras (matplotlib, modo claro). Os decks HTML usam as mesmas cores.

Paleta: a paleta categórica de referência do guia de visualização, validada para daltonismo
(slots 1-7 passam CVD e visão normal nos pares adjacentes; e-mail real × spam passa em todos os pares).
- Classe: e-mail real = azul (slot 1); spam = laranja (slot 2). Sempre, em toda figura.
- Famílias do Stacking 7: slots 1-7 na ordem fixa de FAMILIAS. Três delas (aqua, amarelo, magenta) têm
  contraste < 3:1 com o fundo: por isso todo gráfico por família leva o nome escrito ao lado da marca.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if "ipykernel" not in sys.modules:          # rodando como script: nenhuma janela, só arquivos
    os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt

from . import caminhos

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"; TINTA_2 = "#52514e"; TINTA_MUDA = "#898781"
GRADE = "#e1e0d9"; EIXO = "#c3c2b7"

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REAL, SPAM = CATEGORICA[0], CATEGORICA[1]
CLASSE = {0: REAL, 1: SPAM}
NOME_CLASSE = {0: "e-mail real", 1: "spam"}

FAMILIA_COR = {   # mesma ordem de modelos.FAMILIAS (e os nomes do torneio)
    "linear": CATEGORICA[0], "kernel": CATEGORICA[1], "vizinhos": CATEGORICA[2], "rede neural": CATEGORICA[3],
    "probabilística": CATEGORICA[4], "bagging": CATEGORICA[5], "boosting": CATEGORICA[6], "combinação": CATEGORICA[7],
    "referência": EIXO, "árvore": TINTA_MUDA,
}
FAMILIA_DO_MEMBRO = {"Reg. Logística": "linear", "SVM RBF": "kernel", "kNN cosseno": "vizinhos", "MLP": "rede neural",
                     "Naive Bayes": "probabilística", "Extra Trees": "bagging", "LightGBM": "boosting"}

# rampa sequencial (azul) para mapas de calor
SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PASTA_FIGURAS = caminhos.RESULTADOS / "figuras"


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": TINTA_2, "axes.edgecolor": EIXO, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": TINTA_MUDA, "ytick.color": TINTA_MUDA, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "text.color": TINTA, "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def salvar(fig, nome: str, pasta: Path | None = None) -> Path:
    pasta = Path(pasta or PASTA_FIGURAS); pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"{nome}.png"
    fig.savefig(destino)
    return destino


def mostrar(fig, nome: str) -> None:
    """Salva em resultados/figuras/<nome>.png; no notebook também exibe. Não devolve nada, para o notebook não
    imprimir o caminho do arquivo."""
    salvar(fig, nome)
    if "inline" in matplotlib.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


def legenda_classes(ax, **kw) -> None:
    """Legenda e-mail real × spam (sempre presente quando as duas classes aparecem)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=CLASSE[c], mec=SUPERFICIE, mew=1.5, label=NOME_CLASSE[c]) for c in (0, 1)]
    ax.legend(handles=h, **({"loc": "best"} | kw))


estilo()
