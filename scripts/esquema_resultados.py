"""Lista as chaves de resultados/metricas_finais.json (para quem escreve README, decks e demo)."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = json.loads((RAIZ / "resultados" / "metricas_finais.json").read_text(encoding="utf-8"))


def esq(o, p="", saida=None):
    saida = [] if saida is None else saida
    if isinstance(o, dict):
        for k, v in o.items():
            esq(v, f"{p}.{k}" if p else k, saida)
    elif isinstance(o, list):
        if o and isinstance(o[0], dict):
            saida.append(f"{p}: lista[{len(o)}] de {{{', '.join(list(o[0].keys())[:14])}}}")
        else:
            saida.append(f"{p}: lista[{len(o)}] ex: {str(o[:3])[:80]}")
    else:
        saida.append(f"{p}: {str(o)[:70]}")
    return saida


linhas = [l for l in esq(R) if "por_base.Enron" not in l]
(RAIZ / "resultados" / "metricas_finais_esquema.txt").write_text("\n".join(linhas), encoding="utf-8")
print(f"{len(linhas)} linhas -> resultados/metricas_finais_esquema.txt")
