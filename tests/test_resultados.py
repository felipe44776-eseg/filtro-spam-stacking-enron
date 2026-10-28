"""Coerência de resultados/metricas_finais.json: as contas agregadas usam as mesmas definições das métricas."""
import json

import pytest

from spamfilter import caminhos

R = json.loads((caminhos.RESULTADOS / "metricas_finais.json").read_text(encoding="utf-8"))
F = R["finais"]
FILTROS = [k for k in F if k != "nao_filtrar"]


@pytest.mark.parametrize("onde", ["cv20", "externo", "spamassassin"])
def test_tcr_agregado_e_razao_dos_custos(onde):
    """TCR = custo de não filtrar ÷ custo do filtro, com os custos publicados (médias), e não a média das razões."""
    ref = F["nao_filtrar"][onde]["custo_por_mil"]
    for k in FILTROS:
        assert F[k][onde]["tcr"] == pytest.approx(ref / F[k][onde]["custo_por_mil"], rel=1e-9), k


def test_nao_filtrar_custa_mil_vezes_a_fracao_de_spam():
    tb = R["terceira_base"]
    assert F["nao_filtrar"]["spamassassin"]["custo_por_mil"] == pytest.approx(1000 * tb["n_spam"] / tb["n"], rel=1e-9)
    tt = R["protocolo"]["teste_trancado"]
    assert F["nao_filtrar"]["teste_trancado"]["spam_passou_n"] == tt["spam"]


def test_sinal_das_comparacoes_de_custo():
    """Diferença negativa = o primeiro filtro é mais barato; o t tem o mesmo sinal da diferença."""
    for r in R["comparacoes_custo_20_folds"]:
        d = F[r["A"]]["cv20"]["custo_por_mil"] - F[r["B"]]["cv20"]["custo_por_mil"]
        assert r["dif"] == pytest.approx(d, abs=1e-9)
        assert r["t"] * r["dif"] >= 0
