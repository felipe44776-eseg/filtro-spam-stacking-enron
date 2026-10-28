"""O filtro salvo (resultados/modelos/*.joblib): carregar, explicar e decidir dão a mesma resposta."""
import numpy as np
import pytest
from scipy import sparse

from spamfilter import caminhos, conversao, dados, publicados
from spamfilter.filtro import FiltroStacking

ARQ = caminhos.RESULTADOS / "modelos" / "stacking7_padrao.joblib"


@pytest.fixture(scope="module")
def filtro():
    if not ARQ.exists():
        pytest.skip("filtros não treinados: baixe de Releases (spamfilter.publicados.garantir) ou rode scripts/09_modelos_finais.py")
    return FiltroStacking.carregar(ARQ)


def test_arquivo_local_e_o_publicado():
    """O sha256 de cada filtro local é o da Release v1.0 (a mesma referência que o download confere)."""
    presentes = [n for n in publicados.SHA256 if (caminhos.RESULTADOS / "modelos" / f"{n}.joblib").exists()]
    if not presentes:
        pytest.skip("filtros ausentes")
    for n in presentes:
        assert publicados._sha256(caminhos.RESULTADOS / "modelos" / f"{n}.joblib") == publicados.SHA256[n], n


def test_explicar_bate_com_probabilidade_e_decisao(filtro):
    texto = "Subject: cheap meds online\nget your prescription drugs now , no doctor needed !"
    e = filtro.explicar(texto)
    X = sparse.csr_matrix(conversao.vetor(conversao.para_aueb(texto), filtro.vocab).astype(np.float32)[None, :])
    assert e["prob_spam"] == pytest.approx(float(filtro.probabilidade(X)[0]), abs=1e-9)
    assert e["spam"] == bool(filtro.decidir(X)[0])
    assert sum(e["contribuicoes"].values()) + e["intercepto"] == pytest.approx(e["logit"], abs=1e-9)


def test_vocabulario_do_filtro_e_o_do_csv(filtro):
    assert tuple(filtro.vocab) == tuple(dados.carregar().vocab)
