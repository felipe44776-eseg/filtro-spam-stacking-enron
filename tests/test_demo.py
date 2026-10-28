"""A demo roda de ponta a ponta com um exemplo (sem abrir o navegador)."""
import pytest

from spamfilter import caminhos

pytestmark = pytest.mark.lento


def test_demo_classifica_os_exemplos():
    if not (caminhos.RESULTADOS / "modelos" / "especialista.joblib").exists():
        pytest.skip("filtros não treinados: rode scripts/09_modelos_finais.py ou baixe de Releases")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(caminhos.RAIZ / "demo" / "app.py"), default_timeout=120).run()
    assert not app.exception
    opcoes = app.selectbox[0].options
    spam = next(o for o in opcoes if o.startswith("spam"))
    app.selectbox[0].select(spam).run()
    assert not app.exception
    assert any("SPAM" in e.value for e in app.error)            # o spam de exemplo vai para a lixeira
    real = next(o for o in opcoes if o.startswith("e-mail real —"))
    app.selectbox[0].select(real).run()
    assert not app.exception and any("E-MAIL REAL" in s.value for s in app.success)
