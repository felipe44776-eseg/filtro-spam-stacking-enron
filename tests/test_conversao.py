"""Conversão de um e-mail qualquer para o formato da AUEB (usada na demo e na 3ª base)."""
import tarfile
from collections import Counter

import numpy as np
import pytest

from spamfilter import caminhos, conversao


def test_regras_de_tokenizacao():
    # minúsculas, pontuação separada, letras × dígitos separados; 0 e 1 colados a letras viram "o" e "l"
    assert conversao.tokenizar_linha("Starts at 2:00 p.m. today") == "starts at 2 : 00 p . m . today"
    assert conversao.tokenizar_linha("George x3-6992") == "george x 3 - 6992"
    assert conversao.tokenizar_linha("EI305reveff061200.xls") == "ei 305 reveffo 61200 . xls"
    assert conversao.tokenizar_linha("V1agra 100%") == "vlagra 100 %"


def test_email_bruto_vira_assunto_mais_corpo():
    bruto = ("From: a@b.com\nTo: c@d.com\nSubject: Meter 7268 Nov Allocation\n\n"
             "Lauri, see <mailto:x@y.com> the file << File: plan.xls >>\n\n\nThanks.\n")
    assert conversao.para_aueb(bruto) == "Subject: meter 7268 nov allocation\nlauri , see the file >\nthanks ."


def test_texto_colado_sem_cabecalho():
    assert conversao.para_aueb("Assunto: Oi\nTudo bem?").startswith("Subject: oi\ntudo bem ?")
    assert conversao.para_aueb("cheap meds!!").startswith("Subject: \ncheap meds ! !")


def test_cobertura_baixa_em_portugues(base):
    en = conversao.cobertura(conversao.para_aueb("Subject: meeting\nplease see the attached file for the gas deal"), base.vocab)
    pt = conversao.cobertura(conversao.para_aueb("Assunto: reunião\nsegue em anexo a planilha da entrega do projeto"), base.vocab)
    assert en > 0.8 > 0.5 > pt


@pytest.mark.lento
def test_reproduz_os_emails_brutos_da_farmer_d(base):
    """90,5% dos 3669 e-mails brutos da farmer-d viram exatamente uma linha do CSV (dados/brutos_aueb_raw, fora do git)."""
    arq = caminhos.DADOS / "brutos_aueb_raw" / "farmer-d.tar.gz"
    if not arq.exists():
        pytest.skip("baixe http://www2.aueb.gr/users/ion/data/enron-spam/raw/ham/farmer-d.tar.gz em dados/brutos_aueb_raw/")
    csv = Counter(r.astype(np.int64).tobytes() for r in base.X.toarray())
    with tarfile.open(arq, "r:gz") as t:
        V = [conversao.vetor(conversao.para_aueb(t.extractfile(m).read()), base.vocab).astype(np.int64).tobytes()
             for m in t.getmembers() if m.isfile()]
    assert np.mean([v in csv for v in V]) >= 0.90
