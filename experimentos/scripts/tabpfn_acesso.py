"""Confere o acesso aos modelos gated do TabPFN sem abrir navegador. A chave vem da variavel TABPFN_TOKEN
(nunca deste arquivo). Nao imprime a chave."""
import os, sys
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from tabpfn import browser_auth as ba
from tabpfn.settings import settings

tok = os.environ.get("TABPFN_TOKEN", "").strip()
print("chave presente:", bool(tok), "| formato tabpfn_sk_:", tok.startswith("tabpfn_sk_"))
if not tok:
    sys.exit("TABPFN_TOKEN ausente")
api = settings.tabpfn.auth_api_url
print("chave aceita pelo servidor:", ba.verify_token(tok, api))
for repo in ["tabpfn_3_5", "tabpfn_3", "tabpfn_2_5"]:
    try:
        lic = ba._get_license_name(repo)
        print(f"{repo:12s} licença '{lic}' aceita:", ba.check_license_accepted(tok, api, lic))
    except Exception as e:
        print(f"{repo:12s} erro ao consultar licença: {type(e).__name__}: {str(e)[:200]}")
