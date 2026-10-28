"""Notas do TabPFN no formato do motor: nota out-of-fold no treino (5 folds internos) + nota em cada teste.
Assim o TabPFN entra no stacking como 8a familia sem retreinar os outros 7 modelos.

  nuvem: TabPFN-3.5-Plus via tabpfn-client  -> rodar com .venv-tabpfn-nuvem\\Scripts\\python.exe
  local: TabPFN-3.5-Fast em CPU             -> rodar com .venv\\Scripts\\python.exe

A chave vem da variavel TABPFN_TOKEN (nunca deste arquivo). Retoma do cache se for interrompido.
Exemplos:  --modo nuvem --chaves c201_0          (piloto)
           --modo nuvem --chaves todas --paralelo 3
           --modo local --chaves c201              (prefixo: as 5 divisoes da semente 201)"""
import os, sys, time, argparse, ctypes, threading, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--modo", choices=["nuvem", "local"], required=True)
ap.add_argument("--chaves", default="todas", help="todas | lista separada por virgula | prefixo")
ap.add_argument("--paralelo", type=int, default=1)
ap.add_argument("--so-um-interno", action="store_true", help="piloto: so o 1o treino interno de cada divisao, sem gravar cache")
a = ap.parse_args()

RAIZ = Path(__file__).resolve().parents[2]; E = RAIZ / "experimentos"; D = E / "05_tabpfn"
CACHE = D / "cache_tabpfn" / a.modo; CACHE.mkdir(parents=True, exist_ok=True)
LOG = D / f"progresso_{a.modo}.log"; trava = threading.Lock(); T0 = time.time()
def log(m):
    linha = f"{time.strftime('%H:%M:%S')} [{time.time() - T0:7.0f}s] {m}"
    with trava:
        print(linha, flush=True)
        with LOG.open("a", encoding="utf-8") as fh: fh.write(linha + "\n")

if os.name == "nt":   # pede ao Windows para nao suspender enquanto este processo roda (sem mudar configuracao)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
if not os.environ.get("TABPFN_TOKEN"):
    sys.exit("TABPFN_TOKEN ausente: rode via rodar_tabpfn.ps1 ou defina a variavel")

if a.modo == "nuvem":
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
    novo = lambda: TabPFNClassifier()
    def uso():
        try: return tabpfn_client.get_api_usage()
        except Exception as e: return f"(uso indisponível: {type(e).__name__})"
    VERSAO = f"tabpfn-client {getattr(tabpfn_client, '__version__', '?')} (TabPFN-3.5-Plus)"
else:
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    import tabpfn, psutil
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    novo = lambda: TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, device="cpu", random_state=42)
    uso = lambda: f"RAM livre {psutil.virtual_memory().available / 2**30:.1f} GB"
    VERSAO = f"tabpfn {tabpfn.__version__} (TabPFN-3.5-Fast, CPU)"

df = pd.read_csv(RAIZ / "dados" / "emails.csv"); y = df.pop("Prediction").values; df.pop("Email No.")
X = df.values.astype(np.float32); del df
S = np.load(D / "splits_tabpfn.npz")
todas = [str(c) for c in S["chaves"]]
if a.chaves == "todas": sel = todas
elif "," in a.chaves or a.chaves in todas: sel = [c for c in a.chaves.split(",") if c in todas]
else: sel = [c for c in todas if c.startswith(a.chaves)]
EXT = {}
if "externo" in sel:
    z = np.load(RAIZ / "dados" / "enron_vetores.npz")
    EXT = {f"Enron{k}": z[f"X{k}"][S[f"ext_Enron{k}"]].astype(np.float32) for k in range(2, 7)}

def ajustar_prever(Xtr, ytr, testes, tentativas=4):
    """Na nuvem, repete em caso de falha de rede (ja houve queda de DNS no meio de uma chamada)."""
    for t in range(tentativas):
        try:
            m = novo(); m.fit(Xtr, ytr)
            return [m.predict_proba(A)[:, 1] for A in testes]
        except Exception as e:
            if a.modo != "nuvem" or t == tentativas - 1:
                raise
            espera = 30 * (t + 1)
            log(f"  falha ({type(e).__name__}: {str(e)[:80]}); nova tentativa em {espera}s")
            time.sleep(espera)

def processar(chave):
    destino = CACHE / f"{chave}.npz"
    if destino.exists() and not a.so_um_interno: return chave, "cache"
    tr = S[f"{chave}__tr"]; Xtr, ytr = X[tr], y[tr]; t0 = time.time(); ztr = np.zeros(len(tr))
    for k in range(1 if a.so_um_interno else 5):
        itr, ite = S[f"{chave}__in{k}_tr"], S[f"{chave}__in{k}_te"]; t1 = time.time()
        ztr[ite] = ajustar_prever(Xtr[itr], ytr[itr], [Xtr[ite]])[0]
        log(f"{chave} interno {k + 1}/5: treino {len(itr)} | prevê {len(ite)} | {time.time() - t1:.0f}s | {uso()}")
    if a.so_um_interno: return chave, "piloto"
    testes = {"te": X[S[f"{chave}__te"]]} if f"{chave}__te" in S.files else EXT
    t1 = time.time(); zs = ajustar_prever(Xtr, ytr, list(testes.values()))
    log(f"{chave} final: treino {len(tr)} | prevê {sum(len(v) for v in testes.values())} | {time.time() - t1:.0f}s")
    tmp = destino.with_name(destino.stem + "_tmp.npz")
    np.savez(tmp, ztr=ztr, t=time.time() - t0, **{f"zte_{n}": v for n, v in zip(testes, zs)})
    os.replace(tmp, destino)
    return chave, f"{time.time() - t0:.0f}s"

log(f"inicio | {VERSAO} | {len(sel)} divisões | paralelo {a.paralelo} | {uso()}")
with ThreadPoolExecutor(max_workers=max(1, a.paralelo)) as ex:
    futuros = {ex.submit(processar, c): c for c in sel}
    for f in as_completed(futuros):
        try:
            c, st = f.result(); log(f"OK {c} ({st}) | {uso()}")
        except Exception as e:
            log(f"ERRO {futuros[f]}: {type(e).__name__}: {str(e)[:300]}")
log("fim")
