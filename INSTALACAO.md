# Instalação

Três caminhos, do mais simples ao mais completo.

| Caminho | Para quê | Python | Tempo |
|---|---|---|---|
| A. Google Colab | abrir e rodar qualquer notebook sem instalar nada | o do Colab (3.12) | cerca de 5 min na primeira vez |
| B. Windows (PowerShell) | tudo: testes, notebooks, demo | 3.11 ou 3.12 | cerca de 10 min |
| C. macOS / Linux | tudo: testes, notebooks, demo | 3.11 ou 3.12 | cerca de 10 min |

---

## A. Google Colab (sem instalar nada)

1. No [README](README.md#os-notebooks), clique em **Abrir no Colab** ao lado do notebook que quiser (o notebook-história é o `Projeto3_Classificacao_Spam.ipynb`).
   - Ou: em https://colab.research.google.com, **Arquivo → Abrir notebook → GitHub**, cole `felipe44776-eseg/filtro-spam-stacking-enron` e escolha o notebook.
2. Use **Ambiente de execução → Executar tudo**. A primeira célula de código:
   - baixa o projeto para `/content/filtro-spam-stacking-enron`;
   - instala os pacotes de `requirements-colab.txt` (as mesmas versões do projeto, sem mexer nos pacotes do Jupyter do Colab);
   - **reinicia o ambiente**, para carregar as versões novas.
3. Quando o Colab reconectar, use **Executar tudo** de novo. Da segunda vez em diante, a célula só entra na pasta do projeto.

Os filtros treinados que o notebook-história usa são baixados da Release v1.0 automaticamente.

---

## B. Windows (PowerShell)

```powershell
# 1. Python 3.11 ou 3.12: https://www.python.org/downloads/ (marque "Add python.exe to PATH")
py -3.11 --version

# 2. Baixar o projeto numa pasta de caminho curto (ex.: C:\projetos), por causa do limite de 260 caracteres do Windows
git clone https://github.com/felipe44776-eseg/filtro-spam-stacking-enron.git
cd filtro-spam-stacking-enron

# 3. Ambiente isolado
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 4. Dependências (cerca de 5 min, inclui o Jupyter) e o pacote do projeto
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps

# 5. Conferir
pytest
```

Resultado esperado no passo 5: `42 passed, 3 skipped, 11 deselected` em menos de 1 minuto. Os 3 testes pulados conferem os filtros treinados, que ainda não foram baixados; depois do passo 3 do [COMO_TESTAR.md](COMO_TESTAR.md) (ou de `python -c "from spamfilter import publicados; [publicados.garantir(n) for n in publicados.SHA256]"`), passam a ser `45 passed`.

Para abrir os notebooks: `jupyter notebook` (o Jupyter já vem do `requirements.txt`, no mesmo ambiente, e enxerga o pacote `spamfilter`).

---

## C. macOS / Linux

```bash
python3.11 --version
git clone https://github.com/felipe44776-eseg/filtro-spam-stacking-enron.git
cd filtro-spam-stacking-enron
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

No macOS, o LightGBM e o XGBoost precisam do OpenMP: `brew install libomp`.

---

## Os filtros treinados

Os modelos finais (`resultados/modelos/*.joblib`) têm dezenas de MB e ficam fora do repositório, na página *Releases* do GitHub. Há três jeitos de obtê-los:
- **automático:** o notebook-história e a demo baixam o que faltar e conferem o sha256 (`spamfilter.publicados`);
- **manual:** baixe da página *Releases* e coloque em `resultados/modelos/`;
- **treinar de novo:** `python scripts/09_modelos_finais.py` (cerca de 5 min).

---

## Opcional: TabPFN

O TabPFN (Prior Labs) é usado nos experimentos 05, 07, 08 e 09. **Nada disso é necessário para rodar o projeto**: as notas que ele gerou estão gravadas em `experimentos/05_tabpfn/` e `resultados/cache_tabpfn/`. Para refazer essas chamadas:

1. Crie uma conta em https://platform.priorlabs.ai, aceite a licença (uso não comercial) e gere uma chave.
2. Guarde a chave **só** numa variável de ambiente, nunca num arquivo:
   - Windows: `[Environment]::SetEnvironmentVariable("TABPFN_TOKEN", "<sua chave>", "User")`
   - macOS/Linux: `export TABPFN_TOKEN=<sua chave>`
3. **Nuvem**, que é rápida: crie um ambiente separado, porque o cliente exige outras versões de pandas e scikit-learn.
   ```powershell
   py -3.11 -m venv .venv-tabpfn-nuvem
   .venv-tabpfn-nuvem\Scripts\pip install -r requirements-tabpfn-nuvem.txt
   ```
4. **Local, em CPU**, que é lenta (cerca de 3 h para 5 divisões): `pip install -r requirements-tabpfn-local.txt` no ambiente principal.

---

## Problemas comuns

| Sintoma | Causa provável e solução |
|---|---|
| `pip` falha com "Long Path" ou "No such file or directory" (Windows) | O caminho da pasta ficou longo demais: o Windows limita caminhos a 260 caracteres, e alguns pacotes têm pastas profundas. Clone numa pasta curta, como `C:\projetos\`, ou ative caminhos longos: https://pip.pypa.io/warnings/enable-long-paths |
| `pip` instala, mas `import spamfilter` falha | Faltou `pip install -e . --no-deps` com o ambiente ativado, ou o Jupyter aberto é outro (de fora do `.venv`) |
| No Colab, "ModuleNotFoundError" depois de instalar | O ambiente não reiniciou: use *Ambiente de execução → Reiniciar sessão* e *Executar tudo* de novo |
| Erro de versão do Python | Use 3.11 ou 3.12. No Windows, `py -0` lista as versões instaladas |
| Execução lenta ou "MemoryError" | Feche o navegador. O motor escolhe sozinho de 1 a 4 processos conforme a memória livre, mas abaixo de 2 GB tudo fica lento |
| Antivírus deixa a instalação muito lenta (Windows) | Ele examina cada arquivo do `.venv`. Espere, ou adicione a pasta do projeto às exclusões |
| `Activate.ps1` bloqueado | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| A demo ou o notebook não acham os filtros e não há internet | Veja "Os filtros treinados", acima |
