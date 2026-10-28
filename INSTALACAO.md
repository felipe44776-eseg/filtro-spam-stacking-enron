# Instalação

Três caminhos, do mais simples ao mais completo. Todos usam **Python 3.11**: outras versões não foram testadas, e o `pyproject.toml` recusa a instalação nelas.

| Caminho | Para quê | Tempo |
|---|---|---|
| A. Google Colab | abrir e rodar o notebook-história sem instalar nada | cerca de 5 min |
| B. Windows (PowerShell) | tudo: testes, notebooks, demo | cerca de 10 min |
| C. macOS / Linux | tudo: testes, notebooks, demo | cerca de 10 min |

---

## A. Google Colab (sem instalar nada)

1. Abra https://colab.research.google.com e escolha **Arquivo → Abrir notebook → GitHub**.
2. Cole `felipe44776-eseg/filtro-spam-stacking-enron` e abra `Projeto3_Classificacao_Spam.ipynb`.
3. Siga o quadro "No Google Colab" no início do notebook: crie uma célula com os três comandos de instalação, rode-a e depois use **Ambiente de execução → Executar tudo**.

O Colab costuma ter outra versão do Python. Se a instalação reclamar da versão, use o caminho B ou C.

---

## B. Windows (PowerShell)

```powershell
# 1. Python 3.11: https://www.python.org/downloads/ (marque "Add python.exe to PATH")
py -3.11 --version

# 2. Baixar o projeto
git clone https://github.com/felipe44776-eseg/filtro-spam-stacking-enron.git
cd filtro-spam-stacking-enron

# 3. Ambiente isolado
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 4. Dependências (cerca de 5 min) e o pacote do projeto
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps

# 5. Conferir
pytest
```

Resultado esperado no passo 5: `31 passed, 10 deselected` em menos de 1 minuto.

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

Os modelos finais (`resultados/modelos/*.joblib`) têm dezenas de MB e ficam fora do repositório. Há dois jeitos de obtê-los:
- **baixar** da página *Releases* do GitHub e colocar em `resultados/modelos/`;
- **treinar de novo** com `python scripts/09_modelos_finais.py` (cerca de 5 min).

A demo e o notebook-história precisam deles.

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
| `pip` instala, mas `import spamfilter` falha | Faltou `pip install -e . --no-deps` com o ambiente ativado |
| Erro de versão do Python | Use 3.11. No Windows, `py -0` lista as versões instaladas |
| Execução lenta ou "MemoryError" | Feche o navegador. O motor escolhe sozinho de 1 a 4 processos conforme a memória livre, mas abaixo de 2 GB tudo fica lento |
| Antivírus deixa a instalação muito lenta (Windows) | Ele examina cada arquivo do `.venv`. Espere, ou adicione a pasta do projeto às exclusões |
| `Activate.ps1` bloqueado | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| A demo diz "filtros ainda não foram treinados" | Veja "Os filtros treinados", acima |
