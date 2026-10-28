# Rodada noturna: TabPFN-3.5-Fast LOCAL (CPU) nas 5 divisoes da semente 201, depois o notebook de analise.
# Antes de rodar: feche o navegador, o Claude Code e outros programas pesados. Tempo estimado: 6 a 7 h.
# O script pede ao Windows para nao suspender enquanto roda. Se cair, rode de novo: retoma do cache.
#
#   pwsh -ExecutionPolicy Bypass -File experimentos\rodar_noite_tabpfn.ps1   (a partir da raiz do projeto)

$raiz = Split-Path $PSScriptRoot -Parent
$py = Join-Path $raiz ".venv\Scripts\python.exe"
$env:TABPFN_TOKEN = [Environment]::GetEnvironmentVariable("TABPFN_TOKEN", "User")
$env:TABPFN_NO_BROWSER = "1"
if (-not $env:TABPFN_TOKEN) { Write-Host "TABPFN_TOKEN nao encontrado nas variaveis do usuario" -ForegroundColor Red; exit 1 }

$nb = Join-Path $PSScriptRoot "05_tabpfn\05_tabpfn_stacking.ipynb"
Write-Host "Inicio: $(Get-Date -Format 'dd/MM HH:mm'). Progresso em experimentos\05_tabpfn\progresso_local.log" -ForegroundColor Cyan
# rede de seguranca: se a base externa da nuvem ainda nao estiver no cache, refaz (cerca de 15 min)
if (-not (Test-Path (Join-Path $PSScriptRoot "05_tabpfn\cache_tabpfn\nuvem\externo.npz"))) {
    & (Join-Path $raiz ".venv-tabpfn-nuvem\Scripts\python.exe") (Join-Path $PSScriptRoot "scripts\tabpfn_notas.py") --modo nuvem --chaves externo
}
& $py (Join-Path $PSScriptRoot "scripts\tabpfn_notas.py") --modo local --chaves c201
Write-Host "Notas locais prontas: $(Get-Date -Format 'dd/MM HH:mm'). Gerando e executando o notebook de analise..." -ForegroundColor Cyan
& $py (Join-Path $PSScriptRoot "scripts\build_tabpfn.py") $nb
& $py (Join-Path $PSScriptRoot "scripts\run_nb.py") $nb
Write-Host "Fim: $(Get-Date -Format 'dd/MM HH:mm'). Notebook: $nb" -ForegroundColor Green
