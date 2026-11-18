<#
.SYNOPSIS
    Baixa e configura automaticamente os binários CUDA do llama.cpp (ou compila fork TurboQuant)
    diretamente em engine_bin\ sem poluir o disco C:.
#>
param(
    [switch]$BuildTurboQuantFromSource = $false
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $RootDir "engine_bin"
$TmpDir = Join-Path $RootDir ".cache\tmp"

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

if ($BuildTurboQuantFromSource) {
    Write-Host "🔧 Compilando fork llama-cpp-turboquant com suporte nativo CUDA (-DGGML_CUDA=ON)..." -ForegroundColor Cyan
    $SrcDir = Join-Path $TmpDir "llama-cpp-turboquant"
    if (-not (Test-Path $SrcDir)) {
        git clone --depth 1 https://github.com/TheTom/llama-cpp-turboquant.git $SrcDir
    }
    Push-Location $SrcDir
    try {
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release -j 6
        Copy-Item -Path "build\bin\Release\*" -Destination $BinDir -Recurse -Force
        Write-Host "✅ Compilação CUDA + TurboQuant concluída em $BinDir!" -ForegroundColor Green
    } finally {
        Pop-Location
    }
    exit 0
}

Write-Host "⚡ Consultando última release oficial CUDA 12.4 (Windows x64) do ggml-org/llama.cpp..." -ForegroundColor Cyan
$ReleaseApi = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
$Release = Invoke-RestMethod -Uri $ReleaseApi -Headers @{ "User-Agent" = "New-Gen-Local-AI" }

$CudaAsset = $Release.assets | Where-Object { $_.name -match "bin-win-cuda-12.*-x64\.zip$" } | Select-Object -First 1
$CuArtAsset = $Release.assets | Where-Object { $_.name -match "cudart-llama-bin-win-cuda-12.*-x64\.zip$" } | Select-Object -First 1

if (-not $CudaAsset) {
    $CudaAsset = $Release.assets | Where-Object { $_.name -match "bin-win-vulkan-x64\.zip$" } | Select-Object -First 1
}

if (-not $CudaAsset) {
    throw "Nenhum pacote binário compatível encontrado na última release do llama.cpp."
}

if ($CuArtAsset) {
    $CuZip = Join-Path $TmpDir $CuArtAsset.name
    Write-Host "⬇ Baixando runtime CUDA ($($CuArtAsset.name))..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $CuArtAsset.browser_download_url -OutFile $CuZip
    Expand-Archive -Path $CuZip -DestinationPath $BinDir -Force
}

$EngineZip = Join-Path $TmpDir $CudaAsset.name
Write-Host "⬇ Baixando motor llama-server ($($CudaAsset.name))..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $CudaAsset.browser_download_url -OutFile $EngineZip
Expand-Archive -Path $EngineZip -DestinationPath $BinDir -Force

Write-Host "✅ Motor instalado em: $BinDir" -ForegroundColor Green
Write-Host "💡 Dica: Para ativar kernels experimentais 'turbo3' nativos, rode: .\scripts\setup_engine.ps1 -BuildTurboQuantFromSource (requer CUDA Toolkit + CMake)." -ForegroundColor Gray
