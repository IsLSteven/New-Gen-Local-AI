<#
.SYNOPSIS
    Instala automaticamente os binários pré-compilados do llama.cpp (CUDA / Vulkan / CPU)
    ou compila opcionalmente o fork experimental llama-cpp-turboquant a partir do código-fonte.
#>
param(
    [ValidateSet("auto", "cuda", "vulkan", "cpu")]
    [string]$Backend = "auto",
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

& (Join-Path $RootDir "nglai.bat") engine install --backend $Backend
