[CmdletBinding()]
param(
    [switch]$Modeling
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$results = [System.Collections.Generic.List[object]]::new()

function Show-Summary {
    Write-Host ""
    Write-Host "Verification summary"
    Write-Host "--------------------"
    foreach ($result in $results) {
        Write-Host ("{0,-32} {1}" -f $result.Name, $result.Status)
    }
}

function Invoke-Native {
    $command = [string]$args[0]
    $arguments = [string[]]$args[1..($args.Count - 1)]

    & $command @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $command $($arguments -join ' ')"
    }
}

function Invoke-Gate {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Name"
    try {
        & $Action
        $results.Add([pscustomobject]@{ Name = $Name; Status = "PASS" })
    }
    catch {
        $results.Add([pscustomobject]@{ Name = $Name; Status = "FAIL" })
        Show-Summary
        Write-Error $_
        exit 1
    }
}

Push-Location $repoRoot
try {
    Invoke-Gate "core: create Python 3.12 env" {
        Push-Location core
        try { Invoke-Native uv venv --python 3.12 --clear }
        finally { Pop-Location }
    }
    Invoke-Gate "core: install engine dependencies" {
        Push-Location core
        try { Invoke-Native uv pip install -e ".[dev,engine]" }
        finally { Pop-Location }
    }
    Invoke-Gate "core: KV + kernel tests" {
        Push-Location core
        try { Invoke-Native uv run --no-sync pytest -q tests/kv tests/kernels }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: fmt" {
        Push-Location scheduler
        try { Invoke-Native cargo fmt --check }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: clippy" {
        Push-Location scheduler
        try { Invoke-Native cargo clippy --all-targets "--" -D warnings }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: test" {
        Push-Location scheduler
        try { Invoke-Native cargo test }
        finally { Pop-Location }
    }
    Invoke-Gate "scheduler: python feature" {
        Push-Location scheduler
        try { Invoke-Native cargo check --features python }
        finally { Pop-Location }
    }
    Invoke-Gate "real engine: build + install scheduler wheel" {
        $wheelDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("autotree-wheel-" + [System.Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $wheelDirectory | Out-Null
        $previousVirtualEnv = $env:VIRTUAL_ENV
        try {
            $env:VIRTUAL_ENV = Join-Path $repoRoot "core/.venv"
            Invoke-Native uv pip install -e "core[dev,engine]" -e "serve[dev]"
            Invoke-Native uv pip install maturin
            Invoke-Native uv run --active maturin build -m scheduler/Cargo.toml --features python --release --out $wheelDirectory
            $wheels = @(Get-ChildItem -LiteralPath $wheelDirectory -Filter "*.whl" -File)
            if ($wheels.Count -ne 1) {
                throw "Expected exactly one scheduler wheel in $wheelDirectory; found $($wheels.Count)."
            }
            Invoke-Native uv pip install $wheels[0].FullName
        }
        finally {
            $env:VIRTUAL_ENV = $previousVirtualEnv
            Remove-Item -LiteralPath $wheelDirectory -Recurse -Force
        }
    }
    Invoke-Gate "real engine: core engine tests" {
        Push-Location core
        try { Invoke-Native uv run --no-sync pytest -q tests/engine }
        finally { Pop-Location }
    }
    Invoke-Gate "real engine: serve TreeKV E2E" {
        $previousVirtualEnv = $env:VIRTUAL_ENV
        try {
            $env:VIRTUAL_ENV = Join-Path $repoRoot "core/.venv"
            Push-Location serve
            try { Invoke-Native uv run --active --no-sync pytest -q tests/test_treekv_e2e.py }
            finally { Pop-Location }
        }
        finally { $env:VIRTUAL_ENV = $previousVirtualEnv }
    }

    if (Test-Path -LiteralPath (Join-Path $repoRoot "serve")) {
        Invoke-Gate "serve: create Python 3.12 env" {
            Push-Location serve
            try { Invoke-Native uv venv --python 3.12 --clear }
            finally { Pop-Location }
        }
        Invoke-Gate "serve: install dev dependencies" {
            Push-Location serve
            try { Invoke-Native uv pip install -e ".[dev]" }
            finally { Pop-Location }
        }
        Invoke-Gate "serve: tests outside real-engine gate" {
            Push-Location serve
            try { Invoke-Native uv run --no-sync pytest -q --ignore=tests/test_treekv_e2e.py }
            finally { Pop-Location }
        }
    }
    else {
        Write-Host ""
        Write-Host "==> serve: tests"
        Write-Host "serve/ is absent; skipping the serve gate until that package merges."
        $results.Add([pscustomobject]@{ Name = "serve: tests"; Status = "SKIP (serve/ absent)" })
    }

    Invoke-Gate "sdk: create Python 3.12 env" {
        Push-Location sdk
        try { Invoke-Native uv venv --python 3.12 --clear }
        finally { Pop-Location }
    }
    Invoke-Gate "sdk: install test dependencies" {
        Push-Location sdk
        try {
            Invoke-Native uv pip install -e . -e ../serve "hypothesis>=6.100,<7" "pytest>=8.2,<9"
        }
        finally { Pop-Location }
    }
    Invoke-Gate "sdk: unit tests" {
        Push-Location sdk
        try { Invoke-Native uv run --no-sync pytest -q --ignore=tests/test_server_contract.py }
        finally { Pop-Location }
    }
    Invoke-Gate "wire: real serve + SDK contract" {
        Push-Location sdk
        try { Invoke-Native uv run --no-sync pytest -q tests/test_server_contract.py }
        finally { Pop-Location }
    }

    Invoke-Gate "thoughtbench: create Python 3.12 env" {
        Push-Location thoughtbench
        try { Invoke-Native uv venv --python 3.12 --clear }
        finally { Pop-Location }
    }
    Invoke-Gate "thoughtbench: install test dependencies" {
        Push-Location thoughtbench
        try {
            Invoke-Native uv pip install -e . -e ../sdk -e ../serve "httpx>=0.28,<1" "pytest>=8.3,<9" "uvicorn>=0.34,<1"
        }
        finally { Pop-Location }
    }
    Invoke-Gate "thoughtbench: tests" {
        Push-Location thoughtbench
        try { Invoke-Native uv run --no-sync pytest -q }
        finally { Pop-Location }
    }

    Invoke-Gate "workflow YAML parse" {
        $code = "import pathlib,yaml; files=[pathlib.Path('.github/workflows/ci.yml'),pathlib.Path('.github/workflows/gpu-parity.yml')]; [yaml.safe_load(path.read_text(encoding='utf-8')) for path in files]; print('Parsed workflow YAML:', ', '.join(map(str, files)))"
        Invoke-Native uv run --with pyyaml python -c $code
    }

    if ($Modeling) {
        Invoke-Gate "modeling: install dependencies" {
            Push-Location core
            try { Invoke-Native uv pip install -e ".[dev,engine,modeling]" }
            finally { Pop-Location }
        }
        Invoke-Gate "modeling: core tests" {
            Push-Location core
            try {
                $env:HF_HOME = Join-Path ([System.IO.Path]::GetTempPath()) "autotree-huggingface"
                Invoke-Native uv run --no-sync pytest -q tests
            }
            finally { Pop-Location }
        }
    }
    else {
        $results.Add([pscustomobject]@{ Name = "modeling: scheduled/manual"; Status = "SKIP (use -Modeling)" })
    }

    Show-Summary
}
finally {
    Pop-Location
}
