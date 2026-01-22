param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Title"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Title"
    }
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipBackend) {
    Invoke-Checked "Backend: ruff" { python -m ruff check backend }
    Invoke-Checked "Backend: bandit" { python -m bandit -r backend -x backend/tests }
    Invoke-Checked "Backend: pytest + coverage" {
        Push-Location backend
        try {
            python -m pytest -q --cov=. --cov-report=term-missing
        } finally {
            Pop-Location
        }
    }
}

if (-not $SkipFrontend) {
    Invoke-Checked "Frontend: lint" {
        Push-Location frontend
        try {
            npm run lint
        } finally {
            Pop-Location
        }
    }
    Invoke-Checked "Frontend: build" {
        Push-Location frontend
        try {
            npm run build
        } finally {
            Pop-Location
        }
    }
}

Write-Host ""
Write-Host "OK"

