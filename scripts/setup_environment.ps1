param(
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $WorkspaceRoot "MEIDNet-main"

if (-not $VenvPath) {
    $VenvPath = Join-Path $WorkspaceRoot ".venv"
}
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Invoke-CheckedPython {
    param([string[]]$PythonArguments)

    & $VenvPython @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($PythonArguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BasePython = $null
    foreach ($Version in @("3.12", "3.11", "3.10")) {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & py "-$Version" -c "import sys; print(sys.executable)" *> $null
        $ProbeExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousPreference
        if ($ProbeExitCode -eq 0) {
            $BasePython = @("py", "-$Version")
            break
        }
    }
    if (-not $BasePython) {
        throw "No usable Python 3.10-3.12 interpreter was found."
    }

    Write-Host "Creating virtual environment at $VenvPath"
    & $BasePython[0] $BasePython[1] -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual-environment creation failed."
    }
}

Invoke-CheckedPython -PythonArguments @(
    "-c",
    "import sys; assert sys.version_info[:2] in [(3, 10), (3, 11), (3, 12)], sys.version"
)
Invoke-CheckedPython -PythonArguments @("-m", "pip", "install", "--upgrade", "pip>=25,<27")
Invoke-CheckedPython -PythonArguments @(
    "-m", "pip", "install", "torch==2.10.0",
    "--index-url", "https://download.pytorch.org/whl/cpu"
)

$LockFile = Join-Path $WorkspaceRoot "requirements-lock.txt"
$RequirementFile = if (Test-Path -LiteralPath $LockFile) {
    $LockFile
} else {
    Join-Path $WorkspaceRoot "requirements-setup.txt"
}
Invoke-CheckedPython -PythonArguments @("-m", "pip", "install", "-r", $RequirementFile)

# Do not install editable metadata into MEIDNet-main: that directory is kept
# byte-for-byte identical to the official checkout. Validation scripts add the
# source directory to sys.path without writing bytecode there.
$env:PYTHONDONTWRITEBYTECODE = "1"
Invoke-CheckedPython -PythonArguments @("-m", "pip", "check")
Invoke-CheckedPython -PythonArguments @(
    (Join-Path $PSScriptRoot "check_environment.py"),
    "--source-root", $SourceRoot
)

Write-Host "Environment ready. Interpreter: $VenvPython"
