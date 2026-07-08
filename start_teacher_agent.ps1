param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [switch]$InstallDeps,
    [switch]$NoBrowser,
    [switch]$ShowLogs,
    [switch]$DisablePedagogicalRag,
    [string]$LogRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $RepoRoot "data\runtime"
$PidFile = Join-Path $RuntimeDir "teacher_agent_pids.json"
$ApiUrl = "http://127.0.0.1:$ApiPort"
$UiUrl = "http://127.0.0.1:$UiPort"
$PedagogicalRagValue = if ($DisablePedagogicalRag) { "false" } else { "true" }
$SessionId = Get-Date -Format "yyyyMMdd-HHmmss"
$ResolvedLogRoot = if ($LogRoot) { $LogRoot } else { Join-Path $RepoRoot "logs\sessions" }
$SessionLogDir = Join-Path $ResolvedLogRoot $SessionId
$TurnLogPath = Join-Path $SessionLogDir "turn_logs.jsonl"

function Test-PythonImport {
    param([string]$Module)
    & python -c "import $Module" *> $null
    return $LASTEXITCODE -eq 0
}

function Test-PortOpen {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $connection
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $SessionLogDir | Out-Null

Write-Host "Teacher Agent launcher" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
Write-Host "Session log dir: $SessionLogDir"

if ($InstallDeps) {
    Write-Host "Installing project dependencies..." -ForegroundColor Yellow
    & python -m pip install -e ".[dev,ui]"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

$missing = @()
foreach ($module in @("fastapi", "uvicorn", "streamlit", "langgraph", "pydantic")) {
    if (-not (Test-PythonImport $module)) {
        $missing += $module
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing Python modules: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "Run this once:"
    Write-Host "  .\start_teacher_agent.ps1 -InstallDeps"
    exit 1
}

if (Test-PortOpen $ApiPort) {
    Write-Host "API port $ApiPort is already in use; assuming backend is running." -ForegroundColor Yellow
    Write-Host "Existing backend keeps its original TEACHER_LOG_DIR; restart it to use this new session log." -ForegroundColor Yellow
    $apiProcess = $null
} else {
    Write-Host "Starting FastAPI backend on $ApiUrl ..."
    $apiArgs = "-NoExit -Command `"cd '$RepoRoot'; `$env:USE_PEDAGOGICAL_RAG='$PedagogicalRagValue'; `$env:TEACHER_LOG_DIR='$SessionLogDir'; python -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port $ApiPort`""
    if ($ShowLogs) {
        $apiProcess = Start-Process powershell -ArgumentList $apiArgs -PassThru
    } else {
        $apiProcess = Start-Process powershell -ArgumentList $apiArgs -WindowStyle Hidden -PassThru
    }
}

Start-Sleep -Seconds 2

if (Test-PortOpen $UiPort) {
    Write-Host "UI port $UiPort is already in use; assuming Streamlit is running." -ForegroundColor Yellow
    $uiProcess = $null
} else {
    Write-Host "Starting Streamlit UI on $UiUrl ..."
    $uiArgs = "-NoExit -Command `"cd '$RepoRoot'; `$env:TEACHER_API_URL='$ApiUrl'; python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port $UiPort`""
    if ($ShowLogs) {
        $uiProcess = Start-Process powershell -ArgumentList $uiArgs -PassThru
    } else {
        $uiProcess = Start-Process powershell -ArgumentList $uiArgs -WindowStyle Hidden -PassThru
    }
}

$pids = [ordered]@{
    api_port = $ApiPort
    ui_port = $UiPort
    api_url = $ApiUrl
    ui_url = $UiUrl
    api_pid = if ($apiProcess) { $apiProcess.Id } else { $null }
    ui_pid = if ($uiProcess) { $uiProcess.Id } else { $null }
    use_pedagogical_rag = (-not $DisablePedagogicalRag)
    session_id = $SessionId
    session_log_dir = $SessionLogDir
    turn_log_path = $TurnLogPath
    started_at = (Get-Date).ToString("o")
}
$pids | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8

Write-Host ""
Write-Host "Teacher Agent is starting." -ForegroundColor Green
Write-Host "API: $ApiUrl/health"
Write-Host "UI : $UiUrl"
Write-Host "This session log: $TurnLogPath"
Write-Host "PID file: $PidFile"
Write-Host ""
Write-Host "Try in the UI: 老师 3x+6=10 怎么解？"

if (-not $NoBrowser) {
    Start-Process $UiUrl
}
