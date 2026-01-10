param(
  [ValidateSet("ingest","index","api","test","all")]
  [string]$Step = "all"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

if ($Step -eq "ingest" -or $Step -eq "all") {
  python -m src.legalrag.ingest
}

if ($Step -eq "index" -or $Step -eq "all") {
  python -m src.legalrag.index
}

if ($Step -eq "test" -or $Step -eq "all") {
  pytest -q
}

if ($Step -eq "api" -or $Step -eq "all") {
  uvicorn src.legalrag.api.app:app --host 127.0.0.1 --port 8000
}
