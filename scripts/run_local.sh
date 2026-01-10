@'
#!/usr/bin/env bash
set -euo pipefail

STEP="${1:-all}"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if [[ "$STEP" == "ingest" || "$STEP" == "all" ]]; then
  python -m src.legalrag.ingest
fi

if [[ "$STEP" == "index" || "$STEP" == "all" ]]; then
  python -m src.legalrag.index
fi

if [[ "$STEP" == "test" || "$STEP" == "all" ]]; then
  pytest -q
fi

if [[ "$STEP" == "api" || "$STEP" == "all" ]]; then
  uvicorn src.legalrag.api.app:app --host 127.0.0.1 --port 8000
fi
'@ | Set-Content -Encoding UTF8 scripts\run_local.sh
