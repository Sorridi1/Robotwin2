#!/usr/bin/env bash
set -euo pipefail
FPOPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" "${FPOPP_DIR}/train.py" "$@"
