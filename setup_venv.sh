#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv cgn_visualize_venv
cgn_visualize_venv/bin/python -m pip install --upgrade pip
cgn_visualize_venv/bin/python -m pip install 'numpy>=1.24,<2' open3d matplotlib pillow

printf '%s\n' 'Run: cgn_visualize_venv/bin/python -m p1.visualize_predictions'
