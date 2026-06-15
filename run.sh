#!/usr/bin/env bash
# Launch the Pendulum Decay Analyzer. Creates the venv on first run.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run: setting up virtual environment…"
  if python3 -m venv .venv 2>/dev/null; then
    :
  else
    # Fall back when python3-venv's ensurepip is unavailable.
    python3 -m venv .venv --without-pip
    . .venv/bin/activate
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
    deactivate
  fi
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi

exec python server.py "$@"
