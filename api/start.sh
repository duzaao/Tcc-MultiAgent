#!/bin/sh
set -e

# Start the API processes (auth + flights) using the provided run_services script
cd /app
python3 scripts/run_services.py
