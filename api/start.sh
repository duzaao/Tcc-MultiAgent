#!/bin/sh
set -e

# Start Auth and Flights API services
cd /app
python3 scripts/run_services.py
