#!/usr/bin/env python3
"""Test script para validar a stack dockerizada e tentar login via agent.

Uso: python3 test_stack.py

O script faz:
- checa /openapi.json da API (8001/8002)
- checa / da MCP (8003) e /health do agent (8000)
- envia POST /message ao agent com a frase de login solicitada
"""
import os
import time
import requests
from requests.exceptions import RequestException

AGENT_URL = os.getenv('AGENT_URL', 'http://localhost:8000')
API_URLS = [os.getenv('API_URL1', 'http://localhost:8001'), os.getenv('API_URL2', 'http://localhost:8002')]
MCP_URL = os.getenv('MCP_URL', 'http://localhost:8003')

TIMEOUT = 60


def wait_for(url, timeout=TIMEOUT, interval=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                import os
                import time
                import json
                import subprocess
                import requests
                from requests.exceptions import RequestException

                AGENT_URL = os.getenv('AGENT_URL', 'http://localhost:8000')
                API_URLS = [os.getenv('API_URL1', 'http://localhost:8001'), os.getenv('API_URL2', 'http://localhost:8002')]
                MCP_URL = os.getenv('MCP_URL', 'http://localhost:8003')
                COMPOSE_FILE = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')

                TIMEOUT = 60


                def wait_for(url, timeout=TIMEOUT, interval=2):
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        try:
                            r = requests.get(url, timeout=5)
                            if r.status_code == 200:
                                return True, r
                        except RequestException:
                            pass
                        time.sleep(interval)
                    return False, None


                def run_cmd(cmd):
                    try:
                        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                        return out
                    except subprocess.CalledProcessError as e:
                        return e.output


                def print_docker_agent_info():
                    print('\n--- Agent container env (docker compose) ---')
                    cmd_env = f"docker compose -f {COMPOSE_FILE} exec -T agent env"
                    out_env = run_cmd(cmd_env)
                    print(out_env)

                    print('\n--- Agent recent logs (docker compose) ---')
                    cmd_logs = f"docker compose -f {COMPOSE_FILE} logs --no-color --tail=300 agent"
                    out_logs = run_cmd(cmd_logs)
                    print(out_logs)


                def main():
                    print('1) Checando APIs (openapi)...')
                    ok_any = False
                    for u in API_URLS:
                        url = u.rstrip('/') + '/openapi.json'
                        print(f' - testando {url} ...', end=' ')
                        ok, resp = wait_for(url, timeout=15)
                        if ok:
                            print('OK')
                            ok_any = True
                        else:
                            print('NO')
                    if not ok_any:
                        print('Nenhuma das APIs respondeu /openapi.json. Continuando diagnóstico...')

                    print('\n2) Checando MCP (root /)...')
                    ok, resp = wait_for(MCP_URL, timeout=10)
                    print(f' - MCP {MCP_URL} ->', 'OK' if ok else 'NO (may be normal)')

                    print('\n3) Checando Agent /health...')
                    *** End Patch
                    ok, resp = wait_for(agent_health, timeout=15)
