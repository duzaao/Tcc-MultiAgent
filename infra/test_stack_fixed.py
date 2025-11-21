#!/usr/bin/env python3
"""Versão corrigida do test_stack; roda checagens e tenta o login via /message.

Uso: python3 test_stack_fixed.py

Observação: este script usa docker compose na pasta deploy/infra para coletar logs/environ
em caso de erro, então rode-o a partir da raiz do repositório (ou ajuste COMPOSE_DIR).
"""
import os
import time
import json
import subprocess
import requests
from requests.exceptions import RequestException

COMPOSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
COMPOSE_FILE = os.path.join(COMPOSE_DIR, 'docker-compose.yml')
AGENT_URL = os.getenv('AGENT_URL', 'http://localhost:8000')
API_URLS = [os.getenv('API_URL1', 'http://localhost:8001'), os.getenv('API_URL2', 'http://localhost:8002')]
MCP_URL = os.getenv('MCP_URL', 'http://localhost:8003')

TIMEOUT = 60


def run_cmd(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
        return out
    except subprocess.CalledProcessError as e:
        return e.output


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


def dump_agent_diagnostics():
    print('\n--- Agent container env (docker compose) ---')
    out_env = run_cmd(f"docker compose -f {COMPOSE_FILE} exec -T agent env", cwd=COMPOSE_DIR)
    print(out_env)

    print('\n--- Agent recent logs (docker compose, tail 500) ---')
    out_logs = run_cmd(f"docker compose -f {COMPOSE_FILE} logs --no-color --tail=500 agent", cwd=COMPOSE_DIR)
    print(out_logs)

    print('\n--- Agent process list inside container ---')
    out_ps = run_cmd(f"docker compose -f {COMPOSE_FILE} exec -T agent ps aux", cwd=COMPOSE_DIR)
    print(out_ps)

    # If .env exists inside container, print it
    print('\n--- /app/.env inside agent container (if exists) ---')
    out_dotenv = run_cmd(f"docker compose -f {COMPOSE_FILE} exec -T agent bash -lc 'cat /app/.env 2>/dev/null || true'", cwd=COMPOSE_DIR)
    print(out_dotenv)


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
    agent_health = AGENT_URL.rstrip('/') + '/health'
    ok, resp = wait_for(agent_health, timeout=15)
    print(f' - Agent {agent_health} ->', 'OK' if ok else 'NO')

    # Enviar mensagem de login
    login_phrase = 'i wanna login. aloha and password eduardo1234'
    payload = {'message': login_phrase, 'confirm_sensitive': False}
    post_url = AGENT_URL.rstrip('/') + '/message'
    print(f"\n4) Enviando frase de login para agente: {post_url}")
    try:
        r = requests.post(post_url, json=payload, timeout=20)
    except RequestException as e:
        print('Erro ao postar no agent:', e)
        dump_agent_diagnostics()
        return 2

    if r.status_code == 404:
        print('\nAgent retornou 404 Not Found para /message. Coletando logs e env do container...')
        dump_agent_diagnostics()
        print('\nResposta HTTP 404 body:\n', r.text)
        return 5

    try:
        data = r.json()
    except Exception:
        print('Resposta não-JSON do agent:', r.status_code, r.text)
        dump_agent_diagnostics()
        return 3

    print('\nResposta do agent (raw):')
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # Heurística de sucesso: agent devolve 'reply' sem 'error'
    if isinstance(data, dict):
        if 'error' in data:
            print('\nAgent returned error:', data.get('error'))
            return 4
        if 'reply' in data:
            print('\nLogin attempt reply:')
            print(data.get('reply'))
            print('\nTeste concluído com sucesso (reply recebido).')
            return 0

    print('\nNão foi possível detectar login bem-sucedido automaticamente. Verifique logs do agent/mcp.')
    return 6


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
