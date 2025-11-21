#!/usr/bin/env python3
"""
Script para executar todos os serviços do sistema simultaneamente
Copied into deploy for containerized run.
"""

import subprocess
import time
import signal
import sys
import os

def run_service(name, command, port, color_code):
    """Executa um serviço em subprocess"""
    print(f"\033[{color_code}m🚀 Iniciando {name} na porta {port}...\033[0m")
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def main():
    """Executa todos os serviços"""
    print("=" * 60)
    print("🎯 SISTEMA DE VOOS - INICIANDO TODOS OS SERVIÇOS")
    print("=" * 60)
    
    # Mudar para o diretório correto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    processes = []
    names = []
    try:
        # Auth Service (porta 8001)
        auth_process = run_service(
            "Auth Service",
            "uvicorn src.auth.service:app --host 0.0.0.0 --port 8001",
            8001,
            "32"  # Verde
        )
        processes.append(auth_process)
        names.append("Auth Service")
        time.sleep(2)
        
        # Flight Service (porta 8002)  
        flight_process = run_service(
            "Flight Service",
            "uvicorn src.flights.service:app --host 0.0.0.0 --port 8002",
            8002,
            "34"  # Azul
        )
        processes.append(flight_process)
        names.append("Flight Service")
        time.sleep(2)
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS SERVIÇOS INICIADOS COM SUCESSO!")
        print("=" * 60)
        print("\n📡 URLs dos serviços:")
        print("   🔐 Auth Service:   http://localhost:8001")
        print("   ✈️  Flight Service: http://localhost:8002")
        
        print("\n⚠️  Pressione Ctrl+C para parar todos os serviços")
        print("-" * 60)
        
        # Monitorar processos e exibir logs
        while True:
            for i, process in enumerate(processes):
                name = names[i]
                # Print new lines from stdout
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if line == '':
                            break
                        print(f"[{name}] {line}", end='')
                if process.poll() is not None:
                    print(f"\033[31m❌ {name} parou inesperadamente!\033[0m")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Recebido sinal de interrupção. Parando serviços...")
        # Parar todos os processos
        for i, process in enumerate(processes):
            name = names[i]
            print(f"🔴 Parando {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"✅ {name} parado com sucesso")
            except subprocess.TimeoutExpired:
                print(f"⏰ {name} não respondeu, forçando parada...")
                process.kill()
                process.wait()
                print(f"🔪 {name} terminado forçadamente")
        print("\n✅ Todos os serviços foram parados com sucesso!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        # Parar processos em caso de erro
        for i, process in enumerate(processes):
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
        sys.exit(1)

if __name__ == "__main__":
    main()
