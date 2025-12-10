#!/usr/bin/env python3
"""Analyzer que usa Groq (GROQ_API_KEY do .env) para determinar quais respostas estão ERRADAS.

Comportamento:
- Usa o provider `groq` via `agent.multi.llm.ensure_llm`.
- Para cada linha JSONL, pede ao LLM apenas `true` ou `false` (true = resposta satisfaz o prompt, false = resposta ERRADA).
- Considera negativas válidas (ex: "você não pode comprar esse ticket porque já tem ele") como CORRETAS — o prompt do verificador instrui isso.
- Calcula latências e tokens médios como antes; acurácia é calculada somente pela decisão do LLM.
- Imprime um resumo por modelo e lista os itens marcados como ERRADOS.

Uso:
  python3 analyze.py multi_eval2 --llm-model gpt-4o-mini

"""
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional, List, Any
import types


def _load_dotenv_if_any():
    # tenta python-dotenv, senão carregamento simples
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except Exception:
        pass

    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as ef:
                for ln in ef:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def classify_with_groq(prompt_text: str, reply: str, llm) -> Optional[bool]:
    """Pede ao LLM (Groq) para retornar estritamente 'true' ou 'false'.

    - True: resposta satisfaz o pedido do usuário (inclui negativos válidos)
    - False: resposta incorreta (ex: 'I could not match your request to any available tool.')
    - None: LLM não retornou algo interpretável
    """
    if llm is None:
        return None

    system_msg = (
        "Você é um verificador de acurácia. Receberá um prompt do usuário e a resposta do modelo."
        " Responda estritamente com 'true' se a resposta satisfaz corretamente o pedido do usuário, ou 'false' caso contrário."
        " NÃO forneça explicações. Considere respostas negativas que façam sentido (ex: 'você não pode comprar esse ticket porque já tem ele') como CORRETAS."
        " Considere como INCORRETAS respostas que indiquem falha de mapeamento ou incapacidade de executar, por exemplo: 'I could not match your request to any available tool.'"
    )

    user_msg = (
        f"Prompt do usuário:\n{prompt_text}\n\nResposta do modelo:\n{reply}\n\nRetorne apenas: true ou false"
    )

    msgs = [
        types.SimpleNamespace(type="system", content=system_msg),
        types.SimpleNamespace(type="user", content=user_msg),
    ]

    try:
        resp = llm.invoke(msgs)
        text = getattr(resp, "content", None)
        if text is None:
            text = str(resp)
        low = (text or "").lower()

        # procurar palavra inteira true/false
        if re.search(r"\btrue\b", low):
            return True
        if re.search(r"\bfalse\b", low):
            return False

        # fallback: extrair JSON se o modelo retornou um objeto
        parsed = _extract_json_object(text)
        if parsed is not None:
            c = parsed.get("correct")
            if isinstance(c, bool):
                return c
            if isinstance(c, str):
                if c.lower() in ("true", "yes", "sim"):
                    return True
                if c.lower() in ("false", "no", "nao", "não"):
                    return False

        return None
    except Exception:
        return None


def analyze_file(path: str, llm_model: Optional[str] = None):
    # stats por modelo
    stats = defaultdict(lambda: {
        "latencies_with_mcp": [],
        "latencies_without_mcp": [],
        "tokens_input": [],
        "tokens_output": [],
        "tokens_total": [],
        "accuracies": [],
        "count": 0,
    })

    wrong_items: List[dict] = []

    # carregar .env
    _load_dotenv_if_any()

    # importar fábrica LLM do projeto
    try:
        from agent.multi.llm import ensure_llm
    except Exception:
        print("Erro: não foi possível importar agent.multi.llm.ensure_llm. Assegure que o módulo exista.")
        sys.exit(1)

    # exigir chave GROQ_API_KEY
    if not os.environ.get("GROQ_API_KEY"):
        print("Erro: variável de ambiente GROQ_API_KEY não encontrada. Coloque a chave no .env ou no ambiente.")
        sys.exit(1)

    try:
        llm = ensure_llm(model=llm_model, provider="groq")
    except Exception as e:
        print("Erro ao instanciar LLM (groq):", e)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print("Linha inválida ignorada (não JSON):", line[:120])
                continue

            model = item.get("model") or "unknown"
            reply = item.get("reply", "")
            latency = item.get("latency_ms", 0)
            has_mcp = item.get("mcp_tool") is not None

            stats[model]["tokens_input"].append(item.get("tokens_input", 0))
            stats[model]["tokens_output"].append(item.get("tokens_output", 0))
            stats[model]["tokens_total"].append(item.get("tokens_total", 0))
            if has_mcp:
                stats[model]["latencies_with_mcp"].append(latency)
            else:
                stats[model]["latencies_without_mcp"].append(latency)

            prompt_text = item.get("prompt") or item.get("input") or ""

            decision = classify_with_groq(prompt_text, reply, llm)
            if decision is None:
                # tratar indecisão como incorreto (por pedido do usuário: quero saber o que está errado)
                acc = 0.0
                wrong_items.append({
                    "line": idx,
                    "id": item.get("id"),
                    "model": model,
                    "reason": "undecided",
                    "prompt": prompt_text,
                    "reply": reply,
                })
            elif decision is False:
                acc = 0.0
                wrong_items.append({
                    "line": idx,
                    "id": item.get("id"),
                    "model": model,
                    "reason": "llm_says_false",
                    "prompt": prompt_text,
                    "reply": reply,
                })
            else:
                acc = 1.0

            stats[model]["accuracies"].append(acc)
            stats[model]["count"] += 1

    # calcular médias
    results = {}
    for model, data in stats.items():
        n = data["count"]
        if n == 0:
            continue

        def avg(lst: List[float]):
            return sum(lst) / len(lst) if lst else 0.0

        results[model] = {
            "samples": n,
            "avg_latency_with_mcp": avg(data["latencies_with_mcp"]),
            "avg_latency_without_mcp": avg(data["latencies_without_mcp"]),
            "avg_tokens_input": avg(data["tokens_input"]),
            "avg_tokens_output": avg(data["tokens_output"]),
            "avg_tokens_total": avg(data["tokens_total"]),
            "accuracy": avg(data["accuracies"]),
        }

    return results, wrong_items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze LLM benchmark results using Groq to mark incorrect replies")
    parser.add_argument("file", help="Caminho do arquivo JSONL (ex: multi_eval2)")
    parser.add_argument("--llm-model", default=None, help="Modelo Groq a usar (ex: llama-3.1-8b-instant)")
    args = parser.parse_args()

    results, wrong = analyze_file(args.file, llm_model=args.llm_model)

    print("\n=== MÉDIAS POR MODELO ===\n")
    for model, m in results.items():
        print(f"Modelo: {model}")
        print(f"  Amostras:               {m['samples']}")
        print(f"  Latência c/ MCP:        {m['avg_latency_with_mcp']:.2f} ms")
        print(f"  Latência s/ MCP:        {m['avg_latency_without_mcp']:.2f} ms")
        print(f"  Tokens input médios:    {m['avg_tokens_input']:.2f}")
        print(f"  Tokens output médios:   {m['avg_tokens_output']:.2f}")
        print(f"  Tokens total médios:    {m['avg_tokens_total']:.2f}")
        print(f"  Acurácia (LLM):         {m['accuracy']:.2f}")
        print()

    print(f"=== ITENS MARCADOS COMO ERRADOS: {len(wrong)} ===\n")
    for w in wrong:
        # imprimir id e trecho de prompt/reply (cortar para legibilidade)
        pid = w.get("id")
        line = w.get("line")
        reason = w.get("reason")
        p = (w.get("prompt") or "").replace("\n", " ")[:200]
        r = (w.get("reply") or "").replace("\n", " ")[:300]
        print(f"line={line} id={pid} reason={reason}\n  prompt: {p}\n  reply: {r}\n")
import json
import re
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional
import types
import sys

# Tentar carregar .env se estiver presente
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # fallback simples: carregar arquivo .env na raiz do workspace se existir
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as ef:
                for ln in ef:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass

# Import LLM factory se disponível
try:
    from agent.multi.llm import ensure_llm
except Exception:
    ensure_llm = None


# ============================
# Palavras que indicam erro real (FAILURE)
# ============================
ERROR_KEYWORDS = [
    "não foi possível", "nao foi possivel", "erro", "error",
    "failed", "unauthorized", "invalid", "incorrect",
    "can't", "cannot", "exception", "fail", "timeout"
]

# Palavras que NÃO são erro (NEGATIVE BUT VALID)
VALID_NEGATIVE_KEYWORDS = [
    "não tem voos", "no flights", "no active", "sem voos",
    "nenhum voo encontrado", "no tickets", "zero tickets"
]


def is_error(reply: str) -> bool:
    """Determina se uma resposta realmente indica erro."""
    if not reply:
        return False

    text = reply.lower()

    # se contém algo que claramente NÃO é erro
    for kw in VALID_NEGATIVE_KEYWORDS:
        if kw in text:
            return False

    # se contém palavras típicas de erro → erro
    for kw in ERROR_KEYWORDS:
        if kw in text:
            return True

    return False


def _extract_json_object(text: str) -> Optional[dict]:
    """Extrai o primeiro objeto JSON bem formado do texto retornado pelo LLM."""
    if not text:
        return None
    # encontra primeiro '{'
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def classify_with_llm(user_prompt: str, reply: str, llm) -> Optional[bool]:
    """Usa um LLM para decidir se a `reply` é correta para o `user_prompt`.

    Retorna True se correto, False se incorreto, ou None se falhar (usar fallback depois).
    """
    if llm is None:
        return None

    # Prompt simples: responder apenas 'true' ou 'false' (sem explicações)
    system_msg = (
        "Você é um verificador de acurácia. Receberá um prompt do usuário e a resposta do modelo."
        " Responda estritamente com 'true' se a resposta satisfaz corretamente o pedido do usuário, ou 'false' caso contrário."
        " Não forneça explicações adicionais."
    )

    user_msg = (
        f"Prompt do usuário:\n{user_prompt}\n\nResposta do modelo:\n{reply}\n\n"
        "Retorne apenas: true ou false"
    )
    # Mensagens no estilo LangChain (SimpleNamespace com .type e .content)
    msgs = [
        types.SimpleNamespace(type="system", content=system_msg),
        types.SimpleNamespace(type="user", content=user_msg),
    ]

    try:
        resp = llm.invoke(msgs)
        text = getattr(resp, "content", None)
        if text is None:
            text = str(resp)

        # Esperamos um retorno simples: 'true' ou 'false'
        low = (text or "").lower()
        if re.search(r"\btrue\b|\btrue\.|\btrue\!|\bcorrect\b|\bsim\b", low):
            return True
        if re.search(r"\bfalse\b|\bfalse\.|\bfalse\!|\bincorrect\b|\bnao\b|\bnão\b", low):
            return False
        # tentar extrair JSON por segurança
        parsed = _extract_json_object(text)
        if parsed is not None:
            correct = parsed.get("correct")
            if isinstance(correct, bool):
                return correct
            if isinstance(correct, str):
                if correct.lower() in ("true", "yes", "sim"):
                    return True
                if correct.lower() in ("false", "no", "nao", "não"):
                    return False
        return None
    except Exception:
        return None


# ============================
# Função principal
# ============================
def analyze_file(path: str, use_llm: bool = False, llm_model: Optional[str] = None, llm_provider: Optional[str] = None):
    stats = defaultdict(lambda: {
        "latencies_with_mcp": [],
        "latencies_without_mcp": [],
        "tokens_input": [],
        "tokens_output": [],
        "tokens_total": [],
        "accuracies": [],
        "count": 0
    })

    with open(path, "r", encoding="utf-8") as f:
        llm = None
        if use_llm:
            if ensure_llm is None:
                print("Erro: fábrica de LLM (agent.multi.llm) não disponível. Para usar --use-llm é necessário ter o módulo 'agent.multi.llm' e a dependência da Groq instalada.")
                sys.exit(1)
            try:
                # Forçar provider groq conforme solicitado
                provider = "groq"
                llm = ensure_llm(model=llm_model, provider=provider)
            except Exception as e:
                print("Erro ao instanciar LLM (groq):", e)
                sys.exit(1)
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print("Linha inválida ignorada:", line[:80])
                continue

            model = item.get("model")
            if not model:
                continue

            reply = item.get("reply", "")
            latency = item.get("latency_ms", 0)
            has_mcp = item.get("mcp_tool") is not None

            # Coleta tokens
            stats[model]["tokens_input"].append(item.get("tokens_input", 0))
            stats[model]["tokens_output"].append(item.get("tokens_output", 0))
            stats[model]["tokens_total"].append(item.get("tokens_total", 0))

            # Separa latência pelo uso de MCP
            if has_mcp:
                stats[model]["latencies_with_mcp"].append(latency)
            else:
                stats[model]["latencies_without_mcp"].append(latency)

            # Acurácia: quando --use-llm, usar somente decisão do LLM (true/false).
            if use_llm:
                prompt_text = item.get("prompt") or item.get("input") or ""
                try:
                    acc_bool = classify_with_llm(prompt_text, reply, llm)
                except Exception:
                    acc_bool = None
                if acc_bool is None:
                    # se LLM não pôde decidir, considerar como incorreto e avisar
                    print("Aviso: LLM não decidiu (contando como incorreto) para uma linha.")
                    acc = 0.0
                else:
                    acc = 1.0 if acc_bool else 0.0
            else:
                acc = 0.0 if is_error(reply) else 1.0
            stats[model]["accuracies"].append(acc)

            stats[model]["count"] += 1

    # ========================
    # Cálculo final das médias
    # ========================
    results = {}

    for model, data in stats.items():
        n = data["count"]
        if n == 0:
            continue

        def avg(lst: List[float]):
            return sum(lst) / len(lst) if lst else 0.0

        results[model] = {
            "samples": n,
            "avg_latency_with_mcp": avg(data["latencies_with_mcp"]),
            "avg_latency_without_mcp": avg(data["latencies_without_mcp"]),
            "avg_tokens_input": avg(data["tokens_input"]),
            "avg_tokens_output": avg(data["tokens_output"]),
            "avg_tokens_total": avg(data["tokens_total"]),
            "accuracy": avg(data["accuracies"]),
        }

    return results


# ============================
# Execução CLI
# ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze LLM benchmark results")
    parser.add_argument("file", help="Caminho do arquivo de logs JSONL")
    parser.add_argument("--use-llm", action="store_true", help="Usar LLM para avaliar acurácia por item")
    parser.add_argument("--llm-model", default=None, help="Modelo LLM a ser passado para a fábrica (ex: gpt-4o-mini)")
    parser.add_argument("--llm-provider", default=None, help="Provedor LLM (ex: groq, openai)")
    args = parser.parse_args()

    results = analyze_file(args.file, use_llm=args.use_llm, llm_model=args.llm_model, llm_provider=args.llm_provider)

    print("\n=== MÉDIAS POR MODELO ===\n")
    for model, m in results.items():
        print(f"Modelo: {model}")
        print(f"  Amostras:               {m['samples']}")
        print(f"  Latência c/ MCP:        {m['avg_latency_with_mcp']:.2f} ms")
        print(f"  Latência s/ MCP:        {m['avg_latency_without_mcp']:.2f} ms")
        print(f"  Tokens input médios:    {m['avg_tokens_input']:.2f}")
        print(f"  Tokens output médios:   {m['avg_tokens_output']:.2f}")
        print(f"  Tokens total médios:    {m['avg_tokens_total']:.2f}")
        print(f"  Acurácia:               {m['accuracy']:.2f}")
        print()
