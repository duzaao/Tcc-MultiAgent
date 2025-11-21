import json
from collections import defaultdict
from typing import Dict, Any, List


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


# ============================
# Função principal
# ============================
def analyze_file(path: str):
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

            # Acurácia: erro real = 0, caso contrário = 1
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
    args = parser.parse_args()

    results = analyze_file(args.file)

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
