# Scripts

Utility scripts for evaluation and analysis.

## Analyze Results

```bash
# Overall accuracy by model
python analyze.py multi_eval.jsonl --llm-model gpt-4o-mini

# Breakdown by category
python analyze2.py multi_eval.jsonl --llm-model gpt-4o-mini
```

## Files

- `analyze.py` - LLM-based evaluation (uses Groq)
- `analyze2.py` - Category-based metrics
