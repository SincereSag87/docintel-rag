# Benchmarks

`rag_eval.json` is a safe synthetic employee-policy benchmark for repeatable DocIntel RAG evaluation.

It contains small demo documents and answer expectations only. It does not include private,
client, or copyrighted documents.

Run full RAG evaluation:

```bash
uv run python -m app.main --evaluate benchmarks/rag_eval.json --model llama3.2
```

Run retrieval-only evaluation:

```bash
uv run python -m app.main --evaluate benchmarks/rag_eval.json --evaluate-retrieval
```
