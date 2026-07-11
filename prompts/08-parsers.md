# 08 — Parser Strategies

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add the `langchain` and `llamaindex` parsers from SPEC.md §2 alongside `manual`.
LangChainParser wraps PyPDFLoader/UnstructuredFileLoader; LlamaIndexParser wraps
SimpleDirectoryReader/PDFReader. Both must lazy-import their package inside parse()
so the tool runs without them, with a friendly UI error if missing. Output the same
ParsedDocument contract. New files in backend/stages/parsers/ + registry lines only.
Ask before adding langchain/llama-index deps. Add tests against sample_data.
```
