# Input

Drop documents here (`.pdf`, `.txt`, or chunk `.json` files), then run:

```bash
csr-lab build-graph --dry-run           # what would run / be skipped
csr-lab build-graph --llm lmstudio      # analyse and merge into graphs/main.json
```

- Each **subfolder** is analysed as one bundle, so put a CSR together with the documents it
  references in the same subfolder. Loose files in this folder form one more bundle.
  `--group file` analyses every file on its own.
- Files already merged into the graph (same bytes) are skipped; `--force` re-analyses them.
- `--graph PATH` picks a different graph; `--no-update-graph` reads it without changing it.

Everything in this folder except this README is git-ignored.
