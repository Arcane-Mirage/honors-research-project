# Input

Drop PDFs here, then run:

```bash
csr-lab build-graph --dry-run           # what would run / be skipped
csr-lab build-graph --llm lmstudio      # analyse and merge into graphs/main.json
```

- Each **subfolder** is analysed as one bundle, so put a CSR together with the documents it
  references in the same subfolder. Loose files in this folder form one more bundle.
  `--group file` analyses every file on its own.
- Only PDFs are analysed (`input_formats` in `configs/default.yaml`); other files are listed
  as skipped. A document you do not have, or that is not a PDF, still becomes a node in the
  graph when an analysed document references it ("content not provided").
- A document without chapters (a contract, a form) gives no requirements: it is cut into
  passages and analysed for references and conflicts only.
- Each new bundle is also checked for conflicts against what is already in the graph.
- Files already merged into the graph (same bytes) are skipped; `--force` re-analyses them.
- `--graph PATH` picks a different graph; `--no-update-graph` reads it without changing it.
- A `documents.yaml` in a bundle folder (or this folder) states what the PDFs do not: identifier,
  title, revision and other names of each file, names of documents you do not have, which
  document follows another's chapter structure, and which takes precedence. Example:

  ```yaml
  documents:
    - file: Formel Q Konkret.pdf
      identifier: FORMEL-Q-KONKRET
      title: Formel Q konkret
      aliases: [Formula Q concretely]
    - file: CVS10_2024-08.pdf
      follows_structure_of: FORMEL-Q-KONKRET     # optional: CVS10 also says so in its Scope
      takes_precedence_over: [FORMEL-Q-KONKRET, ISO-9001]
    - identifier: FORMEL-Q-NEW-PARTS-INTEGRAL     # not a file here: names only
      title: Formel Q New Parts Integral
      aliases: [Formel Q Neuteile Integral]
  ```

Everything in this folder except this README is git-ignored.
