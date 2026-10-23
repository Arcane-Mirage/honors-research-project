# Public datasets

Nothing here is required for the offline demo. Public requirement collections can be added as
extra datasets once their licence and access terms have been checked (for example PURE, a
public collection of requirements documents).

To add one:

1. Place the documents (UTF-8 `.txt`, text-based `.pdf`, or `.json` with a `text` field) in a
   folder under `data/public/<name>/documents/`. Do not commit files whose licence does not
   allow redistribution; keep a download script instead.
2. Write a `manifest.json` in the same format as `data/synthetic/manifest.json`. Cases without
   `ground_truth` can be analysed (`csr-lab analyze`) but not scored.
3. For scored cases, write gold files that validate against `CaseGold` in
   `src/csr_lab/schemas.py`; annotate them by hand, never from a pipeline's output.
4. Point an experiment at it: `dataset: data/public/<name>/manifest.json`.
