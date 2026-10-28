"""scripts/generate_synthetic_data.py

Regenerate the synthetic dataset (same as `csr-lab generate-data`).
"""

from __future__ import annotations

import argparse
import sys

from csr_lab.config import PROJECT_ROOT
from csr_lab.datasets.synthetic import write_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "synthetic"))
    args = parser.parse_args(argv)
    manifest = write_dataset(args.out)
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
