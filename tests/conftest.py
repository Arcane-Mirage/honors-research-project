from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from csr_lab.config import AppConfig, load_config  # noqa: E402
from csr_lab.datasets.loader import load_dataset  # noqa: E402
from csr_lab.datasets.synthetic import write_dataset  # noqa: E402
from csr_lab.ingestion.loader import load_bundle  # noqa: E402

SIMPLE_DOC = """\
CORVANE MOTORS
Test Requirements
Document ID: CVM-CSR-900
Revision: A

1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 The supplier shall comply with Specification CVM-QS-110 and Drawing DRW-045.
2.2 Packaging Standard CVM-PKG-050 does not apply and shall not be used.
2.3 Parts shall be validated at a maximum test temperature of 85 °C according to
CVM-TM-320.
2.4 Thermal validation according to CVM-TM-320 shall be performed at a minimum test
temperature of 105 °C.
2.5 The supplier shall keep records for five years.
"""


@pytest.fixture
def config() -> AppConfig:
    """Default config with a perfect mock (no injected errors), for exact assertions."""
    return load_config(
        overrides={
            "llm": {
                "mock": {
                    "reference_miss_rate": 0.0,
                    "hallucination_rate": 0.0,
                    "quote_paraphrase_rate": 0.0,
                    "conflict_miss_rate": 0.0,
                    "conflict_false_positive_rate": 0.0,
                }
            }
        }
    )


@pytest.fixture
def noisy_config() -> AppConfig:
    """Default config with the default (noisy) mock."""
    return load_config()


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    write_dataset(out)
    return out


@pytest.fixture(scope="session")
def dataset(dataset_dir):
    return load_dataset(dataset_dir / "manifest.json")


@pytest.fixture
def simple_doc_path(tmp_path) -> Path:
    path = tmp_path / "simple.txt"
    path.write_text(SIMPLE_DOC, encoding="utf-8")
    return path


@pytest.fixture
def simple_docs(simple_doc_path):
    return load_bundle([simple_doc_path])
