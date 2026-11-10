"""src/csr_lab/prompts/system_prompt.py

Extra system-prompt text: who the documents are read for.

Edit the sentences below freely. Placeholders are filled from ``system_prompt:`` in
``configs/default.yaml``:

    {supplier}   the company the documents are read for
    {customer}   its customer
    {aliases}    other names of the customer, quoted and comma-separated
    {ambiguous}  aliases that are also ordinary words, quoted

The parts are joined in the order of ``SYSTEM_PROMPT`` and appended to the system prompt
of the tasks in ``TASKS``. ``AMBIGUOUS_NAME_RULES`` is left out when there are no
ambiguous aliases; ``other_customers`` (``skip`` or ``keep``) picks one of the two
``OTHER_CUSTOMERS_*`` sentences. The fingerprint of the rendered text is stored in
``metadata.prompt_versions['system_prompt']``, so results show which text was used.
"""

from __future__ import annotations

import hashlib

from csr_lab.config import SystemPromptSettings

VERSION = "system_prompt.v1"

# Tasks that decide what applies to the supplier. Chunking and reference merging do not get it.
TASKS = frozenset({"single_pass_extraction", "reference_extraction", "reference_verification", "conflict_analysis"})

SUPPLIER_ROLE = "You read these documents for {supplier}, a direct supplier to {customer}."
SCOPE = "Report only obligations of {supplier}; obligations of {supplier}'s own suppliers are out of scope."
NAME_RULES = "{customer} is also called {aliases}."
AMBIGUOUS_NAME_RULES = "{ambiguous} means {customer} only when it names the company, not when it is used as an ordinary word."
OTHER_CUSTOMERS_SKIP = "Ignore text that applies only to customers other than {customer}."
OTHER_CUSTOMERS_KEEP = "Analyse text for other customers too, and say when an obligation applies only to them."

SYSTEM_PROMPT = [SUPPLIER_ROLE, SCOPE, NAME_RULES, AMBIGUOUS_NAME_RULES, "{other_customers}"]


def _quoted(names: list[str]) -> str:
    return ", ".join(f'"{n}"' for n in names)


def render(settings: SystemPromptSettings) -> str | None:
    """The extra system-prompt text for these settings, or None when it is switched off."""
    if not settings.enabled:
        return None
    other = OTHER_CUSTOMERS_SKIP if settings.other_customers == "skip" else OTHER_CUSTOMERS_KEEP
    parts = [other if p == "{other_customers}" else p for p in SYSTEM_PROMPT]
    if not settings.ambiguous_aliases:
        parts = [p for p in parts if p is not AMBIGUOUS_NAME_RULES]
    if not settings.customer_aliases:
        parts = [p for p in parts if p is not NAME_RULES]
    values = {
        "supplier": settings.supplier,
        "customer": settings.customer,
        "aliases": _quoted(settings.customer_aliases),
        "ambiguous": _quoted(settings.ambiguous_aliases),
    }
    return "\n".join(f"- {p.format(**values)}" for p in parts)


def fingerprint(settings: SystemPromptSettings) -> str:
    text = render(settings)
    if text is None:
        return "off"
    return f"{VERSION}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"
