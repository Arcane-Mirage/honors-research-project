"""src/csr_lab/prompts/system_prompt.py

Extra system-prompt text: who the documents are read for.

Edit the sentences below freely. Placeholders are filled from ``system_prompt:`` in
``configs/default.yaml`` and the customers listed under ``issuers.known``:

    {supplier}        the company the documents are read for
    {customers}       its known customers, all treated alike ('Navistar, MAN and Scania'); not a complete list
    {non_customers}   companies listed as not being customers (``non_customer_names``), if any
    {customer}        one customer (NAME_RULES, AMBIGUOUS_NAME_RULES: one sentence per customer)
    {aliases}         that customer's other names, quoted and comma-separated
    {ambiguous}       its aliases that are also ordinary words, quoted

The parts are joined in the order of ``SYSTEM_PROMPT`` and appended to the system prompt
of the tasks in ``TASKS``. ``NAME_RULES`` is written for each customer with aliases and
``AMBIGUOUS_NAME_RULES`` for each customer with ambiguous aliases. ``UNLISTED_COMPANIES`` says the
customer list is not complete. Only when ``non_customer_names`` lists companies, ``non_customers``
(``skip`` or ``keep``) adds one of the two ``NON_CUSTOMERS_*`` sentences, naming them. The fingerprint of
the rendered text is stored in ``metadata.prompt_versions['system_prompt']``, so results show
which text was used.
"""

from __future__ import annotations

import hashlib

from csr_lab.config import SystemPromptSettings

VERSION = "system_prompt.v3"

# Tasks that decide what applies to the supplier. Chunking and reference merging do not get it.
TASKS = frozenset({"single_pass_extraction", "reference_extraction", "reference_verification", "conflict_analysis"})

SUPPLIER_ROLE = "You read these documents for {supplier}, a direct supplier. Its customers include {customers}."
SCOPE = "Report only obligations of {supplier}; obligations of {supplier}'s own suppliers are out of scope."
EQUAL_CUSTOMERS = "Treat every one of these customers alike: text for any of them applies to {supplier}."
NAME_RULES = "{customer} is also called {aliases}."
AMBIGUOUS_NAME_RULES = "{ambiguous} means {customer} only when it names the company, not when it is used as an ordinary word."
UNLISTED_COMPANIES = "Other companies may also be customers of {supplier}: do not skip text because the company it names is not listed here."
NON_CUSTOMERS_SKIP = "Ignore text that applies only to {non_customers}; they are not customers of {supplier}."
NON_CUSTOMERS_KEEP = "Analyse text for {non_customers} too, and say when an obligation applies only to them."

SYSTEM_PROMPT = [SUPPLIER_ROLE, SCOPE, EQUAL_CUSTOMERS, "{name_rules}", UNLISTED_COMPANIES, "{non_customers}"]


def _quoted(names: list[str]) -> str:
    return ", ".join(f'"{n}"' for n in names)


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def render(settings: SystemPromptSettings) -> str | None:
    """The extra system-prompt text for these settings, or None when it is switched off."""
    if not settings.enabled:
        return None
    customers = settings.customers
    values = {
        "supplier": settings.supplier,
        "customers": _listed(settings.customer_names) if customers else "the companies the documents name",
        "non_customers": _listed(list(settings.non_customer_names)) if settings.non_customer_names else "",
    }
    lines: list[str] = []
    for part in SYSTEM_PROMPT:
        if part == "{name_rules}":
            for c in customers:
                aliases = [a for a in c.aliases if a != c.name]
                if aliases:
                    lines.append(NAME_RULES.format(customer=c.name, aliases=_quoted(aliases)))
                if c.ambiguous_aliases:
                    lines.append(AMBIGUOUS_NAME_RULES.format(customer=c.name, ambiguous=_quoted(c.ambiguous_aliases)))
        elif part == "{non_customers}":
            if settings.non_customer_names:  # never inferred: only companies the config names
                lines.append((NON_CUSTOMERS_SKIP if settings.non_customers == "skip" else NON_CUSTOMERS_KEEP).format(**values))
        elif part is EQUAL_CUSTOMERS and len(customers) < 2:
            continue
        else:
            lines.append(part.format(**values))
    return "\n".join(f"- {line}" for line in lines)


def fingerprint(settings: SystemPromptSettings) -> str:
    text = render(settings)
    if text is None:
        return "off"
    return f"{VERSION}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"
