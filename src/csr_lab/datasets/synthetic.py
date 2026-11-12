"""src/csr_lab/datasets/synthetic.py

Deterministic synthetic CSR dataset with gold annotations, shaped like real CSRs.

The case is built like CVS10 on top of Formel Q Konkret, so experiments meet the same
difficulties as real documents:

* name-only documents: the base "Halvard Q Agreement" has no identifier in its text; the
  case's ``documents.yaml`` gives its title and aliases, and names a document nobody has under
  two names ("Halvard New Parts Program" = "Halvard Neuteil Program");
* a delta document: HVG-CVS-20 "follows the chapter structure of the Halvard Q Agreement";
  three chapters only say "See Halvard Q Agreement. No additional requirements." (gold
  ``chapter_links``: inherits_from / amends);
* stated precedence: HVG-CVS-20 takes precedence over the Agreement, so the two
  cross-document conflicts are ``resolved_by_precedence``; one inside HVG-CVS-20 is ``unresolved``;
* customer-labelled paragraphs ("Global: ...", "Ostra: ...", a chapter titled "Brennic"), with
  "Global" also an ordinary word ("global supply chain"); gold references carry ``applies_to``;
  the dataset is read for Global Motors (``READER``, written to the manifest);
* chapter-qualified references ("HVG-CVS-31, chapter 4.2"; gold ``location``); contents pages.

All companies, documents and identifiers are fictional. Documents are rendered from the
structured specification below, and the gold annotations are produced by the same rendering
code, so gold offsets and requirement texts always match the generated files exactly. The
ground truth is written by hand in the specification and never derived from pipeline output;
``build_case`` checks it independently of the pipeline (labels match ``applies_to``, chapter
links cover exactly the shared chapter numbers, pointer chapters are ``inherits_from``, the
prevailing document is one of the conflict's).

Splits: ``dev`` cases may be used for prompt debugging; ``heldout`` cases should not be
edited to suit a pipeline. Bump ``DATASET_VERSION`` whenever the content changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from csr_lab.ids import document_family_id, document_node_id, slug
from csr_lab.schemas import (
    CaseGold,
    DocumentType,
    GoldChapterLink,
    GoldConflict,
    GoldEvidence,
    GoldInputDocument,
    GoldPair,
    GoldReference,
    GoldReferencedDocument,
    GoldRequirement,
    ReferenceRelation,
)

DATASET_NAME = "synthetic_csr"
DATASET_VERSION = "2.0.0"  # 2.0.0: the real-CSR-shaped case replaces the earlier cases
WRAP_WIDTH = 92
FICTION_NOTE = "All names, companies and document identifiers in this document are fictional."

# ---------------------------------------------------------------------------
# Specification building blocks
# ---------------------------------------------------------------------------


@dataclass
class Ref:
    mention: str
    canonical: str
    revision: str | None = None
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    resolution: str = "identifier"
    applies_to: list[str] = field(default_factory=list)  # customers of the labelled paragraph holding the mention
    location: str | None = None  # chapter of the referenced document the mention names


@dataclass
class H:
    """Heading line (never a requirement). An empty label renders the title alone ('Scope')."""

    label: str
    title: str


@dataclass
class R:
    """A requirement: the text below its heading (``H`` with the same chapter number)."""

    chapter: str
    text: str
    refs: list[Ref] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    more: list[str] = field(default_factory=list)  # further paragraphs, each on a new line (no blank line, as in PDFs)


@dataclass
class DocSpec:
    identifier: str
    company: str
    title: str
    revision: str
    date: str
    filename: str
    blocks: list
    header: list[str] | None = None  # replaces the default header (company, title, Document ID, ...)


@dataclass
class ConflictSpec:
    a: str
    b: str
    conflict_type: str
    quotes: tuple[str, str]
    category: str = "conflict"
    note: str = ""
    precedence: str | None = None  # resolved_by_precedence | mutual_precedence_claims | unresolved
    prevailing: str | None = None  # document-level node ID of the prevailing document


@dataclass
class CaseSpec:
    case_id: str
    split: str
    description: str
    documents: list[DocSpec]
    conflicts: list[ConflictSpec] = field(default_factory=list)
    non_conflicting: list[tuple[str, str, str]] = field(default_factory=list)
    chapter_links: list[tuple[str, str, str]] = field(default_factory=list)  # (source gid, target gid, kind)
    registry: str | None = None  # documents.yaml written next to the documents
    customers: dict[str, str] = field(default_factory=dict)  # label as printed -> customer ('Global' -> 'Global Motors')


def gid(doc: str, chapter: str) -> str:
    return f"{doc}#{chapter}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _safe_line_start(word: str, next_word: str | None) -> bool:
    """Wrapped lines must not start like a section label or table row."""
    if re.match(r"\d", word) or word.startswith("|"):
        return False
    return not (re.fullmatch(r"[A-Z]{2,6}", word) and next_word and re.match(r"\d", next_word))


def wrap(text: str, width: int = WRAP_WIDTH) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > width and _safe_line_start(word, nxt):
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines


@dataclass
class _Rendered:
    text: str
    requirements: list[GoldRequirement]
    headings: dict[str, str] = field(default_factory=dict)  # requirement gid -> title of the heading above it
    paragraphs: dict[str, list[str]] = field(default_factory=dict)  # requirement gid -> its paragraphs


def render_document(doc: DocSpec) -> _Rendered:
    out = ""
    reqs: list[GoldRequirement] = []
    doc_id = document_node_id(doc.identifier, doc.revision)

    def emit(line: str) -> int:
        nonlocal out
        start = len(out)
        out += line + "\n"
        return start

    header = doc.header or [doc.company, doc.title, f"Document ID: {doc.identifier}", f"Revision: {doc.revision}", f"Effective Date: {doc.date}"]
    for line in (*header, FICTION_NOTE):
        emit(line)
    emit("")
    emit("Contents")  # dot leaders, as in CVS10 / Formel Q; the chapter splitter reads it
    headings = [b for b in doc.blocks if isinstance(b, H)]
    for i, h in enumerate(headings):
        entry = f"{h.label} {h.title}" if h.label else h.title
        emit(f"{entry} {'.' * max(4, 60 - len(entry))} {i // 4 + 1}")
    emit("")
    headings_above: dict[str, str] = {}
    paragraphs: dict[str, list[str]] = {}
    last_title = ""
    for block in doc.blocks:
        if isinstance(block, H):
            emit(f"{block.label} {block.title}" if block.label else block.title)
            last_title = block.title
        elif isinstance(block, R):
            lines = wrap(block.text)
            for para in block.more:
                lines += wrap(para)
            text_start = emit(lines[0])
            for line in lines[1:]:
                emit(line)
            text_end = len(out) - 1  # exclude trailing newline
            emit("")
            req_text = out[text_start:text_end]
            if req_text.split() != " ".join([block.text, *block.more]).split():
                raise AssertionError(f"rendering changed requirement text: {block.chapter}")
            refs = []
            for ref in block.refs:
                if " ".join(ref.mention.split()) not in " ".join(req_text.split()):
                    raise AssertionError(f"{doc.identifier} {block.chapter}: mention {ref.mention!r} not in text")
                refs.append(
                    GoldReference(
                        canonical_identifier=ref.canonical,
                        revision=ref.revision,
                        mention_text=ref.mention,
                        relation=ref.relation,
                        resolution=ref.resolution,  # type: ignore[arg-type]
                        applies_to=list(ref.applies_to),
                        location=ref.location,
                    )
                )
            gold_id = gid(doc.identifier, block.chapter)
            if gold_id in paragraphs:
                raise AssertionError(f"{gold_id}: a chapter number is used twice")
            headings_above[gold_id] = last_title
            paragraphs[gold_id] = [block.text, *block.more]
            reqs.append(
                GoldRequirement(
                    gold_id=gold_id,
                    source_document_id=doc_id,
                    chapter=block.chapter,
                    requirement_text=req_text,
                    start_offset=text_start,
                    end_offset=text_end,
                    references=refs,
                    tags=block.tags,
                )
            )
        else:  # pragma: no cover - programming error in the spec
            raise TypeError(block)
    return _Rendered(text=out, requirements=reqs, headings=headings_above, paragraphs=paragraphs)


def build_case(case: CaseSpec) -> tuple[dict[str, str], CaseGold]:
    files: dict[str, str] = {}
    documents: list[GoldInputDocument] = []
    requirements: list[GoldRequirement] = []
    provided = {d.identifier: d.revision for d in case.documents}
    headings: dict[str, str] = {}
    paragraphs: dict[str, list[str]] = {}
    for doc in case.documents:
        rendered = render_document(doc)
        headings |= rendered.headings
        paragraphs |= rendered.paragraphs
        files[doc.filename] = rendered.text
        documents.append(
            GoldInputDocument(
                document_id=document_node_id(doc.identifier, doc.revision),
                canonical_identifier=doc.identifier,
                filename=doc.filename,
                title=doc.title,
                revision=doc.revision or None,
            )
        )
        requirements += rendered.requirements

    referenced: dict[tuple[str, str | None], GoldReferencedDocument] = {}
    for req in requirements:
        for ref in req.references:
            key = (ref.canonical_identifier, ref.revision)
            referenced.setdefault(
                key,
                GoldReferencedDocument(
                    canonical_identifier=ref.canonical_identifier,
                    revision=ref.revision,
                    # Unrevisioned: the provided revision is meant. Revision-qualified: only that revision.
                    content_available=ref.canonical_identifier in provided
                    and (ref.revision is None or slug(ref.revision) == slug(provided[ref.canonical_identifier])),
                    document_type=DocumentType.UNKNOWN,
                ),
            )
    _check_customer_scope(case, requirements, headings, paragraphs)
    by_id = {r.gold_id: r for r in requirements}
    links = _chapter_links(case, by_id)
    conflicts = []
    for i, c in enumerate(case.conflicts, start=1):
        for rid, quote in zip((c.a, c.b), c.quotes):
            if quote not in " ".join(by_id[rid].requirement_text.split()):
                raise AssertionError(f"{case.case_id}: conflict quote {quote!r} not in {rid}")
        conflicts.append(
            GoldConflict(
                gold_conflict_id=f"{case.case_id}-C{i:02d}",
                requirement_ids=[c.a, c.b],
                conflict_type=c.conflict_type,  # type: ignore[arg-type]
                category=c.category,  # type: ignore[arg-type]
                cross_document=by_id[c.a].source_document_id != by_id[c.b].source_document_id,
                evidence=[GoldEvidence(requirement_id=c.a, quoted_text=c.quotes[0]), GoldEvidence(requirement_id=c.b, quoted_text=c.quotes[1])],
                note=c.note,
                precedence=c.precedence,  # type: ignore[arg-type]
                prevailing_document_id=c.prevailing,
            )
        )
        own = {document_family_id(c.a.split("#")[0]), document_family_id(c.b.split("#")[0])}
        if c.precedence == "resolved_by_precedence" and c.prevailing not in own:
            raise AssertionError(f"{case.case_id}: prevailing document {c.prevailing} is not one of the conflict's documents {sorted(own)}")
        if c.precedence == "unresolved" and len(own) == 1 and c.prevailing:
            raise AssertionError(f"{case.case_id}: a conflict inside one document has no prevailing document")
    gold = CaseGold(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        case_id=case.case_id,
        split=case.split,  # type: ignore[arg-type]
        description=case.description,
        documents=documents,
        requirements=requirements,
        referenced_documents=sorted(referenced.values(), key=lambda d: (d.canonical_identifier, d.revision or "")),
        conflicts=conflicts,
        non_conflicting_pairs=[GoldPair(requirement_ids=[a, b], note=n) for a, b, n in case.non_conflicting],
        chapter_links=links,
    )
    return files, gold


def _label_customers(paragraph: str, case: CaseSpec) -> list[str] | None:
    """Customers of a paragraph that starts with a label ('Global: ...', 'Brennic/Ostra: ...'), else None."""
    head, sep, _rest = paragraph.partition(":")
    if not sep or len(head) > 60:
        return None
    parts = [p.strip() for p in head.split("/")]
    if all(p in case.customers for p in parts):
        return [case.customers[p] for p in parts]
    return None


def _check_customer_scope(case: CaseSpec, requirements: list[GoldRequirement], headings: dict[str, str], paragraphs: dict[str, list[str]]) -> None:
    """Every reference's ``applies_to`` must equal the label of the paragraph holding it (or the
    customer the chapter is titled with); unlabelled text applies to everyone. A guard against
    hand-written gold mistakes, independent of the pipeline code."""
    if not case.customers:
        for r in requirements:
            if any(ref.applies_to for ref in r.references):
                raise AssertionError(f"{case.case_id}: applies_to used without case.customers")
        return
    for r in requirements:
        title = headings.get(r.gold_id, "")
        parts = [p.strip() for p in title.split("/")] if title else []
        chapter_scope = [case.customers[p] for p in parts] if parts and all(p in case.customers for p in parts) else None
        for ref in r.references:
            holder = next((p for p in paragraphs[r.gold_id] if ref.mention_text in p), None)
            if holder is None:
                raise AssertionError(f"{r.gold_id}: mention {ref.mention_text!r} not inside one paragraph")
            expected = _label_customers(holder, case) or chapter_scope or []
            if sorted(expected) != sorted(ref.applies_to):
                raise AssertionError(f"{r.gold_id}: {ref.mention_text!r} applies_to {ref.applies_to} but its paragraph/chapter says {expected}")
        for p in paragraphs[r.gold_id]:
            if _label_customers(p, case) and ". " in p.rstrip()[:-1]:
                raise AssertionError(f"{r.gold_id}: a labelled paragraph must be one sentence (a block ends at a paragraph end): {p[:60]!r}")


_NO_ADDITIONAL = "No additional requirements."


def _chapter_links(case: CaseSpec, by_id: dict[str, GoldRequirement]) -> list[GoldChapterLink]:
    """Gold structure edges, checked against the specification:
    * every chapter number the delta and base documents share has exactly one inherits_from or amends link;
    * inherits_from exactly when the delta chapter is a pointer ('See <base>. No additional requirements.');
    * every references_chapter link comes from a reference with that location to the base document."""
    links = [GoldChapterLink(source=a, target=b, kind=k) for a, b, k in case.chapter_links]  # type: ignore[arg-type]
    for link in links:
        if link.source not in by_id or link.target not in by_id:
            raise AssertionError(f"{case.case_id}: chapter link names an unknown requirement: {link}")
    aligned = [x for x in links if x.kind in ("inherits_from", "amends")]
    pairs = {(x.source.split("#")[0], x.target.split("#")[0]) for x in aligned}
    for delta, base in pairs:
        delta_ch = {g.chapter: g.gold_id for g in by_id.values() if g.gold_id.startswith(delta + "#")}
        base_ch = {g.chapter: g.gold_id for g in by_id.values() if g.gold_id.startswith(base + "#")}
        expected = {(delta_ch[c], base_ch[c]) for c in delta_ch.keys() & base_ch.keys()}
        got = {(x.source, x.target) for x in aligned if x.source.startswith(delta + "#")}
        if got != expected:
            raise AssertionError(f"{case.case_id}: aligned chapters {sorted(got)} != shared chapter numbers {sorted(expected)}")
    for x in aligned:
        text = by_id[x.source].requirement_text.strip()
        pointer = text.endswith(_NO_ADDITIONAL) and len(text.split()) <= 12
        if (x.kind == "inherits_from") != pointer:
            raise AssertionError(f"{case.case_id}: {x.source} is {'a' if pointer else 'not a'} pointer chapter but labelled {x.kind}")
    for x in links:
        if x.kind != "references_chapter":
            continue
        target_doc, target_ch = x.target.split("#")
        if not any(ref.location == target_ch and ref.canonical_identifier == target_doc for ref in by_id[x.source].references):
            raise AssertionError(f"{case.case_id}: {x.source} has no reference to {target_doc} chapter {target_ch}")
    return links


# ---------------------------------------------------------------------------
# The case
# ---------------------------------------------------------------------------

PREC = ReferenceRelation.TAKES_PRECEDENCE_OVER

REGISTRY = """\
# Fictional document metadata, as a user would write it (see ingestion/loader.py).
documents:
  - file: halvard_q_agreement.txt
    title: Halvard Q Agreement
    aliases: [Q Agreement]
  - identifier: HALVARD-NEW-PARTS-PROGRAM
    title: Halvard New Parts Program
    aliases: [Halvard Neuteil Program]
"""

# Who the dataset is read for. Written to the manifest as `reader`; experiments start from it
# (``config.apply_experiment``), so references for Brennic or Ostra only are not expected.
READER_SYSTEM_PROMPT = {
    "enabled": True,
    "supplier": "the supplier",
    "customer": "Global Motors",
    "customer_aliases": ["Global", "Global Motors LLC"],
    "ambiguous_aliases": ["Global"],
    "other_customers": "skip",
    "other_customers_names": {"Brennic": ["Brennic Trucks"], "Ostra": ["Ostra AB"]},
}


def _case_halvard_delta() -> CaseSpec:
    q, c = "HALVARD-Q-AGREEMENT", "HVG-CVS-20"
    hqa = Ref("Halvard Q Agreement", q, resolution="name_alias")
    base = DocSpec(
        identifier=q,
        company="HALVARD GROUP",
        title="Halvard Q Agreement",
        revision="",
        date="",
        filename="halvard_q_agreement.txt",
        header=["HALVARD GROUP", "Halvard Q Agreement", "Quality management agreement between the Halvard Group companies and their suppliers", "6th edition, March 2025"],
        blocks=[
            H("0", "General provisions"),
            R("0", "The supplier shall comply with ISO 9001 and shall apply the Halvard New Parts Program to every new part.",
              [Ref("ISO 9001", "ISO-9001"), Ref("Halvard New Parts Program", "HALVARD-NEW-PARTS-PROGRAM", resolution="name_alias")],
              ["name_only_reference", "registry_only_document"]),
            H("1", "Request and offer"),
            H("1.1", "Offer prerequisites"),
            R("1.1", "The supplier shall submit a complete offer that covers all documents listed in the request.", [], ["no_reference"]),
            H("1.2", "Records"),
            R("1.2", "The supplier shall retain quality records for a minimum of 15 years after the end of series production.", [], ["conflict"]),
            H("2", "Approval"),
            H("2.1", "Part approval"),
            R("2.1", "Part approval shall be requested according to the Part Approval Manual HVG-PA-100 before the first delivery.",
              [Ref("HVG-PA-100", "HVG-PA-100")], ["simple"]),
            H("2.2", "Initial samples"),
            R("2.2", "Initial samples shall be delivered no earlier than 10 working days before the planned start of production.", [], ["conflict"]),
            H("2.3", "Part labels"),
            R("2.3", "Part labels shall show the part number and the date of manufacture.", [], ["no_reference", "chapter_target"]),
            H("3", "Change management"),
            H("3.1", "Change notification"),
            R("3.1", "The supplier shall notify the customer in writing at least 30 days before any change to a manufacturing process.", [], ["no_reference"]),
            H("3.2", "Field observation"),
            R("3.2", "The supplier shall observe the field performance of its products and report field failures to the customer.", [], ["no_reference"]),
        ],
    )
    delta = DocSpec(
        identifier=c,
        company="HALVARD VEHICLE GROUP",
        title="Halvard Customer Requirements",
        revision="2025-04",
        date="2025-04-01",
        filename="hvg_cvs_20.txt",
        blocks=[
            H("", "Scope"),
            R("Scope", "This standard follows the chapter structure of the Halvard Q Agreement and contains the requirements of Brennic, Ostra and Global Motors.",
              [Ref("Halvard Q Agreement", q, relation=PREC, resolution="name_alias")], ["structure_statement", "precedence"],
              more=["Where this standard and the Halvard Q Agreement contradict each other, this standard takes precedence."]),
            H("0", "General provisions"),
            R("0", "In addition to the Halvard Q Agreement, the supplier shall comply with IATF 16949 throughout its global supply chain.",
              [hqa, Ref("IATF 16949", "IATF-16949"),
               Ref("GLB-SWR-001", "GLB-SWR-001", applies_to=["Global Motors"]),
               Ref("OST-STD-3868", "OST-STD-3868", applies_to=["Ostra"]),
               Ref("BRN-PC-001", "BRN-PC-001", applies_to=["Brennic"])],
              ["customer_labels", "ambiguous_customer_word"],
              more=["Global: Unless exempted in writing, the supplier shall comply with the Global Motors Supplier Warranty Requirements GLB-SWR-001.",
                    "Ostra: The supplier shall comply with the Ostra purchasing standard OST-STD-3868.",
                    "Brennic: The Brennic purchasing conditions BRN-PC-001 apply in addition."]),
            H("1", "Request and offer"),
            H("1.1", "Offer prerequisites"),
            R("1.1", "See Halvard Q Agreement. No additional requirements.", [hqa], ["pointer_chapter"]),
            H("1.2", "Records"),
            R("1.2", "Quality records shall be destroyed no later than 10 years after the end of series production.", [], ["conflict", "amends"]),
            H("2", "Approval"),
            H("2.1", "Part approval"),
            R("2.1", "See Halvard Q Agreement. No additional requirements.", [hqa], ["pointer_chapter"]),
            H("2.2", "Initial samples"),
            R("2.2", "Initial samples shall be delivered at least 20 working days before the planned start of production, and samples of new parts shall follow the Halvard Neuteil Program.",
              [Ref("Halvard Neuteil Program", "HALVARD-NEW-PARTS-PROGRAM", resolution="name_alias"), Ref("OST-PR-410", "OST-PR-410", applies_to=["Ostra"])],
              ["conflict", "amends", "alias_of_registry_document", "customer_labels"],
              more=["Ostra: Sample reports for Ostra shall be uploaded according to the Ostra portal procedure OST-PR-410."]),
            H("3", "Change management"),
            H("3.1", "Change notification"),
            R("3.1", "Process changes shall be handled in accordance with HVG-CVS-31, chapter 4.2.",
              [Ref("HVG-CVS-31", "HVG-CVS-31", location="4.2")], ["chapter_reference", "missing_external", "customer_labels"],
              more=["Global: The supplier shall also inform the Global supplier quality representative of every process change."]),
            H("3.2", "Field observation"),
            R("3.2", "See Halvard Q Agreement. No additional requirements.", [hqa], ["pointer_chapter"]),
            H("3.3", "Brennic"),
            R("3.3", "Field failures shall be reported to Brennic within 24 hours according to BRN-FF-120.",
              [Ref("BRN-FF-120", "BRN-FF-120", applies_to=["Brennic"])], ["customer_chapter"]),
            H("4", "Labels"),
            H("4.1", "Label size"),
            R("4.1", "Container labels shall be printed with a minimum barcode height of 25 mm.", [], ["conflict"]),
            H("4.2", "Label layout"),
            R("4.2", "Labels shall follow the Halvard Q Agreement, chapter 2.3 and shall have a barcode height of no more than 20 mm.",
              [Ref("Halvard Q Agreement", q, resolution="name_alias", location="2.3")], ["conflict", "chapter_reference"]),
        ],
    )
    return CaseSpec(
        case_id="heldout_halvard_delta",
        split="heldout",
        description=(
            "Real-CSR-shaped: a name-only base agreement and a delta standard that follows its chapter structure "
            "(pointer chapters), stated precedence, customer-labelled paragraphs, chapter-qualified references, contents pages."
        ),
        documents=[delta, base],  # the CSR first (the incremental variant analyses it as the new document)
        conflicts=[
            ConflictSpec(gid(c, "1.2"), gid(q, "1.2"), "timing_incompatibility",
                         ("destroyed no later than 10 years after the end of series production", "retain quality records for a minimum of 15 years"),
                         note="the delta chapter changes the base chapter; the delta standard takes precedence",
                         precedence="resolved_by_precedence", prevailing=document_family_id(c)),
            ConflictSpec(gid(c, "2.2"), gid(q, "2.2"), "timing_incompatibility",
                         ("delivered at least 20 working days before the planned start of production", "delivered no earlier than 10 working days before the planned start of production"),
                         precedence="resolved_by_precedence", prevailing=document_family_id(c)),
            ConflictSpec(gid(c, "4.1"), gid(c, "4.2"), "numeric_incompatibility",
                         ("minimum barcode height of 25 mm", "barcode height of no more than 20 mm"),
                         note="inside one document: precedence does not decide it", precedence="unresolved"),
        ],
        non_conflicting=[
            (gid(c, "1.1"), gid(q, "1.1"), "the delta chapter adopts the base chapter unchanged"),
            (gid(c, "3.1"), gid(q, "3.1"), "names another procedure; no contradictory obligation"),
            (gid(c, "0"), gid(q, "0"), "IATF 16949 is required in addition to ISO 9001"),
        ],
        chapter_links=[
            (gid(c, "0"), gid(q, "0"), "amends"),
            (gid(c, "1.1"), gid(q, "1.1"), "inherits_from"),
            (gid(c, "1.2"), gid(q, "1.2"), "amends"),
            (gid(c, "2.1"), gid(q, "2.1"), "inherits_from"),
            (gid(c, "2.2"), gid(q, "2.2"), "amends"),
            (gid(c, "3.1"), gid(q, "3.1"), "amends"),
            (gid(c, "3.2"), gid(q, "3.2"), "inherits_from"),
            (gid(c, "4.2"), gid(q, "2.3"), "references_chapter"),
        ],
        registry=REGISTRY,
        customers={"Global": "Global Motors", "Brennic": "Brennic", "Ostra": "Ostra"},
    )


READER = {"system_prompt": READER_SYSTEM_PROMPT, "evaluation": {"customer": READER_SYSTEM_PROMPT["customer"]}}


def all_cases() -> list[CaseSpec]:
    return [_case_halvard_delta()]


# ---------------------------------------------------------------------------
# Writing the dataset
# ---------------------------------------------------------------------------


def write_manifest(
    out: Path, *, name: str, description: str, generator: str, cases: list[dict], version: str = DATASET_VERSION, extra: dict | None = None
) -> Path:
    """``manifest.json`` of a generated dataset (every generator writes it this way), with the
    dataset's ``reader`` (who it is read for)."""
    manifest = {"name": name, "version": version, "description": description, "generator": generator, "reader": READER, **(extra or {}), "cases": cases}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def write_dataset(
    out_dir: str | Path,
    cases: list[CaseSpec] | None = None,
    name: str = DATASET_NAME,
    *,
    version: str = DATASET_VERSION,
    description: str = "Fictional CSRs shaped like real ones (a delta document on a name-only base agreement, customer labels, stated precedence).",
    generator: str = "csr_lab.datasets.synthetic",
) -> Path:
    """Documents (and a case's ``documents.yaml``), gold files and the manifest."""
    out = Path(out_dir)
    (out / "documents").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in cases or all_cases():
        files, gold = build_case(case)
        doc_paths = []
        for filename, text in files.items():
            path = out / "documents" / case.case_id / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            doc_paths.append(path.relative_to(out).as_posix())
        if case.registry:
            (out / "documents" / case.case_id / "documents.yaml").write_text(case.registry, encoding="utf-8", newline="\n")
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.write_text(gold.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest_cases.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "description": case.description,
                "documents": doc_paths,
                "ground_truth": gold_path.relative_to(out).as_posix(),
            }
        )
    return write_manifest(out, name=name, version=version, description=description, generator=generator, cases=manifest_cases)
