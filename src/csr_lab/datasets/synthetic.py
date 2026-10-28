"""src/csr_lab/datasets/synthetic.py

Deterministic synthetic CSR-like dataset with gold annotations.

All companies (Corvane Motors, Tessaly Automotive), documents and identifiers
are fictional. Documents are rendered from the structured specification below,
and the gold annotations are produced by the same rendering code, so gold
offsets and requirement texts always match the generated files exactly. The
ground truth is written by hand in this specification; it is never derived
from any pipeline output.

Splits: ``dev`` cases may be used for prompt debugging; ``heldout`` cases
should not be edited to suit a pipeline. Bump ``DATASET_VERSION`` whenever the
content changes.

``generate_scaled_case`` builds parameterised documents (length, reference
density, reference position) for ablations.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from csr_lab.ids import document_node_id, slug
from csr_lab.schemas import (
    CaseGold,
    DocumentType,
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
DATASET_VERSION = "1.1.0"  # 1.1.0: input document IDs include the revision
WRAP_WIDTH = 92
FICTION_NOTE = "All names, companies and document identifiers in this document are fictional."

# Document types as a reader of the requirement would classify them (gold only).
DOCUMENT_TYPES: dict[str, DocumentType] = {
    "CVM-SQM-001": DocumentType.QUALITY_MANUAL,
    "CVM-QS-110": DocumentType.TECHNICAL_SPECIFICATION,
    "DRW-045": DocumentType.ENGINEERING_DRAWING,
    "QP-12": DocumentType.PROCEDURE,
    "CVM-TM-305": DocumentType.TECHNICAL_SPECIFICATION,
    "CVM-TM-311": DocumentType.TECHNICAL_SPECIFICATION,
    "CVM-TM-320": DocumentType.TECHNICAL_SPECIFICATION,
    "ES-4.12": DocumentType.STANDARD,
    "CVM-ES-210": DocumentType.STANDARD,
    "CVM-PKG-210": DocumentType.TECHNICAL_SPECIFICATION,
    "CVM-PKG-050": DocumentType.STANDARD,
    "CVM-LBL-220": DocumentType.STANDARD,
    "CVM-LG-140": DocumentType.TECHNICAL_SPECIFICATION,
    "CVM-CSR-200": DocumentType.CUSTOMER_SPECIFICATION,
    "CVM-CSR-450": DocumentType.CUSTOMER_SPECIFICATION,
    "TSA-PPAP-410": DocumentType.PROCEDURE,
    "TSA-CSR-400": DocumentType.CUSTOMER_SPECIFICATION,
    "TSA-QF-031A": DocumentType.FORM,
    "TSA-DRW-1188": DocumentType.ENGINEERING_DRAWING,
    "TSA-DRW-2210": DocumentType.ENGINEERING_DRAWING,
    "DRW-7731": DocumentType.ENGINEERING_DRAWING,
}


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


@dataclass
class H:
    """Heading line (never a requirement)."""

    label: str
    title: str


@dataclass
class Info:
    """Numbered informational paragraph (not a requirement)."""

    label: str
    text: str


@dataclass
class R:
    """A requirement. ``under_heading`` renders it as a paragraph below the previous heading."""

    chapter: str
    text: str
    refs: list[Ref] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    table: list[str] = field(default_factory=list)
    under_heading: bool = False


@dataclass
class DocSpec:
    identifier: str
    company: str
    title: str
    revision: str
    date: str
    filename: str
    blocks: list


@dataclass
class ConflictSpec:
    a: str
    b: str
    conflict_type: str
    quotes: tuple[str, str]
    category: str = "conflict"
    note: str = ""


@dataclass
class CaseSpec:
    case_id: str
    split: str
    description: str
    documents: list[DocSpec]
    conflicts: list[ConflictSpec] = field(default_factory=list)
    non_conflicting: list[tuple[str, str, str]] = field(default_factory=list)


def gid(doc: str, chapter: str) -> str:
    return f"{doc}#{chapter}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _safe_line_start(word: str, next_word: str | None) -> bool:
    """Wrapped lines must not start like a section label or table row."""
    if re.match(r"\d", word) or word.startswith("|"):
        return False
    if re.fullmatch(r"[A-Z]{2,6}", word) and next_word and re.match(r"\d", next_word):
        return False
    return True


def wrap(first_prefix: str, text: str, width: int = WRAP_WIDTH) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = first_prefix
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


def render_document(doc: DocSpec) -> _Rendered:
    out = ""
    reqs: list[GoldRequirement] = []
    doc_id = document_node_id(doc.identifier, doc.revision)

    def emit(line: str) -> int:
        nonlocal out
        start = len(out)
        out += line + "\n"
        return start

    for line in (doc.company, doc.title, f"Document ID: {doc.identifier}", f"Revision: {doc.revision}", f"Effective Date: {doc.date}", FICTION_NOTE):
        emit(line)
    emit("")
    counts: dict[str, int] = {}
    for block in doc.blocks:
        if isinstance(block, H):
            emit(f"{block.label} {block.title}")
        elif isinstance(block, Info):
            for line in wrap(block.label, block.text):
                emit(line)
            emit("")
        elif isinstance(block, R):
            prefix = "" if block.under_heading else block.chapter
            lines = wrap(prefix, block.text)
            first_start = emit(lines[0])
            text_start = first_start + (len(prefix) + 1 if prefix else 0)
            for line in lines[1:]:
                emit(line)
            for row in block.table:
                emit(row)
            text_end = len(out) - 1  # exclude trailing newline
            emit("")
            req_text = out[text_start:text_end]
            if req_text.split() != " ".join([block.text, *block.table]).split():
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
                    )
                )
            counts[block.chapter] = counts.get(block.chapter, 0) + 1
            suffix = "" if counts[block.chapter] == 1 else f"-{counts[block.chapter]}"
            reqs.append(
                GoldRequirement(
                    gold_id=gid(doc.identifier, block.chapter) + suffix,
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
    return _Rendered(text=out, requirements=reqs)


def build_case(case: CaseSpec) -> tuple[dict[str, str], CaseGold]:
    files: dict[str, str] = {}
    documents: list[GoldInputDocument] = []
    requirements: list[GoldRequirement] = []
    provided = {d.identifier: d.revision for d in case.documents}
    for doc in case.documents:
        rendered = render_document(doc)
        files[doc.filename] = rendered.text
        documents.append(
            GoldInputDocument(
                document_id=document_node_id(doc.identifier, doc.revision),
                canonical_identifier=doc.identifier,
                filename=doc.filename,
                title=doc.title,
                revision=doc.revision,
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
                    document_type=DOCUMENT_TYPES.get(ref.canonical_identifier, DocumentType.UNKNOWN),
                ),
            )

    by_id = {r.gold_id: r for r in requirements}
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
            )
        )
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
    )
    return files, gold


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

REF, EXC, PREC, SUP = (
    ReferenceRelation.REFERENCES,
    ReferenceRelation.EXCLUDES,
    ReferenceRelation.TAKES_PRECEDENCE_OVER,
    ReferenceRelation.SUPERSEDES,
)


def _case_dev_production() -> CaseSpec:
    d = "CVM-CSR-100"
    doc = DocSpec(
        identifier=d,
        company="CORVANE MOTORS",
        title="Customer-Specific Requirements for Production Part Suppliers",
        revision="C",
        date="2026-02-15",
        filename="cvm_csr_100.txt",
        blocks=[
            H("1", "Scope"),
            Info("1.1", "This document defines the customer-specific requirements of Corvane Motors for suppliers of production parts and supplements the purchase order terms."),
            H("2", "Quality Management System"),
            R("2.1", "The supplier shall maintain a quality management system that conforms to the Corvane Supplier Quality Manual CVM-SQM-001.", [Ref("CVM-SQM-001", "CVM-SQM-001")], ["simple", "named_identifier"]),
            R("2.2", "The supplier shall notify the Corvane supplier quality engineer of any change to its quality management system certification status within 10 working days.", [], ["no_reference"]),
            H("3", "Product Quality Planning"),
            H("3.1", "Control Plans"),
            R("3.1", "The supplier shall prepare control plans in accordance with Specification CVM-QS-110 and shall include all special characteristics identified on Drawing DRW-045.", [Ref("CVM-QS-110", "CVM-QS-110"), Ref("DRW-045", "DRW-045")], ["multiple_references", "heading_paragraph"], under_heading=True),
            R("3.2", "The supplier shall perform a process failure mode and effects analysis for every new part number before the production trial run, using the worksheet format defined in Procedure QP-12. Part number 55-1020 and part number 55-1021 are excluded from this requirement during the 2026 model year.", [Ref("QP-12", "QP-12")], ["distractor_numbers", "long"]),
            H("4", "Validation Testing"),
            R("4.1", "The supplier shall perform the validation tests listed in the following table:", [Ref("CVM-TM-305", "CVM-TM-305"), Ref("CVM-TM-311 Rev B", "CVM-TM-311", "B")], ["table", "revision"], table=["| Test | Method | Frequency |", "| Salt spray corrosion | CVM-TM-305 | Each production lot |", "| Coating adhesion | CVM-TM-311 Rev B | Weekly |"]),
            R("4.2", "Production parts shall be validated at a maximum test temperature of 85 °C according to Test Method CVM-TM-320.", [Ref("CVM-TM-320", "CVM-TM-320")], ["conflict"]),
            R("4.3", "Thermal validation according to CVM-TM-320 shall be performed at a minimum test temperature of 105 °C for 500 hours.", [Ref("CVM-TM-320", "CVM-TM-320")], ["conflict", "repeated_reference"]),
            H("5", "Change Management"),
            R("5.1", "The supplier shall notify Corvane Motors in writing at least 90 days before implementing any change to a manufacturing process, as required by the Corvane Supplier Quality Manual CVM-SQM-001.", [Ref("CVM-SQM-001", "CVM-SQM-001")], ["conflict", "repeated_reference"]),
            R("5.2", "Minor process changes shall be implemented no later than 30 days after written notification to Corvane Motors and shall be documented according to Procedure QP-12.", [Ref("QP-12", "QP-12")], ["conflict"]),
            H("6", "Records"),
            R("6.1", "The supplier shall retain quality records for a minimum of 15 years after the end of series production, unless a longer period is specified on the applicable drawing.", [], ["no_reference", "generic_mention", "conflict"]),
            R("6.2", "Inspection records described in Specification CVM-QS-110 shall be destroyed no later than 10 years after the end of series production.", [Ref("CVM-QS-110", "CVM-QS-110")], ["conflict"]),
            H("7", "Engineering Standards"),
            H("7.1", "Fastener Compliance"),
            R("7.1", "Fasteners shall conform to Engineering Standard ES 4.12, and coated fasteners shall additionally meet CVM-ES-210 Rev. B.", [Ref("ES 4.12", "ES-4.12"), Ref("CVM-ES-210 Rev. B", "CVM-ES-210", "B")], ["section_like_identifier", "revision"], under_heading=True),
            R("7.2", "Replacement fasteners supplied for service shall conform to CVM-ES-210 Revision D. Section 4.12 of this document does not apply to service fasteners.", [Ref("CVM-ES-210 Revision D", "CVM-ES-210", "D")], ["revision", "distractor_section"]),
        ],
    )
    return CaseSpec(
        case_id="dev_corvane_production",
        split="dev",
        description="Single production-part CSR: simple, multiple, table, revision, distractor and section-like references; three internal conflicts.",
        documents=[doc],
        conflicts=[
            ConflictSpec(gid(d, "4.2"), gid(d, "4.3"), "numeric_incompatibility", ("maximum test temperature of 85 °C", "minimum test temperature of 105 °C")),
            ConflictSpec(gid(d, "5.1"), gid(d, "5.2"), "timing_incompatibility", ("at least 90 days before implementing any change", "no later than 30 days after written notification")),
            ConflictSpec(gid(d, "6.1"), gid(d, "6.2"), "timing_incompatibility", ("retain quality records for a minimum of 15 years", "destroyed no later than 10 years after the end of series production")),
        ],
        non_conflicting=[
            (gid(d, "2.2"), gid(d, "5.1"), "both concern notifications, but about different subjects"),
            (gid(d, "7.1"), gid(d, "7.2"), "different revisions of CVM-ES-210 for production versus service fasteners"),
            (gid(d, "3.1"), gid(d, "6.2"), "share CVM-QS-110 without contradiction"),
        ],
    )


def _case_dev_packaging() -> CaseSpec:
    c, p = "CVM-CSR-200", "CVM-PKG-210"
    csr = DocSpec(
        identifier=c,
        company="CORVANE MOTORS",
        title="Logistics and Packaging Requirements for Production Suppliers",
        revision="A",
        date="2026-01-10",
        filename="cvm_csr_200.txt",
        blocks=[
            H("1", "General"),
            Info("1.1", "This document applies to all shipments of production parts to Corvane Motors assembly plants."),
            R("1.2", "The supplier shall package all production parts in accordance with the Packaging and Labeling Specification CVM-PKG-210.", [Ref("CVM-PKG-210", "CVM-PKG-210")], ["available_document", "named_identifier"]),
            R("1.3", "Packaging Standard CVM-PKG-050 does not apply to parts shipped under this document and shall not be used for packaging design.", [Ref("CVM-PKG-050", "CVM-PKG-050", relation=EXC)], ["exclusion"]),
            R("1.4", "The supplier shall comply with the shipping requirements of the Corvane Supplier Quality Manual CVM-SQM-001.", [Ref("CVM-SQM-001", "CVM-SQM-001")], ["named_identifier"]),
            H("2", "Labels"),
            R("2.1", "Each container shall carry a shipping label that conforms to CVM-LBL-220.", [Ref("CVM-LBL-220", "CVM-LBL-220")], ["simple"]),
            R("2.2", "Container labels shall be printed with a minimum barcode height of 25 mm.", [], ["no_reference", "conflict"]),
            H("3", "Containers"),
            R("3.1", "Production parts shall be shipped only in returnable containers approved by Corvane Logistics.", [], ["no_reference", "conflict"]),
            H("4", "Deliveries"),
            R("4.1", "The supplier shall transmit an advance shipping notice no later than 2 hours after shipment departure, using the message format defined in CVM-LG-140.", [Ref("CVM-LG-140", "CVM-LG-140")], ["position_end"]),
            R("4.2", "Damaged packaging shall be reported to Corvane Logistics using the reporting procedure described in the Packaging and Labeling Specification.", [Ref("Packaging and Labeling Specification", "CVM-PKG-210", resolution="name_alias")], ["name_alias"]),
            H("5", "Precedence"),
            R("5.1", "Where this document conflicts with CVM-PKG-210, the requirements of this document shall take precedence.", [Ref("CVM-PKG-210", "CVM-PKG-210", relation=PREC)], ["precedence"]),
        ],
    )
    pkg = DocSpec(
        identifier=p,
        company="CORVANE MOTORS",
        title="Packaging and Labeling Specification",
        revision="D",
        date="2025-11-03",
        filename="cvm_pkg_210.txt",
        blocks=[
            H("1", "Scope"),
            Info("1.1", "This specification defines packaging and labeling requirements for production parts delivered to Corvane Motors."),
            H("2", "Packaging Design"),
            R("2.1", "Packaging design shall follow the dimensional guidelines of Packaging Standard CVM-PKG-050.", [Ref("CVM-PKG-050", "CVM-PKG-050")], ["nested_reference", "conflict"]),
            R("2.2", "Expendable (one-way) cardboard containers shall be used for all parts weighing less than 2 kg.", [], ["no_reference", "conflict"]),
            H("3", "Labeling"),
            R("3.1", "Labels shall conform to CVM-LBL-220 and shall have a barcode height of no more than 20 mm.", [Ref("CVM-LBL-220", "CVM-LBL-220")], ["conflict", "repeated_reference"]),
            R("3.2", "Label data content shall follow the requirements of the Corvane Supplier Quality Manual.", [Ref("Corvane Supplier Quality Manual", "CVM-SQM-001", resolution="name_alias")], ["name_alias"]),
            H("4", "Precedence"),
            R("4.1", "In the event of a conflict between this specification and CVM-CSR-200, the requirements of this specification shall take precedence.", [Ref("CVM-CSR-200", "CVM-CSR-200", relation=PREC)], ["precedence", "nested_reference"]),
        ],
    )
    return CaseSpec(
        case_id="dev_corvane_packaging",
        split="dev",
        description="Two provided documents: a logistics CSR and the packaging specification it references; exclusion, precedence, name aliases and cross-document conflicts.",
        documents=[csr, pkg],
        conflicts=[
            ConflictSpec(gid(c, "1.3"), gid(p, "2.1"), "applicability_conflict", ("Packaging Standard CVM-PKG-050 does not apply", "follow the dimensional guidelines of Packaging Standard CVM-PKG-050"), note="CSR excludes a standard that its referenced specification requires"),
            ConflictSpec(gid(c, "3.1"), gid(p, "2.2"), "contradictory_obligation", ("shipped only in returnable containers", "Expendable (one-way) cardboard containers shall be used")),
            ConflictSpec(gid(c, "2.2"), gid(p, "3.1"), "numeric_incompatibility", ("minimum barcode height of 25 mm", "barcode height of no more than 20 mm")),
            ConflictSpec(gid(c, "5.1"), gid(p, "4.1"), "potential_precedence_conflict", ("the requirements of this document shall take precedence", "the requirements of this specification shall take precedence")),
        ],
        non_conflicting=[
            (gid(c, "2.1"), gid(p, "3.1"), "both require CVM-LBL-220 labels; compatible"),
            (gid(c, "1.2"), gid(c, "4.2"), "both point to the packaging specification"),
        ],
    )


def _case_heldout_electronics() -> CaseSpec:
    d = "TSA-CSR-300"
    doc = DocSpec(
        identifier=d,
        company="TESSALY AUTOMOTIVE",
        title="Customer-Specific Requirements: Electronic Control Units",
        revision="2",
        date="2026-05-04",
        filename="tsa_csr_300.txt",
        blocks=[
            H("TSA 1", "Introduction"),
            Info("TSA 1.1", "This document supplements the general purchasing terms of Tessaly Automotive for suppliers of electronic control units."),
            H("TSA 2", "Design and Development"),
            R("TSA 2.1", "The supplier shall submit the design failure mode and effects analysis to Tessaly Automotive at each design review, and the analysis shall follow the methodology described in Tessaly procedure TSA-QP-031; where the analysis identifies a severity rating of 9 or 10, the supplier shall additionally document the mitigation actions according to TSA-QP-032 and shall report the results of all associated validation tests using the template in TSA-QF-031A.", [Ref("TSA-QP-031", "TSA-QP-031"), Ref("TSA-QP-032", "TSA-QP-032"), Ref("TSA-QF-031A", "TSA-QF-031A")], ["long", "multiple_references", "similar_identifiers", "position_middle", "position_end"]),
            R("TSA 2.2", "Software for electronic control units shall be developed according to TSA-SW-500 rev.3.", [Ref("TSA-SW-500 rev.3", "TSA-SW-500", "3")], ["revision"]),
            R("TSA 2.3", "Software release notes shall reference TSA-SW-500 Rev 4 for all releases issued after 1 July 2026.", [Ref("TSA-SW-500 Rev 4", "TSA-SW-500", "4")], ["revision", "ambiguous"]),
            H("TSA 3", "Manufacturing"),
            R("TSA 3.1", "Printed circuit board assemblies shall be cleaned according to the TSA-MP-210 specification, and ionic contamination shall not exceed 1.0 µg/cm² NaCl equivalent.", [Ref("TSA-MP-210", "TSA-MP-210")], ["alias_form"]),
            R("TSA 3.2", "Conformal coating shall be applied in accordance with Specification TSA MP 215 and cured for at least 30 minutes.", [Ref("TSA MP 215", "TSA-MP-215")], ["normalization"]),
            R("TSA 3.3", "Coated assemblies shall be inspected under ultraviolet light as described in TSA-MP-215, and assemblies with coating voids larger than 0.5 mm shall be rejected.", [Ref("TSA-MP-215", "TSA-MP-215")], ["normalization", "repeated_reference"]),
            R("TSA 3.4", "All solder joints shall be inspected by automated optical inspection on 100% of production units.", [], ["no_reference", "conflict"]),
            R("TSA 3.5", "Solder joint inspection shall be performed on a sampling basis of 5 units per production lot, as defined in TSA-QP-040.", [Ref("TSA-QP-040", "TSA-QP-040")], ["conflict", "similar_identifiers"]),
            H("TSA 4", "Traceability"),
            R("TSA 4.1", "Each electronic control unit shall carry a serial number label containing the Tessaly part number, the hardware revision, and the date of manufacture in the format defined in TSA-LB-077. The label shall remain legible for the service life of the vehicle. The label position is shown on drawing TSA-DRW-1188, sheet 2.", [Ref("TSA-LB-077", "TSA-LB-077"), Ref("TSA-DRW-1188", "TSA-DRW-1188")], ["long", "multiple_references", "distractor_numbers"]),
            R("TSA 4.2", "Traceability records shall be retained for at least 10 years.", [], ["no_reference"]),
            H("TSA 5", "Problem Solving"),
            R("TSA 5.1", "The supplier shall provide an initial containment response within 24 hours of notification of a nonconformance and shall complete the root cause analysis using the 8D format defined in TSA-QP-050.", [Ref("TSA-QP-050", "TSA-QP-050")], ["conflict", "distractor_token"]),
            R("TSA 5.2", "Initial containment responses shall be submitted no earlier than 72 hours after notification of a nonconformance, once sorting results are available.", [], ["no_reference", "conflict"]),
            R("TSA 5.3", "Warranty part analysis shall be performed according to TSA-QP-050 and TSA-WR-090.", [Ref("TSA-QP-050", "TSA-QP-050"), Ref("TSA-WR-090", "TSA-WR-090")], ["multiple_references", "repeated_reference"]),
        ],
    )
    return CaseSpec(
        case_id="heldout_tessaly_electronics",
        split="heldout",
        description="ECU CSR with prefixed chapter labels (TSA 2.1), long requirements, similar identifiers, revision variants and spacing aliases.",
        documents=[doc],
        conflicts=[
            ConflictSpec(gid(d, "TSA 3.4"), gid(d, "TSA 3.5"), "process_incompatibility", ("on 100% of production units", "on a sampling basis of 5 units per production lot")),
            ConflictSpec(gid(d, "TSA 5.1"), gid(d, "TSA 5.2"), "timing_incompatibility", ("within 24 hours of notification of a nonconformance", "no earlier than 72 hours after notification of a nonconformance")),
            ConflictSpec(gid(d, "TSA 2.2"), gid(d, "TSA 2.3"), "other_or_uncertain", ("TSA-SW-500 rev.3", "TSA-SW-500 Rev 4"), category="ambiguous", note="different software standard revisions; may be intentional by date"),
        ],
        non_conflicting=[
            (gid(d, "TSA 3.2"), gid(d, "TSA 3.3"), "both cite TSA-MP-215 for different steps"),
            (gid(d, "TSA 5.1"), gid(d, "TSA 5.3"), "both cite TSA-QP-050 without contradiction"),
        ],
    )


def _case_heldout_battery() -> CaseSpec:
    c, pp, m = "TSA-CSR-400", "TSA-PPAP-410", "TSA-MAT-420"
    csr = DocSpec(
        identifier=c,
        company="TESSALY AUTOMOTIVE",
        title="Customer-Specific Requirements for Battery Module Suppliers",
        revision="B",
        date="2026-03-20",
        filename="tsa_csr_400.txt",
        blocks=[
            H("1", "General"),
            Info("1.1", "This document applies to suppliers of high-voltage battery modules for Tessaly Automotive."),
            R("1.2", "Production part approval shall be obtained according to the Production Part Approval Procedure TSA-PPAP-410 before the first production shipment.", [Ref("TSA-PPAP-410", "TSA-PPAP-410")], ["available_document", "named_identifier"]),
            R("1.3", "Materials used in battery modules shall conform to TSA-MAT-420.", [Ref("TSA-MAT-420", "TSA-MAT-420")], ["available_document"]),
            R("1.4", "Where this document conflicts with TSA-PPAP-410, the requirements of this document shall take precedence.", [Ref("TSA-PPAP-410", "TSA-PPAP-410", relation=PREC)], ["precedence"]),
            H("2", "Materials"),
            R("2.1", "Specification TSA-CT-077 does not apply to battery modules and shall not be used for cell coatings.", [Ref("TSA-CT-077", "TSA-CT-077", relation=EXC)], ["exclusion", "conflict"]),
            R("2.2", "Cell housing adhesives shall have a maximum curing temperature of 60 °C.", [], ["no_reference", "conflict"]),
            R("2.3", "Thermal interface materials shall be qualified according to TSA-MAT-425, and the qualification report shall be included in the production part approval package described in TSA-PPAP-410.", [Ref("TSA-MAT-425", "TSA-MAT-425"), Ref("TSA-PPAP-410", "TSA-PPAP-410")], ["multiple_references", "missing_external"]),
            H("3", "Approval"),
            R("3.1", "The supplier shall submit the production part approval package at least 20 working days before the planned start of production.", [], ["no_reference", "conflict"]),
            R("3.2", "The approval package shall include the dimensional results report described in the Production Part Approval Procedure.", [Ref("Production Part Approval Procedure", "TSA-PPAP-410", resolution="name_alias")], ["name_alias"]),
            R("3.3", "Run-at-rate results shall be documented on form TSA-QF-031A.", [Ref("TSA-QF-031A", "TSA-QF-031A")], ["simple"]),
            R("3.4", "The supplier shall perform the run-at-rate verification during the first week of series production.", [], ["no_reference", "ambiguous"]),
        ],
    )
    ppap = DocSpec(
        identifier=pp,
        company="TESSALY AUTOMOTIVE",
        title="Production Part Approval Procedure",
        revision="F",
        date="2025-09-12",
        filename="tsa_ppap_410.txt",
        blocks=[
            H("1", "Purpose"),
            Info("1.1", "This procedure describes the production part approval process for Tessaly Automotive suppliers."),
            H("2", "Submission"),
            R("2.1", "The production part approval package shall be submitted no earlier than 10 working days before the planned start of production.", [], ["no_reference", "conflict"]),
            R("2.2", "The package shall contain material certifications according to TSA-MAT-420 and a control plan prepared according to TSA-QP-020.", [Ref("TSA-MAT-420", "TSA-MAT-420"), Ref("TSA-QP-020", "TSA-QP-020")], ["nested_reference", "multiple_references"]),
            R("2.3", "Dimensional results shall be reported for all characteristics shown on drawing TSA-DRW-2210.", [Ref("TSA-DRW-2210", "TSA-DRW-2210")], ["simple"]),
            R("2.4", "Run-at-rate verification shall be completed before the production part approval package is submitted.", [], ["no_reference", "ambiguous"]),
            H("3", "Precedence"),
            R("3.1", "Where this procedure conflicts with any customer-specific requirements document, including TSA-CSR-400, this procedure shall govern.", [Ref("TSA-CSR-400", "TSA-CSR-400", relation=PREC)], ["precedence"]),
        ],
    )
    mat = DocSpec(
        identifier=m,
        company="TESSALY AUTOMOTIVE",
        title="Material Specification for Battery Module Components",
        revision="C",
        date="2025-12-01",
        filename="tsa_mat_420.txt",
        blocks=[
            H("1", "Scope"),
            Info("1.1", "This specification applies to materials used in battery module assemblies."),
            H("2", "Adhesives"),
            R("2.1", "Cell housing adhesives shall be cured at a minimum temperature of 80 °C for 20 minutes.", [], ["no_reference", "conflict"]),
            R("2.2", "Adhesive lot certificates shall be retained for at least 5 years.", [], ["no_reference"]),
            H("3", "Coatings"),
            R("3.1", "Cell coatings shall meet the dielectric strength requirements of Specification TSA-CT-077.", [Ref("TSA-CT-077", "TSA-CT-077")], ["conflict"]),
            R("3.2", "Coating thickness shall be measured according to TSA-CT-080 at a frequency of one sample per shift.", [Ref("TSA-CT-080", "TSA-CT-080")], ["simple"]),
        ],
    )
    return CaseSpec(
        case_id="heldout_tessaly_battery",
        split="heldout",
        description="Three provided documents (CSR, approval procedure, material specification) with nested references, mutual precedence, an exclusion contradicted downstream and cross-document conflicts.",
        documents=[csr, ppap, mat],
        conflicts=[
            ConflictSpec(gid(c, "1.4"), gid(pp, "3.1"), "potential_precedence_conflict", ("the requirements of this document shall take precedence", "this procedure shall govern")),
            ConflictSpec(gid(c, "2.1"), gid(m, "3.1"), "applicability_conflict", ("Specification TSA-CT-077 does not apply to battery modules", "Cell coatings shall meet the dielectric strength requirements of Specification TSA-CT-077"), note="exclusion in the CSR contradicted by a document the CSR references"),
            ConflictSpec(gid(c, "2.2"), gid(m, "2.1"), "numeric_incompatibility", ("maximum curing temperature of 60 °C", "cured at a minimum temperature of 80 °C")),
            ConflictSpec(gid(c, "3.1"), gid(pp, "2.1"), "timing_incompatibility", ("at least 20 working days before the planned start of production", "no earlier than 10 working days before the planned start of production")),
            ConflictSpec(gid(c, "3.4"), gid(pp, "2.4"), "timing_incompatibility", ("during the first week of series production", "completed before the production part approval package is submitted"), category="ambiguous", note="order of run-at-rate and submission is unclear rather than strictly contradictory"),
        ],
        non_conflicting=[
            (gid(c, "1.2"), gid(c, "2.3"), "both cite TSA-PPAP-410 without contradiction"),
            (gid(c, "1.3"), gid(pp, "2.2"), "both require TSA-MAT-420"),
        ],
    )


def _case_heldout_interior() -> CaseSpec:
    d = "CVM-CSR-500"
    doc = DocSpec(
        identifier=d,
        company="CORVANE MOTORS",
        title="Customer-Specific Requirements: Interior Trim Components",
        revision="A",
        date="2026-07-01",
        filename="cvm_csr_500.txt",
        blocks=[
            H("1", "Scope"),
            Info("1.1", "This document applies to suppliers of interior trim components delivered to Corvane Motors."),
            R("1.2", "Suppliers shall apply this document in place of CVM-CSR-450 Revision C, which this document supersedes.", [Ref("CVM-CSR-450 Revision C", "CVM-CSR-450", "C", relation=SUP)], ["supersession", "revision"]),
            H("2", "Materials"),
            R("2.1", "Interior trim materials shall meet the flammability requirements of CVM-MS-610 and the odor requirements of CVM-MS-612, and the supplier shall submit test results from a laboratory accredited according to the Corvane laboratory policy CVM-LAB-005.", [Ref("CVM-MS-610", "CVM-MS-610"), Ref("CVM-MS-612", "CVM-MS-612"), Ref("CVM-LAB-005", "CVM-LAB-005")], ["long", "multiple_references", "similar_identifiers"]),
            R("2.2", "Volatile organic compound emissions shall be tested according to CVM-MS-615 at a test temperature of 65 °C. Results shall be reported in µg/m³ to Corvane Motors within 5 working days of test completion.", [Ref("CVM-MS-615", "CVM-MS-615")], ["similar_identifiers"]),
            R("2.3", "Materials that contain recycled content shall be declared in the material data system before part submission. The declaration shall include the supplier name, material grade, and recycled content percentage.", [], ["no_reference"]),
            H("3", "Appearance"),
            R("3.1", "Color and gloss shall match the master sample approved under CVM-AP-300. Visual evaluation shall be performed under D65 lighting at 1000 lux.", [Ref("CVM-AP-300", "CVM-AP-300")], ["distractor_token"]),
            R("3.2", "Grain and texture shall conform to CVM-AP-310 as shown on drawing DRW-7731.", [Ref("CVM-AP-310", "CVM-AP-310"), Ref("DRW-7731", "DRW-7731")], ["multiple_references", "similar_identifiers"]),
            H("4", "Approval of Changes"),
            R("4.1", "Any change to a material formulation shall require written approval from Corvane Motors before implementation.", [], ["no_reference", "conflict"]),
            R("4.2", "Changes to a material formulation shall be implemented without approval from Corvane Motors when they are recorded in the supplier's internal change log according to CVM-QS-110.", [Ref("CVM-QS-110", "CVM-QS-110")], ["conflict"]),
            H("5", "Shipping"),
            R("5.1", "Finished trim parts shall be packaged according to CVM-PKG-210 and shall be protected from ultraviolet exposure during transport.", [Ref("CVM-PKG-210", "CVM-PKG-210")], ["missing_external"]),
            R("5.2", "Parts shall be labeled according to CVM-LBL-220 and CVM-PKG-210.", [Ref("CVM-LBL-220", "CVM-LBL-220"), Ref("CVM-PKG-210", "CVM-PKG-210")], ["multiple_references", "repeated_reference"]),
            H("6", "Testing"),
            R("6.1", "Test results from internal supplier laboratories shall not be accepted for flammability testing under CVM-MS-610.", [Ref("CVM-MS-610", "CVM-MS-610")], ["repeated_reference"]),
            R("6.2", "Odor testing according to CVM-MS-612 shall be performed at 80 °C.", [Ref("CVM-MS-612", "CVM-MS-612")], ["conflict", "repeated_reference"]),
            R("6.3", "All material tests shall be performed at a maximum test temperature of 70 °C.", [], ["no_reference", "conflict"]),
        ],
    )
    return CaseSpec(
        case_id="heldout_corvane_interior",
        split="heldout",
        description="Interior trim CSR with supersession, dense similar identifiers, distractor tokens, unavailable external documents and non-conflicting overlaps.",
        documents=[doc],
        conflicts=[
            ConflictSpec(gid(d, "4.1"), gid(d, "4.2"), "contradictory_obligation", ("shall require written approval from Corvane Motors before implementation", "shall be implemented without approval from Corvane Motors")),
            ConflictSpec(gid(d, "6.2"), gid(d, "6.3"), "numeric_incompatibility", ("shall be performed at 80 °C", "maximum test temperature of 70 °C")),
        ],
        non_conflicting=[
            (gid(d, "5.1"), gid(d, "5.2"), "both cite CVM-PKG-210"),
            (gid(d, "2.1"), gid(d, "6.1"), "both concern flammability testing under CVM-MS-610; compatible"),
            (gid(d, "2.2"), gid(d, "6.3"), "65 °C is within the 70 °C maximum"),
            (gid(d, "2.1"), gid(d, "6.2"), "both cite CVM-MS-612"),
        ],
    )


def all_cases() -> list[CaseSpec]:
    return [_case_dev_production(), _case_dev_packaging(), _case_heldout_electronics(), _case_heldout_battery(), _case_heldout_interior()]


# ---------------------------------------------------------------------------
# Parameterised documents for later ablations
# ---------------------------------------------------------------------------

_FILLER = [
    "The supplier shall keep the related records available for review during audits.",
    "Deviations from this clause require documented justification by the supplier quality team.",
    "Measurement equipment used for this activity shall be calibrated at defined intervals.",
    "Results shall be communicated to the responsible Corvane plant contact on request.",
    "Personnel performing this activity shall be trained and qualified.",
]


def generate_scaled_case(
    case_id: str = "scaled_case",
    *,
    n_requirements: int = 40,
    refs_per_requirement: int = 2,
    filler_sentences: int = 2,
    position: str = "mixed",
    seed: int = 0,
    split: str = "heldout",
) -> CaseSpec:
    """A long single-document case with controlled reference density and position.

    ``position``: start | middle | end | mixed (where in a requirement the references appear).
    """
    rng = random.Random(seed)
    ident = f"CVM-CSR-{900 + seed % 100}"
    blocks: list = [H("1", "Scope"), Info("1.1", "This generated document is used to measure the effect of document length and reference density.")]
    section = 2
    for i in range(n_requirements):
        if i % 10 == 0:
            blocks.append(H(str(section), f"Generated Section {section}"))
            section += 1
        chapter = f"{section - 1}.{i % 10 + 1}"
        ids = [f"SYN-{rng.choice(['QS', 'TM', 'ES', 'MS'])}-{rng.randint(100, 999)}" for _ in range(refs_per_requirement)]
        ids = list(dict.fromkeys(ids))
        ref_clause = "according to " + " and ".join(ids) if ids else ""
        fillers = " ".join(rng.choice(_FILLER) for _ in range(filler_sentences))
        core = "The supplier shall perform the generated activity"
        where = position if position != "mixed" else rng.choice(["start", "middle", "end"])
        if not ids:
            text = f"{core}. {fillers}"
        elif where == "start":
            text = f"{core} {ref_clause}. {fillers}"
        elif where == "middle":
            half = filler_sentences // 2
            first = " ".join(fillers.split(". ")[:half])
            text = f"{first + ' ' if first else ''}{core} {ref_clause}. {fillers}"
        else:
            text = f"{fillers} {core} {ref_clause}."
        text = re.sub(r"\.\.+", ".", text).strip()
        blocks.append(R(chapter, text, [Ref(x, x) for x in ids], [f"position_{where}", f"refs_{len(ids)}"]))
    doc = DocSpec(ident, "CORVANE MOTORS", "Generated Requirements for Length Experiments", "A", "2026-01-01", f"{case_id}.txt", blocks)
    return CaseSpec(case_id=case_id, split=split, description=f"generated: n={n_requirements}, refs={refs_per_requirement}, filler={filler_sentences}, position={position}, seed={seed}", documents=[doc])


# ---------------------------------------------------------------------------
# Writing the dataset
# ---------------------------------------------------------------------------


def write_dataset(out_dir: str | Path, cases: list[CaseSpec] | None = None, name: str = DATASET_NAME) -> Path:
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
    manifest = {
        "name": name,
        "version": DATASET_VERSION,
        "description": "Fictional CSR-like documents with hand-specified gold annotations.",
        "generator": "csr_lab.datasets.synthetic",
        "cases": manifest_cases,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
