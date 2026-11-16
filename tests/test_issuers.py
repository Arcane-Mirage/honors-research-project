"""tests/test_issuers.py

Issuer classification (extraction/issuers.py) and issuer groups (relationships/issuers.py).
"""

from __future__ import annotations

import pytest
from helpers import ScriptedProvider

from csr_lab.config import load_config
from csr_lab.extraction.issuers import IssuerCatalog
from csr_lab.ingestion.loader import load_bundle
from csr_lab.pipelines.registry import build_workflow, derive_strategy, get_strategy
from csr_lab.pipelines.run import load_inputs, run_analysis
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.queries import to_mermaid
from csr_lab.relationships.render import to_llm_markdown
from csr_lab.relationships.store import GraphStore, _merge_attributes
from csr_lab.schemas import AnalysisResult

CSR = """\
TRATON TEST STANDARD
Document ID: CVS900
Revision: A

1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 The supplier shall be certified to Standard ISO 9001.
2.2 Material data shall be reported according to Procedure ISQ-011.
2.3 Packaging Standard HVG-PKG-050 applies.
"""


@pytest.fixture
def csr_docs(tmp_path):
    path = tmp_path / "cvs900.txt"
    path.write_text(CSR, encoding="utf-8")
    return load_bundle([path])


def _staged(**issuer_options):
    return derive_strategy("staged_issuers_test", "staged", {"issuer_classification": issuer_options})


def _by_doc(result: AnalysisResult) -> dict[str, str | None]:
    return {a.document_id: a.issuer for a in result.issuers.assignments}


# -- the shipped issuer list ----------------------------------------------------------


def test_default_catalog_has_the_seed_issuers_in_order():
    names = [s.name for s in load_config().issuers.known]
    assert names == [
        "TRATON", "International Motors", "MAN", "Scania", "Volkswagen Group", "VDA", "AIAG", "IATF", "ISO/IEC",
        "European Standards (EN/DIN)", "EU Legislation", "US Legislation", "International Treaties",
    ]


@pytest.mark.parametrize(
    ("text", "issuer"),
    [
        ("CVS10", "TRATON"),
        ("CVS-168", "TRATON"),
        ("M-3335", "MAN"),
        ("MAN-239-1", "MAN"),
        ("STD4158", "Scania"),
        ("VW-50156", "Volkswagen Group"),
        ("FORMEL-Q-KONKRET", "Volkswagen Group"),
        ("VDA-6.3", "VDA"),
        ("CQI-9", "AIAG"),
        ("APQP Manual", "AIAG"),
        ("IATF-16949", "IATF"),
        ("ISO-9001", "ISO/IEC"),
        ("ISO/IEC 17025", "ISO/IEC"),
        ("DIN-EN-ISO-9001", "European Standards (EN/DIN)"),
        ("EN-61340-5-1", "European Standards (EN/DIN)"),
        ("2000/53/EC", "EU Legislation"),
        ("Regulation (EC) No 1907/2006", "EU Legislation"),
        ("DIRECTIVE-200053EC-OF-THE-EUROPEAN-PARLIAMENT", "EU Legislation"),  # identifier from a EUR-Lex file name
        ("2003138EC-COMMISSION-DECISION-OF-27", "EU Legislation"),
        ("REACH", "EU Legislation"),
        ("CFR-751", "US Legislation"),
        ("Stockholm Convention", "International Treaties"),
        ("ISQ-011-PR", "International Motors"),
        ("MPAPS-B-50", "International Motors"),
        ("APQP Workbook", None),  # a customer's workbook, not the AIAG manual
        ("HVG-PKG-050", None),
    ],
)
def test_seed_patterns(text, issuer):
    assert IssuerCatalog(specs=load_config().issuers.known).by_rules([text]) == issuer


def test_answers_match_names_leniently_and_new_issuers_are_added():
    catalog = IssuerCatalog(specs=load_config().issuers.known)
    assert catalog.resolve("european standards") == "European Standards (EN/DIN)"  # without its parenthesis
    assert catalog.resolve("ISO") == "ISO/IEC"  # an alias
    assert catalog.resolve("Traton Group") == "TRATON"
    assert catalog.resolve("Audi") == "Volkswagen Group"  # a customer's name means its issuer
    assert catalog.resolve("Navistar") == "International Motors"
    assert catalog.resolve("International") is None  # ambiguous: also an ordinary word
    assert catalog.resolve("Acme Corp") is None
    assert catalog.add("Acme Corp") == "Acme Corp"
    assert catalog.add("acme corp") == "Acme Corp"  # no duplicate spelling
    assert catalog.names[-1] == "Acme Corp" and not catalog.is_known("Acme Corp")


def test_every_customer_belongs_to_an_issuer():
    from csr_lab.config import apply_experiment, load_experiment

    cfg = load_config()
    issuers = {i.name for i in cfg.issuers.known}
    assert {c.name: c.issuer for c in cfg.system_prompt.customers} == {
        "VWTB": "TRATON", "Navistar": "International Motors", "MAN": "MAN", "Scania": "Scania",
        **dict.fromkeys(["Volkswagen", "Audi", "Porsche", "ŠKODA AUTO", "Bentley", "SEAT"], "Volkswagen Group"),
    }
    assert all(c.issuer in issuers for c in cfg.system_prompt.customers)
    assert "TRATON" not in cfg.system_prompt.customer_names  # an issuer need not be a customer
    # A dataset's reader names its own customer: it becomes an issuer when it is not one yet
    exp = apply_experiment(cfg, load_experiment("demo_mock"))
    assert [(c.name, c.issuer) for c in exp.system_prompt.customers] == [("Global Motors", "Global Motors")]
    auto = exp.issuers.known[-1]
    assert auto.name == "Global Motors" and auto.aliases == ["Global Motors LLC"]  # 'Global' is ambiguous: not an issuer name


# -- in the pipeline ------------------------------------------------------------------


def test_llm_method_groups_documents_and_records_cross_group_references(csr_docs, config):
    result = run_analysis(csr_docs, strategy=_staged(method="llm"), config=config)
    AnalysisResult.model_validate_json(result.model_dump_json())
    rep = result.issuers
    assert rep.enabled and rep.method == "llm"
    assert _by_doc(result) == {
        "DOC-CVS900": "TRATON",
        "DOC-ISO-9001": "ISO/IEC",  # the mock answered 'ISO': matched by alias
        "DOC-ISQ-011": "International Motors",
        "DOC-HVG-PKG-050": None,
    }
    assert rep.new_issuers == []
    assert rep.unclassified_document_ids == ["DOC-HVG-PKG-050"]
    assert rep.by_source == {"llm": 3, "none": 1}
    assert {g.issuer: (g.known, g.input_document_ids) for g in rep.groups} == {
        "TRATON": (True, ["DOC-CVS900"]),
        "International Motors": (True, []),
        "ISO/IEC": (True, []),
    }
    assert [g.issuer for g in rep.groups] == ["TRATON", "International Motors", "ISO/IEC"]  # in list order
    links = {(x.source_issuer, x.target_issuer): x.document_pairs for x in rep.links}
    assert links == {("TRATON", "ISO/IEC"): 1, ("TRATON", "International Motors"): 1, ("TRATON", "unknown"): 1}
    # On the exported records and the graph
    assert result.input_documents[0].issuer == "TRATON" and result.input_documents[0].issuer_source == "llm"
    assert {r.document_id: r.issuer for r in result.referenced_documents}["DOC-ISQ-011"] == "International Motors"
    graph = graph_from_result(result)
    assert graph.node("DOC-ISO-9001").attributes["issuer"] == "ISO/IEC"
    # One batched call, its own stage cost
    calls = [c for c in result.metadata.llm_calls if c.stage == "classify_issuers"]
    assert len(calls) == 1 and result.metadata.operational.stage_costs["classify_issuers"].calls == 1
    assert "issuer_classification" in result.metadata.prompt_versions


def test_new_issuers_can_be_switched_off(csr_docs, config):
    def respond(task, payload, request):
        if task == "issuer_classification":
            return {"results": [{"key": d["key"], "issuer": "Acme Corp", "new_issuer": True} for d in payload["documents"]]}
        return {"results": [], "assessments": []}

    result = run_analysis(csr_docs, strategy=_staged(method="llm", allow_new=False), config=config, provider=ScriptedProvider(respond, scripts_issuers=True))
    assert result.issuers.new_issuers == [] and result.issuers.groups == []
    a = result.issuers.assignments[0]
    assert a.issuer is None and a.model_answer == "Acme Corp"


def test_rules_method_makes_no_model_call(csr_docs, config):
    result = run_analysis(csr_docs, strategy=_staged(method="rules"), config=config)
    assert "classify_issuers" not in result.metadata.operational.stage_costs
    assert _by_doc(result) == {"DOC-CVS900": "TRATON", "DOC-ISO-9001": "ISO/IEC", "DOC-ISQ-011": "International Motors", "DOC-HVG-PKG-050": None}
    assert result.issuers.by_source == {"rules": 3, "none": 1}


def test_hybrid_asks_the_model_only_about_what_the_rules_leave(csr_docs, config):
    asked: list[list[str]] = []

    def respond(task, payload, request):
        if task == "issuer_classification":
            asked.append([d["identifier"] for d in payload["documents"]])
            return {"results": [{"key": d["key"], "issuer": "unknown"} for d in payload["documents"]]}
        if task == "reference_extraction":
            refs = {"2.1": "ISO 9001", "2.2": "ISQ-011", "2.3": "HVG-PKG-050"}
            return {
                "results": [
                    {"requirement_id": r["requirement_id"], "references": [{"identifier": refs[r["chapter"]], "quoted_text": refs[r["chapter"]]}] if r.get("chapter") in refs else []}
                    for r in payload["requirements"]
                ]
            }
        return {"assessments": []}

    result = run_analysis(csr_docs, strategy=_staged(method="hybrid"), config=config, provider=ScriptedProvider(respond, scripts_issuers=True))
    assert asked == [["HVG-PKG-050"]]
    assert result.issuers.by_source == {"none": 1, "rules": 3}


def test_new_issuer_is_offered_to_later_batches(csr_docs, config):
    seen: list[list[str]] = []

    def respond(task, payload, request):
        if task == "issuer_classification":
            seen.append([i["name"] for i in payload["issuers"]])
            return {"results": [{"key": d["key"], "issuer": "Acme Corp" if len(seen) == 1 else "ACME corp.", "new_issuer": True} for d in payload["documents"]]}
        if task == "reference_extraction":
            return {
                "results": [
                    {"requirement_id": r["requirement_id"], "references": [{"identifier": "ISQ-011", "quoted_text": "ISQ-011"}] if "ISQ-011" in r["text"] else []}
                    for r in payload["requirements"]
                ]
            }
        return {"assessments": []}

    result = run_analysis(csr_docs, strategy=_staged(method="llm", batch_size=1), config=config, provider=ScriptedProvider(respond, scripts_issuers=True))
    assert len(seen) == 2 and "Acme Corp" not in seen[0] and seen[1][-1] == "Acme Corp"
    assert set(_by_doc(result).values()) == {"Acme Corp"}  # 'ACME corp.' matched the name added for the first document
    assert result.issuers.new_issuers == ["Acme Corp"]
    assert all(a.new_issuer for a in result.issuers.assignments)


def test_failed_call_leaves_documents_unclassified(csr_docs, config):
    def respond(task, payload, request):
        return "not json" if task == "issuer_classification" else {"results": [], "assessments": []}

    result = run_analysis(csr_docs, strategy=_staged(method="llm"), config=config, provider=ScriptedProvider(respond, scripts_issuers=True))
    assert result.status == "completed_with_errors"
    assert result.issuers.groups == [] and result.issuers.unclassified_document_ids == ["DOC-CVS900"]
    assert any("issuer classification failed" in lim.message for lim in result.coverage.limitations)


def test_registry_issuer_wins_and_is_not_asked(tmp_path, config):
    (tmp_path / "cvs900.txt").write_text(CSR, encoding="utf-8")
    (tmp_path / "documents.yaml").write_text(
        "documents:\n"
        "  - file: cvs900.txt\n    issuer: TRATON Group\n"  # an alias of a known issuer
        "  - identifier: ISQ-011\n    issuer: International Motors\n",  # a document known only by identifier
        encoding="utf-8",
    )
    docs, registry = load_inputs([tmp_path / "cvs900.txt"], config)
    assert docs[0].issuer == "TRATON Group" and docs[0].issuer_source == "registry"
    result = run_analysis(docs, strategy=_staged(), config=config, registry=registry)
    by_doc = {a.document_id: (a.issuer, a.source) for a in result.issuers.assignments}
    assert by_doc["DOC-CVS900"] == ("TRATON", "registry")
    assert by_doc["DOC-ISQ-011"] == ("International Motors", "registry")
    assert result.issuers.new_issuers == []  # named in documents.yaml, not by the model
    assert result.input_documents[0].issuer == "TRATON"


def test_saved_graph_issuers_are_reused(tmp_path, csr_docs, config):
    first = run_analysis(csr_docs, strategy="graph_assisted", config=config)
    store = GraphStore.load_or_new(tmp_path / "g.json")
    store.merge_result(first)
    assert store.summary()["documents_by_issuer"] == {"TRATON": 1, "ISO/IEC": 1, "International Motors": 1, "unknown": 1}
    again = run_analysis(csr_docs, strategy="graph_assisted", config=config, saved_graph=store)
    by_source = again.issuers.by_source
    assert by_source.get("saved_graph") == 3  # the unknown one is asked again
    assert again.metadata.operational.stage_costs["classify_issuers"].calls == 1
    assert again.issuers.new_issuers == []  # International Motors was already in the saved graph


def test_store_keeps_first_issuer_unless_documents_yaml_says_otherwise():
    attrs = {"issuer": "ISO/IEC", "issuer_source": "llm"}
    _merge_attributes(attrs, {"issuer": "European Standards (EN/DIN)", "issuer_source": "llm"})
    assert attrs == {"issuer": "ISO/IEC", "issuer_source": "llm"}
    _merge_attributes(attrs, {"issuer": "European Standards (EN/DIN)", "issuer_source": "registry"})
    assert attrs == {"issuer": "European Standards (EN/DIN)", "issuer_source": "registry"}
    empty: dict = {}
    _merge_attributes(empty, {"issuer": "TRATON", "issuer_source": "rules"})
    assert empty == {"issuer": "TRATON", "issuer_source": "rules"}


def test_views_group_documents_by_issuer(csr_docs, config):
    graph = graph_from_result(run_analysis(csr_docs, strategy=_staged(), config=config))
    md = to_llm_markdown(graph)
    assert "### TRATON" in md and "### Issuer not identified" in md
    assert "## References between issuer groups" in md and "- TRATON → ISO/IEC: 1 document (CVS900 → ISO-9001)" in md
    mmd = to_mermaid(graph, documents_only=True)
    assert 'subgraph g_TRATON["TRATON"]' in mmd and 'subgraph g_ISO_IEC["ISO/IEC"]' in mmd


def test_stage_follows_the_option(csr_docs, config):
    for name in ("single_pass", "staged", "graph_assisted"):
        assert "classify_issuers" in build_workflow(get_strategy(name)).get_graph().nodes
    off = _staged(enabled=False)
    assert "classify_issuers" not in build_workflow(off).get_graph().nodes
    result = run_analysis(csr_docs, strategy=off, config=config)
    assert not result.issuers.enabled and result.issuers.assignments == []
    assert result.input_documents[0].issuer is None


# -- uses of the issuer groups ---------------------------------------------------------


def _conflict(a: str, b: str):
    from csr_lab.schemas import ConflictType, PotentialConflict

    return PotentialConflict(conflict_id="C1", related_requirement_ids=[a, b], conflict_type=ConflictType.OTHER_OR_UNCERTAIN, description="d", reason_for_review="r")


def test_issuer_tier_label_is_separate_from_stated_precedence():
    from csr_lab.conflicts.detector import classify_issuer_tiers

    cfg = load_config()
    tiers = {s.name: s.tier for s in cfg.issuers.known if s.tier}
    assert cfg.issuers.tier_order == ["law", "customer", "industry_standard"]
    assert tiers["ISO/IEC"] == tiers["IATF"] == "industry_standard"  # ISO counts as an industry standard
    docs = {"R1": "DOC-CVS55", "R2": "DOC-2000-53-EC", "R3": "DOC-ISO-9001", "R4": "DOC-IATF-16949", "R5": "DOC-X"}
    issuer = {"DOC-CVS55": "TRATON", "DOC-2000-53-EC": "EU Legislation", "DOC-ISO-9001": "ISO/IEC", "DOC-IATF-16949": "IATF", "DOC-X": "Acme"}
    law, same, unknown, customer = _conflict("R1", "R2"), _conflict("R3", "R4"), _conflict("R1", "R5"), _conflict("R1", "R3")
    classify_issuer_tiers([law, same, unknown, customer], docs, issuer, tiers, cfg.issuers.tier_order, {"DOC-CVS55": "CVS55"})
    assert law.issuer_tier_status == "implied_by_issuer_tier" and law.issuer_tier_prevailing_document_id == "DOC-2000-53-EC"
    assert law.issuer_tier_basis == ["CVS55: TRATON (customer)", "DOC-2000-53-EC: EU Legislation (law)"]
    assert customer.issuer_tier_prevailing_document_id == "DOC-CVS55"  # customer above industry standard
    assert same.issuer_tier_status == "same_tier" and same.issuer_tier_prevailing_document_id is None
    assert unknown.issuer_tier_status == "not_assessed"  # an issuer without a tier
    assert {c.precedence_status for c in (law, same, unknown, customer)} == {"not_assessed"}  # stated precedence untouched


def test_tiers_must_be_in_the_tier_order():
    from csr_lab.config import IssuerCatalogSettings

    with pytest.raises(ValueError, match="not in issuers.tier_order"):
        IssuerCatalogSettings(known=[{"name": "X", "tier": "law"}], tier_order=["customer"])


def test_saved_graph_check_can_pair_documents_of_the_same_issuer():
    from csr_lab.conflicts.detector import saved_graph_candidates
    from csr_lab.drafts import RequirementDraft
    from csr_lab.relationships.store import SavedUnit

    new = RequirementDraft(requirement_id="REQ-NEW-1", document_id="DOC-CVS10-REV-2025", chapter="4.6", text="Suppliers shall report all changes of the production process to the customer.")
    saved = {"REQ-OLD-1": SavedUnit(unit_id="REQ-OLD-1", kind="chapter", document_id="DOC-CVS168", family_id="DOC-CVS168", identifier="CVS168", revision=None, chapter="2", text="Suppliers shall report changes of the production process.", issuer="TRATON")}
    args = {"max_pairs": 10, "min_score": 0.0, "max_fanout": 8, "families": {"DOC-CVS10-REV-2025": "DOC-CVS10"}, "issuers": {"DOC-CVS10-REV-2025": "TRATON"}}
    off, _ = saved_graph_candidates([new], {}, saved, **args)
    on, _ = saved_graph_candidates([new], {}, saved, **args, same_issuer=True)
    assert off == []  # no reference links them
    assert [c.selectors for c in on] == [["saved_graph:same_issuer:TRATON"]]
    from csr_lab.config import GraphSettings

    assert GraphSettings().saved_same_issuer is False  # a toggle, off unless asked for


LABELLED_CSR = """\
TRATON TEST STANDARD
Document ID: CVS901
Revision: A

1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 Material data shall be reported according to Standard MAN 239.
2.2 Packaging
Scania: Packaging shall follow Standard SCA 4172.
"""


def test_reference_to_another_customers_standard_in_unlabelled_text_is_flagged(tmp_path, config):
    path = tmp_path / "cvs901.txt"
    path.write_text(LABELLED_CSR, encoding="utf-8")
    result = run_analysis(load_bundle([path]), strategy=_staged(), config=config)
    refs = result.coverage.references
    assert result.coverage.customer_scope.blocks == 1  # the document uses customer labels
    assert refs.customer_specific_unlabelled == 1
    (flag,) = refs.customer_specific_mentions
    assert (flag.cited_document_id, flag.cited_issuer, flag.customers) == ("DOC-MAN-239", "MAN", ["MAN"])
    assert "MAN 239" in flag.quoted_text
    off = run_analysis(load_bundle([path]), strategy=derive_strategy("no_scope", "staged", {"customer_scoping": "off"}), config=config)
    assert off.coverage.references.customer_specific_unlabelled is None


def test_registry_labels_can_be_hidden_from_the_classifier(tmp_path, config):
    (tmp_path / "cvs900.txt").write_text(CSR, encoding="utf-8")
    (tmp_path / "documents.yaml").write_text("documents:\n  - file: cvs900.txt\n    issuer: Secret Label Corp\n", encoding="utf-8")
    docs, registry = load_inputs([tmp_path / "cvs900.txt"], config)
    offered: list[list[str]] = []

    def respond(task, payload, request):
        if task == "issuer_classification":
            offered.append([i["name"] for i in payload["issuers"]])
            return {"results": [{"key": d["key"], "issuer": "TRATON"} for d in payload["documents"]]}
        return {"results": [], "assessments": []}

    hidden = _staged(method="llm", use_registry=False)
    result = run_analysis(docs, strategy=hidden, config=config, registry=registry, provider=ScriptedProvider(respond, scripts_issuers=True))
    assert offered and all("Secret Label Corp" not in names for names in offered)  # the label does not leak into the prompt
    assert _by_doc(result)["DOC-CVS900"] == "TRATON"


def test_issuer_eval_scores_methods_against_documents_yaml(tmp_path, config):
    from csr_lab.evaluation.issuers import evaluate_issuers, report_markdown

    (tmp_path / "cvs900.txt").write_text(CSR, encoding="utf-8")
    (tmp_path / "documents.yaml").write_text(
        "documents:\n"
        "  - file: cvs900.txt\n    issuer: TRATON\n"
        "  - identifier: ISO 9001\n    issuer: ISO/IEC\n"
        "  - identifier: HVG-PKG-050\n    issuer: Halvard\n",  # not a configured issuer
        encoding="utf-8",
    )
    results, info = evaluate_issuers([tmp_path / "cvs900.txt"], config=config, methods=["rules", "llm"])
    assert info["labelled"] == 3
    rules, llm = results
    assert rules.calls == 0 and llm.calls == 1
    assert rules.summary["all"] == {"labelled": 3, "classified": 2, "correct": 2, "accuracy": 0.6667, "coverage": 0.6667, "precision": 1.0}
    assert rules.summary["gold_in_list"]["accuracy"] == 1.0
    cited = next(r for r in rules.rows if r.document_id == "DOC-ISO-9001")
    assert not cited.loaded and cited.correct
    assert "| HVG-PKG-050 | Halvard (not in list) | unclassified |" in report_markdown(results, info)


def test_cited_only_documents_get_the_citing_sentence_as_context(csr_docs):
    from csr_lab.evaluation.issuers import citing_context

    text = citing_context(csr_docs, ["ISO-9001", "ISO 9001"], 60)
    assert "Standard ISO 9001" in text
