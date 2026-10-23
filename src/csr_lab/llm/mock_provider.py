"""Deterministic offline mock model.

The mock implements ``LLMProvider`` exactly like a real provider: it receives
the rendered prompt messages, reads the ``<input_json>`` block, and returns
JSON text that then goes through the normal parsing, validation and retry
path. It dispatches on the task name of the request.

Error injection (``MockSettings``) makes the output imperfect so that the
verification and evaluation code paths are exercised. Each injected error is a
seeded draw keyed by (seed, run seed, task kind, item). Because the key
contains the item and not the strategy, two strategies asking the same
question about the same item get the same answer, and a different kind of
question (for example a targeted verification) is an independent draw.

Mock results only demonstrate that the system works end to end.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from csr_lab.config import MockSettings
from csr_lab.ingestion.segmentation import segment_document
from csr_lab.llm import mock_rules as rules
from csr_lab.llm.interface import GenerationRequest, GenerationResponse, LLMProvider
from csr_lab.prompts import parse_input_payload
from csr_lab.schemas import InputDocument


def _collapse(text: str) -> str:
    return " ".join(text.split())


class MockProvider(LLMProvider):
    is_mock = True

    def __init__(self, settings: MockSettings, run_seed: int = 0, model: str = "mock-heuristic-v1"):
        self.settings = settings
        self.run_seed = run_seed
        self.model = model

    def describe(self) -> dict[str, Any]:
        return {"provider": "mock", "model": self.model, "run_seed": self.run_seed, **self.settings.model_dump()}

    # -- seeded randomness -------------------------------------------------

    def _chance(self, rate: float, *key: object) -> bool:
        if rate <= 0:
            return False
        digest = hashlib.sha256("|".join(map(str, (self.settings.seed, self.run_seed, *key))).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < rate

    def _pick(self, options: list[Any], *key: object) -> Any:
        digest = hashlib.sha256("|".join(map(str, (self.settings.seed, self.run_seed, *key))).encode()).digest()
        return options[int.from_bytes(digest[:4], "big") % len(options)]

    # -- dispatch ----------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        user = next(m["content"] for m in reversed(request.messages) if m["role"] == "user" and "<input_json>" in m["content"])
        payload = parse_input_payload(user)
        handler = {
            "single_pass_extraction": self._single_pass,
            "requirement_identification": self._identify,
            "reference_extraction": self._references,
            "reference_verification": self._verify,
            "conflict_analysis": self._conflicts,
        }.get(request.task)
        if handler is None:
            raise ValueError(f"mock provider has no behaviour for task {request.task!r}")
        text = json.dumps(handler(payload), ensure_ascii=False)
        if self._chance(self.settings.invalid_json_rate, "invalid_json", request.task, hash_text(user), len(request.messages)):
            text = text[: max(1, len(text) // 2)]  # truncated JSON, as a real model sometimes returns
        return GenerationResponse(text=text, input_tokens=None, output_tokens=None, model=self.model)

    # -- reference behaviour shared by several tasks ------------------------

    def _mock_references(self, text: str) -> list[dict[str, Any]]:
        out = []
        text_key = hash_text(text)
        for mention in rules.find_references(text):
            if self._chance(self.settings.reference_miss_rate, "reference", text_key, mention.identifier, mention.start):
                continue
            quote = mention.quote
            if self._chance(self.settings.quote_paraphrase_rate, "paraphrase", text_key, mention.identifier):
                quote = f"refers to {mention.identifier} as required"
            out.append(
                {
                    "identifier": mention.identifier,
                    "revision": mention.revision,
                    "document_name": mention.name if mention.name != mention.identifier else None,
                    "document_type": mention.doc_type,
                    "relation": mention.relation,
                    "quoted_text": quote,
                }
            )
        if self._chance(self.settings.hallucination_rate, "hallucination", text_key):
            out.append(self._hallucinated_reference(text, text_key))
        return out

    def _hallucinated_reference(self, text: str, text_key: str) -> dict[str, Any]:
        mentions = rules.find_references(text)
        distractors = rules.distractor_tokens(text)
        kinds = (["corrupted"] if mentions else []) + (["distractor"] if distractors else []) + ["invented"]
        kind = self._pick(kinds, "hallucination_kind", text_key)
        if kind == "corrupted":
            base = self._pick(mentions, "hallucination_base", text_key).identifier
            last = base[-1]
            identifier = base[:-1] + (str((int(last) + 1) % 10) if last.isdigit() else "X")
        elif kind == "distractor":
            identifier = self._pick(distractors, "hallucination_token", text_key)
        else:
            identifier = "GEN-SPEC-001"
        return {
            "identifier": identifier,
            "revision": None,
            "document_name": None,
            "document_type": None,
            "relation": "references",
            "quoted_text": f"according to {identifier}",
        }

    # -- tasks ---------------------------------------------------------------

    def _single_pass(self, payload: dict[str, Any]) -> dict[str, Any]:
        requirements: list[dict[str, Any]] = []
        texts: dict[str, tuple[str, str]] = {}
        for doc in payload.get("documents", []):
            temp = InputDocument(
                document_id=doc["document_id"],
                canonical_identifier=doc.get("identifier"),
                identifier_source="manifest",
                filename="(prompt)",
                format="txt",
                sha256="",
                char_count=len(doc["text"]),
                parse_status="ok",
                text=doc["text"],
            )
            for seg in segment_document(temp):
                if not seg.is_candidate or not rules.has_modal(seg.text):
                    continue
                if self._chance(self.settings.requirement_miss_rate, "requirement", hash_text(seg.text)):
                    continue
                key = f"R{len(requirements) + 1}"
                requirements.append(
                    {
                        "key": key,
                        "source_document_id": doc["document_id"],
                        "chapter": seg.chapter,
                        "requirement_text": _collapse(seg.text),
                        "references": self._mock_references(seg.text),
                    }
                )
                texts[key] = (seg.text, doc.get("identifier") or "")
        conflicts = []
        keys = list(texts)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                judgement = self._judge(texts[ka], texts[kb])
                if judgement:
                    conflicts.append(
                        {
                            "requirement_keys": [ka, kb],
                            "conflict_type": judgement.conflict_type,
                            "description": judgement.description,
                            "ambiguous": judgement.ambiguous,
                            "evidence": [
                                {"requirement_key": ka, "quoted_text": judgement.quote_a},
                                {"requirement_key": kb, "quoted_text": judgement.quote_b},
                            ],
                        }
                    )
        return {"requirements": requirements, "potential_conflicts": conflicts}

    def _identify(self, payload: dict[str, Any]) -> dict[str, Any]:
        found = []
        for seg in payload.get("segments", []):
            if seg.get("context_only") or not rules.has_modal(seg["text"]):
                continue
            if self._chance(self.settings.requirement_miss_rate, "requirement", hash_text(seg["text"])):
                continue
            found.append({"segment_id": seg["segment_id"], "chapter": seg.get("chapter"), "requirement_text": _collapse(seg["text"])})
        return {"requirements": found}

    def _references(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {"requirement_id": req["requirement_id"], "references": self._mock_references(req["text"])}
                for req in payload.get("requirements", [])
            ]
        }

    def _verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload["requirement"]["text"]
        text_key = hash_text(text)
        found = rules.find_references(text)
        decisions = []
        for cand in payload.get("candidates", []):
            cand_text = cand["text"]
            match = next(
                (m for m in found if cand_text in text[m.start : m.end + 12] or m.identifier in cand_text), None
            )
            missed_again = cand.get("kind") != "doubtful_existing" and self._chance(
                self.settings.reference_miss_rate, "verify", text_key, cand_text
            )
            if match is None or missed_again:
                decisions.append(
                    {"candidate_id": cand["candidate_id"], "is_document_reference": False, "reason": "not a document reference"}
                )
                continue
            decisions.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "is_document_reference": True,
                    "identifier": match.identifier,
                    "revision": match.revision,
                    "document_name": match.name if match.name != match.identifier else None,
                    "relation": match.relation,
                    "quoted_text": match.quote,
                    "reason": "explicit document mention",
                }
            )
        return {"decisions": decisions}

    def _judge(self, a: tuple[str, str], b: tuple[str, str]):
        (text_a, doc_a), (text_b, doc_b) = a, b
        pair_key = "|".join(sorted([hash_text(text_a), hash_text(text_b)]))
        judgement = rules.assess_pair(text_a, doc_a, text_b, doc_b)
        if judgement is not None:
            if self._chance(self.settings.conflict_miss_rate, "conflict_miss", pair_key):
                return None
            if self._chance(self.settings.quote_paraphrase_rate, "conflict_paraphrase", pair_key):
                judgement.quote_a = "the first requirement sets a different limit"
            return judgement
        if rules.overlap(text_a, text_b) >= 0.3 and self._chance(
            self.settings.conflict_false_positive_rate, "conflict_fp", pair_key
        ):
            return rules.PairJudgement(
                "other_or_uncertain",
                "Both requirements address the same subject with different wording.",
                rules.first_clause(text_a),
                rules.first_clause(text_b),
                ambiguous=True,
            )
        return None

    def _conflicts(self, payload: dict[str, Any]) -> dict[str, Any]:
        assessments = []
        for pair in payload.get("pairs", []):
            ra, rb = pair["requirement_a"], pair["requirement_b"]
            judgement = self._judge(
                (ra["text"], ra.get("document_identifier") or ""), (rb["text"], rb.get("document_identifier") or "")
            )
            if judgement is None:
                assessments.append({"pair_id": pair["pair_id"], "potential_conflict": False})
                continue
            assessments.append(
                {
                    "pair_id": pair["pair_id"],
                    "potential_conflict": True,
                    "conflict_type": judgement.conflict_type,
                    "ambiguous": judgement.ambiguous,
                    "description": judgement.description,
                    "evidence": [
                        {"requirement_id": ra["requirement_id"], "quoted_text": judgement.quote_a},
                        {"requirement_id": rb["requirement_id"], "quoted_text": judgement.quote_b},
                    ],
                }
            )
        return {"assessments": assessments}


def hash_text(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()[:12]
