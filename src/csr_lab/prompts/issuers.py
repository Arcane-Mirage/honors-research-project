"""src/csr_lab/prompts/issuers.py

Stage prompt: who issued each document of a bundle (``extraction/issuers.py``).

The known issuers (``issuers.known`` in default.yaml) are listed first, so the model picks from
them instead of spending tokens on naming groups; it names a new issuer only when the input
allows it (``new_issuers_allowed``) and none of the listed ones fits.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import PromptSpec

UNKNOWN = "unknown"


class IssuerAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    issuer: str = Field(description="An issuer name exactly as listed in `issuers`, a new issuer name, or 'unknown'.")
    new_issuer: bool = Field(default=False, description="True only when `issuer` is not in the `issuers` list.")


class IssuerClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[IssuerAnswer] = Field(default_factory=list)


SPEC = PromptSpec(
    name="issuer_classification",
    version="issuers.v1",
    system=(
        "You identify who issued documents that are provided with, or cited in, customer-specific "
        "requirements (CSR) for automotive suppliers. The issuer is the organisation that publishes "
        "and maintains a document: a customer (vehicle manufacturer), an industry association, a "
        "standards body or a government."
    ),
    instructions=(
        "For each entry in `documents`, choose its issuer from `issuers` and return the issuer name "
        "exactly as listed. Decide from the identifier, the names, and the `header` (the first text "
        "of a provided document) or the `context` (the sentence that cites it). A cited document "
        "belongs to whoever publishes it, not to the document that cites it: ISO 9001 cited in a "
        "customer standard is still an ISO document. If `new_issuers_allowed` is true and no listed "
        "issuer fits but the information names the publisher clearly, return a short name for that "
        "organisation and set new_issuer to true. Otherwise, and whenever the issuer cannot be told "
        "from the information given, return 'unknown'. Return one result per key."
    ),
)
