"""src/csr_lab/prompts/chunking.py

Stage prompt: split a batch of document text into semantic chunks (the ``llm`` chunking method).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import PromptSpec


class SemanticChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="", description="Short title for the chunk.")
    text: str = Field(description="The chunk's text, copied from the input.")


class SemanticChunkingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunks: list[SemanticChunk] = Field(default_factory=list)


SPEC = PromptSpec(
    name="semantic_chunking",
    version="chunking.v1",
    system=(
        "You split sections of customer-specific requirements (CSR) documents for automotive supplier "
        "quality engineering into self-contained chunks for later review. A chunk groups text about one "
        "topic or one requirement; related sentences and their tables stay together."
    ),
    instructions=(
        "Split the text in `text` into chunks of roughly `target_chunk_words` words, cutting only at "
        "topic boundaries. Copy every chunk's text exactly from the input: same words, same order, no "
        "summaries or corrections. Cover all of the input text; leave out only a table of contents. "
        "Give each chunk a short title."
    ),
)
