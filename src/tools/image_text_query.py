# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from src.services.llms.llm import create_llm
from src.schema.types import LLMType
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf.yaml"


def _build_data_url(raw_base64: str, media_type: str) -> str:
    """Ensure the provided base64 payload is wrapped in a proper data URL."""

    payload = raw_base64.strip()
    if payload.lower().startswith("data:"):
        return payload
    media = media_type if media_type else "image/png"
    return f"data:{media};base64,{payload}"


def _load_image_from_path(path_value: str, media_type: str) -> str:
    """Read a local image file and return a data URL."""

    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path must be a file: {path}")

    guessed_type, _ = mimetypes.guess_type(path)
    media = guessed_type or media_type or "image/png"

    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return _build_data_url(encoded, media)


class MultiModalVisionInput(BaseModel):
    """Structured input for the multimodal tool."""

    query: str = Field(..., description="The textual instruction or question for the model.")
    image_path: str = Field(
        description="Local filesystem path to an image file.",
    )
    image_media_type: str = Field(
        default="image/png",
        description="MIME type used for local images when a data URL must be generated.",
    )

    @field_validator("image_media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        if value and not value.startswith("image/"):
            raise ValueError("image_media_type must start with 'image/'.")
        return value

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _ensure_image_source(self) -> "MultiModalVisionInput":
        if not self.image_path:
            raise ValueError("Provide at least one image source: image_path.")
        return self


class MultiModalVisionOutput(BaseModel):
    """Normalized output returned to the caller."""

    answer: str = Field(..., description="Natural language answer produced by the multimodal model.")
    model: str = Field(..., description="Model that generated the answer.")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason returned by the API for terminating generation.",
    )
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage metadata if available.")
    response_id: str = Field(..., description="Identifier returned by the provider for tracing.")


def _build_message_content(input_data: MultiModalVisionInput) -> List[Dict[str, Any]]:
    """Construct the multimodal message payload."""

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": input_data.query.strip(),
        }
    ]


    if input_data.image_path:
        data_url = _load_image_from_path(input_data.image_path, input_data.image_media_type)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            }
        )

    return content


def _extract_answer_text(message_content: Any) -> str:
    """Extract plain text from the completion message content."""

    if isinstance(message_content, str):
        return message_content.strip()

    collected: List[str] = []
    if isinstance(message_content, list):
        for chunk in message_content:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")
                if chunk_type in {"text", "output_text"}:
                    collected.append(str(chunk.get("text", "")))
            elif isinstance(chunk, str):
                collected.append(chunk)

    joined = "\n".join(part.strip() for part in collected if part.strip())
    return joined.strip()


@tool("image_text_query", args_schema=MultiModalVisionInput.model_json_schema())
def image_text_query(**tool_kwargs) -> MultiModalVisionOutput:
    """Answer detailed questions that mix natural-language instructions with one or more images. Use this tool whenever the you needs help describing an image, extracting structured information from a screenshot (e.g. OCR), comparing multiple photos, or reasoning about charts, diagrams, documents, UI mockups, receipts, etc."""

    input_data = MultiModalVisionInput(**tool_kwargs)

    llm = create_llm(LLMType.VISION)


    content = _build_message_content(input_data)
    logger.info(
        "Invoking multimodal model with %d image source(s).", max(len(content) - 1, 0)
    )

    response = llm.invoke([HumanMessage(content=content)])
    answer_text = _extract_answer_text(response.content)

    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = response.usage_metadata or response_metadata.get("token_usage") or {}

    model_used = (
        response_metadata.get("model_name")
        or response_metadata.get("model")
    )
    finish_reason = response_metadata.get("finish_reason")

    response_id = getattr(response, "id", "") or response_metadata.get("id", "")

    output = MultiModalVisionOutput(
        answer=answer_text or "",
        model=model_used,
        finish_reason=finish_reason,
        usage=dict(usage_metadata) if isinstance(usage_metadata, dict) else {},
        response_id=response_id,
    )

    return output


# Explicitly set description attribute for consistency with dynamic tools
# The @tool decorator sets the name attribute, but we ensure description is also explicitly set
# This ensures worker_team can reliably access both name and description attributes
if not hasattr(image_text_query, "description") or not getattr(image_text_query, "description", "").strip():
    image_text_query.description = (
        "Answer detailed questions that mix natural-language instructions with one or more images. "
        "Use this tool whenever the agent needs help describing an image, extracting structured "
        "information from a screenshot, comparing multiple photos, or reasoning about charts, "
        "diagrams, documents, UI mockups, receipts, etc. The caller supplies a text prompt plus "
        "an image (local path) and receives a concise textual answer whose tone "
        "matches the rest of the agent's response."
    )


__all__ = ["image_text_query", "MultiModalVisionInput", "MultiModalVisionOutput"]
