# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from functools import lru_cache
from src.config.config import load_yaml_config
from src.schema.types import LLMType

logger = logging.getLogger(__name__)

LLM_CONFIG_MAP = {
    LLMType.BASIC: "BASIC_MODEL",
    LLMType.VISION: "VISION_MODEL",
    LLMType.SUMMARIZE: "SUMMARIZE_MODEL",
    LLMType.CLUSTER: "CLUSTER_MODEL",
    LLMType.TOOL_ANALYZE: "TOOL_ANALYZE_MODEL",    
}

@lru_cache(maxsize=1)
def get_full_config() -> Dict[str, Any]:
    config_path = (Path(__file__).parent.parent.parent.parent / "conf.yaml").resolve()
    return load_yaml_config(config_path)

def _prepare_llm_kwargs(conf: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = conf.copy()

    provider = str(kwargs.pop("provider", "")).strip().lower()

    # Default to Qwen via DashScope's OpenAI-compatible endpoint when provider indicates Qwen.
    # This keeps the rest of the codebase (LangChain ChatOpenAI + structured output) unchanged.
    if provider in {"qwen", "tongyi", "dashscope"}:
        kwargs.setdefault(
            "base_url",
            os.environ.get(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )
        kwargs.setdefault("api_key", os.environ.get("DASHSCOPE_API_KEY"))
    else:
        # Generic OpenAI-compatible defaults
        if os.environ.get("OPENAI_BASE_URL") and "base_url" not in kwargs:
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY"))

    # Final fallback: allow either env var to satisfy api_key.
    if not kwargs.get("api_key"):
        kwargs["api_key"] = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
    
    kwargs.setdefault("max_retries", 3)
    
    kwargs.pop("token_limit", None)
    if not kwargs.pop("verify_ssl", True):
        kwargs["http_client"] = httpx.Client(verify=False)
        kwargs["http_async_client"] = httpx.AsyncClient(verify=False)
        
    return kwargs

def create_llm(llm_type: LLMType) -> BaseChatModel:
    config_key = LLM_CONFIG_MAP.get(llm_type)
    if not config_key:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    full_conf = get_full_config()
    llm_conf = full_conf.get(config_key)

    if not isinstance(llm_conf, dict):
        raise ValueError(f"Configuration for {llm_type} ({config_key}) must be a dictionary.")

    llm_kwargs = _prepare_llm_kwargs(llm_conf)
    if not llm_kwargs.get("model"):
        raise ValueError(f"{config_key}.model is required")
    if not llm_kwargs.get("api_key"):
        raise ValueError(
            f"{config_key}.api_key is required (or set DASHSCOPE_API_KEY / OPENAI_API_KEY)"
        )
    return ChatOpenAI(**llm_kwargs)

def get_max_tokens(llm_type: LLMType) -> Optional[int]:
    config_key = LLM_CONFIG_MAP.get(llm_type)
    full_conf = get_full_config()
    
    return full_conf.get(config_key, {}).get("token_limit")
