# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from langchain_openai import ChatOpenAI
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
    
    kwargs.setdefault("max_retries", 3)
    
    kwargs.pop("token_limit", None)
    if not kwargs.pop("verify_ssl", True):
        kwargs["http_client"] = httpx.Client(verify=False)
        kwargs["http_async_client"] = httpx.AsyncClient(verify=False)
        
    return kwargs

def create_llm(llm_type: LLMType) -> ChatOpenAI:
    config_key = LLM_CONFIG_MAP.get(llm_type)
    if not config_key:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    full_conf = get_full_config()
    llm_conf = full_conf.get(config_key)

    if not isinstance(llm_conf, dict):
        raise ValueError(f"Configuration for {llm_type} ({config_key}) must be a dictionary.")

    llm_kwargs = _prepare_llm_kwargs(llm_conf)
    return ChatOpenAI(**llm_kwargs)

def get_max_tokens(llm_type: LLMType) -> Optional[int]:
    config_key = LLM_CONFIG_MAP.get(llm_type)
    full_conf = get_full_config()
    
    return full_conf.get(config_key, {}).get("token_limit")
