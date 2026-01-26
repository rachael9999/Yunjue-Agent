# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict

from langchain_core.runnables import RunnableConfig
import yaml
logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class Configuration:

    dynamic_tools_dir: str  # Directory for private dynamic tools
    dynamic_tools_public_dir: str = "dynamic_tools_public"  # Directory for public dynamic tools
    max_task_execution_cnt: int = 5

    @classmethod
    def resolve(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.init:
                env_value = os.environ.get(f.name.upper())
                env_value = f.type(env_value) if env_value is not None else None
                # Prefer configurable value if provided, otherwise use env value
                if f.name in configurable and configurable[f.name] is not None:
                    values[f.name] = configurable[f.name]
                elif env_value is not None:
                    values[f.name] = env_value
        return cls(**{k: v for k, v in values.items() if v is not None})


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        logger.error(f"YAML config file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return data
    except yaml.YAMLError as e:
        logger.error(f"Failed to load YAML configuration file when read file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unknown error when load YAML config: {e}")
        return {}