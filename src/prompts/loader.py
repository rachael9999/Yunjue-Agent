# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import os
import logging
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

class PromptLoader:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_prompt(self, template_name: str, **kwargs) -> str:
        if not template_name.endswith(".md"):
            template_name += ".md"
            
        template = self.env.get_template(template_name)
        prompt = template.render(**kwargs)
        return prompt

prompt_loader = PromptLoader()