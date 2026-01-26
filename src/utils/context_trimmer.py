# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from typing import Any, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage

from src.tools.dynamic_tool_loader import count_text_tokens
from src.utils.utils import summarize_context, extract_tool_calls_from_messages
import asyncio
logger = logging.getLogger(__name__)

class ContextTrimmer:

    def __init__(
        self,
        token_limit: int,
        user_query: str = "",
    ):
        self.token_limit = token_limit
        self.user_query = user_query
    
    def count_tokens(self, messages: List[BaseMessage]) -> int:
        total_tokens = 0
        for message in messages:
            total_tokens += self.count_message_tokens(message)
        return total_tokens

    def count_message_tokens(self, message: BaseMessage) -> int:
        token_count = 0

        attrs_to_count = [
            ("type", str),
            ("name", str),
            ("content", self.normalize_content)
        ]
        for attr, transform in attrs_to_count:
            if val := getattr(message, attr, None):
                token_count += count_text_tokens(transform(val))

        if kwargs := getattr(message, "additional_kwargs", None):
            try:
                extra_str = json.dumps(kwargs, ensure_ascii=False)
            except TypeError:
                extra_str = str(kwargs)
            token_count += count_text_tokens(extra_str)

        return max(1, token_count)

    def normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (list, dict)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)
        return str(content)


    def is_exceeded(self, messages: List[BaseMessage]) -> bool:
        return self.count_tokens(messages) > self.token_limit

    def trim(self, state: dict) -> List[BaseMessage]:
        if self.token_limit is None:
            logger.info("No token_limit set, the context management doesn't work.")
            return state

        if not isinstance(state, dict) or "messages" not in state:
            logger.warning("No messages found in state")
            return state

        messages = state["messages"]

        if not self.is_exceeded(messages):
            return state

        # 2. Compress messages
        compressed_messages = self._trim_internal(messages)

        logger.info(
            f"Message trimming completed: {self.count_tokens(messages)} -> {self.count_tokens(compressed_messages)} tokens"
        )

        state["messages"] = compressed_messages
        return state

    def _trim_internal(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        current_context_summary = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                if message.content.startswith("## Context Summary"):
                    current_context_summary = message.content
                    break
        tool_calls = extract_tool_calls_from_messages(messages)

        output_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                if message.content.startswith("## Context Summary"):
                    output_messages.append(message)
            if not isinstance(message, SystemMessage) and not isinstance(message, HumanMessage):
                output_messages.append(message)
        context_summary = asyncio.run(summarize_context(self.user_query, tool_calls, current_context_summary))
        return [RemoveMessage(id=message.id) for message in output_messages] + [HumanMessage(content=f"## Context Summary\n{context_summary}")]

