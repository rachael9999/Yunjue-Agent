# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
from langgraph.graph import END, START, StateGraph
from .nodes import (
    integrator_node,
    manager_node,
    executor_node,
    tool_developer_node,
)
from src.schema.types import State

def build_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "manager")
    builder.add_node("integrator", integrator_node)
    builder.add_node("manager", manager_node)
    builder.add_node("executor", executor_node)
    builder.add_node("tool_developer", tool_developer_node)
    builder.add_edge("integrator", END)
    return builder


