"""
graph.py — LangGraph StateGraph: router → tools → composer.
Stage 1: placeholder. Stage 4 will implement:
  - router node: decides which retrieval mode to use and whether date tool is needed
  - tools node: executes the chosen retrieval function and/or MCP date tool
  - composer node: produces the final grounded answer with citations, or 'not found'
State: typed TypedDict passed between nodes for full inspectability.
"""
# TODO (Stage 4): implement the LangGraph StateGraph with router, tools, composer nodes
