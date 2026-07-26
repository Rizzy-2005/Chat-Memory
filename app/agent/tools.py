"""
tools.py — LangChain @tool-decorated retrieval functions + MCP tool loader.
Stage 1: placeholder. Stage 4 will implement:
  - semantic_search(query): similarity search over all stored sessions
  - filter_by_participant(query, participant): restrict to one sender
  - pinpoint_message(query): retrieve the single best-matching message
  - date_range_search(query, start, end): filter by timestamp window
  - load_mcp_tools(): loads resolve_date_reference from the MCP server via
    langchain-mcp-adapters (stdio transport)
"""
# TODO (Stage 4): implement @tool-wrapped retrieval functions and MCP tool loader
