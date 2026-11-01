"""rag_search 組み込みツール。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.rag.service import K_MAX, RAGService
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema


class RagSearch(Tool):
    """ワークスペース索引の意味検索。"""

    class Input(BaseModel):
        query: str = Field(description="Natural language query to search the workspace index.")
        k: int = Field(default=8, description=f"Number of chunks to return (1-{K_MAX}).")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="rag_search",
        description=(
            "Search the indexed workspace knowledge base and return the most relevant "
            "code or document chunks with their file paths and line ranges."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, RagSearch.Input)
        rag = ctx.rag
        if not isinstance(rag, RAGService):
            return ToolResult(
                output="RAG is disabled or the workspace has not been indexed "
                "(run `satsumaestro rag ingest`).",
                is_error=True,
            )
        try:
            output = await rag.query(inp.query, inp.k)
        except Exception as e:
            return ToolResult(output=f"RAG unavailable: {e}", is_error=True)
        return ToolResult(output=output)
