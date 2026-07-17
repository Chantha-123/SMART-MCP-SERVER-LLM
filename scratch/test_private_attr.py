import asyncio
from pydantic import PrivateAttr
from langchain_core.tools import BaseTool, tool
from typing import Any

class SanitizedTool(BaseTool):
    _original_tool: BaseTool = PrivateAttr()
    _schema_dict: dict[str, Any] = PrivateAttr()
    
    def __init__(self, original_tool: BaseTool, schema_dict: dict[str, Any], **kwargs: Any):
        super().__init__(
            name=original_tool.name,
            description=original_tool.description,
            args_schema=original_tool.args_schema,
            **kwargs
        )
        self._original_tool = original_tool
        self._schema_dict = schema_dict
        
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_tool.invoke(kwargs)
        
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._original_tool.ainvoke(kwargs)

@tool
def dummy_tool(x: int) -> int:
    """A dummy tool."""
    return x + 1

wrapped = SanitizedTool(dummy_tool, {})
print("Success! Wrapped tool:", wrapped.name, wrapped.description)
res = asyncio.run(wrapped.ainvoke({"x": 5}))
print("Result of ainvoke:", res)
