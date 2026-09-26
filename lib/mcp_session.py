import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

__all__ = ["PersistentMCPSession"]


class PersistentMCPSession:
    """Keeps one MCP session open for the lifetime of an agent.

    Tools from `MultiServerMCPClient.get_tools()` open a new session per call, which
    for stdio servers means spawning a new process (uv/npx) on every tool call and for
    remote servers a new HTTP handshake. Here the session lives in a background task
    (the MCP client's anyio scopes must be entered and exited in the same task) and
    tools are bound to it, so calls reuse the running server.
    """

    def __init__(self, service_name: str, connection: dict[str, Any]):
        self.service_name = service_name
        self.client = MultiServerMCPClient({service_name: connection})
        self.tools: list[BaseTool] = []
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> list[BaseTool]:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[list[BaseTool]] = loop.create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(ready), name=f"mcp-session-{self.service_name}")
        self.tools = await ready
        return self.tools

    async def _run(self, ready: "asyncio.Future[list[BaseTool]]") -> None:
        try:
            async with self.client.session(self.service_name) as session:
                tools = await load_mcp_tools(session, server_name=self.service_name)
                ready.set_result(tools)
                assert self._stop is not None
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning(f"MCP session for '{self.service_name}' ended: {exc!r}")
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def aclose(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, Exception):
                self._task.cancel()
        self._task = None
