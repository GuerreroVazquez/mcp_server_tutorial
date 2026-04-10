import json
from typing import Any, Optional, List
from mcp.types import CallToolResult, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        """Build Gemini-compatible function declarations from MCP tools."""
        function_declarations: list[dict[str, Any]] = []

        for client in clients.values():
            tool_models = await client.list_tools()
            for t in tool_models:
                function_declarations.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    }
                )

        return [{"function_declarations": function_declarations}] if function_declarations else []

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that exposes the named tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    def _extract_function_calls(cls, response: Any) -> list[dict[str, Any]]:
        """Extract Gemini function calls from a GenerateContentResponse."""
        function_calls: list[dict[str, Any]] = []

        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    fc = getattr(part, "functionCall", None)

                if fc is None:
                    continue

                # Handle SDK object or dict-like serialization
                if hasattr(fc, "model_dump"):
                    fc = fc.model_dump(exclude_none=True)
                elif not isinstance(fc, dict):
                    fc = {
                        "name": getattr(fc, "name", None),
                        "args": getattr(fc, "args", None),
                        "id": getattr(fc, "id", None),
                    }

                function_calls.append(fc)

        return function_calls

    @classmethod
    def _build_function_response_part(
        cls,
        tool_name: str,
        response_payload: Any,
        call_id: str | None = None,
        is_error: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "name": tool_name,
            "response": {
                "result": response_payload,
                "is_error": is_error,
            },
        }

        if call_id:
            payload["id"] = call_id

        return {"functionResponse": payload}

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], response: Any
    ) -> List[dict[str, Any]]:
        """Execute Gemini function calls and return Gemini functionResponse parts."""
        function_calls = cls._extract_function_calls(response)
        function_response_parts: list[dict[str, Any]] = []

        for function_call in function_calls:
            tool_name = function_call.get("name")
            tool_input = function_call.get("args", {}) or {}
            call_id = function_call.get("id")

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                function_response_parts.append(
                    cls._build_function_response_part(
                        tool_name=tool_name or "unknown_tool",
                        response_payload={"error": "Could not find that tool"},
                        call_id=call_id,
                        is_error=True,
                    )
                )
                continue

            tool_output = None
            try:
                tool_output = await client.call_tool(tool_name, tool_input)

                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]

                response_payload: Any
                if len(content_list) == 1:
                    # Return a single string directly when possible
                    response_payload = content_list[0]
                else:
                    response_payload = content_list

                function_response_parts.append(
                    cls._build_function_response_part(
                        tool_name=tool_name,
                        response_payload=response_payload,
                        call_id=call_id,
                        is_error=bool(tool_output and tool_output.isError),
                    )
                )

            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)

                function_response_parts.append(
                    cls._build_function_response_part(
                        tool_name=tool_name or "unknown_tool",
                        response_payload={"error": error_message},
                        call_id=call_id,
                        is_error=True,
                    )
                )

        return function_response_parts