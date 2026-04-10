from core.gemini import GeminiLLM
from mcp_client import MCPClient
from core.tools import ToolManager
from typing import Any


class Chat:
    def __init__(self, gemini_service: GeminiLLM, clients: dict[str, MCPClient]):
        self.gemini_service: GeminiLLM = gemini_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[dict[str, Any]] = []
    def _has_function_calls(self, response) -> bool:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in getattr(content, "parts", []) or []:
                if getattr(part, "function_call", None) is not None:
                    return True
                if getattr(part, "functionCall", None) is not None:
                    return True
                if isinstance(part, dict):
                    if part.get("function_call") is not None:
                        return True
                    if part.get("functionCall") is not None:
                        return True

        return False
    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "parts": [{"text": query}]})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.gemini_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.gemini_service.add_assistant_message(self.messages, response)

            if self._has_function_calls(response):
                print(self.gemini_service.text_from_message(response))

                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                # Send tool results back as a user turn
                self.gemini_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.gemini_service.text_from_message(response)
                break

        return final_text_response
