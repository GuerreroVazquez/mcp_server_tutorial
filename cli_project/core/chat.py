from core.gemini import GeminiLLM
from mcp_client import MCPClient
from core.tools import ToolManager
from typing import Any


class Chat:
    def __init__(self, gemini_service: GeminiLLM, clients: dict[str, MCPClient]):
        self.gemini_service: GeminiLLM = gemini_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[dict[str, Any]] = []

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

            if response.stop_reason == "tool_use":
                print(self.gemini_service.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                self.gemini_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.gemini_service.text_from_message(
                    response
                )
                break

        return final_text_response
