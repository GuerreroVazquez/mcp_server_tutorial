import asyncio
import logging
import socket
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

API_TIMEOUT = 60


def _force_ipv4():
    _orig = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only


class GeminiLLM:
    def __init__(self, model: str, api_key: Optional[str] = None):
        _force_ipv4()
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def verify_connection(self) -> bool:
        logger.info("Verifying Gemini API connection for model '%s'...", self.model)
        try:
            config = types.GenerateContentConfig(temperature=0.0)
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": "Say OK"}]}],
                    config=config,
                ),
                timeout=API_TIMEOUT,
            )
            logger.info("Gemini API connection verified successfully.")
            return True
        except asyncio.TimeoutError:
            logger.error("Gemini API connection timed out after %ds.", API_TIMEOUT)
            return False
        except Exception as e:
            logger.error("Gemini API connection failed: %s", e)
            return False

    def add_user_message(self, messages: List[Dict[str, Any]], message: Union[str, Any]):
        
        text = getattr(message, "content", message)
        user_message = {
            "role": "user",
            "parts": [{"text": str(text)}],
        }

        messages.append(user_message)

    def add_assistant_message(self, messages: List[Dict[str, Any]], message: Union[str, Any]):
        """
        Adds the model response to the conversation history.

        Important: preserve the actual response parts so that function calls
        are not lost.
        """
        candidates = getattr(message, "candidates", []) or []

        if candidates:
            content = getattr(candidates[0], "content", None)
            if content and getattr(content, "parts", None):
                assistant_parts = []

                for part in content.parts:
                    if hasattr(part, "model_dump"):
                        assistant_parts.append(part.model_dump(exclude_none=True))
                    elif isinstance(part, dict):
                        assistant_parts.append(part)
                    else:
                        part_dict = {}

                        if getattr(part, "text", None) is not None:
                            part_dict["text"] = part.text

                        function_call = getattr(part, "function_call", None)
                        if function_call is None:
                            function_call = getattr(part, "functionCall", None)

                        if function_call is not None:
                            if hasattr(function_call, "model_dump"):
                                part_dict["functionCall"] = function_call.model_dump(
                                    exclude_none=True
                                )
                            elif isinstance(function_call, dict):
                                part_dict["functionCall"] = function_call
                            else:
                                part_dict["functionCall"] = {
                                    "id": getattr(function_call, "id", None),
                                    "name": getattr(function_call, "name", None),
                                    "args": getattr(function_call, "args", {}) or {},
                                }

                        if part_dict:
                            assistant_parts.append(part_dict)

                assistant_message = {
                    "role": "model",
                    "parts": assistant_parts,
                }
                messages.append(assistant_message)
                return

        # Fallback: plain text only
        text = self.text_from_message(message)
        assistant_message = {
            "role": "model",
            "parts": [{"text": text}],
        }
        messages.append(assistant_message)

    def text_from_message(self, response) -> str:
        # In the normal case, this is the simplest path.
        if hasattr(response, "text") and response.text:
            return response.text

        # Fallback for structured responses.
        texts = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                elif isinstance(part, dict) and part.get("text"):
                    texts.append(part["text"])
        return "\n".join(texts)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[list] = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
    ):
        # `thinking` is kept only for compatibility with the previous interface.
        # It is not mapped here.

        config_kwargs = {
            "temperature": temperature,
        }

        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences

        if system:
            config_kwargs["system_instruction"] = system

        if tools:
            config_kwargs["tools"] = tools

        config = types.GenerateContentConfig(**config_kwargs)

        logger.debug(
            "Gemini request: model=%s messages=%d tools=%s",
            self.model,
            len(messages),
            bool(tools),
        )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=messages,
                    config=config,
                ),
                timeout=API_TIMEOUT,
            )
            return response
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Gemini API request timed out after {API_TIMEOUT}s. "
                "Check your network connection and API key."
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}")