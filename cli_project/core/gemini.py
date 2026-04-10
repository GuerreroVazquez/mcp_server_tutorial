from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types


class GeminiLLM:
    def __init__(self, model: str, api_key: Optional[str] = None):
        # If api_key is None, the SDK can also read GOOGLE_API_KEY from env.
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def add_user_message(self, messages: List[Dict[str, Any]], message: Union[str, Any]):
        text = getattr(message, "content", message)
        user_message = {
            "role": "user",
            "parts": [{"text": text}],
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: List[Dict[str, Any]], message: Union[str, Any]):
        text = getattr(message, "content", message)
        # Gemini uses "model" rather than "assistant" in chat history.
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
        return "\n".join(texts)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[list] = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
    ):
        # Note:
        # - `thinking` is kept in the signature for compatibility with tutorial,
        #   but it is not mapped here because Gemini does not expose a direct drop-in
        #   equivalent in this same request shape.

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

        response = self.client.models.generate_content(
            model=self.model,
            contents=messages,
            config=config,
        )
        return response