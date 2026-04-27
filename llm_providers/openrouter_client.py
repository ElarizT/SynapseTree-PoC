"""
OpenRouter Client
=================

Async client for OpenRouter API.
Provides access to multiple LLM providers through a unified OpenAI-compatible interface.

Design Principles:
- Async-first using httpx
- No retry logic (Phase-1)
- No streaming (Phase-1)
- Provider-agnostic: no OpenRouter-specific logic leaks upward
"""

import os
from typing import Optional

import httpx


# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT = 60.0


class OpenRouterError(Exception):
    """Exception raised for OpenRouter API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def call_openrouter(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Make an async call to OpenRouter API.
    
    Args:
        model: Model identifier (e.g., "qwen/qwen3-coder:free")
        messages: List of message dicts in OpenAI format:
                  [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Optional max tokens for response
        api_key: API key (if None, reads from OPENROUTER_API_KEY env var)
    
    Returns:
        The assistant's response text
    
    Raises:
        OpenRouterError: If API call fails
        ValueError: If API key is not provided or found
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenRouter API key not provided. "
            "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
        )
    
    # Build request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    # Build headers
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://synapsetree.ai",  # Required by OpenRouter
        "X-Title": "SynapseTree",  # App identifier
    }
    
    # Make async request
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", {}).get("message", str(error_data))
                except Exception:
                    error_detail = response.text[:500]
                
                raise OpenRouterError(
                    f"OpenRouter API error: {error_detail}",
                    status_code=response.status_code
                )
            
            # Parse response
            data = response.json()
            
            # Extract assistant message
            choices = data.get("choices", [])
            if not choices:
                raise OpenRouterError("No response choices returned from OpenRouter")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            return content
            
        except httpx.TimeoutException:
            raise OpenRouterError("OpenRouter request timed out")
        except httpx.RequestError as e:
            raise OpenRouterError(f"OpenRouter request failed: {str(e)}")
