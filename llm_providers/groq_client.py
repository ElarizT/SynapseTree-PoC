"""
Groq Cloud Client
==================

Async client for Groq Cloud API.
Provides access to fast inference models including Kimi-K2.

Design Principles:
- Async-first using httpx
- Latency instrumentation for benchmarking
- No retry logic (Phase-1)
- No streaming (Phase-1)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import httpx


# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT = 120.0


@dataclass
class GroqLatencyMetrics:
    """Latency metrics from a Groq API call."""
    request_start: datetime
    request_end: datetime
    latency_ms: float
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class GroqError(Exception):
    """Exception raised for Groq API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def call_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    benchmark_mode: bool = False
) -> str | tuple[str, GroqLatencyMetrics]:
    """
    Make an async call to Groq Cloud API.
    
    Args:
        model: Model identifier (e.g., "moonshotai/kimi-k2-instruct-0905")
        messages: List of message dicts in OpenAI format
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Optional max tokens for response
        api_key: API key (if None, reads from GROQ_API_KEY env var)
        benchmark_mode: If True, return latency metrics with response
    
    Returns:
        If benchmark_mode=False: The assistant's response text
        If benchmark_mode=True: Tuple of (response_text, GroqLatencyMetrics)
    
    Raises:
        GroqError: If API call fails
        ValueError: If API key is not provided or found
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Groq API key not provided. "
            "Set GROQ_API_KEY environment variable or pass api_key parameter."
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
    }
    
    # Record timing
    request_start = datetime.now()
    start_monotonic = time.monotonic()
    
    # Make async request
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.post(
                GROQ_API_URL,
                json=payload,
                headers=headers
            )
            
            end_monotonic = time.monotonic()
            request_end = datetime.now()
            latency_ms = (end_monotonic - start_monotonic) * 1000
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", {}).get("message", str(error_data))
                except Exception:
                    error_detail = response.text[:500]
                
                raise GroqError(
                    f"Groq API error: {error_detail}",
                    status_code=response.status_code
                )
            
            # Parse response
            data = response.json()
            
            # Extract assistant message
            choices = data.get("choices", [])
            if not choices:
                raise GroqError("No response choices returned from Groq")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            # Extract usage metrics
            usage = data.get("usage", {})
            
            if benchmark_mode:
                metrics = GroqLatencyMetrics(
                    request_start=request_start,
                    request_end=request_end,
                    latency_ms=latency_ms,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
                return content, metrics
            
            return content
            
        except httpx.TimeoutException:
            raise GroqError("Groq request timed out")
        except httpx.RequestError as e:
            raise GroqError(f"Groq request failed: {str(e)}")


# Supported Groq models
GROQ_MODELS = {
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
    "llama3-8b": "llama3-8b-8192",
    "llama3-70b": "llama3-70b-8192",
    "mixtral": "mixtral-8x7b-32768",
    "gemma": "gemma-7b-it",
}


def get_groq_model_id(alias: str) -> str:
    """Get full Groq model ID from alias."""
    return GROQ_MODELS.get(alias.lower(), alias)
