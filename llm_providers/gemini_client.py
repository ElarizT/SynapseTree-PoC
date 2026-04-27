"""
Gemini Client
=============

Client for Google AI Studio (Gemini) API.
Supports Gemini 2.5 Flash and other Gemini models.
"""

import os
import time
import aiohttp
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class GeminiCallMetrics:
    """Metrics from a Gemini API call."""
    request_start: datetime
    request_end: datetime
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class GeminiError(Exception):
    """Exception for Gemini API errors."""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def call_gemini(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    benchmark_mode: bool = False
) -> str | tuple[str, GeminiCallMetrics]:
    """
    Call Gemini API.
    
    Args:
        model: Gemini model name (e.g., "gemini-2.5-flash-preview-05-20")
        messages: List of message dicts in OpenAI format
        temperature: Sampling temperature
        max_tokens: Max tokens for response
        api_key: Gemini API key
        benchmark_mode: If True, return metrics with response
    
    Returns:
        Response text, or tuple of (text, metrics) if benchmark_mode
    """
    # Get API key
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise GeminiError("GEMINI_API_KEY not found in environment")
    
    # Build Gemini URL
    # Model format: gemini-2.0-flash, gemini-2.5-flash-preview-05-20, etc.
    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"
    
    # Convert OpenAI message format to Gemini format
    contents = []
    system_instruction = None
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            # Gemini uses system_instruction separately
            system_instruction = content
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": content}]
            })
        elif role == "assistant":
            contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })
    
    # Build request body
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        }
    }
    
    if max_tokens:
        body["generationConfig"]["maxOutputTokens"] = max_tokens
    
    if system_instruction:
        body["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    # Record timing
    request_start = datetime.now()
    start_mono = time.monotonic()
    
    # Make request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                error_msg = result.get("error", {}).get("message", str(result))
                raise GeminiError(
                    f"Gemini API error: {error_msg}",
                    status_code=response.status
                )
    
    # Record end time
    end_mono = time.monotonic()
    request_end = datetime.now()
    latency_ms = (end_mono - start_mono) * 1000
    
    # Extract response text
    try:
        candidates = result.get("candidates", [])
        if not candidates:
            raise GeminiError("No candidates in Gemini response")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise GeminiError("No parts in Gemini response")
        
        response_text = parts[0].get("text", "")
    except (KeyError, IndexError) as e:
        raise GeminiError(f"Failed to parse Gemini response: {e}")
    
    # Extract token usage if available
    usage = result.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount")
    completion_tokens = usage.get("candidatesTokenCount")
    total_tokens = usage.get("totalTokenCount")
    
    if benchmark_mode:
        metrics = GeminiCallMetrics(
            request_start=request_start,
            request_end=request_end,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        return response_text, metrics
    
    return response_text


def is_gemini_available() -> bool:
    """Check if Gemini API key is available."""
    return bool(os.environ.get("GEMINI_API_KEY"))
