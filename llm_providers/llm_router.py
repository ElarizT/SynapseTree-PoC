"""
LLM Router
==========

Provider-agnostic routing layer for LLM calls.
Routes requests to appropriate provider clients based on configuration.

Design Principles:
- Provider abstraction: calling code doesn't know which provider is used
- API key resolution from environment variables
- Extensible: easy to add new providers (Groq, Azure, local, etc.)
- No hardcoded models or provider-specific logic
- Benchmark mode for latency measurement
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .openrouter_client import call_openrouter, OpenRouterError
from .groq_client import call_groq, GroqError, GroqLatencyMetrics
from .gemini_client import call_gemini, GeminiError, GeminiCallMetrics


# Default model for SynapseTree (Gemini 2.5 Flash for speed)
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROVIDER = "gemini"

# Benchmark model (Kimi-K2 via Groq)
BENCHMARK_MODEL = "moonshotai/kimi-k2-instruct-0905"
BENCHMARK_PROVIDER = "groq"


class LLMError(Exception):
    """Base exception for LLM routing errors."""
    pass


class UnsupportedProviderError(LLMError):
    """Raised when an unsupported provider is requested."""
    pass


@dataclass
class LLMCallMetrics:
    """Latency metrics from an LLM call."""
    provider: str
    model: str
    request_start: datetime
    request_end: datetime
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    is_real_llm: bool = True  # False for mock calls
    
    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "request_start": self.request_start.isoformat(),
            "request_end": self.request_end.isoformat(),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "is_real_llm": self.is_real_llm,
        }


def resolve_api_key(api_key_ref: Optional[str]) -> Optional[str]:
    """
    Resolve an API key reference to its actual value.
    
    API key references are environment variable names.
    Example: "OPENROUTER_API_KEY" -> os.environ.get("OPENROUTER_API_KEY")
    
    Args:
        api_key_ref: Environment variable name or None
    
    Returns:
        The resolved API key value, or None if not found
    """
    if not api_key_ref:
        return None
    
    # Handle common prefixes (normalize)
    ref = api_key_ref
    if ref.startswith("ENV_"):
        ref = ref[4:]  # Remove ENV_ prefix
    
    return os.environ.get(ref) or os.environ.get(api_key_ref)


async def call_llm(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    api_key_ref: Optional[str] = None,
    max_tokens: Optional[int] = None,
    benchmark_mode: bool = False
) -> str | tuple[str, LLMCallMetrics]:
    """
    Route an LLM call to the appropriate provider.
    
    This is the main entry point for all LLM calls in SynapseTree.
    Provider-specific logic is encapsulated in individual client modules.
    
    Args:
        provider: Provider identifier ("openrouter", "groq")
        model: Model identifier (provider-specific format)
        messages: List of message dicts in OpenAI format:
                  [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: Sampling temperature
        api_key_ref: Reference to API key (environment variable name)
        max_tokens: Optional max tokens for response
        benchmark_mode: If True, return latency metrics with response
    
    Returns:
        If benchmark_mode=False: The assistant's response text
        If benchmark_mode=True: Tuple of (response_text, LLMCallMetrics)
    
    Raises:
        UnsupportedProviderError: If provider is not supported
        LLMError: For other LLM-related errors
    """
    # Resolve API key from reference
    api_key = resolve_api_key(api_key_ref)
    
    # Route to appropriate provider
    provider_lower = provider.lower()
    
    # Record timing for all providers
    request_start = datetime.now()
    start_monotonic = time.monotonic()
    
    if provider_lower == "openrouter":
        try:
            result = await call_openrouter(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
            
            if benchmark_mode:
                end_monotonic = time.monotonic()
                metrics = LLMCallMetrics(
                    provider=provider,
                    model=model,
                    request_start=request_start,
                    request_end=datetime.now(),
                    latency_ms=(end_monotonic - start_monotonic) * 1000,
                    is_real_llm=True,
                )
                return result, metrics
            
            return result
            
        except OpenRouterError as e:
            raise LLMError(f"OpenRouter error: {e.message}") from e
    
    elif provider_lower == "groq":
        try:
            # Use Groq's built-in benchmark mode
            result = await call_groq(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                benchmark_mode=True  # Always get metrics from Groq
            )
            
            if isinstance(result, tuple):
                content, groq_metrics = result
                
                if benchmark_mode:
                    metrics = LLMCallMetrics(
                        provider=provider,
                        model=model,
                        request_start=groq_metrics.request_start,
                        request_end=groq_metrics.request_end,
                        latency_ms=groq_metrics.latency_ms,
                        prompt_tokens=groq_metrics.prompt_tokens,
                        completion_tokens=groq_metrics.completion_tokens,
                        total_tokens=groq_metrics.total_tokens,
                        is_real_llm=True,
                    )
                    return content, metrics
                
                return content
            
            return result
            
        except GroqError as e:
            raise LLMError(f"Groq error: {e.message}") from e
    
    # Mock provider for testing
    elif provider_lower == "mock":
        import asyncio
        await asyncio.sleep(0.1)  # Simulate latency
        
        mock_response = f"[Mock response for: {messages[-1].get('content', '')[:50]}...]"
        
        if benchmark_mode:
            end_monotonic = time.monotonic()
            metrics = LLMCallMetrics(
                provider="mock",
                model=model,
                request_start=request_start,
                request_end=datetime.now(),
                latency_ms=(end_monotonic - start_monotonic) * 1000,
                is_real_llm=False,
            )
            return mock_response, metrics
        
        return mock_response
    
    elif provider_lower == "gemini" or provider_lower == "google":
        try:
            result = await call_gemini(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                benchmark_mode=True  # Always get metrics
            )
            
            if isinstance(result, tuple):
                content, gemini_metrics = result
                
                if benchmark_mode:
                    metrics = LLMCallMetrics(
                        provider=provider,
                        model=model,
                        request_start=gemini_metrics.request_start,
                        request_end=gemini_metrics.request_end,
                        latency_ms=gemini_metrics.latency_ms,
                        prompt_tokens=gemini_metrics.prompt_tokens,
                        completion_tokens=gemini_metrics.completion_tokens,
                        total_tokens=gemini_metrics.total_tokens,
                        is_real_llm=True,
                    )
                    return content, metrics
                
                return content
            
            return result
            
        except GeminiError as e:
            raise LLMError(f"Gemini error: {e.message}") from e
    
    else:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: openrouter, groq, gemini, mock"
        )


def get_default_model_config() -> dict:
    """
    Get default model configuration for SynapseTree.
    
    Returns:
        Dict with default provider, model, and temperature
    """
    return {
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "api_key_ref": "OPENROUTER_API_KEY"
    }


def get_benchmark_model_config() -> dict:
    """
    Get benchmark model configuration (Kimi-K2 via Groq).
    
    Returns:
        Dict with Groq provider and Kimi-K2 model
    """
    return {
        "provider": BENCHMARK_PROVIDER,
        "model": BENCHMARK_MODEL,
        "temperature": 0.7,
        "api_key_ref": "GROQ_API_KEY"
    }


def is_groq_available() -> bool:
    """Check if Groq API key is available."""
    return bool(os.environ.get("GROQ_API_KEY"))

