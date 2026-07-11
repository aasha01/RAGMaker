"""LLM provider strategy registry.

Maps the config key -> concrete `BaseLLMProvider` subclass. Adding a provider is
a new file in this package plus one line here — nothing else changes.

The concrete providers lazy-import their SDK inside `generate`, so importing this
registry (e.g. for the discovery endpoint) never drags in openai/anthropic/httpx
or requires an API key.
"""

from __future__ import annotations

from .base import BaseLLMProvider, GenerationResult
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,      # local, no API key, zero cost — the default
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

__all__ = ["BaseLLMProvider", "GenerationResult", "REGISTRY"]
