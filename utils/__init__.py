"""
Utils Package Initialization.

This file exposes the key functions and classes from the underlying modules 
so they can be imported directly from the `utils` package. 

Example:
    from utils import get_api_key, count_tokens_in_text, LLMProvider
"""

# Expose configuration utilities
from .config_utils import (
    load_yaml_configuration,
    get_model_configuration,
    get_parameter_configuration,
    get_api_key,
    is_api_key_valid,
)

# Expose token utilities
from .token_utils import (
    get_token_counter,
    count_tokens_in_text,
    count_tokens_in_messages,
)

# Expose LLM utilities
from .llm_utils import (
    LLMProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ToolRegistry,
    LLM,
    create_llm_provider,
)

# Define what gets imported when someone uses `from utils import *`
__all__ = [
    "load_yaml_configuration",
    "get_model_configuration",
    "get_parameter_configuration",
    "get_api_key",
    "is_api_key_valid",
    "get_token_counter",
    "count_tokens_in_text",
    "count_tokens_in_messages",
    "LLMProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ToolRegistry",
    "LLM",
    "create_llm_provider",
]
