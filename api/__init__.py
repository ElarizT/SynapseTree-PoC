"""
SynapseTree Public API
=======================

Clean, stable API layer that exposes SynapseTree's full agentic capabilities
to external clients (web UI, CLI, SDKs) without leaking internal details.

Core Principles:
- Blueprint remains the single executable authority
- No execution logic duplicated outside the engine
- API layer is thin, deterministic, and stateless
- All responses are serializable (JSON-safe)
- No UI assumptions
- API functions never throw exceptions outward
"""

from api.runtime_api import (
    # Request/Response types
    APIRunRequest,
    APIRunResponse,
    APIError,
    
    # Core API functions
    validate_blueprint_api,
    run_blueprint_api,
    generate_blueprint_api,
    
    # Analysis APIs
    replay_execution_api,
    critical_path_api,
    optimization_insights_api,
)

from api.serialization import (
    blueprint_to_dict,
    dict_to_blueprint,
    execution_result_to_dict,
    dict_to_execution_result,
)

__all__ = [
    # Types
    "APIRunRequest",
    "APIRunResponse",
    "APIError",
    
    # Core API
    "validate_blueprint_api",
    "run_blueprint_api",
    "generate_blueprint_api",
    
    # Analysis API
    "replay_execution_api",
    "critical_path_api",
    "optimization_insights_api",
    
    # Serialization
    "blueprint_to_dict",
    "dict_to_blueprint",
    "execution_result_to_dict",
    "dict_to_execution_result",
]
