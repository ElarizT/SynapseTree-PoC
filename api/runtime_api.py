"""
Runtime API (Phase S)
======================

Official SynapseTree runtime contract.

Core Principles:
- API functions NEVER throw exceptions outward
- Always return APIError on failure
- Validation errors are not exceptions
- Internal stack traces must not leak
- All responses are JSON-serializable
"""

import traceback
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

from api.serialization import (
    blueprint_to_dict,
    dict_to_blueprint,
    execution_result_to_dict,
    dict_to_execution_result,
)


# =============================================================================
# API TYPES
# =============================================================================

@dataclass
class APIRunRequest:
    """Request for executing a Blueprint."""
    blueprint: dict                              # Serialized Blueprint
    execution_options: Optional[dict] = None     # Overrides (max_concurrency, fail_fast)


@dataclass
class APIRunResponse:
    """Response from Blueprint execution."""
    execution_id: str
    execution_result: dict                       # Serialized ExecutionResult
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class APIError:
    """Error response from API."""
    code: str
    message: str
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# ERROR CODES
# =============================================================================

ERR_INVALID_BLUEPRINT = "INVALID_BLUEPRINT"
ERR_VALIDATION_FAILED = "VALIDATION_FAILED"
ERR_EXECUTION_FAILED = "EXECUTION_FAILED"
ERR_SERIALIZATION_FAILED = "SERIALIZATION_FAILED"
ERR_INTERNAL_ERROR = "INTERNAL_ERROR"
ERR_MISSING_DEPENDENCY = "MISSING_DEPENDENCY"


# =============================================================================
# VALIDATION API
# =============================================================================

def validate_blueprint_api(blueprint_dict: dict) -> dict:
    """
    Validate a serialized Blueprint.
    
    Args:
        blueprint_dict: Serialized Blueprint dictionary
        
    Returns:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str]
        }
        
    Never raises exceptions.
    """
    try:
        errors = []
        warnings = []
        
        # Basic structure validation
        if not blueprint_dict:
            errors.append("Blueprint is empty")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if "agents" not in blueprint_dict:
            errors.append("Blueprint must have 'agents' field")
        elif not blueprint_dict["agents"]:
            errors.append("Blueprint must have at least one agent")
        
        if "blueprint_id" not in blueprint_dict:
            errors.append("Blueprint must have 'blueprint_id' field")
        
        if "name" not in blueprint_dict:
            warnings.append("Blueprint should have a 'name' field")
        
        # Try to deserialize
        try:
            blueprint = dict_to_blueprint(blueprint_dict)
        except Exception as e:
            errors.append(f"Failed to parse blueprint: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Validate using blueprint_validator if available
        try:
            from blueprint_validator import validate_blueprint
            validation_result = validate_blueprint(blueprint)
            if hasattr(validation_result, 'errors'):
                errors.extend(validation_result.errors)
            if hasattr(validation_result, 'warnings'):
                warnings.extend(validation_result.warnings)
        except ImportError:
            # Validator not available, do basic checks
            pass
        except Exception as e:
            warnings.append(f"Validator warning: {str(e)}")
        
        # Agent-specific validation
        agent_ids = set()
        for agent in blueprint_dict.get("agents", []):
            agent_id = agent.get("agent_id")
            if not agent_id:
                errors.append("Agent missing 'agent_id'")
            elif agent_id in agent_ids:
                errors.append(f"Duplicate agent_id: {agent_id}")
            else:
                agent_ids.add(agent_id)
            
            if not agent.get("name"):
                warnings.append(f"Agent {agent_id} missing 'name'")
            if not agent.get("role"):
                errors.append(f"Agent {agent_id} missing 'role'")
        
        # Dependency validation
        for agent in blueprint_dict.get("agents", []):
            deps = agent.get("execution", {}).get("depends_on", [])
            for dep in deps:
                if dep not in agent_ids:
                    errors.append(f"Agent {agent.get('agent_id')} depends on non-existent agent: {dep}")
        
        # Single agent warning
        if len(agent_ids) == 1:
            warnings.append("Blueprint has only one agent - consider adding parallel agents")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
        }


# =============================================================================
# EXECUTION API
# =============================================================================

async def run_blueprint_api(request: APIRunRequest) -> APIRunResponse | APIError:
    """
    Execute a Blueprint via the API.
    
    Args:
        request: APIRunRequest with blueprint and options
        
    Returns:
        APIRunResponse on success, APIError on failure
        
    Never raises exceptions.
    """
    try:
        start_time = datetime.now()
        warnings = []
        
        # Validate blueprint
        validation = validate_blueprint_api(request.blueprint)
        if not validation["valid"]:
            return APIError(
                code=ERR_VALIDATION_FAILED,
                message="Blueprint validation failed",
                details={"errors": validation["errors"]}
            )
        warnings.extend(validation.get("warnings", []))
        
        # Deserialize blueprint
        try:
            blueprint = dict_to_blueprint(request.blueprint)
        except Exception as e:
            return APIError(
                code=ERR_SERIALIZATION_FAILED,
                message=f"Failed to deserialize blueprint: {str(e)}",
                details=None
            )
        
        # Import execution engine
        try:
            from execution_engine import ExecutionEngine
        except ImportError as e:
            return APIError(
                code=ERR_MISSING_DEPENDENCY,
                message="ExecutionEngine not available",
                details={"error": str(e)}
            )
        
        # Apply execution options
        engine_kwargs = {}
        if request.execution_options:
            if "max_concurrency" in request.execution_options:
                engine_kwargs["max_concurrency"] = request.execution_options["max_concurrency"]
            if "fail_fast" in request.execution_options:
                engine_kwargs["fail_fast"] = request.execution_options["fail_fast"]
        
        # Execute
        try:
            engine = ExecutionEngine(**engine_kwargs) if engine_kwargs else ExecutionEngine()
            execution_result = await engine.run(blueprint)
        except Exception as e:
            return APIError(
                code=ERR_EXECUTION_FAILED,
                message=f"Execution failed: {str(e)}",
                details=None
            )
        
        # Serialize result
        try:
            result_dict = execution_result_to_dict(execution_result)
        except Exception as e:
            return APIError(
                code=ERR_SERIALIZATION_FAILED,
                message=f"Failed to serialize execution result: {str(e)}",
                details=None
            )
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return APIRunResponse(
            execution_id=execution_result.execution_id,
            execution_result=result_dict,
            warnings=warnings,
            duration_ms=duration_ms,
        )
        
    except Exception as e:
        return APIError(
            code=ERR_INTERNAL_ERROR,
            message=f"Internal error: {str(e)}",
            details=None
        )


# =============================================================================
# NL TO BLUEPRINT API
# =============================================================================

def generate_blueprint_api(prompt: str, global_settings: dict = None) -> dict:
    """
    Generate a Blueprint from natural language.
    
    Args:
        prompt: Natural language description
        global_settings: Optional global settings
        
    Returns:
        {
            "success": bool,
            "blueprint": dict | None,
            "warnings": list[str],
            "error": str | None
        }
        
    Never raises exceptions.
    """
    try:
        # Import NL translator
        try:
            from nl_to_blueprint_agent import generate_blueprint_from_nl
        except ImportError:
            return {
                "success": False,
                "blueprint": None,
                "warnings": [],
                "error": "NL translator not available (Phase E)"
            }
        
        # Generate blueprint
        try:
            blueprint = generate_blueprint_from_nl(prompt, global_settings or {})
        except Exception as e:
            return {
                "success": False,
                "blueprint": None,
                "warnings": [],
                "error": f"Generation failed: {str(e)}"
            }
        
        if not blueprint:
            return {
                "success": False,
                "blueprint": None,
                "warnings": [],
                "error": "No blueprint generated"
            }
        
        # Serialize
        try:
            blueprint_dict = blueprint_to_dict(blueprint)
        except Exception as e:
            return {
                "success": False,
                "blueprint": None,
                "warnings": [],
                "error": f"Serialization failed: {str(e)}"
            }
        
        # Validate
        validation = validate_blueprint_api(blueprint_dict)
        
        return {
            "success": validation["valid"],
            "blueprint": blueprint_dict if validation["valid"] else None,
            "warnings": validation.get("warnings", []),
            "error": "; ".join(validation["errors"]) if validation["errors"] else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "blueprint": None,
            "warnings": [],
            "error": f"Internal error: {str(e)}"
        }


# =============================================================================
# ANALYSIS APIs (READ-ONLY)
# =============================================================================

def replay_execution_api(
    blueprint_dict: dict,
    execution_result_dict: dict
) -> dict:
    """
    Replay an execution (Phase H).
    
    Args:
        blueprint_dict: Serialized Blueprint
        execution_result_dict: Serialized ExecutionResult
        
    Returns:
        {
            "success": bool,
            "replay": dict | None,
            "error": str | None
        }
        
    Never raises exceptions.
    """
    try:
        # Import Phase H
        try:
            from execution_replay import build_execution_replay, ExecutionReplay
        except ImportError:
            return {
                "success": False,
                "replay": None,
                "error": "Execution replay not available (Phase H)"
            }
        
        # Deserialize
        try:
            blueprint = dict_to_blueprint(blueprint_dict)
            execution_result = dict_to_execution_result(execution_result_dict)
        except Exception as e:
            return {
                "success": False,
                "replay": None,
                "error": f"Deserialization failed: {str(e)}"
            }
        
        # Build replay
        try:
            replay = build_execution_replay(blueprint, execution_result)
        except Exception as e:
            return {
                "success": False,
                "replay": None,
                "error": f"Replay build failed: {str(e)}"
            }
        
        # Serialize replay
        replay_dict = {
            "execution_id": replay.execution_id,
            "total_duration_ms": replay.total_duration_ms,
            "agent_count": len(replay.agent_events) if hasattr(replay, 'agent_events') else 0,
            "timeline": replay.timeline if hasattr(replay, 'timeline') else [],
        }
        
        return {
            "success": True,
            "replay": replay_dict,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "replay": None,
            "error": f"Internal error: {str(e)}"
        }


def critical_path_api(
    blueprint_dict: dict,
    execution_result_dict: dict
) -> dict:
    """
    Analyze critical path (Phase H).
    
    Args:
        blueprint_dict: Serialized Blueprint
        execution_result_dict: Serialized ExecutionResult
        
    Returns:
        {
            "success": bool,
            "critical_path": dict | None,
            "error": str | None
        }
        
    Never raises exceptions.
    """
    try:
        # Import Phase H
        try:
            from execution_replay import analyze_critical_path
        except ImportError:
            return {
                "success": False,
                "critical_path": None,
                "error": "Critical path analysis not available (Phase H)"
            }
        
        # Deserialize
        try:
            blueprint = dict_to_blueprint(blueprint_dict)
            execution_result = dict_to_execution_result(execution_result_dict)
        except Exception as e:
            return {
                "success": False,
                "critical_path": None,
                "error": f"Deserialization failed: {str(e)}"
            }
        
        # Analyze critical path
        try:
            cp_result = analyze_critical_path(blueprint, execution_result)
        except Exception as e:
            return {
                "success": False,
                "critical_path": None,
                "error": f"Analysis failed: {str(e)}"
            }
        
        # Serialize result
        cp_dict = {
            "critical_path_ms": cp_result.critical_path_ms if hasattr(cp_result, 'critical_path_ms') else 0,
            "critical_agents": cp_result.critical_agents if hasattr(cp_result, 'critical_agents') else [],
            "bottleneck_agent": cp_result.bottleneck_agent if hasattr(cp_result, 'bottleneck_agent') else None,
            "explanation": cp_result.explanation if hasattr(cp_result, 'explanation') else "",
        }
        
        return {
            "success": True,
            "critical_path": cp_dict,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "critical_path": None,
            "error": f"Internal error: {str(e)}"
        }


def optimization_insights_api(
    blueprint_dict: dict,
    execution_results: list[dict]
) -> dict:
    """
    Get optimization insights (Phase I).
    
    Args:
        blueprint_dict: Serialized Blueprint
        execution_results: List of serialized ExecutionResults
        
    Returns:
        {
            "success": bool,
            "insights": dict | None,
            "error": str | None
        }
        
    Never raises exceptions.
    """
    try:
        # Import Phase I
        try:
            from optimization_analysis import analyze_executions
        except ImportError:
            return {
                "success": False,
                "insights": None,
                "error": "Optimization analysis not available (Phase I)"
            }
        
        # Deserialize
        try:
            blueprint = dict_to_blueprint(blueprint_dict)
            results = [dict_to_execution_result(r) for r in execution_results]
        except Exception as e:
            return {
                "success": False,
                "insights": None,
                "error": f"Deserialization failed: {str(e)}"
            }
        
        if not results:
            return {
                "success": False,
                "insights": None,
                "error": "At least one execution result required"
            }
        
        # Analyze
        try:
            report = analyze_executions(blueprint, results)
        except Exception as e:
            return {
                "success": False,
                "insights": None,
                "error": f"Analysis failed: {str(e)}"
            }
        
        # Serialize suggestions
        suggestions = []
        for s in report.suggestions:
            suggestions.append({
                "suggestion_id": s.suggestion_id,
                "type": s.type,
                "description": s.description,
                "expected_speedup_pct": s.expected_speedup_pct,
                "confidence": s.confidence,
                "target_agents": list(s.target_agents),
            })
        
        insights_dict = {
            "blueprint_id": report.blueprint_id,
            "total_runs_analyzed": report.total_runs_analyzed,
            "primary_bottleneck": report.primary_bottleneck,
            "recurring_critical_agents": list(report.recurring_critical_agents),
            "suggestions": suggestions,
            "summary": report.summary,
        }
        
        return {
            "success": True,
            "insights": insights_dict,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "insights": None,
            "error": f"Internal error: {str(e)}"
        }
