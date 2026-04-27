"""
Serialization Utilities (Phase S)
==================================

JSON-safe serialization for all SynapseTree types.

Rules:
- ISO timestamps only
- No non-JSON objects
- Deterministic field ordering
"""

import json
from datetime import datetime
from typing import Any, Optional

from blueprint_schema import (
    Blueprint, AgentDefinition, AgentExecution, ModelConfig,
    Edge, EdgeType, AgentStatus
)
from execution_result import (
    ExecutionResult, AgentExecutionResult, AgentExecutionStatus,
    ParallelismMetrics
)


# =============================================================================
# BLUEPRINT SERIALIZATION
# =============================================================================

def blueprint_to_dict(blueprint: Blueprint) -> dict:
    """
    Serialize a Blueprint to a JSON-safe dictionary.
    
    Args:
        blueprint: Blueprint to serialize
        
    Returns:
        JSON-safe dictionary
    """
    agents = []
    for agent in blueprint.agents:
        agent_dict = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "role": agent.role,
            "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
        }
        
        # Execution config
        if agent.execution:
            agent_dict["execution"] = {
                "parallelizable": agent.execution.parallelizable,
                "depends_on": list(agent.execution.depends_on) if agent.execution.depends_on else [],
                "timeout_seconds": agent.execution.timeout_seconds,
                "retry_count": agent.execution.retry_count,
            }
        
        # Model config
        if agent.model_config:
            agent_dict["model_config"] = {
                "model": agent.model_config.model,
                "provider": agent.model_config.provider,
                "temperature": agent.model_config.temperature,
                "max_tokens": agent.model_config.max_tokens,
            }
        
        # Optional fields
        if agent.can_spawn_agents:
            agent_dict["can_spawn_agents"] = True
        if agent.tools:
            agent_dict["tools"] = list(agent.tools)
        if agent.behavior:
            agent_dict["behavior"] = agent.behavior
        if agent.metadata:
            agent_dict["metadata"] = agent.metadata
        
        agents.append(agent_dict)
    
    # Edges
    edges = []
    for edge in blueprint.edges:
        edges.append({
            "from_agent": edge.from_agent,
            "to_agent": edge.to_agent,
            "type": edge.type.value if hasattr(edge.type, 'value') else str(edge.type),
        })
    
    return {
        "blueprint_id": blueprint.blueprint_id,
        "name": blueprint.name,
        "description": blueprint.description,
        "coordinator_agent_id": blueprint.coordinator_agent_id,
        "agents": agents,
        "edges": edges,
        "schema_version": getattr(blueprint, 'schema_version', None),
    }


def dict_to_blueprint(data: dict) -> Blueprint:
    """
    Deserialize a dictionary to a Blueprint.
    
    Args:
        data: Dictionary representation
        
    Returns:
        Blueprint object
        
    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Empty blueprint data")
    
    if "agents" not in data or not data["agents"]:
        raise ValueError("Blueprint must have at least one agent")
    
    agents = []
    for agent_dict in data["agents"]:
        # Parse execution
        execution = None
        if "execution" in agent_dict:
            exec_dict = agent_dict["execution"]
            execution = AgentExecution(
                parallelizable=exec_dict.get("parallelizable", True),
                depends_on=exec_dict.get("depends_on", []),
                timeout_seconds=exec_dict.get("timeout_seconds", 30),
                retry_count=exec_dict.get("retry_count", 0),
            )
        
        # Parse model config
        model_config = None
        if "model_config" in agent_dict:
            mc = agent_dict["model_config"]
            model_config = ModelConfig(
                model=mc.get("model", "default"),
                provider=mc.get("provider", "openai"),
                temperature=mc.get("temperature", 0.7),
                max_tokens=mc.get("max_tokens"),
            )
        
        # Parse status
        status = AgentStatus.PENDING
        if "status" in agent_dict:
            try:
                status = AgentStatus(agent_dict["status"])
            except ValueError:
                pass
        
        agents.append(AgentDefinition(
            agent_id=agent_dict["agent_id"],
            name=agent_dict["name"],
            role=agent_dict["role"],
            execution=execution,
            model_config=model_config,
            status=status,
            can_spawn_agents=agent_dict.get("can_spawn_agents", False),
            tools=agent_dict.get("tools", []),
            behavior=agent_dict.get("behavior", {}),
            metadata=agent_dict.get("metadata", {}),
        ))
    
    # Parse edges
    edges = []
    for edge_dict in data.get("edges", []):
        edge_type_str = edge_dict.get("type", "feeds")
        try:
            edge_type = EdgeType(edge_type_str)
        except ValueError:
            edge_type = EdgeType.FEEDS
        
        edges.append(Edge(
            from_agent=edge_dict.get("from_agent", ""),
            to_agent=edge_dict.get("to_agent", ""),
            type=edge_type,
        ))
    
    # Determine coordinator
    coordinator_id = data.get("coordinator_agent_id")
    if not coordinator_id and agents:
        coordinator_id = agents[-1].agent_id
    
    return Blueprint(
        blueprint_id=data.get("blueprint_id", ""),
        name=data.get("name", "Unnamed"),
        description=data.get("description"),
        coordinator_agent_id=coordinator_id,
        agents=agents,
        edges=edges,
    )


# =============================================================================
# EXECUTION RESULT SERIALIZATION
# =============================================================================

def execution_result_to_dict(result: ExecutionResult) -> dict:
    """
    Serialize an ExecutionResult to a JSON-safe dictionary.
    
    Args:
        result: ExecutionResult to serialize
        
    Returns:
        JSON-safe dictionary
    """
    # Serialize agent results
    agent_results = {}
    for agent_id, agent_result in result.agent_results.items():
        agent_results[agent_id] = {
            "agent_id": agent_result.agent_id,
            "status": agent_result.status.value if hasattr(agent_result.status, 'value') else str(agent_result.status),
            "start_time": agent_result.start_time,
            "end_time": agent_result.end_time,
            "duration_ms": agent_result.duration_ms,
            "output": agent_result.output,
            "error_message": agent_result.error_message,
            "retry_count": agent_result.retry_count,
            "confidence_score": agent_result.confidence_score,
        }
    
    # Serialize parallelism metrics
    parallelism_metrics = None
    if result.parallelism_metrics:
        pm = result.parallelism_metrics
        parallelism_metrics = {
            "max_parallel_agents": pm.max_parallel_agents,
            "average_parallel_agents": pm.average_parallel_agents,
            "critical_path_ms": pm.critical_path_ms,
            "estimated_sequential_ms": pm.estimated_sequential_ms,
            "parallelism_speedup": pm.parallelism_speedup,
        }
    
    # Serialize timestamps
    started_at = None
    completed_at = None
    if result.started_at:
        started_at = result.started_at.isoformat() if isinstance(result.started_at, datetime) else str(result.started_at)
    if result.completed_at:
        completed_at = result.completed_at.isoformat() if isinstance(result.completed_at, datetime) else str(result.completed_at)
    
    return {
        "execution_id": result.execution_id,
        "blueprint_id": result.blueprint_id,
        "blueprint_name": result.blueprint_name,
        "success": result.success,
        "agent_results": agent_results,
        "execution_start_time": result.execution_start_time,
        "execution_end_time": result.execution_end_time,
        "total_execution_ms": result.total_execution_ms,
        "started_at": started_at,
        "completed_at": completed_at,
        "parallelism_metrics": parallelism_metrics,
        "coordinator_synthesis": result.coordinator_synthesis,
        "conflicts": result.conflicts,
        "speedup_ratio": result.speedup_ratio,
        "speedup_display": result.speedup_display,
    }


def dict_to_execution_result(data: dict) -> ExecutionResult:
    """
    Deserialize a dictionary to an ExecutionResult.
    
    Args:
        data: Dictionary representation
        
    Returns:
        ExecutionResult object
        
    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Empty execution result data")
    
    # Parse agent results
    agent_results = {}
    for agent_id, ar_dict in data.get("agent_results", {}).items():
        status = AgentExecutionStatus.PENDING
        if "status" in ar_dict:
            try:
                status = AgentExecutionStatus(ar_dict["status"])
            except ValueError:
                pass
        
        agent_results[agent_id] = AgentExecutionResult(
            agent_id=ar_dict.get("agent_id", agent_id),
            status=status,
            start_time=ar_dict.get("start_time"),
            end_time=ar_dict.get("end_time"),
            output=ar_dict.get("output"),
            error_message=ar_dict.get("error_message"),
            retry_count=ar_dict.get("retry_count", 0),
            confidence_score=ar_dict.get("confidence_score", 0.0),
        )
    
    # Parse parallelism metrics
    parallelism_metrics = ParallelismMetrics()
    if data.get("parallelism_metrics"):
        pm = data["parallelism_metrics"]
        parallelism_metrics = ParallelismMetrics(
            max_parallel_agents=pm.get("max_parallel_agents", 0),
            average_parallel_agents=pm.get("average_parallel_agents", 0.0),
            critical_path_ms=pm.get("critical_path_ms", 0.0),
            estimated_sequential_ms=pm.get("estimated_sequential_ms", 0.0),
            parallelism_speedup=pm.get("parallelism_speedup", 1.0),
        )
    
    # Parse timestamps
    started_at = None
    completed_at = None
    if data.get("started_at"):
        try:
            started_at = datetime.fromisoformat(data["started_at"])
        except (ValueError, TypeError):
            pass
    if data.get("completed_at"):
        try:
            completed_at = datetime.fromisoformat(data["completed_at"])
        except (ValueError, TypeError):
            pass
    
    return ExecutionResult(
        execution_id=data.get("execution_id", ""),
        blueprint_id=data.get("blueprint_id", ""),
        blueprint_name=data.get("blueprint_name", ""),
        success=data.get("success", False),
        agent_results=agent_results,
        execution_start_time=data.get("execution_start_time", 0.0),
        execution_end_time=data.get("execution_end_time", 0.0),
        started_at=started_at,
        completed_at=completed_at,
        parallelism_metrics=parallelism_metrics,
        coordinator_synthesis=data.get("coordinator_synthesis"),
        conflicts=data.get("conflicts", []),
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_json(obj: Any, indent: int = 2) -> str:
    """Convert any serializable object to JSON string."""
    return json.dumps(obj, indent=indent, default=str, sort_keys=True)


def from_json(json_str: str) -> Any:
    """Parse JSON string to object."""
    return json.loads(json_str)
