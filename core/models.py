"""
Phase Z Core Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib
import json

@dataclass
class AgenticApp:
    """
    A saved agentic application.
    An app is a container for blueprint versions.
    """
    id: str
    name: str
    description: str
    created_at: datetime
    current_version_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "current_version_id": self.current_version_id,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgenticApp":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            current_version_id=data.get("current_version_id"),
            tags=data.get("tags", []),
        )


# BLUEPRINT VERSION (IMMUTABLE)

@dataclass
class BlueprintVersion:
    """
    An immutable blueprint version.
    
    Any edit creates a new version.
    Parent lineage is preserved for diffing and forking.
    """
    id: str
    app_id: str
    blueprint_json: dict
    created_at: datetime
    parent_version_id: Optional[str] = None
    message: str = ""  
    depth: int = 0  
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "app_id": self.app_id,
            "parent_version_id": self.parent_version_id,
            "blueprint_json": self.blueprint_json,
            "created_at": self.created_at.isoformat(),
            "message": self.message,
            "depth": self.depth,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BlueprintVersion":
        return cls(
            id=data["id"],
            app_id=data["app_id"],
            parent_version_id=data.get("parent_version_id"),
            blueprint_json=data["blueprint_json"],
            created_at=datetime.fromisoformat(data["created_at"]),
            message=data.get("message", ""),
            depth=data.get("depth", 0),
        )
    
    @property
    def blueprint_hash(self) -> str:
        """Content hash for comparing blueprints."""
        content = json.dumps(self.blueprint_json, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# EXECUTION RUN

@dataclass
class ExecutionRun:
    """
    A single execution of a blueprint version.
    
    Includes inputs_hash for deterministic replay validation.
    """
    id: str
    blueprint_version_id: str
    execution_trace: dict
    created_at: datetime
    
    # Metrics
    total_latency_ms: float = 0.0
    critical_path_ms: float = 0.0
    speedup: float = 1.0
    
    # For deterministic replay validation
    inputs_hash: str = ""
    
    # Agent-level metrics
    agent_latencies: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "blueprint_version_id": self.blueprint_version_id,
            "execution_trace": self.execution_trace,
            "created_at": self.created_at.isoformat(),
            "total_latency_ms": self.total_latency_ms,
            "critical_path_ms": self.critical_path_ms,
            "speedup": self.speedup,
            "inputs_hash": self.inputs_hash,
            "agent_latencies": self.agent_latencies,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionRun":
        return cls(
            id=data["id"],
            blueprint_version_id=data["blueprint_version_id"],
            execution_trace=data["execution_trace"],
            created_at=datetime.fromisoformat(data["created_at"]),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            critical_path_ms=data.get("critical_path_ms", 0.0),
            speedup=data.get("speedup", 1.0),
            inputs_hash=data.get("inputs_hash", ""),
            agent_latencies=data.get("agent_latencies", {}),
        )


# HELPERS

def compute_inputs_hash(inputs: dict) -> str:
    """Compute hash of execution inputs for replay validation."""
    content = json.dumps(inputs, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
