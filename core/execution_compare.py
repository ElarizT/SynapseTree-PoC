"""
Execution Comparison - Phase Z
"""

from dataclasses import dataclass, field
from typing import Optional

from core.models import ExecutionRun


# COMPARISON RESULT

@dataclass
class ExecutionComparison:
    """Comparison between two execution runs."""
    run1_id: str
    run2_id: str
    
    # Deltas
    latency_delta_ms: float = 0.0
    critical_path_delta_ms: float = 0.0
    speedup_percent: float = 0.0
    
    # Validation
    inputs_match: bool = True
    comparable: bool = True
    warning: str = ""
    
    # Per-agent deltas
    agent_deltas: dict[str, float] = field(default_factory=dict)
    
    # Causal attribution
    causes: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "run1_id": self.run1_id,
            "run2_id": self.run2_id,
            "latency_delta_ms": self.latency_delta_ms,
            "critical_path_delta_ms": self.critical_path_delta_ms,
            "speedup_percent": self.speedup_percent,
            "inputs_match": self.inputs_match,
            "comparable": self.comparable,
            "warning": self.warning,
            "agent_deltas": self.agent_deltas,
            "causes": self.causes,
        }


# COMPARISON ENGINE

def compare_executions(run1: ExecutionRun, run2: ExecutionRun) -> ExecutionComparison:
    """
    Compare two execution runs.
    
    Returns latency deltas, speedup %, and causal attribution.
    """
    result = ExecutionComparison(
        run1_id=run1.id,
        run2_id=run2.id,
    )
    
    # Check inputs match
    if run1.inputs_hash and run2.inputs_hash:
        result.inputs_match = run1.inputs_hash == run2.inputs_hash
        if not result.inputs_match:
            result.warning = "Inputs differ between runs - comparison may not be meaningful"
    
    # Calculate deltas
    result.latency_delta_ms = run2.total_latency_ms - run1.total_latency_ms
    result.critical_path_delta_ms = run2.critical_path_ms - run1.critical_path_ms
    
    # Speedup (positive = run2 is faster)
    if run1.total_latency_ms > 0:
        speedup = (run1.total_latency_ms - run2.total_latency_ms) / run1.total_latency_ms
        result.speedup_percent = speedup * 100
    
    # Per-agent deltas
    all_agents = set(run1.agent_latencies.keys()) | set(run2.agent_latencies.keys())
    
    for agent_id in all_agents:
        lat1 = run1.agent_latencies.get(agent_id, 0.0)
        lat2 = run2.agent_latencies.get(agent_id, 0.0)
        result.agent_deltas[agent_id] = lat2 - lat1
    
    # Causal attribution
    result.causes = _attribute_causes(run1, run2, result.agent_deltas)
    
    return result


def _attribute_causes(
    run1: ExecutionRun, 
    run2: ExecutionRun,
    agent_deltas: dict[str, float],
) -> list[dict]:
    """
    Attribute performance changes to specific causes.
    
    Returns list of causal explanations.
    """
    causes = []
    
    # Find biggest improvements
    sorted_deltas = sorted(agent_deltas.items(), key=lambda x: x[1])
    
    for agent_id, delta in sorted_deltas[:3]: 
        if delta < -100: 
            causes.append({
                "type": "agent_speedup",
                "agent_id": agent_id,
                "improvement_ms": abs(delta),
                "description": f"{agent_id} is {abs(delta):.0f}ms faster",
            })
    
    # Find regressions
    for agent_id, delta in sorted_deltas[-3:]: 
        if delta > 100: 
            causes.append({
                "type": "agent_regression",
                "agent_id": agent_id,
                "regression_ms": delta,
                "description": f"{agent_id} is {delta:.0f}ms slower",
            })
    
    # Check parallelism changes
    run1_agents = set(run1.agent_latencies.keys())
    run2_agents = set(run2.agent_latencies.keys())
    
    new_agents = run2_agents - run1_agents
    removed_agents = run1_agents - run2_agents
    
    if new_agents:
        causes.append({
            "type": "agents_added",
            "agent_ids": list(new_agents),
            "description": f"Added agents: {', '.join(new_agents)}",
        })
    
    if removed_agents:
        causes.append({
            "type": "agents_removed",
            "agent_ids": list(removed_agents),
            "description": f"Removed agents: {', '.join(removed_agents)}",
        })
    
    # Parallel execution detection
    # If total time decreased more than sum of agent improvements,
    # parallelism likely improved
    total_agent_improvement = sum(-d for d in agent_deltas.values() if d < 0)
    total_improvement = run1.total_latency_ms - run2.total_latency_ms
    
    if total_improvement > total_agent_improvement * 1.2: 
        causes.append({
            "type": "parallelism_improved",
            "description": "Improved parallel execution",
            "extra_improvement_ms": total_improvement - total_agent_improvement,
        })
    
    return causes


# HELPERS

def summarize_comparison(comparison: ExecutionComparison) -> str:
    """Generate human-readable summary."""
    if comparison.speedup_percent > 0:
        return f"{comparison.speedup_percent:.1f}% faster ({abs(comparison.latency_delta_ms):.0f}ms saved)"
    elif comparison.speedup_percent < 0:
        return f"{abs(comparison.speedup_percent):.1f}% slower ({comparison.latency_delta_ms:.0f}ms added)"
    else:
        return "No significant change"


def is_valid_comparison(run1: ExecutionRun, run2: ExecutionRun) -> tuple[bool, str]:
    """Check if two runs can be meaningfully compared."""
    # Check inputs hash
    if run1.inputs_hash and run2.inputs_hash and run1.inputs_hash != run2.inputs_hash:
        return False, "Different inputs"
    
    # Check versions are related (would need version lineage check)
    
    return True, ""
