"""
Blueprint Diff Engine - Phase Z
=================================

Structural diff with causal hints for smart UI.
"""

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# DIFF TYPES
# =============================================================================

DiffType = Literal[
    "agent_added",
    "agent_removed",
    "agent_modified",
    "dependency_added",
    "dependency_removed",
    "parallelization_changed",
    "prompt_changed",
    "model_changed",
]

CausalEffect = Literal[
    "moved_to_critical_path",
    "moved_off_critical_path",
    "enables_parallelism",
    "blocks_parallelism",
    "faster_execution",
    "slower_execution",
    "no_effect",
]


@dataclass
class DiffItem:
    """A single diff item with causal hint."""
    type: DiffType
    agent_id: str
    field: str = ""
    old_value: any = None
    new_value: any = None
    effect: CausalEffect = "no_effect"
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "effect": self.effect,
            "description": self.description,
        }


@dataclass
class BlueprintDiff:
    """Complete diff between two blueprints."""
    source_version_id: str
    target_version_id: str
    items: list[DiffItem] = field(default_factory=list)
    
    # Summary
    agents_added: int = 0
    agents_removed: int = 0
    agents_modified: int = 0
    dependencies_changed: int = 0
    
    def to_dict(self) -> dict:
        return {
            "source_version_id": self.source_version_id,
            "target_version_id": self.target_version_id,
            "items": [i.to_dict() for i in self.items],
            "summary": {
                "agents_added": self.agents_added,
                "agents_removed": self.agents_removed,
                "agents_modified": self.agents_modified,
                "dependencies_changed": self.dependencies_changed,
            }
        }


# =============================================================================
# DIFF ENGINE
# =============================================================================

def compute_blueprint_diff(
    source_blueprint: dict,
    target_blueprint: dict,
    source_version_id: str = "",
    target_version_id: str = "",
) -> BlueprintDiff:
    """
    Compute structural diff between two blueprints.
    
    Emits causal hints for smart UI display.
    """
    diff = BlueprintDiff(
        source_version_id=source_version_id,
        target_version_id=target_version_id,
    )
    
    source_agents = {a["agent_id"]: a for a in source_blueprint.get("agents", [])}
    target_agents = {a["agent_id"]: a for a in target_blueprint.get("agents", [])}
    
    source_ids = set(source_agents.keys())
    target_ids = set(target_agents.keys())
    
    # Added agents
    for agent_id in target_ids - source_ids:
        agent = target_agents[agent_id]
        deps = agent.get("execution", {}).get("depends_on", [])
        
        effect = "enables_parallelism" if not deps else "no_effect"
        
        diff.items.append(DiffItem(
            type="agent_added",
            agent_id=agent_id,
            new_value=agent,
            effect=effect,
            description=f"Added agent '{agent.get('name', agent_id)}'",
        ))
        diff.agents_added += 1
    
    # Removed agents
    for agent_id in source_ids - target_ids:
        agent = source_agents[agent_id]
        
        # Check if this was on critical path (heuristic: coordinator dependency)
        was_coordinator_dep = any(
            agent_id in target_agents.get(a, {}).get("execution", {}).get("depends_on", [])
            for a in target_ids
        )
        effect = "moved_off_critical_path" if was_coordinator_dep else "no_effect"
        
        diff.items.append(DiffItem(
            type="agent_removed",
            agent_id=agent_id,
            old_value=agent,
            effect=effect,
            description=f"Removed agent '{agent.get('name', agent_id)}'",
        ))
        diff.agents_removed += 1
    
    # Modified agents
    for agent_id in source_ids & target_ids:
        source_agent = source_agents[agent_id]
        target_agent = target_agents[agent_id]
        
        modified = False
        
        # Check dependencies
        source_deps = set(source_agent.get("execution", {}).get("depends_on", []))
        target_deps = set(target_agent.get("execution", {}).get("depends_on", []))
        
        if source_deps != target_deps:
            added_deps = target_deps - source_deps
            removed_deps = source_deps - target_deps
            
            if removed_deps:
                effect = "enables_parallelism"
                desc = f"Removed dependencies: {', '.join(removed_deps)}"
            elif added_deps:
                effect = "blocks_parallelism"
                desc = f"Added dependencies: {', '.join(added_deps)}"
            else:
                effect = "no_effect"
                desc = "Dependencies changed"
            
            diff.items.append(DiffItem(
                type="dependency_removed" if removed_deps else "dependency_added",
                agent_id=agent_id,
                field="depends_on",
                old_value=list(source_deps),
                new_value=list(target_deps),
                effect=effect,
                description=desc,
            ))
            diff.dependencies_changed += 1
            modified = True
        
        # Check parallelizable flag
        source_parallel = source_agent.get("execution", {}).get("parallelizable", True)
        target_parallel = target_agent.get("execution", {}).get("parallelizable", True)
        
        if source_parallel != target_parallel:
            effect = "enables_parallelism" if target_parallel else "blocks_parallelism"
            
            diff.items.append(DiffItem(
                type="parallelization_changed",
                agent_id=agent_id,
                field="parallelizable",
                old_value=source_parallel,
                new_value=target_parallel,
                effect=effect,
                description=f"Parallelization {'enabled' if target_parallel else 'disabled'}",
            ))
            modified = True
        
        # Check model changes
        source_model = source_agent.get("model_config", {}).get("model", "")
        target_model = target_agent.get("model_config", {}).get("model", "")
        
        if source_model != target_model:
            # Heuristic: faster models
            faster_models = ["qwen", "claude-instant", "gpt-3.5"]
            is_faster = any(m in target_model.lower() for m in faster_models)
            effect = "faster_execution" if is_faster else "slower_execution"
            
            diff.items.append(DiffItem(
                type="model_changed",
                agent_id=agent_id,
                field="model",
                old_value=source_model,
                new_value=target_model,
                effect=effect,
                description=f"Model changed from {source_model.split('/')[-1]} to {target_model.split('/')[-1]}",
            ))
            modified = True
        
        # Check prompt changes
        source_prompt = source_agent.get("prompt_template", "")
        target_prompt = target_agent.get("prompt_template", "")
        
        if source_prompt != target_prompt:
            diff.items.append(DiffItem(
                type="prompt_changed",
                agent_id=agent_id,
                field="prompt_template",
                old_value=source_prompt[:100] + "..." if len(source_prompt) > 100 else source_prompt,
                new_value=target_prompt[:100] + "..." if len(target_prompt) > 100 else target_prompt,
                effect="no_effect",
                description="Prompt template updated",
            ))
            modified = True
        
        if modified:
            diff.agents_modified += 1
    
    return diff


# =============================================================================
# HELPERS
# =============================================================================

def summarize_diff(diff: BlueprintDiff) -> str:
    """Generate human-readable summary of diff."""
    parts = []
    
    if diff.agents_added:
        parts.append(f"+{diff.agents_added} agents")
    if diff.agents_removed:
        parts.append(f"-{diff.agents_removed} agents")
    if diff.agents_modified:
        parts.append(f"~{diff.agents_modified} modified")
    
    if not parts:
        return "No changes"
    
    return ", ".join(parts)


def get_causal_summary(diff: BlueprintDiff) -> list[str]:
    """Get list of causal effect summaries."""
    effects = {}
    
    for item in diff.items:
        if item.effect != "no_effect":
            if item.effect not in effects:
                effects[item.effect] = []
            effects[item.effect].append(item.agent_id)
    
    summaries = []
    
    if "enables_parallelism" in effects:
        summaries.append(f"⚡ Enables parallel execution for: {', '.join(effects['enables_parallelism'])}")
    if "blocks_parallelism" in effects:
        summaries.append(f"🔒 Blocks parallelism for: {', '.join(effects['blocks_parallelism'])}")
    if "moved_off_critical_path" in effects:
        summaries.append(f"🏎️ Moved off critical path: {', '.join(effects['moved_off_critical_path'])}")
    if "faster_execution" in effects:
        summaries.append(f"⚡ Faster model for: {', '.join(effects['faster_execution'])}")
    
    return summaries
