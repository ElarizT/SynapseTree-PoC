"""
Web API for SynapseTree
========================

REST endpoints exposing SynapseTree capabilities.
All data persisted via AppStore - no in-memory shortcuts.
"""

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

# Import AppStore for persistence
from storage.app_store import get_app_store
from core.models import ExecutionRun, compute_inputs_hash


# REQUEST/RESPONSE MODELS

class GenerateBlueprintRequest(BaseModel):
    description: str
    app_id: Optional[str] = None 
    app_name: Optional[str] = None


class GenerateBlueprintResponse(BaseModel):
    app_id: str
    version_id: str
    blueprint_id: str  # Alias for version_id 
    blueprint: dict


class RunExecutionRequest(BaseModel):
    blueprint_id: str


class RunExecutionResponse(BaseModel):
    execution_id: str
    execution_result: dict
    metadata: dict


class OptimizationPreviewRequest(BaseModel):
    blueprint_id: str
    suggestion_id: str


class OptimizationApplyRequest(BaseModel):
    blueprint_id: str
    suggestion_id: str


# ROUTER

router = APIRouter(prefix="/api", tags=["web-mvp"])


# BLUEPRINT ENDPOINTS

@router.post("/blueprint/generate")
async def generate_blueprint(request: GenerateBlueprintRequest) -> GenerateBlueprintResponse:
    """
    Generate a Blueprint from natural language description.
    Creates or uses an App, creates a new BlueprintVersion.
    """
    from templates.template_engine import generate_blueprint_from_template
    
    store = get_app_store()
    description = request.description.lower()
    
    # Select template based on description
    if "research" in description or "analyze" in description:
        template_name = "research-synthesis"
    elif "code" in description or "review" in description:
        template_name = "code-review"
    elif "summarize" in description or "summary" in description:
        template_name = "summarization"
    else:
        template_name = "rapid-analysis"
    
    # Generate blueprint from template
    blueprint = generate_blueprint_from_template(
        template_name,
        user_prompt=request.description
    )
    
    # Get or create app
    if request.app_id:
        app = store.get_app(request.app_id)
        if not app:
            raise HTTPException(status_code=404, detail=f"App {request.app_id} not found")
    else:
        # Create new app
        app_name = request.app_name or f"App: {request.description[:50]}"
        app = store.create_app(
            name=app_name,
            description=request.description,
        )
    
    # Create version (persisted)
    version = store.create_version(
        app_id=app.id,
        blueprint=blueprint,
        message=f"Generated from: {template_name}",
    )
    
    return GenerateBlueprintResponse(
        app_id=app.id,
        version_id=version.id,
        blueprint_id=version.id,  # Backwards compat
        blueprint=blueprint
    )


@router.get("/blueprint/{blueprint_id}")
async def get_blueprint(blueprint_id: str, app_id: Optional[str] = None) -> dict:
    """Get a stored blueprint by version ID."""
    store = get_app_store()
    
    # Search all apps if app_id not provided
    if app_id:
        version = store.get_version(app_id, blueprint_id)
        if version:
            return version.blueprint_json
    else:
        # Search all apps for this version
        for app in store.list_apps():
            version = store.get_version(app.id, blueprint_id)
            if version:
                return version.blueprint_json
    
    raise HTTPException(status_code=404, detail=f"Blueprint {blueprint_id} not found")


# EXECUTION ENDPOINTS

@router.post("/execution/run")
async def run_execution(request: RunExecutionRequest) -> RunExecutionResponse:
    """
    Execute a blueprint with real LLMs.
    Persists ExecutionRun to AppStore.
    """
    store = get_app_store()
    
    # Find the version (blueprint_id = version_id)
    version = None
    app_id = None
    for app in store.list_apps():
        v = store.get_version(app.id, request.blueprint_id)
        if v:
            version = v
            app_id = app.id
            break
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Blueprint {request.blueprint_id} not found")
    
    blueprint = version.blueprint_json
    execution_id = f"exec-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now()
    
    try:
        # Import and run execution engine
        from api import run_blueprint_api, APIRunRequest
        
        api_request = APIRunRequest(blueprint=blueprint)
        result = await run_blueprint_api(api_request)
        
        # Handle API error
        if hasattr(result, 'code'):
            raise HTTPException(status_code=500, detail=result.message)
        
        completed_at = datetime.now()
        duration_ms = (completed_at - started_at).total_seconds() * 1000
        
        # Extract agent latencies from result
        agent_latencies = {}
        agent_results = result.execution_result.get("agent_results", {})
        for agent_id, agent_result in agent_results.items():
            started = agent_result.get("started_at")
            completed = agent_result.get("completed_at")
            if started and completed:
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                agent_latencies[agent_id] = (end_dt - start_dt).total_seconds() * 1000
        
        # Build metadata
        models_used = {}
        for agent in blueprint.get("agents", []):
            agent_id = agent.get("agent_id")
            model_config = agent.get("model_config", {})
            models_used[agent_id] = model_config.get("model", "default")
        
        # Create and persist ExecutionRun
        run = ExecutionRun(
            id=execution_id,
            blueprint_version_id=version.id,
            execution_trace=result.execution_result,
            created_at=started_at,
            total_latency_ms=duration_ms,
            critical_path_ms=0.0,  # Computed on demand
            speedup=1.0,
            inputs_hash=compute_inputs_hash({"prompt": blueprint.get("prompt", "")}),
            agent_latencies=agent_latencies,
        )
        store.save_run(run)
        
        metadata = {
            "execution_id": execution_id,
            "app_id": app_id,
            "blueprint_version_id": version.id,
            "models_used": models_used,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": duration_ms,
        }
        
        return RunExecutionResponse(
            execution_id=execution_id,
            execution_result=result.execution_result,
            metadata=metadata,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/{execution_id}")
async def get_execution(execution_id: str, app_id: Optional[str] = None) -> dict:
    """Get execution result by ID."""
    store = get_app_store()
    
    # Search all apps if app_id not provided
    run = None
    found_app_id = app_id
    
    if app_id:
        run = store.get_run(app_id, execution_id)
    else:
        for app in store.list_apps():
            r = store.get_run(app.id, execution_id)
            if r:
                run = r
                found_app_id = app.id
                break
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    return {
        "execution_id": run.id,
        "app_id": found_app_id,
        "blueprint_version_id": run.blueprint_version_id,
        "result": run.execution_trace,
        "metadata": {
            "total_latency_ms": run.total_latency_ms,
            "critical_path_ms": run.critical_path_ms,
            "speedup": run.speedup,
            "created_at": run.created_at.isoformat(),
        },
    }


@router.get("/execution/replay/{execution_id}")
async def get_execution_replay(execution_id: str, app_id: Optional[str] = None) -> dict:
    """
    Get execution replay with timeline spans.
    """
    store = get_app_store()
    
    # Find execution
    run = None
    found_app_id = app_id
    
    if app_id:
        run = store.get_run(app_id, execution_id)
    else:
        for app in store.list_apps():
            r = store.get_run(app.id, execution_id)
            if r:
                run = r
                found_app_id = app.id
                break
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    # Get blueprint
    version = store.get_version(found_app_id, run.blueprint_version_id)
    blueprint = version.blueprint_json if version else {}
    
    try:
        from api import replay_execution_api
        
        replay_result = replay_execution_api(blueprint, run.execution_trace)
        
        if not replay_result.get("success"):
            return {
                "execution_id": execution_id,
                "error": replay_result.get("error"),
                "timeline_spans": [],
            }
        
        return {
            "execution_id": execution_id,
            "metadata": {
                "total_latency_ms": run.total_latency_ms,
                "created_at": run.created_at.isoformat(),
            },
            "replay": replay_result.get("replay"),
            "timeline_spans": _extract_timeline_spans(run.execution_trace),
        }
        
    except Exception as e:
        return {
            "execution_id": execution_id,
            "error": str(e),
            "timeline_spans": _extract_timeline_spans(run.execution_trace) if run else [],
        }


@router.get("/execution/critical-path/{execution_id}")
async def get_critical_path(execution_id: str, app_id: Optional[str] = None) -> dict:
    """
    Get critical path analysis for an execution.
    """
    store = get_app_store()
    
    # Find execution
    run = None
    found_app_id = app_id
    
    if app_id:
        run = store.get_run(app_id, execution_id)
    else:
        for app in store.list_apps():
            r = store.get_run(app.id, execution_id)
            if r:
                run = r
                found_app_id = app.id
                break
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    # Get blueprint
    version = store.get_version(found_app_id, run.blueprint_version_id)
    blueprint = version.blueprint_json if version else {}
    
    try:
        from api import critical_path_api
        
        cp_result = critical_path_api(blueprint, run.execution_trace)
        
        return {
            "execution_id": execution_id,
            "metadata": {
                "total_latency_ms": run.total_latency_ms,
            },
            "critical_path": cp_result.get("critical_path"),
            "success": cp_result.get("success"),
        }
        
    except Exception as e:
        return {
            "execution_id": execution_id,
            "error": str(e),
        }


# OPTIMIZATION ENDPOINTS

@router.get("/optimization/insights/{blueprint_id}")
async def get_optimization_insights(blueprint_id: str, app_id: Optional[str] = None) -> dict:
    """
    Get optimization suggestions for a blueprint.
    """
    store = get_app_store()
    
    # Find version
    version = None
    found_app_id = app_id
    
    if app_id:
        version = store.get_version(app_id, blueprint_id)
    else:
        for app in store.list_apps():
            v = store.get_version(app.id, blueprint_id)
            if v:
                version = v
                found_app_id = app.id
                break
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Blueprint {blueprint_id} not found")
    
    blueprint = version.blueprint_json
    
    # Find executions for this version
    runs = store.get_runs_for_version(found_app_id, blueprint_id)
    executions = [r.execution_trace for r in runs]
    
    if not executions:
        return {
            "blueprint_id": blueprint_id,
            "suggestions": [],
            "message": "No executions found. Run the blueprint first.",
        }
    
    try:
        from api import optimization_insights_api
        
        insights = optimization_insights_api(blueprint, executions)
        
        return {
            "blueprint_id": blueprint_id,
            "insights": insights.get("insights"),
            "success": insights.get("success"),
        }
        
    except Exception as e:
        return {
            "blueprint_id": blueprint_id,
            "error": str(e),
            "suggestions": [],
        }


@router.post("/optimization/preview")
async def preview_optimization(request: OptimizationPreviewRequest) -> dict:
    """
    Preview optimization changes before applying.
    """
    store = get_app_store()
    
    # Find version
    version = None
    for app in store.list_apps():
        v = store.get_version(app.id, request.blueprint_id)
        if v:
            version = v
            break
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Blueprint {request.blueprint_id} not found")
    
    blueprint = version.blueprint_json
    
    # Generate preview based on suggestion
    preview = _generate_optimization_preview(blueprint, request.suggestion_id)
    
    return {
        "blueprint_id": request.blueprint_id,
        "suggestion_id": request.suggestion_id,
        "preview": preview,
    }


@router.post("/optimization/apply")
async def apply_optimization(request: OptimizationApplyRequest) -> dict:
    """
    Apply optimization and create new blueprint version (child of original).
    """
    store = get_app_store()
    
    # Find version
    version = None
    found_app_id = None
    for app in store.list_apps():
        v = store.get_version(app.id, request.blueprint_id)
        if v:
            version = v
            found_app_id = app.id
            break
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Blueprint {request.blueprint_id} not found")
    
    original = version.blueprint_json
    
    # Apply optimization
    optimized = _apply_optimization(original, request.suggestion_id)
    
    # Create new version as child of original
    new_version = store.create_version(
        app_id=found_app_id,
        blueprint=optimized,
        parent_version_id=version.id,
        message=f"Optimized: {request.suggestion_id}",
    )
    
    return {
        "app_id": found_app_id,
        "original_version_id": request.blueprint_id,
        "new_version_id": new_version.id,
        "new_blueprint_id": new_version.id,  # Backwards compat
        "blueprint": optimized,
    }


# HELPER FUNCTIONS

def _extract_timeline_spans(execution_result: dict) -> list[dict]:
    """Extract timeline spans from execution result."""
    spans = []
    agent_results = execution_result.get("agent_results", {})
    
    # Find earliest start time
    all_starts = []
    for agent_id, result in agent_results.items():
        started_at = result.get("started_at")
        if started_at:
            all_starts.append(datetime.fromisoformat(started_at.replace("Z", "+00:00")))
    
    baseline = min(all_starts) if all_starts else datetime.now()
    
    for agent_id, result in agent_results.items():
        started_at = result.get("started_at")
        completed_at = result.get("completed_at")
        
        if started_at and completed_at:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            
            start_offset_ms = (start_dt - baseline).total_seconds() * 1000
            duration_ms = (end_dt - start_dt).total_seconds() * 1000
            
            spans.append({
                "agent_id": agent_id,
                "start_offset_ms": start_offset_ms,
                "duration_ms": duration_ms,
                "status": result.get("status", "unknown"),
            })
    
    return sorted(spans, key=lambda x: x["start_offset_ms"])


def _generate_optimization_preview(blueprint: dict, suggestion_id: str) -> dict:
    """Generate optimization preview."""
    # Simple optimization: parallelize more agents
    agents = blueprint.get("agents", [])
    
    changes = []
    expected_speedup = 1.0
    
    if suggestion_id == "parallelize":
        for agent in agents:
            if not agent.get("execution", {}).get("parallelizable", True):
                changes.append({
                    "type": "modify",
                    "agent_id": agent["agent_id"],
                    "field": "parallelizable",
                    "from": False,
                    "to": True,
                })
        expected_speedup = 1.3
    
    elif suggestion_id == "faster_model":
        for agent in agents:
            model = agent.get("model_config", {}).get("model", "")
            if "deepseek" in model:
                changes.append({
                    "type": "modify",
                    "agent_id": agent["agent_id"],
                    "field": "model",
                    "from": model,
                    "to": "qwen/qwen3-coder",
                })
        expected_speedup = 1.2
    
    return {
        "changes": changes,
        "expected_speedup": expected_speedup,
        "warnings": [],
    }


def _apply_optimization(blueprint: dict, suggestion_id: str) -> dict:
    """Apply optimization to blueprint."""
    import copy
    optimized = copy.deepcopy(blueprint)
    
    if suggestion_id == "parallelize":
        for agent in optimized.get("agents", []):
            execution = agent.setdefault("execution", {})
            execution["parallelizable"] = True
    
    elif suggestion_id == "faster_model":
        for agent in optimized.get("agents", []):
            model_config = agent.get("model_config", {})
            if "deepseek" in model_config.get("model", ""):
                model_config["model"] = "qwen/qwen3-coder"
    
    return optimized


# PHASE Z: APP MANAGEMENT ENDPOINTS

class CreateAppRequest(BaseModel):
    name: str
    description: str
    tags: list[str] = []


class CreateVersionRequest(BaseModel):
    blueprint: dict
    parent_version_id: str = None
    message: str = ""


class ForkVersionRequest(BaseModel):
    source_version_id: str
    message: str = "Forked"


@router.get("/apps")
async def list_apps() -> dict:
    """List all saved agentic apps."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    apps = store.list_apps()
    
    return {
        "apps": [a.to_dict() for a in apps],
        "count": len(apps),
    }


@router.post("/apps")
async def create_app(request: CreateAppRequest) -> dict:
    """Create a new agentic app."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    app = store.create_app(
        name=request.name,
        description=request.description,
        tags=request.tags,
    )
    
    return {
        "app": app.to_dict(),
    }


@router.get("/apps/{app_id}")
async def get_app(app_id: str) -> dict:
    """Get an app by ID."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    app = store.get_app(app_id)
    
    if not app:
        raise HTTPException(status_code=404, detail=f"App {app_id} not found")
    
    return {
        "app": app.to_dict(),
    }


@router.delete("/apps/{app_id}")
async def delete_app(app_id: str) -> dict:
    """Delete an app."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    deleted = store.delete_app(app_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"App {app_id} not found")
    
    return {"deleted": True, "app_id": app_id}


# PHASE Z: VERSION ENDPOINTS

@router.get("/apps/{app_id}/versions")
async def get_version_history(app_id: str) -> dict:
    """Get version history for an app."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    app = store.get_app(app_id)
    
    if not app:
        raise HTTPException(status_code=404, detail=f"App {app_id} not found")
    
    versions = store.get_version_history(app_id)
    
    return {
        "app_id": app_id,
        "versions": [v.to_dict() for v in versions],
        "count": len(versions),
    }


@router.get("/apps/{app_id}/version-tree")
async def get_version_tree(app_id: str) -> dict:
    """Get version history as a tree structure."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    app = store.get_app(app_id)
    
    if not app:
        raise HTTPException(status_code=404, detail=f"App {app_id} not found")
    
    return store.get_version_tree(app_id)


@router.post("/apps/{app_id}/versions")
async def create_version(app_id: str, request: CreateVersionRequest) -> dict:
    """Create a new version for an app."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    app = store.get_app(app_id)
    
    if not app:
        raise HTTPException(status_code=404, detail=f"App {app_id} not found")
    
    version = store.create_version(
        app_id=app_id,
        blueprint=request.blueprint,
        parent_version_id=request.parent_version_id,
        message=request.message,
    )
    
    return {
        "version": version.to_dict(),
    }


@router.get("/apps/{app_id}/versions/{version_id}")
async def get_version(app_id: str, version_id: str) -> dict:
    """Get a specific version."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    version = store.get_version(app_id, version_id)
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    
    return {
        "version": version.to_dict(),
    }


@router.post("/apps/{app_id}/fork")
async def fork_version(app_id: str, request: ForkVersionRequest) -> dict:
    """Fork a version, creating a new version with the same blueprint."""
    from storage.app_store import get_app_store
    
    store = get_app_store()
    
    version = store.fork_version(
        app_id=app_id,
        source_version_id=request.source_version_id,
        message=request.message,
    )
    
    if not version:
        raise HTTPException(status_code=404, detail=f"Version {request.source_version_id} not found")
    
    return {
        "version": version.to_dict(),
        "forked_from": request.source_version_id,
    }


# PHASE Z: DIFF ENDPOINTS

@router.get("/diff/{app_id}/{version1_id}/{version2_id}")
async def get_blueprint_diff(app_id: str, version1_id: str, version2_id: str) -> dict:
    """Get structural diff between two versions."""
    from storage.app_store import get_app_store
    from core.blueprint_diff import compute_blueprint_diff, get_causal_summary
    
    store = get_app_store()
    
    v1 = store.get_version(app_id, version1_id)
    v2 = store.get_version(app_id, version2_id)
    
    if not v1:
        raise HTTPException(status_code=404, detail=f"Version {version1_id} not found")
    if not v2:
        raise HTTPException(status_code=404, detail=f"Version {version2_id} not found")
    
    diff = compute_blueprint_diff(
        v1.blueprint_json,
        v2.blueprint_json,
        source_version_id=version1_id,
        target_version_id=version2_id,
    )
    
    return {
        "diff": diff.to_dict(),
        "causal_summary": get_causal_summary(diff),
    }


# PHASE Z: EXECUTION COMPARISON ENDPOINTS

@router.get("/compare/{app_id}/{run1_id}/{run2_id}")
async def compare_executions(app_id: str, run1_id: str, run2_id: str) -> dict:
    """Compare two execution runs."""
    from storage.app_store import get_app_store
    from core.execution_compare import compare_executions as do_compare, summarize_comparison
    
    store = get_app_store()
    
    run1 = store.get_run(app_id, run1_id)
    run2 = store.get_run(app_id, run2_id)
    
    if not run1:
        raise HTTPException(status_code=404, detail=f"Run {run1_id} not found")
    if not run2:
        raise HTTPException(status_code=404, detail=f"Run {run2_id} not found")
    
    comparison = do_compare(run1, run2)
    
    return {
        "comparison": comparison.to_dict(),
        "summary": summarize_comparison(comparison),
    }


# APP BUILDER ENDPOINTS

class BuildAppRequest(BaseModel):
    description: str
    app_name: Optional[str] = None


class BuildAppResponse(BaseModel):
    app_id: str
    version_id: str
    execution_id: str
    app_code: dict  
    timeline_ms: float
    agents_used: list[str]


@router.post("/app/build")
async def build_app(request: BuildAppRequest) -> BuildAppResponse:
    """
    Build an actual app from natural language description.
    Uses parallel agents: UI, Style, Logic, Coordinator.
    Returns generated HTML/CSS/JS code.
    """
    from templates.app_builder_template import create_app_builder_blueprint, extract_app_code
    from blueprint_validator import parse_blueprint
    from execution_engine import ExecutionEngine
    from api.serialization import execution_result_to_dict
    
    store = get_app_store()
    
    # Create app builder blueprint
    blueprint_dict = create_app_builder_blueprint(request.description)
    
    # Parse into Blueprint object
    blueprint = parse_blueprint(blueprint_dict)
    
    # Create app in store
    app_name = request.app_name or f"Built App: {request.description[:40]}"
    app = store.create_app(
        name=app_name,
        description=request.description,
    )
    
    # Create version
    version = store.create_version(
        app_id=app.id,
        blueprint=blueprint_dict,
        message="App builder execution",
    )
    
    # Execute blueprint
    engine = ExecutionEngine()
    result = await engine.run(blueprint)
    
    # Serialize result
    result_dict = execution_result_to_dict(result)
    
    # Extract app code
    app_code = extract_app_code(result_dict)
    
    # Get timeline
    total_ms = result.total_execution_ms or 0
    
    # Get agents used
    agents_used = list(result.agent_results.keys())
    
    # Persist run
    from core.models import ExecutionRun
    from datetime import datetime
    run = ExecutionRun(
        id=result.execution_id,
        blueprint_version_id=version.id,
        execution_trace=result_dict,
        total_latency_ms=total_ms,
        critical_path_ms=result.parallelism_metrics.critical_path_ms if result.parallelism_metrics else total_ms,
        speedup=result.parallelism_metrics.parallelism_speedup if result.parallelism_metrics else 1.0,
        created_at=datetime.now(),
    )
    store.save_run(run)
    
    return BuildAppResponse(
        app_id=app.id,
        version_id=version.id,
        execution_id=result.execution_id,
        app_code=app_code,
        timeline_ms=total_ms,
        agents_used=agents_used,
    )


class BuildFullstackRequest(BaseModel):
    """Request to build a full-stack AI app with LLM backend."""
    description: str
    provider: str  # gemini, openai, groq, anthropic
    model: str     # Model name
    app_name: Optional[str] = None


class BuildFullstackResponse(BaseModel):
    """Response with generated full-stack app."""
    app_id: str
    version_id: str
    execution_id: str
    app_code: dict  # {html, css, javascript, combined}
    timeline_ms: float
    agents_used: list[str]
    config: dict    # {provider, model}


@router.post("/app/build-fullstack")
async def build_fullstack_app(request: BuildFullstackRequest) -> BuildFullstackResponse:
    """
    Build a full-stack AI app with LLM backend connectivity.
    
    The generated app includes:
    - Frontend UI with chat interface
    - Backend proxy configuration for LLM calls
    - Selected provider/model embedded in the app
    """
    from templates.app_builder_template import create_app_builder_blueprint, extract_app_code
    from blueprint_validator import parse_blueprint
    from execution_engine import ExecutionEngine
    from api.serialization import execution_result_to_dict
    
    store = get_app_store()
    
    # Enhanced description with LLM config
    enhanced_description = f"""{request.description}

IMPORTANT BACKEND CONFIGURATION:
- This app must include a chat interface that connects to an AI backend
- The AI backend uses {request.provider} with model {request.model}
- Include a function to call the backend API at /api/runtime/chat
- The chat should send messages to the backend and display responses
- Add loading states while waiting for AI responses
"""
    
    # Create app builder blueprint with enhanced description
    blueprint_dict = create_app_builder_blueprint(enhanced_description)
    
    # Parse into Blueprint object
    blueprint = parse_blueprint(blueprint_dict)
    
    # Create app in store
    app_name = request.app_name or f"AI App: {request.description[:40]}"
    app = store.create_app(
        name=app_name,
        description=request.description,
    )
    
    # Store config in app metadata
    app.metadata = {
        "provider": request.provider,
        "model": request.model,
        "is_fullstack": True,
    }
    
    # Create version
    version = store.create_version(
        app_id=app.id,
        blueprint=blueprint_dict,
        message=f"Full-stack app ({request.provider}/{request.model})",
    )
    
    # Execute blueprint
    engine = ExecutionEngine()
    result = await engine.run(blueprint)
    
    # Serialize result
    result_dict = execution_result_to_dict(result)
    
    # Extract app code
    app_code = extract_app_code(result_dict)
    
    # Inject runtime API configuration into the generated JavaScript
    runtime_js = f"""
// SynapseTree Runtime Configuration
const SYNAPSE_CONFIG = {{
    provider: '{request.provider}',
    model: '{request.model}',
    endpoint: '/api/runtime/chat',
    getApiKey: function() {{
        // API key is injected into window by the preview iframe
        return window.SYNAPSE_API_KEY || '';
    }}
}};

// Helper to call the AI backend
async function callAI(message, history) {{
    const apiKey = SYNAPSE_CONFIG.getApiKey();
    if (!apiKey) {{
        throw new Error('No API key found. Please refresh and rebuild the app.');
    }}
    
    const response = await fetch(SYNAPSE_CONFIG.endpoint, {{
        method: 'POST',
        headers: {{ 
            'Content-Type': 'application/json',
            'X-API-Key': apiKey
        }},
        body: JSON.stringify({{
            message: message,
            provider: SYNAPSE_CONFIG.provider,
            model: SYNAPSE_CONFIG.model,
            history: history || []
        }})
    }});
    
    if (!response.ok) {{
        const error = await response.json();
        throw new Error(error.detail || 'Failed to connect to AI');
    }}
    
    return await response.json();
}}

"""
    # Prepend runtime config to generated JavaScript
    if app_code.get('javascript'):
        app_code['javascript'] = runtime_js + app_code['javascript']
    
    # Also inject into combined HTML
    if app_code.get('combined'):
        app_code['combined'] = app_code['combined'].replace(
            '<script>',
            f'<script>\n{runtime_js}'
        )
    
    # Get timeline
    total_ms = result.total_execution_ms or 0
    
    # Get agents used
    agents_used = list(result.agent_results.keys())
    
    # Persist run
    from core.models import ExecutionRun
    from datetime import datetime
    run = ExecutionRun(
        id=result.execution_id,
        blueprint_version_id=version.id,
        execution_trace=result_dict,
        total_latency_ms=total_ms,
        critical_path_ms=result.parallelism_metrics.critical_path_ms if result.parallelism_metrics else total_ms,
        speedup=result.parallelism_metrics.parallelism_speedup if result.parallelism_metrics else 1.0,
        created_at=datetime.now(),
    )
    store.save_run(run)
    
    return BuildFullstackResponse(
        app_id=app.id,
        version_id=version.id,
        execution_id=result.execution_id,
        app_code=app_code,
        timeline_ms=total_ms,
        agents_used=agents_used,
        config={
            "provider": request.provider,
            "model": request.model,
        }
    )

@router.get("/app/preview/{execution_id}")
async def get_app_preview(execution_id: str) -> dict:
    """
    Get the generated app code for preview.
    Returns the combined HTML that can be rendered in an iframe.
    """
    from templates.app_builder_template import extract_app_code
    
    store = get_app_store()
    
    # Find the run
    run = None
    found_app_id = None
    for app in store.list_apps():
        r = store.get_run(app.id, execution_id)
        if r:
            run = r
            found_app_id = app.id
            break
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    # Extract app code
    app_code = extract_app_code(run.execution_trace)
    
    return {
        "execution_id": execution_id,
        "app_id": found_app_id,
        "app_code": app_code,
        "preview_html": app_code.get("combined", ""),
    }


class RuntimeChatRequest(BaseModel):
    """Request to chat with the AI runtime."""
    message: str
    provider: str
    model: str
    history: Optional[list[dict]] = None  # Previous messages


class RuntimeChatResponse(BaseModel):
    """Response from AI runtime."""
    response: str
    provider: str
    model: str


@router.post("/runtime/chat")
async def runtime_chat(
    request: RuntimeChatRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> RuntimeChatResponse:
    """
    Proxy chat messages to the configured LLM provider.
    
    The API key should be passed via the X-API-Key header.
    This endpoint is called by generated apps to make LLM calls.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Pass via X-API-Key header."
        )
    
    # Build messages in OpenAI format (most providers use this)
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    if request.history:
        messages.extend(request.history)
    messages.append({"role": "user", "content": request.message})
    
    # Route to the appropriate provider
    try:
        if request.provider == "gemini":
            from llm_providers.gemini_client import call_gemini
            response = await call_gemini(
                model=request.model,
                messages=messages,
                api_key=x_api_key,
            )
        elif request.provider == "groq":
            from llm_providers.groq_client import call_groq
            response = await call_groq(
                model=request.model,
                messages=messages,
                api_key=x_api_key,
            )
        elif request.provider == "openai":
            # OpenAI client - use OpenRouter instead for flexibility
            from llm_providers.openrouter_client import call_openrouter
            response = await call_openrouter(
                model=f"openai/{request.model}",
                messages=messages,
                api_key=x_api_key,
            )
        elif request.provider == "anthropic":
            # Use OpenRouter for Anthropic models
            from llm_providers.openrouter_client import call_openrouter
            response = await call_openrouter(
                model=f"anthropic/{request.model}",
                messages=messages,
                api_key=x_api_key,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {request.provider}"
            )
        
        return RuntimeChatResponse(
            response=response,
            provider=request.provider,
            model=request.model,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}"
        )

