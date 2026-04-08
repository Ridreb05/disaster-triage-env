# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Disaster Triage Environment.

This module creates an HTTP server that exposes the DisasterTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset:    Reset the environment (supports seed + task_id params)
    - POST /step:     Execute an action
    - GET  /state:    Get current environment state
    - GET  /schema:   Get action/observation schemas
    - GET  /tasks:    List all tasks with action schemas
    - POST /grader:   Grade the current completed episode
    - GET  /baseline: Run baseline inference and return scores for all 3 tasks
    - GET  /health:   Health check
    - WS   /ws:       WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openenv.core.env_server import create_web_interface_app
from models import DisasterAction, DisasterObservation
from server.my_environment import DisasterTriageEnvironment


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

env = DisasterTriageEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: run a reset so state is valid on first ping
    env.reset(seed=0, task_id="task_01_single_zone")
    yield


app = create_web_interface_app(env, DisasterAction, DisasterObservation)

# Patch lifespan onto the app (openenv create_web_interface_app may not set one)
app.router.lifespan_context = lifespan

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models for extra endpoints
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: Optional[str] = "task_01_single_zone"


class StepRequest(BaseModel):
    zone_id: int
    resource_type: str
    quantity: int = 1
    priority: str = "urgent"
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Required hackathon endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["OpenEnv"])
async def get_tasks():
    """
    Returns list of tasks with action schema.
    Required by hackathon spec: /tasks endpoint.
    """
    return {"tasks": env.list_tasks()}


@app.post("/grader", tags=["OpenEnv"])
async def grade_episode():
    """
    Grade the current completed episode.
    Required by hackathon spec: /grader endpoint.
    Returns score in [0.0, 1.0] with breakdown.
    """
    result = env.grade()
    return {
        "task_id": result.task_id,
        "score": result.score,
        "max_score": result.max_score,
        "passed": result.passed,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
    }


@app.get("/baseline", tags=["OpenEnv"])
async def run_baseline():
    """
    Run the baseline inference script against all 3 tasks and return scores.
    Required by hackathon spec: /baseline endpoint.

    Calls inference/baseline_agent.py programmatically.
    Returns reproducible scores using fixed seeds.
    """
    try:
        from inference.baseline_agent import run_baseline_all_tasks
        scores = run_baseline_all_tasks()
        return {
            "status": "success",
            "baseline_scores": scores,
            "note": "Scores produced by GPT-4o-mini with seed=42 for all tasks.",
        }
    except Exception as e:
        # Fallback: return cached baseline scores if inference env not configured
        return {
            "status": "cached",
            "baseline_scores": {
                "task_01_single_zone":     {"score": 0.72, "seed": 42},
                "task_02_multi_zone":      {"score": 0.51, "seed": 42},
                "task_03_cascading_hazards": {"score": 0.34, "seed": 42},
            },
            "note": f"Live inference unavailable ({e}). Showing cached scores from README.",
        }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "environment": "disaster_triage_env", "version": "1.0.0"}


@app.get("/schema", tags=["OpenEnv"])
async def get_schema():
    return {
        "action": DisasterAction.model_json_schema(),
        "observation": DisasterObservation.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# WebSocket session (persistent per-client episode)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent sessions.
    Each WS connection gets its own isolated environment instance.
    """
    await websocket.accept()
    session_env = DisasterTriageEnvironment()

    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")

            if command == "reset":
                obs = session_env.reset(
                    seed=data.get("seed"),
                    task_id=data.get("task_id", "task_01_single_zone"),
                )
                await websocket.send_json({
                    "type": "reset",
                    "observation": obs.model_dump(),
                })

            elif command == "step":
                try:
                    action = DisasterAction(**data.get("action", {}))
                    obs, reward, done, info = session_env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif command == "state":
                state = session_env.state()
                await websocket.send_json({
                    "type": "state",
                    "state": state.model_dump(),
                })

            elif command == "grade":
                result = session_env.grade()
                await websocket.send_json({
                    "type": "grade",
                    "result": {
                        "task_id": result.task_id,
                        "score": result.score,
                        "passed": result.passed,
                        "breakdown": result.breakdown,
                        "feedback": result.feedback,
                    },
                })

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown command: {command}"})

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False, workers=1)

if __name__ == "__main__":
    main()
