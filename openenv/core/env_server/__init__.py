from __future__ import annotations

from typing import Any, Type

from fastapi import FastAPI


def create_web_interface_app(env: Any, action_model: Type[Any], observation_model: Type[Any]) -> FastAPI:
    """
    Minimal stand-in for OpenEnv's server factory.

    This is intentionally lightweight; the project's `server/app.py` adds extra
    endpoints on top.
    """
    app = FastAPI()

    @app.post("/reset")
    async def _reset(payload: dict | None = None):
        payload = payload or {}
        obs = env.reset(seed=payload.get("seed"), task_id=payload.get("task_id"))
        return obs.model_dump() if hasattr(obs, "model_dump") else obs

    @app.post("/step")
    async def _step(payload: dict):
        action = action_model(**payload)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

    @app.get("/state")
    async def _state():
        st = env.state()
        return st.model_dump() if hasattr(st, "model_dump") else st

    return app

