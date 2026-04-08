# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
DisasterTriageEnv client — typed HTTP/WebSocket client.

Usage:
    from disaster_triage_env import DisasterTriageEnv, DisasterAction

    async with DisasterTriageEnv(base_url="http://localhost:8000") as client:
        obs = await client.reset(seed=42, task_id="task_02_multi_zone")
        result = await client.step(DisasterAction(
            zone_id=0, resource_type="medical_team", quantity=2, priority="immediate"
        ))
        print(result.reward)

    # Sync wrapper
    with DisasterTriageEnv(base_url="http://localhost:8000").sync() as client:
        obs = client.reset(seed=42)
        result = client.step(DisasterAction(zone_id=0, resource_type="rescue_unit"))
"""

from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.types import StepResult
from models import DisasterAction, DisasterObservation, DisasterState


class DisasterTriageEnv(EnvClient[DisasterAction, DisasterObservation]):
    """
    Typed client for the Disaster Triage Environment.
    Extends EnvClient — handles HTTP + WebSocket communication.
    """

    async def reset(
        self,
        seed: int | None = None,
        task_id: str = "task_01_single_zone",
    ) -> DisasterObservation:
        payload: dict = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        return await self._reset(payload)

    def _step_payload(self, action: DisasterAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[DisasterObservation]:
        obs = DisasterObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict) -> DisasterState:
        return DisasterState(**payload)
