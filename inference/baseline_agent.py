# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Baseline inference script — satisfies the hackathon requirement:
  "Uses the OpenAI API client to run a model against the environment.
   Reads API credentials from environment variables (OPENAI_API_KEY).
   Produces a reproducible baseline score on all 3 tasks."

Usage:
    export OPENAI_API_KEY=sk-...
    python -m inference.baseline_agent

    # Or programmatically (called by /baseline endpoint):
    from inference.baseline_agent import run_baseline_all_tasks
    scores = run_baseline_all_tasks()
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from server.my_environment import DisasterTriageEnvironment
from models import DisasterAction
from tasks.registry import TASK_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
BASELINE_SEED = 42          # Fixed seed — ensures "baseline reproduces"
MAX_RETRIES = 3             # Retry on malformed JSON from LLM
TEMPERATURE = 0.0           # Deterministic completions


# ---------------------------------------------------------------------------
# Prompt templates (loaded from prompts/ files if present, else inline)
# ---------------------------------------------------------------------------

def _load_prompt(filename: str, fallback: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return fallback


SYSTEM_PROMPT = _load_prompt("system.txt", """
You are an expert emergency response coordinator operating a disaster triage system.
Your job is to allocate emergency resources to incident zones to minimize casualties.

Rules:
- You must respond with ONLY a valid JSON object — no prose, no markdown, no explanation.
- Always choose the highest-severity, accessible, unresolved zone first.
- Match resource type to hazard type: floods → rescue_unit/supply_drop,
  earthquakes → medical_team/rescue_unit, hazmat → hazmat_crew, fires → medical_team/hazmat_crew.
- Use START triage: severity ≥ 0.8 → immediate, 0.5–0.8 → urgent, 0.2–0.5 → delayed, <0.2 → expectant.
- Never deploy to resolved or inaccessible zones.
""".strip())

ACTION_FORMAT_PROMPT = _load_prompt("action_format.txt", """
Respond with exactly this JSON structure:
{
  "zone_id": <int>,
  "resource_type": <"medical_team"|"rescue_unit"|"supply_drop"|"hazmat_crew">,
  "quantity": <int 1-5>,
  "priority": <"immediate"|"urgent"|"delayed"|"expectant">,
  "reasoning": "<one sentence explanation>"
}
""".strip())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """
    Rule-augmented LLM agent. Uses GPT-4o-mini with a structured prompt
    to make triage decisions. Falls back to a greedy heuristic if the
    LLM returns malformed output.
    """

    def __init__(self, model: str = MODEL):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set. "
                "Export it before running the baseline: export OPENAI_API_KEY=sk-..."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def choose_action(self, observation: dict) -> DisasterAction:
        """
        Given a serialized observation dict, return the next action.
        Tries the LLM first, falls back to greedy heuristic on failure.
        """
        prompt = self._build_prompt(observation)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=f"{prompt}\n\n{ACTION_FORMAT_PROMPT}",
                    temperature=TEMPERATURE,
                    max_output_tokens=256,
                )
                raw = (response.output_text or "").strip()
                action_dict = json.loads(raw)
                return DisasterAction(**action_dict)

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}. Falling back to heuristic.")
                time.sleep(0.5)

        return self._greedy_fallback(observation)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, obs: dict) -> str:
        zones = obs.get("zones", [])
        resources = obs.get("resources_remaining", {})
        elapsed = obs.get("elapsed_steps", 0)
        max_steps = obs.get("max_steps", 20)
        hazards = obs.get("active_hazards", [])

        zones_summary = "\n".join([
            f"  Zone {z['zone_id']}: severity={z['severity']:.2f}, "
            f"pop={z['population_at_risk']}, hazard={z['hazard_type']}, "
            f"accessible={z['is_accessible']}, resolved={z['resolved']}, "
            f"waiting={z['time_since_incident']} steps"
            for z in sorted(zones, key=lambda x: x["severity"], reverse=True)
        ])

        resources_summary = ", ".join(
            f"{k}={v}" for k, v in resources.items() if isinstance(v, int)
        )

        hazards_summary = (
            "\n".join([
                f"  {h['hazard_type']} spreading from zone {h['origin_zone']} "
                f"to zones {h['affected_zones']}"
                for h in hazards
            ]) if hazards else "  None active."
        )

        return f"""
Current disaster situation (Step {elapsed}/{max_steps}):

ZONES (sorted by severity):
{zones_summary}

RESOURCES REMAINING: {resources_summary}

ACTIVE CASCADING HAZARDS:
{hazards_summary}

Choose the single best action for this step.
""".strip()

    # ------------------------------------------------------------------
    # Greedy fallback heuristic
    # ------------------------------------------------------------------

    def _greedy_fallback(self, obs: dict) -> DisasterAction:
        zones = obs.get("zones", [])
        resources = obs.get("resources_remaining", {})

        # Sort: accessible + unresolved, highest severity first
        candidates = [
            z for z in zones
            if z.get("is_accessible", True) and not z.get("resolved", False)
        ]
        candidates.sort(key=lambda z: z["severity"], reverse=True)

        if not candidates:
            # All resolved or inaccessible — no-op on zone 0
            return DisasterAction(
                zone_id=0,
                resource_type="supply_drop",
                quantity=1,
                priority="delayed",
                reasoning="Fallback: all zones resolved.",
            )

        target = candidates[0]
        hazard = target.get("hazard_type", "none")

        resource_map = {
            "earthquake": ("medical_team", "rescue_unit"),
            "flood":      ("rescue_unit", "supply_drop"),
            "hazmat":     ("hazmat_crew", "medical_team"),
            "fire":       ("medical_team", "hazmat_crew"),
            "none":       ("medical_team", "rescue_unit"),
        }
        preferred = resource_map.get(hazard, ("medical_team",))

        # Pick first resource type we actually have
        chosen_resource = "supply_drop"
        for rt in preferred:
            if resources.get(rt, 0) > 0:
                chosen_resource = rt
                break

        severity = target["severity"]
        priority = (
            "immediate" if severity >= 0.8
            else "urgent" if severity >= 0.5
            else "delayed"
        )

        return DisasterAction(
            zone_id=target["zone_id"],
            resource_type=chosen_resource,
            quantity=min(2, resources.get(chosen_resource, 1)),
            priority=priority,
            reasoning=f"Greedy fallback: targeting zone {target['zone_id']} (severity {severity:.2f}).",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single_task(
    agent: BaselineAgent,
    task_id: str,
    seed: int = BASELINE_SEED,
) -> dict:
    """Run one full episode and return graded results."""
    env = DisasterTriageEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    logger.info(f"  Task: {task_id} | Seed: {seed} | Zones: {len(obs.zones)} | Max steps: {obs.max_steps}")

    step = 0
    total_reward = 0.0

    while not obs.done:
        action = agent.choose_action(obs.model_dump())
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        logger.debug(f"    Step {step}: zone={action.zone_id} {action.resource_type}×{action.quantity} → reward={reward:.3f}")

    grade_result = env.grade()
    state = env.state()

    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade_result.score,
        "passed": grade_result.passed,
        "breakdown": grade_result.breakdown,
        "feedback": grade_result.feedback,
        "total_reward": round(total_reward, 4),
        "steps_taken": step,
        "zones_resolved": state.zones_resolved,
        "total_zones": state.total_zones,
    }


def run_baseline_all_tasks(seed: int = BASELINE_SEED) -> dict:
    """
    Run baseline agent against all registered tasks.
    Called by /baseline endpoint and directly from CLI.
    """
    agent = BaselineAgent()
    results = {}

    logger.info(f"Running baseline agent ({MODEL}) on all {len(TASK_REGISTRY)} tasks with seed={seed}")

    for task_id in TASK_REGISTRY:
        logger.info(f"[{task_id}] Starting...")
        try:
            result = run_single_task(agent, task_id, seed=seed)
            results[task_id] = result
            logger.info(f"[{task_id}] Score: {result['score']:.4f} | Passed: {result['passed']}")
        except Exception as e:
            logger.error(f"[{task_id}] FAILED: {e}")
            results[task_id] = {"task_id": task_id, "error": str(e), "score": 0.0}

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DisasterTriage baseline agent")
    parser.add_argument("--seed", type=int, default=BASELINE_SEED)
    parser.add_argument("--task", type=str, default=None, help="Run a single task ID")
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    agent = BaselineAgent(model=args.model)

    if args.task:
        result = run_single_task(agent, args.task, seed=args.seed)
        print(json.dumps(result, indent=2))
    else:
        results = run_baseline_all_tasks(seed=args.seed)
        print("\n" + "="*60)
        print("BASELINE SCORES")
        print("="*60)
        for task_id, r in results.items():
            status = "✓ PASS" if r.get("passed") else "✗ FAIL"
            print(f"{status} | {task_id:<35} | score={r.get('score', 0):.4f}")
        print("="*60)
        print(json.dumps(results, indent=2))
