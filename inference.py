"""
Inference Script — Disaster Triage Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import sys
import json
import textwrap
import logging
from typing import List, Optional, Dict

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from server.my_environment import DisasterTriageEnvironment
from models import DisasterAction
from tasks.registry import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Mandatory env vars (per hackathon spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 256
BASELINE_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert emergency response coordinator managing a disaster triage system.
    Your job is to allocate emergency resources to incident zones to minimize casualties.

    Rules:
    - Respond with ONLY a valid JSON object. No prose, no markdown, no explanation.
    - Always prioritize the highest-severity accessible unresolved zone first.
    - Match resource to hazard: flood→rescue_unit/supply_drop, earthquake→medical_team/rescue_unit,
      hazmat→hazmat_crew, fire→medical_team/hazmat_crew, none→medical_team/rescue_unit.
    - START triage: severity>=0.8→immediate, 0.5-0.8→urgent, 0.2-0.5→delayed, <0.2→expectant.
    - Never deploy to resolved=true or is_accessible=false zones.
""").strip()

ACTION_FORMAT = textwrap.dedent("""
    Respond with exactly this JSON and nothing else:
    {
      "zone_id": <int>,
      "resource_type": <"medical_team"|"rescue_unit"|"supply_drop"|"hazmat_crew">,
      "quantity": <int 1-5>,
      "priority": <"immediate"|"urgent"|"delayed"|"expectant">,
      "reasoning": "<one sentence>"
    }
""").strip()

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(step: int, observation: dict, history: List[str]) -> str:
    zones = observation.get("zones", [])
    resources = observation.get("resources_remaining", {})
    elapsed = observation.get("elapsed_steps", 0)
    max_steps = observation.get("max_steps", MAX_STEPS)
    hazards = observation.get("active_hazards", [])

    zones_text = "\n".join([
        f"  Zone {z['zone_id']}: severity={z['severity']:.2f}, "
        f"pop={z['population_at_risk']}, hazard={z['hazard_type']}, "
        f"accessible={z['is_accessible']}, resolved={z['resolved']}, "
        f"waiting={z['time_since_incident']} steps"
        for z in sorted(zones, key=lambda x: x["severity"], reverse=True)
    ])

    resources_text = ", ".join(
        f"{k}={v}" for k, v in resources.items() if isinstance(v, int)
    )

    hazards_text = (
        "\n".join([
            f"  {h['hazard_type']} spreading from zone {h['origin_zone']} "
            f"to zones {h['affected_zones']}"
            for h in hazards
        ]) if hazards else "  None active."
    )

    history_text = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Step: {step}/{max_steps}
        Elapsed: {elapsed} steps

        ZONES (sorted by severity desc):
        {zones_text}

        RESOURCES REMAINING: {resources_text}

        ACTIVE CASCADING HAZARDS:
        {hazards_text}

        RECENT ACTIONS:
        {history_text}

        Choose the single best action now.
    """).strip()


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_model_action(response_text: str, observation: dict) -> DisasterAction:
    """Parse LLM JSON response into a DisasterAction. Falls back to greedy heuristic."""
    if response_text:
        match = JSON_BLOCK_RE.search(response_text)
        if match:
            try:
                data = json.loads(match.group(0))
                return DisasterAction(**data)
            except Exception:
                pass

    return _greedy_fallback(observation)


def _greedy_fallback(observation: dict) -> DisasterAction:
    """Deterministic fallback when LLM output is unparseable."""
    zones = observation.get("zones", [])
    resources = observation.get("resources_remaining", {})

    candidates = [
        z for z in zones
        if z.get("is_accessible", True) and not z.get("resolved", False)
    ]
    candidates.sort(key=lambda z: z["severity"], reverse=True)

    if not candidates:
        return DisasterAction(
            zone_id=0, resource_type="supply_drop",
            quantity=1, priority="delayed",
            reasoning="Fallback: no actionable zones found."
        )

    target = candidates[0]
    hazard = target.get("hazard_type", "none")

    resource_map = {
        "earthquake": ["medical_team", "rescue_unit"],
        "flood":      ["rescue_unit", "supply_drop"],
        "hazmat":     ["hazmat_crew", "medical_team"],
        "fire":       ["medical_team", "hazmat_crew"],
        "none":       ["medical_team", "rescue_unit"],
    }
    preferred = resource_map.get(hazard, ["medical_team"])
    chosen = next((r for r in preferred if resources.get(r, 0) > 0), "supply_drop")

    severity = target["severity"]
    priority = (
        "immediate" if severity >= 0.8
        else "urgent" if severity >= 0.5
        else "delayed"
    )

    return DisasterAction(
        zone_id=target["zone_id"],
        resource_type=chosen,
        quantity=min(2, max(1, resources.get(chosen, 1))),
        priority=priority,
        reasoning=f"Greedy: zone {target['zone_id']} severity={severity:.2f} hazard={hazard}.",
    )


# ---------------------------------------------------------------------------
# Structured output helpers — required by the OpenEnv validator
# Validator expects these exact token patterns on stdout:
#   [START] task=<name>
#   [STEP]  step=<n> reward=<float>
#   [END]   task=<name> score=<float> steps=<n>
# ---------------------------------------------------------------------------

def emit(line: str) -> None:
    """Print a structured output line to stdout, flushed immediately."""
    print(line, flush=True)


def emit_start(task_id: str) -> None:
    emit(f"[START] task={task_id}")


def emit_step(step: int, reward: float) -> None:
    emit(f"[STEP] step={step} reward={reward:.4f}")


def emit_end(task_id: str, score: float, steps: int) -> None:
    emit(f"[END] task={task_id} score={score:.4f} steps={steps}")


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str, seed: int = BASELINE_SEED) -> dict:
    env = DisasterTriageEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump()

    history: List[str] = []
    total_reward = 0.0
    step = 0

    logger.info(f"  [{task_id}] seed={seed} zones={len(obs.zones)} max_steps={obs.max_steps}")

    # Signal task start to validator
    emit_start(task_id)

    for step in range(1, obs.max_steps + 1):
        if obs_dict.get("done", False):
            break

        user_prompt = build_user_prompt(step, obs_dict, history)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": ACTION_FORMAT}]},
        ]

        response_text = ""
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"  Model request failed ({exc}). Using greedy fallback.")

        action = parse_model_action(response_text, obs_dict)
        logger.info(
            f"  Step {step}: zone={action.zone_id} {action.resource_type}×{action.quantity} "
            f"priority={action.priority}"
        )

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward

        # Emit per-step structured line to stdout
        emit_step(step, reward)

        history.append(
            f"Step {step}: zone={action.zone_id} {action.resource_type}×{action.quantity} "
            f"→ reward {reward:+.3f}"
        )

        if done:
            break

    else:
        logger.info(f"  [{task_id}] Reached max steps ({MAX_STEPS}).")

    grade = env.grade()
    state = env.state()

    # Signal task end with final score to validator
    emit_end(task_id, grade.score, step)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade.score,
        "passed": grade.passed,
        "breakdown": grade.breakdown,
        "feedback": grade.feedback,
        "total_reward": round(total_reward, 4),
        "steps_taken": step,
        "zones_resolved": state.zones_resolved,
        "total_zones": state.total_zones,
    }


# ---------------------------------------------------------------------------
# run_baseline_all_tasks — called by /baseline endpoint
# ---------------------------------------------------------------------------

def run_baseline_all_tasks(seed: int = BASELINE_SEED) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = {}
    for task_id in TASK_REGISTRY:
        try:
            results[task_id] = run_task(client, task_id, seed=seed)
        except Exception as e:
            results[task_id] = {"task_id": task_id, "error": str(e), "score": 0.0, "passed": False}
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        # Still emit structured blocks so the validator can parse something
        for task_id in TASK_REGISTRY:
            emit_start(task_id)
            emit_step(1, 0.0)
            emit_end(task_id, 0.0, 1)
        print("ERROR: HF_TOKEN (or API_KEY) environment variable not set.", flush=True)
        print("  Export it: export HF_TOKEN=hf_...", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    logger.info(f"Baseline agent | model={MODEL_NAME} | seed={BASELINE_SEED} | base_url={API_BASE_URL}")

    results = {}
    for task_id in TASK_REGISTRY:
        logger.info(f"\n[{task_id}] Starting...")
        try:
            result = run_task(client, task_id, seed=BASELINE_SEED)
            results[task_id] = result
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            logger.info(f"[{task_id}] {status} score={result['score']:.4f}")
        except Exception as e:
            logger.error(f"[{task_id}] FAILED: {e}")
            results[task_id] = {"task_id": task_id, "error": str(e), "score": 0.0, "passed": False}
            # Still emit a valid [END] block so the validator doesn't choke
            emit_end(task_id, 0.0, 0)

    print("\n" + "=" * 60, flush=True)
    print("BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    for task_id, r in results.items():
        status = "✓ PASS" if r.get("passed") else "✗ FAIL"
        print(f"{status} | {task_id:<35} | score={r.get('score', 0):.4f}", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
