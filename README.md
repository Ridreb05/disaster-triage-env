---
title: Disaster Triage Environment
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - disaster-response
---

#  Disaster Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://meta-pytorch.org/OpenEnv/)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces/Ridreb05/disaster-triage-env)
[![CI](https://github.com/Ridreb05/disaster-triage-env/actions/workflows/ci.yml/badge.svg)](https://github.com/Ridreb05/disaster-triage-env/actions)

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as an **emergency response coordinator**, allocating limited resources across disaster-struck zones under time pressure and cascading hazard conditions.

---

## Environment Description

The agent manages a grid of incident zones, each with a severity score, population at risk, hazard type (flood, earthquake, hazmat, fire), and accessibility status. Every step, the agent deploys one resource type to one zone. Hazards can cascade to adjacent zones if not contained in time. The reward signal is rich and partial — every step provides feedback, not just end-of-episode.

**Why this task?**  
Disaster triage is a real, high-stakes decision problem that humans perform under extreme uncertainty. It has clear optimal strategies (severity-first, hazard-matched resources, START triage classification) while remaining complex enough that naive agents score poorly. All scenarios are synthetically generated with seeded randomness — no external APIs required.

---

## Action Space

| Field | Type | Values | Description |
|---|---|---|---|
| `zone_id` | int | 0 to N-1 | Target zone index |
| `resource_type` | string | `medical_team`, `rescue_unit`, `supply_drop`, `hazmat_crew` | Resource to deploy |
| `quantity` | int | 1–5 | Units to deploy |
| `priority` | string | `immediate`, `urgent`, `delayed`, `expectant` | START triage classification |
| `reasoning` | string | optional | Chain-of-thought explanation (earns small bonus) |

**Example action:**
```json
{
  "zone_id": 2,
  "resource_type": "medical_team",
  "quantity": 2,
  "priority": "immediate",
  "reasoning": "Zone 2 has highest severity earthquake requiring immediate medical response."
}
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `zones` | array[ZoneState] | Per-zone state: severity, hazard, population, accessibility, resolution status |
| `resources_remaining` | object | Available units per resource type |
| `elapsed_steps` | int | Steps used so far |
| `max_steps` | int | Episode step budget |
| `active_hazards` | array[HazardEvent] | Cascading hazard events with spread probabilities |
| `cumulative_reward` | float | Running reward total |
| `task_id` | string | Active task identifier |
| `scenario_seed` | int | Seed used — enables exact reproduction |
| `done` | bool | Episode termination flag |

---

## Tasks

### Task 01 — Single Zone Triage (Easy)
**Objective:** Identify the highest-severity zone and resolve it within 5 steps.  
**Zones:** 3 | **Max steps:** 5 | **Hazards:** None  
**Grading:** Correct zone (0.3) + correct resource (0.3) + resolved (0.25) + speed bonus (0.15)  
**Baseline score:** `0.72`

### Task 02 — Multi-Zone Priority Routing (Medium)
**Objective:** Triage 5 simultaneous zones with limited resources in 15 steps.  
**Zones:** 5 | **Max steps:** 15 | **Hazards:** Low probability  
**Grading:** Resolution rate (0.50) + priority ordering (0.25) + efficiency (0.15) + speed (0.10)  
**Baseline score:** `0.51`

### Task 03 — Cascading Hazard Response (Hard)
**Objective:** Manage 10 zones with spreading hazards, blocked roads, resource constraints.  
**Zones:** 10 | **Max steps:** 30 | **Hazards:** High probability, cascading  
**Grading:** Resolution (0.35) + containment (0.25) + population saved (0.20) + efficiency (0.10) + speed (0.10)  
**Baseline score:** `0.34`

---

## Reward Function

The reward signal is **partial at every step** — not binary end-of-episode:

| Component | Weight | Description |
|---|---|---|
| Severity coverage | +2.0× severity | Base reward for acting on a zone |
| Population saved | +0.001× pop | Per person-at-risk in the zone |
| Time decay penalty | −0.05× waiting steps | Penalizes leaving high-severity zones unattended |
| Priority alignment | +0.30 | Bonus for correct START triage classification |
| Hazard affinity | +0.20× qty | Correct resource type for hazard |
| Resource waste | −0.20 | Deploying to already-resolved zone |
| Inaccessible zone | −0.40 | Attempting blocked zone |
| Reasoning bonus | +0.05 | Providing chain-of-thought |
| Terminal bonus | up to +2.5 | Completion ratio + speed bonus at episode end |

---

## Setup & Usage

### Local development

```bash
git clone https://github.com/Ridreb05/disaster-triage-env
cd disaster-triage-env
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t disaster-triage-env:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... disaster-triage-env:latest
```

### Run baseline

```bash
export OPENAI_API_KEY=sk-...
python -m inference.baseline_agent
```

### Run tests

```bash
pytest tests/ -v
```

### Validate OpenEnv spec

```bash
openenv validate openenv.yaml
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start new episode. Body: `{"seed": 42, "task_id": "task_01_single_zone"}` |
| POST | `/step` | Execute action. Body: `DisasterAction` JSON |
| GET | `/state` | Current episode metadata |
| GET | `/tasks` | List all tasks with action schemas |
| POST | `/grader` | Grade current episode → score 0.0–1.0 |
| GET | `/baseline` | Run baseline agent on all 3 tasks |
| GET | `/health` | Liveness check |
| GET | `/schema` | Full action + observation JSON schemas |
| WS | `/ws` | WebSocket for persistent sessions |

---

## Python Client

```python
from client import DisasterTriageEnv
from models import DisasterAction

async with DisasterTriageEnv(base_url="http://localhost:8000") as env:
    obs = await env.reset(seed=42, task_id="task_02_multi_zone")

    while not obs.done:
        action = DisasterAction(
            zone_id=obs.critical_zones[0].zone_id,
            resource_type="medical_team",
            quantity=2,
            priority="immediate",
        )
        result = await env.step(action)
        obs = result.observation
        print(f"Reward: {result.reward:.3f} | Done: {result.done}")
```

---

## Baseline Scores

Produced by `gpt-4o-mini` with `seed=42`, `temperature=0`:

| Task | Score | Passed |
|---|---|---|
| task_01_single_zone | 0.7200 | ✓ |
| task_02_multi_zone | 0.5100 | ✓ |
| task_03_cascading_hazards | 0.3400 | ✗ |

---

## File Structure

```
disaster-triage-openenv/
├── server/
│   ├── app.py               # FastAPI — all endpoints
│   └── my_environment.py    # Core env class (step/reset/state)
├── simulation/
│   ├── incident_generator.py # Seeded scenario generation
│   ├── hazard_physics.py     # Time decay, cascade spread
│   └── reward_calculator.py  # Partial reward signal
├── tasks/
│   ├── base_task.py          # Abstract task + GraderResult
│   ├── registry.py           # TASK_REGISTRY
│   ├── task_01_easy.py
│   ├── task_02_medium.py
│   └── task_03_hard.py
├── inference/
│   ├── baseline_agent.py     # OpenAI client baseline script
│   └── prompts/              # System prompt, action format, few-shot
├── tests/
│   ├── test_openenv_compliance.py
│   ├── test_reward.py
│   └── test_tasks.py
├── models.py                 # Pydantic Action/Observation/State
├── client.py                 # Typed EnvClient
├── openenv.yaml              # OpenEnv spec manifest
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## License

BSD-3-Clause — same as OpenEnv framework.
