---
title: Traffic Optimization OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# Traffic Optimization OpenEnv

Traffic Optimization OpenEnv is a seedable reinforcement-learning environment for adaptive control of a four-way intersection. It models queues on the north, south, east, and west approaches, exposes a binary signal-control action, and reports judge-friendly metrics such as throughput, average queue length, switching count, and reward improvement over a fixed-time baseline.

## Submission Snapshot

- Adaptive controller with deterministic threshold-based switching and anti-oscillation memory.
- Strict hackathon logging in `inference.py` for `easy`, `medium`, and `hard`.
- LLM call attempted on every step through `API_BASE_URL` and `API_KEY` with `HF_TOKEN` fallback.
- Safe fallback agent when credentials are missing or model output is invalid.
- Seeded scenarios and repeatable local benchmarks for debugging and demos.

## Benchmark Snapshot

The local benchmark compares the adaptive controller against a fixed-time baseline on the same seed:

| Task | Adaptive Reward | Fixed Baseline | Improvement |
| --- | ---: | ---: | ---: |
| `easy` | `70.8` | `52.4` | `+18.4` |
| `medium` | `41.9` | `-137.7` | `+179.6` |
| `hard` | `-801.45` | `-946.5` | `+145.05` |

These benchmark rewards are for local comparison only. The official hackathon score emitted by `inference.py` follows the required formula `1 / (1 + abs(total_reward))`.

## Why This Version Is Stronger

- Deterministic task seeds make evaluation reproducible.
- Easy, medium, and hard scenarios now have distinct traffic demand profiles.
- The adaptive controller compares against a fixed-time traffic light baseline.
- The controller understands the current signal, so it does not switch away from a busy green phase.
- Inference uses an LLM proxy when credentials are available, but falls back quickly to a reliable adaptive policy locally.
- API responses include step diagnostics for debugging and demo storytelling.

## Environment

### Observation

```json
{
  "north": 7,
  "south": 6,
  "east": 5,
  "west": 8,
  "signal": "NS",
  "phase_age": 0
}
```

### Actions

- `0`: keep the current signal
- `1`: switch to the other signal

### Reward

The reward balances multiple traffic goals:

- Higher throughput is rewarded.
- Queue growth is penalized.
- Long single-lane queues are penalized.
- Excessive switching is penalized.
- Balanced service across both axes is encouraged.

## Tasks

| Task | Description | Max Steps |
| --- | --- | --- |
| `easy` | Light stochastic arrivals | 20 |
| `medium` | Moderate asymmetric arrivals | 24 |
| `hard` | Sustained congestion | 30 |

Each task is scored by comparing the adaptive policy against a fixed-time baseline on the same deterministic seed.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the judged inference script:

```bash
python inference.py
```

The output format is exactly:

```text
[START] task=<task_name>
[STEP] step=<n> reward=<value>
[END] task=<task_name> score=<0-1 float> steps=<n>
```

Run the task benchmark:

```bash
python -m app.test_tasks
```

Run the tests:

```bash
python -m unittest
```

Start the API:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## API

### Health and Discovery

```http
GET /
GET /health
GET /tasks
GET /tasks/{difficulty}
```

### Reset

```http
POST /reset
```

Optional body:

```json
{
  "difficulty": "medium",
  "seed": 23
}
```

### Step

```http
POST /step
```

Body:

```json
{
  "action": 0
}
```

Example response:

```json
{
  "observation": {
    "north": 4,
    "south": 3,
    "east": 7,
    "west": 9,
    "signal": "NS",
    "phase_age": 1
  },
  "reward": -6.7,
  "done": false,
  "info": {
    "step": 1,
    "scenario": "medium",
    "throughput": 6,
    "total_queue": 23,
    "average_queue": 23.0,
    "switches": 0
  }
}
```

## Project Structure

```text
app/
  agent.py        Adaptive rule-based traffic controller
  env.py          Seedable traffic simulation
  models.py       Pydantic observation, action, reward, and info models
  tasks.py        Fixed-time baseline and task scoring
server/
  app.py          FastAPI service
tests/
  test_environment.py
inference.py      OpenEnv inference entrypoint
openenv.yaml      OpenEnv metadata
Dockerfile        Hugging Face Spaces container
```

## OpenEnv Entry Points

```yaml
environment:
  entrypoint: app.env:TrafficEnv

inference:
  entrypoint: inference:main
```

## Future Extensions

- Multi-intersection grid coordination.
- Emergency vehicle priority.
- Weather or event-driven arrival profiles.
- Learned RL agents trained against the seedable scenarios.
