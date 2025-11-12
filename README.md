# Warehouse Multi‑Agent RL

A lightweight, reproducible research repo for learning pickup‑and‑dropoff policies in a grid‑world warehouse with multiple robots. The environment is Gymnasium‑compatible and supports **parameter‑shared actors** with a **centralized critic**, curriculum/difficulty sampling, color GIF rendering with overlays, and solvability‑aware instance generation.

---

## Table of Contents

* [Features](#features)
* [Environment Overview](#environment-overview)
* [Install](#install)
* [Quickstart](#quickstart)
* [Training (PPO: shared actor + central critic)](#training-ppo-shared-actor--central-critic)
* [Config & Instances](#config--instances)
* [Observation & Action Spaces](#observation--action-spaces)
* [Rewards](#rewards)
* [Logging & Visualization](#logging--visualization)
* [Roadmap](#roadmap)

---

## Features

* **Gymnasium env** for a multi‑robot warehouse pick‑and‑place task.
* **Centralized Critic, Shared Actor**: a single policy network for all agents; critic can see global context.
* **Fixed instances** via YAML/dataclass (no procedural generation or randomization yet).
* **Color GIF renderer** with optional overlays (actions, rewards, dones, info).
* **Checkpointing** and resume training.
* **Deterministic seeding** for reproducible runs.

---

## Environment Overview

Each episode places multiple robots on a 2D grid with obstacles. Every robot must:

1. Navigate to its pickup cell **A**; 2) toggle carrying; 3) reach its drop‑off cell **B**.

Collisions, invalid moves, and dithering are discouraged by the reward. Goals are per‑robot; obstacles and other robots require coordination.

---

## Install

We use **Poetry** for dependency management and **Marimo** for a notebook‑style UI.

```bash
# 1) Install dependencies
poetry install

# 2) Launch the Marimo app (editable notebook)
poetry run marimo edit notebooks/warehouse_app.py

# (Optional) Run CLI scripts via Poetry
poetry run python scripts/train.py --help
```

> Tip: If you prefer plain Python, you can still `poetry run python ...` any script without opening Marimo.

---

## Quickstart

1. Start the app:

   ```bash
   poetry run marimo edit main.py
   ```
2. In the notebook, edit the **Env JSON** cell (see example below) to change map size, robots, A/B targets, and obstacles.
3. Run the training cell. Live logs and inline renders will appear inside the notebook.
4. Use the render cell to export a **color GIF** of a rollout.

Example training logs:

```
[rollout] reward mean/std: -0.0792/0.1174 | V mean/std: -0.0155/0.0092
[0001] loss=0.251 pi=-0.000 v=0.534 H=1.545 adv=0.000 adv_std=1.000 ret=-0.979
```

---

## Training (PPO: shared actor + central critic)

**Policy**: Parameter‑shared actor (\pi_\theta) per agent; inputs are per‑agent local crops and features.
**Critic**: Centralized value network (V_\phi) that can aggregate multi‑agent observations (e.g., stacked crops or a compact global encoding).

Key PPO settings (defaults):

* GAE((\gamma), (\lambda)) w/ advantage normalization
* Clipping (\epsilon) (policy/value)

---

## Config & Instances

The environment currently uses **fixed, hand‑authored instances** (no procedural generation or randomization yet). You can define instances via YAML files or **directly in Marimo as JSON**.

```python
@dataclass
class FastWarehouseInstance:
    height: int
    width: int
    horizon: int
    robots: List[str]
    start_positions: Dict[str, Tuple[int, int]]
    A_targets: Dict[str, Tuple[int, int]]  # pickup
    B_targets: Dict[str, Tuple[int, int]]  # dropoff
    obstacles: List[Tuple[int, int]]
    start_carry: Optional[Dict[str, bool]] = None
    start_delivered: Optional[Dict[str, bool]] = None
```

### Editing the env **in Marimo** (JSON cell)

**Example JSON** you can paste into the Marimo `create_fast_env` cell:

```json
{
  "height": 21,
  "width": 21,
  "horizon": 200,
  "robots": ["r1", "r2", "r3"],
  "start_positions": {"r1": [1, 1], "r2": [1, 19], "r3": [10, 1]},
  "A_targets": {"r1": [2, 2], "r2": [2, 18], "r3": [10, 2]},
  "B_targets": {"r1": [18, 18], "r2": [18, 2], "r3": [10, 18]},
  "obstacles": [[5, 5], [5, 6], [6, 5]]
}
```

---

## Observation & Action Spaces

**Actions** (Discrete 5):

```
0=WAIT, 1=NORTH, 2=SOUTH, 3=WEST, 4=EAST
```

**Per‑agent observation** (default):

* **9x9 crop** centered on the robot (clipped at borders). Channels may include: walls/obstacles, pickups A, dropoffs B, other robots, and a carry flag layer.
* **Global compact features** concatenated to the actor input, e.g. **goal vector** `(Tx, Ty, Tdist, phase)`.

The **central critic** can see concatenated per‑agent tensors or a compact global map embedding.

---

## Rewards

Current coefficients:

* Step penalty: `-0.01`
* Pickup success: `+1.0`
* **Dropoff success: `+2.0`**
* **Collision / invalid move: `-0.1`**
* **Manhattan shaping:** `+0.02` if the Manhattan distance to the current subgoal (A when not carrying, B when carrying) **decreases** vs. the previous step, `-0.02` if it **increases**, and `0` if unchanged.

> Note: the shaping above is potential‑based in spirit and designed not to overpower the sparse goals. Tune `0.02` if you observe dithering or hallway hugging.

All coefficients are configurable in the notebook and scripts.

---

## Generalization (planned)

Randomized instance sampling and curriculum scheduling are **not yet implemented**. Current experiments rely on training across several fixed layouts (multiple YAML configs) and evaluating on held‑out fixed layouts.

Planned: domain randomization (starts/targets/obstacles), solvability filter, and a curriculum that increases map size & obstacle density over time.

---

## Logging & Visualization

* Scalar logs: rollout reward mean/std, value mean/std, policy/val losses, entropy, returns.
* **GIF rendering** (color): configurable palette; overlays for `action`, `reward`, `done`, and step counters.

Rendering script supports reading a JSON episode export and creating a GIF from JSON episode.

---

## Roadmap

* **Procedural instance sampler** with **solvability filter** (BFS/shortest‑path).
* **Domain randomization** and **curriculum scheduler** to improve generalization.
