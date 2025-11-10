from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

Action = int  # 0=WAIT, 1=N, 2=S, 3=W, 4=E

# --------------------- Instance-like config ---------------------


@dataclass
class WarehouseInstance:
    """Instance-like config for the warehouse env (RDDL 'instance' analog).

    Fields:
      height, width: grid size
      horizon: episode step limit
      robots: list of robot IDs (e.g., ["r1","r2","r3"])
      start_positions: {robot: (x, y)}
      A_targets:       {robot: (xA, yA)}  # pickup
      B_targets:       {robot: (xB, yB)}  # dropoff
      obstacles:       [(x,y), ...]
      start_carry:     optional {robot: bool}
      start_delivered: optional {robot: bool}
    """
    height: int
    width: int
    horizon: int = 400
    robots: List[str] = field(default_factory=list)
    start_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    A_targets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    B_targets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    start_carry: Optional[Dict[str, bool]] = None
    start_delivered: Optional[Dict[str, bool]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WarehouseInstance":
        # Robots: if omitted, infer from keys of start_positions/A_targets/B_targets
        robots = d.get("robots")
        if robots is None:
            keys = set()
            for k in ("start_positions", "A_targets", "B_targets"):
                if k in d and isinstance(d[k], dict):
                    keys |= set(d[k].keys())
            robots = sorted(keys)

        return cls(
            height=int(d["height"]),
            width=int(d["width"]),
            horizon=int(d.get("horizon", 400)),
            robots=list(robots),
            start_positions={k: tuple(v) for k, v in d["start_positions"].items()},
            A_targets={k: tuple(v) for k, v in d["A_targets"].items()},
            B_targets={k: tuple(v) for k, v in d["B_targets"].items()},
            obstacles=[tuple(t) for t in d.get("obstacles", [])],
            start_carry={k: bool(v) for k, v in d.get("start_carry", {}).items()} or None,
            start_delivered={k: bool(v) for k, v in d.get("start_delivered", {}).items()} or None,
        )

    def validate(self) -> None:
        H, W = self.height, self.width
        if not (H > 0 and W > 0):
            raise ValueError("height and width must be positive.")

        # Robots set
        R = set(self.robots)
        if not R:
            raise ValueError("robots list must not be empty.")

        # Required maps contain all robots
        def _have_all(name, m):
            missing = R - set(m.keys())
            if missing:
                raise ValueError(f"{name} missing entries for robots: {sorted(missing)}")

        _have_all("start_positions", self.start_positions)
        _have_all("A_targets", self.A_targets)
        _have_all("B_targets", self.B_targets)

        # Bounds and duplicates
        seen = set()
        for label, pairs in [
            ("start_positions", self.start_positions),
            ("A_targets", self.A_targets),
            ("B_targets", self.B_targets),
        ]:
            for rid, (x, y) in pairs.items():
                if not (0 <= x < H and 0 <= y < W):
                    raise ValueError(f"{label}[{rid}] out of bounds: {(x,y)} not in [0,{H-1}]x[0,{W-1}]")
                # (It’s okay if positions/targets overlap each other; env handles collisions at runtime.)

        for (x, y) in self.obstacles:
            if not (0 <= x < H and 0 <= y < W):
                raise ValueError(f"Obstacle {(x,y)} out of bounds.")
            if (x, y) in seen:
                raise ValueError(f"Duplicate obstacle at {(x,y)}.")
            seen.add((x, y))

        # Optional bool maps default to False if missing
        for name, mp in [("start_carry", self.start_carry), ("start_delivered", self.start_delivered)]:
            if mp is not None:
                extra = set(mp.keys()) - R
                if extra:
                    raise ValueError(f"{name} has unknown robots: {sorted(extra)}")

    def make_env(self, **env_kwargs) -> "WarehousePickPlaceMultiEnv":
        self.validate()
        # Fill missing optional maps with False
        start_carry = self.start_carry or {rid: False for rid in self.robots}
        start_delivered = self.start_delivered or {rid: False for rid in self.robots}
        return WarehousePickPlaceMultiEnv(
            H=self.height,
            W=self.width,
            horizon=self.horizon,
            robots=self.robots,
            A_targets=self.A_targets,
            B_targets=self.B_targets,
            start_positions=self.start_positions,
            start_carry=start_carry,
            start_delivered=start_delivered,
            obstacles=list(self.obstacles),
            **env_kwargs,
        )

# --------------------- The environment ---------------------


class WarehousePickPlaceMultiEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    ACTION_MEANINGS = ["WAIT", "NORTH", "SOUTH", "WEST", "EAST"]

    def __init__(
        self,
        H: int,
        W: int,
        horizon: int = 400,
        robots: Optional[List[str]] = None,
        A_targets: Optional[Dict[str, Tuple[int, int]]] = None,
        B_targets: Optional[Dict[str, Tuple[int, int]]] = None,
        start_positions: Optional[Dict[str, Tuple[int, int]]] = None,
        start_carry: Optional[Dict[str, bool]] = None,
        start_delivered: Optional[Dict[str, bool]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.horizon = int(horizon)
        self.render_mode = render_mode

        self.robots = list(robots or [])
        self.A = dict(A_targets or {})
        self.B = dict(B_targets or {})
        self.start_positions = dict(start_positions or {})
        self.start_carry = dict(start_carry or {rid: False for rid in self.robots})
        self.start_delivered = dict(start_delivered or {rid: False for rid in self.robots})
        self.obstacles = set(obstacles or [])

        # State
        self._x: Dict[str, int] = {}
        self._y: Dict[str, int] = {}
        self._carry: Dict[str, bool] = {}
        self._delivered: Dict[str, bool] = {}
        self._t = 0

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Spaces
        self.action_space = spaces.Dict({rid: spaces.Discrete(5, start=0) for rid in self.robots})

        robot_low = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        robot_high = np.array([self.H - 1, self.W - 1, 1, 1, self.H - 1, self.W - 1, self.H - 1, self.W - 1], dtype=np.int32)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.H, self.W), dtype=np.int8),
            "robots": spaces.Dict({
                rid: spaces.Box(low=robot_low, high=robot_high, dtype=np.int32) for rid in self.robots
            }),
            "step": spaces.Box(low=0, high=self.horizon, shape=(1,), dtype=np.int32),
        })

    @classmethod
    def from_instance(cls, inst: WarehouseInstance, **env_kwargs) -> "WarehousePickPlaceMultiEnv":
        return inst.make_env(**env_kwargs)

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._t = 0
        self._x = {rid: int(self.start_positions[rid][0]) for rid in self.robots}
        self._y = {rid: int(self.start_positions[rid][1]) for rid in self.robots}
        self._carry = {rid: bool(self.start_carry.get(rid, False)) for rid in self.robots}
        self._delivered = {rid: bool(self.start_delivered.get(rid, False)) for rid in self.robots}
        return self._obs(), self._info(blocked=None, picked=None, dropped=None)

    def step(self, action: Dict[str, Action]):
        if not isinstance(action, dict):
            action = {rid: int(a) for rid, a in zip(self.robots, list(action))}
        self._t += 1

        # Proposals
        next_x, next_y, attempted = {}, {}, {}
        for rid in self.robots:
            a = int(action.get(rid, 0))
            attempted[rid] = (a != 0)
            cx, cy = self._x[rid], self._y[rid]
            nx, ny = cx, cy
            if a == 1 and cx > 0:
                nx = cx - 1   # N
            elif a == 2 and cx < self.H - 1:
                nx = cx + 1   # S
            elif a == 3 and cy > 0:
                ny = cy - 1   # W
            elif a == 4 and cy < self.W - 1:
                ny = cy + 1   # E
            next_x[rid], next_y[rid] = nx, ny

        # Collisions
        blocked = {rid: False for rid in self.robots}
        counts = {}
        for rid in self.robots:
            key = (next_x[rid], next_y[rid])
            counts[key] = counts.get(key, 0) + 1
        for rid in self.robots:
            if counts[(next_x[rid], next_y[rid])] > 1:
                blocked[rid] = True
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                ri, rj = self.robots[i], self.robots[j]
                if (next_x[ri], next_y[ri]) == (self._x[rj], self._y[rj]) and \
                   (next_x[rj], next_y[rj]) == (self._x[ri], self._y[ri]):
                    blocked[ri] = True
                    blocked[rj] = True
        for rid in self.robots:
            if (next_x[rid], next_y[rid]) in self.obstacles:
                blocked[rid] = True

        # Target (based on carry BEFORE move)
        xtgt, ytgt = {}, {}
        for rid in self.robots:
            if self._carry[rid]:
                xtgt[rid], ytgt[rid] = self.B[rid]
            else:
                xtgt[rid], ytgt[rid] = self.A[rid]

        dist_before = {rid: abs(self._x[rid] - xtgt[rid]) + abs(self._y[rid] - ytgt[rid]) for rid in self.robots}

        # Apply movement
        new_x, new_y, moved = {}, {}, {}
        for rid in self.robots:
            if blocked[rid]:
                new_x[rid], new_y[rid] = self._x[rid], self._y[rid]
            else:
                new_x[rid], new_y[rid] = next_x[rid], next_y[rid]
            moved[rid] = (new_x[rid] != self._x[rid]) or (new_y[rid] != self._y[rid])

        # Pickup / Drop
        picked_up = {rid: False for rid in self.robots}
        dropped = {rid: False for rid in self.robots}
        new_carry = dict(self._carry)
        new_delivered = dict(self._delivered)
        for rid in self.robots:
            if self._carry[rid] and (new_x[rid], new_y[rid]) == self.B[rid]:
                new_carry[rid] = False
                new_delivered[rid] = True
                dropped[rid] = True
            elif (not self._carry[rid]) and (new_x[rid], new_y[rid]) == self.A[rid]:
                new_carry[rid] = True
                picked_up[rid] = True

        dist_after = {rid: abs(new_x[rid] - xtgt[rid]) + abs(new_y[rid] - ytgt[rid]) for rid in self.robots}

        # Rewards
        rewards = {rid: -0.01 for rid in self.robots}
        for rid in self.robots:
            if dist_after[rid] < dist_before[rid]:
                rewards[rid] += 0.05
            elif dist_after[rid] > dist_before[rid]:
                rewards[rid] -= 0.05
            if attempted[rid] and not moved[rid]:
                rewards[rid] -= 0.02
            if picked_up[rid]:
                rewards[rid] += 0.5
            if dropped[rid]:
                rewards[rid] += 1.0

        # Commit
        self._x, self._y = new_x, new_y
        self._carry, self._delivered = new_carry, new_delivered

        terminated = all(self._delivered[rid] for rid in self.robots)
        truncated = (self._t >= self.horizon) and not terminated

        return self._obs(), float(sum(rewards.values())), bool(terminated), bool(truncated), \
            self._info(blocked=blocked, picked=picked_up, dropped=dropped, rewards=rewards)

    # ---------- Helpers ----------

    def _obs(self):
        grid = np.zeros((self.H, self.W), dtype=np.int8)
        for (x, y) in self.obstacles:
            if 0 <= x < self.H and 0 <= y < self.W:
                grid[x, y] = 1
        robots_feats = {}
        for rid in self.robots:
            ax, ay = self.A[rid]
            bx, by = self.B[rid]
            robots_feats[rid] = np.array([
                self._x[rid], self._y[rid],
                int(self._carry[rid]), int(self._delivered[rid]),
                ax, ay, bx, by,
            ], dtype=np.int32)
        return {"grid": grid, "robots": robots_feats, "step": np.array([self._t], dtype=np.int32)}

    def _info(self, *, blocked=None, picked=None, dropped=None, rewards=None):
        return {
            "blocked": blocked,
            "picked_up": picked,
            "dropped": dropped,
            "per_robot_reward": rewards,
            "action_meanings": self.ACTION_MEANINGS,
        }

    def render(self):
        # Only supports ANSI text rendering.
        if self.render_mode != "ansi":
            return None

        # If the env hasn't been reset yet, fall back to instance config (preview mode).
        initialized = (
            bool(getattr(self, "_x", None)) and all(rid in self._x for rid in self.robots) and bool(getattr(self, "_y", None)) and all(rid in self._y for rid in self.robots)
        )

        if initialized:
            xs = self._x
            ys = self._y
            carries = self._carry
            t = self._t
        else:
            # Preview based on start_* config
            xs = {rid: int(self.start_positions[rid][0]) for rid in self.robots}
            ys = {rid: int(self.start_positions[rid][1]) for rid in self.robots}
            carries = {rid: bool(self.start_carry.get(rid, False)) for rid in self.robots}
            t = 0

        # Build a char grid
        G = np.full((self.H, self.W), ".", dtype="<U3")

        # Obstacles
        for (x, y) in self.obstacles:
            if 0 <= x < self.H and 0 <= y < self.W:
                G[x, y] = "###"

        # Targets (don't overwrite obstacles)
        for rid in self.robots:
            ax, ay = self.A[rid]
            bx, by = self.B[rid]
            if 0 <= ax < self.H and 0 <= ay < self.W and G[ax, ay] == ".":
                G[ax, ay] = "A" + rid[-1]
            if 0 <= bx < self.H and 0 <= by < self.W and G[bx, by] == ".":
                G[bx, by] = "B" + rid[-1]

        # Robots (overwrite whatever is underneath for visibility)
        for rid in self.robots:
            rx, ry = xs[rid], ys[rid]
            if 0 <= rx < self.H and 0 <= ry < self.W:
                token = ("C" if carries[rid] else "R") + rid[-1]  # C=carrying, R=not
                G[rx, ry] = token

        # Compose text
        lines = [f"t={t}  (render {'post-reset' if initialized else 'preview'})"]
        for i in range(self.H):
            lines.append(" ".join(f"{G[i, j]:>3}" for j in range(self.W)))
        return "\n".join(lines)

    def close(self):  # noqa: D401
        pass

# --------------------- Quick demo ---------------------
#
#
# if __name__ == "__main__":
#     # Build an instance that mirrors your RDDL init-state
#     inst_dict = {
#         "height": 21, "width": 21, "horizon": 400,
#         "robots": ["r1", "r2", "r3"],
#         "start_positions": {"r1": (1, 1), "r2": (1, 19), "r3": (10, 1)},
#         "A_targets": {"r1": (2, 2), "r2": (2, 18), "r3": (10, 2)},
#         "B_targets": {"r1": (18, 18), "r2": (18, 2), "r3": (10, 18)},
#         "obstacles": [
#             (3, 3), (4, 3), (5, 3),
#             (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
#             (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
#         ],
#         # Optional:
#         # "start_carry": {"r1": False, "r2": False, "r3": False},
#         # "start_delivered": {"r1": False, "r2": False, "r3": False},
#     }
#     inst = WarehouseInstance.from_dict(inst_dict)
#     env = inst.make_env(render_mode="ansi", seed=0)
#
#     obs, info = env.reset()
#     print(env.render())
#
#     for _ in range(5):
#         a = {rid: env.action_space[rid].sample() for rid in env.robots}
#         obs, rew, term, trunc, info = env.step(a)
#         print("\nAction:", {k: env.ACTION_MEANINGS[v] for k, v in a.items()})
#         print(f"Reward={rew:.3f} term={term} trunc={trunc}")
#         print(env.render())
#         if term or trunc:
#             break
