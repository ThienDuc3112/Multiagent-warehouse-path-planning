from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

Action = int  # 0=WAIT, 1=NORTH, 2=SOUTH, 3=WEST, 4=EAST

# ======================================================================
#                           Instance-like config
# ======================================================================


@dataclass
class FastWarehouseInstance:
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
    def from_dict(cls, d: Dict[str, Any]) -> "FastWarehouseInstance":
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

        R = set(self.robots)
        if not R:
            raise ValueError("robots list must not be empty.")

        def _have_all(name, m):
            missing = R - set(m.keys())
            if missing:
                raise ValueError(f"{name} missing entries for robots: {sorted(missing)}")

        _have_all("start_positions", self.start_positions)
        _have_all("A_targets", self.A_targets)
        _have_all("B_targets", self.B_targets)

        # Bounds & duplicates
        for label, pairs in [
            ("start_positions", self.start_positions),
            ("A_targets", self.A_targets),
            ("B_targets", self.B_targets),
        ]:
            for rid, (x, y) in pairs.items():
                if not (0 <= x < H and 0 <= y < W):
                    raise ValueError(f"{label}[{rid}] out of bounds: {(x,y)} not in [0,{H-1}]x[0,{W-1}]")

        seen = set()
        for (x, y) in self.obstacles:
            if not (0 <= x < H and 0 <= y < W):
                raise ValueError(f"Obstacle {(x,y)} out of bounds.")
            if (x, y) in seen:
                raise ValueError(f"Duplicate obstacle at {(x,y)}.")
            seen.add((x, y))

        for name, mp in [("start_carry", self.start_carry), ("start_delivered", self.start_delivered)]:
            if mp is not None:
                extra = set(mp.keys()) - R
                if extra:
                    raise ValueError(f"{name} has unknown robots: {sorted(extra)}")

    def make_env(self, **env_kwargs) -> "FastWarehousePickPlaceMultiEnv":
        self.validate()
        start_carry = self.start_carry or {rid: False for rid in self.robots}
        start_delivered = self.start_delivered or {rid: False for rid in self.robots}
        return FastWarehousePickPlaceMultiEnv(
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

# ======================================================================
#                              Environment
# ======================================================================


class FastWarehousePickPlaceMultiEnv(gym.Env):
    """
    Multi-robot warehouse pick-&-place environment (plain Gymnasium).

    - Coordinates: x=row in [0..H-1] (north: x-1), y=col in [0..W-1] (west: y-1).
    - Actions: 0=WAIT, 1=NORTH, 2=SOUTH, 3=WEST, 4=EAST
    - Pickup at A if not carrying; drop at B if carrying (marks delivered=True).
    - Collisions blocked: same-cell contention and head-on swaps. Obstacles also block.
    - Reward per robot (summed): step -0.01, progress +0.05 / -0.05, blocked-move -0.02,
      pickup +0.5, delivery +1.0
    - Done when all delivered or horizon reached.

    FAST VIEWS:
      get_global_map_fast(): (C,H,W) int8 with channels {0:obs, 1..N:robots, N+1..2N:A, 2N+1..3N:B}
      get_global_scalars_fast(): (2N+1,) float32 [carry_i,delivered_i..., t/horizon]
      get_local_map_fast(rid): (3,crop,crop) int8 centered at agent (obs/others/agent-target)
      get_local_scalars_fast(rid): (6,) float32 [dx/H,dy/W,carry,delivered,tfrac,id_norm(=0)]
    """

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
        crop: int = 11,                 # default egocentric crop (must be odd)
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

        # Spaces (classic observation for compatibility; fast views are separate helpers)
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

        # ----------- FAST STATE CACHES -----------
        self._rid2i = {rid: i for i, rid in enumerate(self.robots)}
        N, Hh, Ww = len(self.robots), self.H, self.W

        # Obstacles grid (static)
        self._obs_grid = np.zeros((Hh, Ww), dtype=np.int8)
        for (ox, oy) in self.obstacles:
            self._obs_grid[ox, oy] = 1

        # Per-agent target one-hots (static)
        self._A_maps = np.zeros((N, Hh, Ww), dtype=np.int8)
        self._B_maps = np.zeros((N, Hh, Ww), dtype=np.int8)
        for i, rid in enumerate(self.robots):
            ax, ay = self.A[rid]
            bx, by = self.B[rid]
            self._A_maps[i, ax, ay] = 1
            self._B_maps[i, bx, by] = 1

        # Per-agent occupancy one-hots (dynamic; filled at reset/step)
        self._occ_maps = np.zeros((N, Hh, Ww), dtype=np.int8)

        # Global map: 0:obs, 1..N:robots, N+1..2N:A, 2N+1..3N:B
        C = 1 + 3 * N
        self._global_map = np.zeros((C, Hh, Ww), dtype=np.int8)
        self._global_map[0] = self._obs_grid
        self._global_map[1 + N:1 + 2 * N] = self._A_maps
        self._global_map[1 + 2 * N:1 + 3 * N] = self._B_maps

        # Crop + padding for O(1) egocentric slices
        self._crop = int(crop)
        assert self._crop % 2 == 1, "crop must be odd"
        self._pad_r = self._crop // 2
        self._build_pads()

    # --------------------- fast cache helpers ---------------------

    def _build_pads(self):
        """Build/refresh padded arrays for O(1) cropping."""
        r = self._pad_r
        pad = ((r, r), (r, r))
        self._obs_pad = np.pad(self._obs_grid, pad, constant_values=0)  # static
        self._A_pad = np.pad(self._A_maps, ((0, 0),) + pad, constant_values=0)  # static
        self._B_pad = np.pad(self._B_maps, ((0, 0),) + pad, constant_values=0)  # static
        # dynamic padded occ will be filled in reset / step
        self._occ_pad = np.pad(self._occ_maps, ((0, 0),) + pad, constant_values=0)

    def set_crop(self, crop: int):
        """Optionally adjust the egocentric crop size (must be odd)."""
        self._crop = int(crop)
        assert self._crop % 2 == 1, "crop must be odd"
        self._pad_r = self._crop // 2
        self._build_pads()

    # --------------------- Gym API ---------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._t = 0
        self._x = {rid: int(self.start_positions[rid][0]) for rid in self.robots}
        self._y = {rid: int(self.start_positions[rid][1]) for rid in self.robots}
        self._carry = {rid: bool(self.start_carry.get(rid, False)) for rid in self.robots}
        self._delivered = {rid: bool(self.start_delivered.get(rid, False)) for rid in self.robots}

        # fast caches: occ maps & global robot layers
        self._occ_maps[...] = 0
        for rid in self.robots:
            i = self._rid2i[rid]
            x, y = self._x[rid], self._y[rid]
            self._occ_maps[i, x, y] = 1
        N = len(self.robots)
        self._global_map[1:1 + N] = self._occ_maps

        # padded dynamic mirror
        self._build_pads()  # resets occ_pad to zeros
        r = self._pad_r
        self._occ_pad[:, r:r + self.H, r:r + self.W] = self._occ_maps

        return self._obs(), self._info(blocked=None, picked=None, dropped=None)

    def step(self, action: Dict[str, Action]):
        if not isinstance(action, dict):
            action = {rid: int(a) for rid, a in zip(self.robots, list(action))}
        self._t += 1

        # Proposed next cells
        next_x, next_y, attempted = {}, {}, {}
        for rid in self.robots:
            a = int(action.get(rid, 0))
            attempted[rid] = (a != 0)
            cx, cy = self._x[rid], self._y[rid]
            nx, ny = cx, cy
            if a == 1 and cx > 0:
                nx = cx - 1  # NORTH
            elif a == 2 and cx < self.H - 1:
                nx = cx + 1  # SOUTH
            elif a == 3 and cy > 0:
                ny = cy - 1  # WEST
            elif a == 4 and cy < self.W - 1:
                ny = cy + 1  # EAST
            next_x[rid], next_y[rid] = nx, ny

        # Collisions: same-cell and head-on
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
        # Obstacles
        for rid in self.robots:
            if self._obs_grid[next_x[rid], next_y[rid]] == 1:
                blocked[rid] = True

        # Targets based on carry BEFORE move
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

        # Commit state
        # Update occ maps & global robot layers only where moved
        for rid in self.robots:
            i = self._rid2i[rid]
            ox, oy = self._x[rid], self._y[rid]
            nx, ny = new_x[rid], new_y[rid]
            if (ox, oy) != (nx, ny):
                self._occ_maps[i, ox, oy] = 0
                self._occ_maps[i, nx, ny] = 1
                self._global_map[1 + i, ox, oy] = 0
                self._global_map[1 + i, nx, ny] = 1
                # padded mirror
                rpad = self._pad_r
                self._occ_pad[i, ox + rpad, oy + rpad] = 0
                self._occ_pad[i, nx + rpad, ny + rpad] = 1

        self._x, self._y = new_x, new_y
        self._carry, self._delivered = new_carry, new_delivered

        terminated = all(self._delivered[rid] for rid in self.robots)
        truncated = (self._t >= self.horizon) and not terminated

        return self._obs(), float(sum(rewards.values())), bool(terminated), bool(truncated), \
            self._info(blocked=blocked, picked=picked_up, dropped=dropped, rewards=rewards)

    # --------------------- Classic observation & info ---------------------

    def _obs(self):
        # grid is static obstacle map
        grid = self._obs_grid
        robots_feats = {}
        for rid in self.robots:
            ax, ay = self.A[rid]
            bx, by = self.B[rid]
            robots_feats[rid] = np.array([
                self._x.get(rid, self.start_positions[rid][0]),
                self._y.get(rid, self.start_positions[rid][1]),
                int(self._carry.get(rid, self.start_carry.get(rid, False))),
                int(self._delivered.get(rid, self.start_delivered.get(rid, False))),
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

    # --------------------- Fast view getters for MAPPO ---------------------

    def get_global_map_fast(self) -> np.ndarray:
        """(C,H,W) int8 for centralized critic: 0:obs, 1..N:robots, N+1..2N:A, 2N+1..3N:B"""
        return self._global_map

    def get_global_scalars_fast(self) -> np.ndarray:
        """(2N+1,) float32: [carry_i, delivered_i for each robot] + time fraction."""
        feats = []
        for rid in self.robots:
            feats += [1.0 if self._carry[rid] else 0.0, 1.0 if self._delivered[rid] else 0.0]
        feats += [self._t / float(self.horizon)]
        return np.asarray(feats, dtype=np.float32)

    def get_local_map_fast(self, rid: str, crop: Optional[int] = None) -> np.ndarray:
        """
        Egocentric 3-channel crop for one agent (int8):
          0: obstacles
          1: other robots (self excluded)
          2: this agent's current target (A if !carry else B)
        """
        if crop is None:
            crop = self._crop
        assert crop == self._crop, "Use set_crop() first to change the global crop."
        r = self._pad_r

        i = self._rid2i[rid]
        rx, ry = self._x[rid], self._y[rid]
        px, py = rx + r, ry + r

        # obstacles (static)
        ch0 = self._obs_pad[px - r:px + r + 1, py - r:py + r + 1]

        # other robots (dynamic): sum all occ, subtract self, clip to {0,1}
        occ_block = self._occ_pad[:, px - r:px + r + 1, py - r:py + r + 1]
        occ_sum = occ_block.sum(axis=0)
        ch1 = np.minimum(occ_sum - occ_block[i], 1).astype(np.int8)

        # target (static per agent; channel chosen by carry)
        if self._carry[rid]:
            ch2 = self._B_pad[i, px - r:px + r + 1, py - r:py + r + 1]
        else:
            ch2 = self._A_pad[i, px - r:px + r + 1, py - r:py + r + 1]

        return np.stack([ch0, ch1, ch2], axis=0)

    def get_local_scalars_fast(self, rid: str) -> np.ndarray:
        """(6,) float32: [dx/H, dy/W, carry, delivered, t/horizon, id_norm(=0)]"""
        rx, ry = self._x[rid], self._y[rid]
        carry = 1.0 if self._carry[rid] else 0.0
        delivered = 1.0 if self._delivered[rid] else 0.0
        ax, ay = self.A[rid]
        bx, by = self.B[rid]
        tx, ty = (bx, by) if self._carry[rid] else (ax, ay)
        dx = (tx - rx) / float(max(1, self.H - 1))
        dy = (ty - ry) / float(max(1, self.W - 1))
        tfrac = self._t / float(self.horizon)
        return np.asarray([dx, dy, carry, delivered, tfrac, 0.0], dtype=np.float32)

    # --------------------- Rendering (pre-reset safe) ---------------------

    def render(self):
        # Only supports ANSI text rendering.
        if self.render_mode != "ansi":
            return None

        initialized = (
            bool(getattr(self, "_x", None)) and all(rid in self._x for rid in self.robots) and bool(getattr(self, "_y", None)) and all(rid in self._y for rid in self.robots)
        )
        if initialized:
            xs = self._x
            ys = self._y
            carries = self._carry
            t = self._t
        else:
            xs = {rid: int(self.start_positions[rid][0]) for rid in self.robots}
            ys = {rid: int(self.start_positions[rid][1]) for rid in self.robots}
            carries = {rid: bool(self.start_carry.get(rid, False)) for rid in self.robots}
            t = 0

        G = np.full((self.H, self.W), ".", dtype="<U3")
        # obstacles
        G[self._obs_grid == 1] = "###"
        # targets
        for rid in self.robots:
            ax, ay = self.A[rid]
            bx, by = self.B[rid]
            if G[ax, ay] == ".":
                G[ax, ay] = "A" + rid[-1]
            if G[bx, by] == ".":
                G[bx, by] = "B" + rid[-1]
        # robots (overwrite)
        for rid in self.robots:
            rx, ry = xs[rid], ys[rid]
            if 0 <= rx < self.H and 0 <= ry < self.W:
                G[rx, ry] = ("C" if carries[rid] else "R") + rid[-1]

        lines = [f"t={t}  (render {'post-reset' if initialized else 'preview'})"]
        for i in range(self.H):
            lines.append(" ".join(f"{G[i, j]:>3}" for j in range(self.W)))
        return "\n".join(lines)

    def close(self):
        pass


# ======================================================================
#                               Quick demo
# ======================================================================

# if __name__ == "__main__":
#     # Mirrors your RDDL example
#     inst = WarehouseInstance.from_dict({
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
#     })
#     env = inst.make_env(render_mode="ansi", seed=0, crop=11)
#
#     # Preview render (pre-reset safe)
#     print(env.render())
#     obs, info = env.reset()
#     print(env.render())
#
#     # A few random steps
#     for _ in range(5):
#         a = {rid: env.action_space[rid].sample() for rid in env.robots}
#         obs, rew, term, trunc, info = env.step(a)
#         print("\nAction:", {k: env.ACTION_MEANINGS[v] for k, v in a.items()})
#         print(f"Reward={rew:.3f} term={term} trunc={trunc}")
#         print(env.render())
#
#     # Fast views example (centralized + egocentric)
#     gmap = env.get_global_map_fast()           # (C,H,W) int8
#     gsc  = env.get_global_scalars_fast()       # (2N+1,) float32
#     lmap = env.get_local_map_fast("r1")        # (3,crop,crop) int8
#     lsc  = env.get_local_scalars_fast("r1")    # (6,) float32
#     print("\nGlobal map shape:", gmap.shape, "Local map shape:", lmap.shape)
