import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing dependencies
    """)
    return


@app.cell
def _():
    import re, math, random, pprint, time
    from typing import Dict, List, Tuple, Any

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    from pyRDDLGym.core.env import RDDLEnv

    from gymnasium.wrappers import RecordEpisodeStatistics

    from utils import DictToListWrapper

    from env import WarehouseInstance, WarehousePickPlaceMultiEnv


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    return (
        Any,
        Dict,
        RecordEpisodeStatistics,
        Tuple,
        WarehouseInstance,
        nn,
        np,
        re,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utilities
    """)
    return


@app.cell
def _(nn, np, re):
    _key_re = re.compile(r"([A-Za-z0-9_]+)\(([^)]*)\)")

    def parse_key(k: str):
        m = _key_re.match(k)
        if m is None:
            return k, []
        return m.group(1), [t.strip() for t in m.group(2).split(",")]

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    return layer_init, parse_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment creation
    See the full definition in `./env.py`
    """)
    return


@app.cell
def _(RecordEpisodeStatistics, WarehouseInstance):
    def create_env():
        inst_dict = {
            "height": 21, "width": 21, "horizon": 400,
            "robots": ["r1", "r2", "r3"],
            "start_positions": {"r1": (1, 1), "r2": (1, 19), "r3": (10, 1)},
            "A_targets": {"r1": (2, 2), "r2": (2, 18), "r3": (10, 2)},
            "B_targets": {"r1": (18, 18), "r2": (18, 2), "r3": (10, 18)},
            "obstacles": [
                (3, 3), (4, 3), (5, 3),
                (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
                (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
            ],
            # Optional:
            # "start_carry": {"r1": False, "r2": False, "r3": False},
            # "start_delivered": {"r1": False, "r2": False, "r3": False},
        }
        inst = WarehouseInstance.from_dict(inst_dict)

        env = inst.make_env(render_mode="ansi", seed=0)
        env = RecordEpisodeStatistics(env)
        return env

    test_env = create_env().env
    return (test_env,)


@app.cell
def _(test_env):
    print(f"Observation space: {test_env.observation_space}")

    print(f"Action space: {test_env.action_space}")
    print(test_env.render())

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agent view helper class
    """)
    return


@app.cell
def _(Any, Dict, Tuple, np, parse_key):
    class EgocentricBuilder:
        """Per-agent 11x11 crops + action masks from full state."""
        def __init__(self, fov):
            self.fov = fov
            self.rad = fov // 2

        def build(self, state: Dict[str, Any], nf: Dict[str, Any]) -> Dict[str, np.ndarray]:
            H = int(nf.get("H", 21))
            W = int(nf.get("W", 21))

            obstacles = np.zeros((H, W), dtype=np.float32)
            for (x, y), val in nf.get("OBSTACLE", {}).items():
                if 0 <= x < H and 0 <= y < W:
                    obstacles[x, y] = 1.0 if val else 0.0

            Axy = {rid: xy for rid, xy in nf.get("A", {}).items()}
            Bxy = {rid: xy for rid, xy in nf.get("B", {}).items()}

            agent_xyc: Dict[str, Tuple[int, int, bool]] = {}
            for k, v in state.items():
                name, args = parse_key(k)
                if name == "agent_x":
                    rid = args[0]
                    x = int(v)
                    y = int(state.get(f"agent_y({rid})", 0))
                    c = bool(state.get(f"carry({rid})", 0))
                    agent_xyc[rid] = (x, y, c)

            rids = sorted(agent_xyc.keys())
            A = len(rids)

            # Global raster for critic: [obstacle, agent_occ, agent_carry, A, B]
            global_map = np.zeros((5, H, W), dtype=np.float32)
            global_map[0] = obstacles
            for rid, (ax, ay) in Axy.items():
                if 0 <= ax < H and 0 <= ay < W:
                    global_map[3, ax, ay] = 1.0
            for rid, (bx, by) in Bxy.items():
                if 0 <= bx < H and 0 <= by < W:
                    global_map[4, bx, by] = 1.0
            for rid, (x, y, c) in agent_xyc.items():
                if 0 <= x < H and 0 <= y < W:
                    global_map[1, x, y] = 1.0
                    if c: global_map[2, x, y] = 1.0

            # Per-agent crops: [obst, others_occ, others_carry, A, B, ego]
            C = 6
            crops = np.zeros((A, C, self.fov, self.fov), dtype=np.float32)
            amasks = np.ones((A, 5), dtype=np.float32)  # {wait,N,S,W,E}

            for i, rid in enumerate(rids):
                x, y, c = agent_xyc[rid]
                ax, ay = Axy[rid]
                bx, by = Bxy[rid]

                x0, x1 = x - self.rad, x + self.rad + 1
                y0, y1 = y - self.rad, y + self.rad + 1

                # Obstacles
                for xx in range(x0, x1):
                    for yy in range(y0, y1):
                        cx, cy = xx - x0, yy - y0
                        if 0 <= xx < H and 0 <= yy < W:
                            crops[i, 0, cx, cy] = obstacles[xx, yy]
                        else:
                            crops[i, 0, cx, cy] = 1.0  # OOB = wall

                # Other robots
                for rid2, (x2, y2, c2) in agent_xyc.items():
                    if rid2 == rid: continue
                    if x0 <= x2 < x1 and y0 <= y2 < y1:
                        cx, cy = x2 - x0, y2 - y0
                        crops[i, 1, cx, cy] = 1.0
                        if c2: crops[i, 2, cx, cy] = 1.0

                # A and B markers for this robot
                if x0 <= ax < x1 and y0 <= ay < y1:
                    crops[i, 3, ax - x0, ay - y0] = 1.0
                if x0 <= bx < x1 and y0 <= by < y1:
                    crops[i, 4, bx - x0, by - y0] = 1.0

                # Ego
                crops[i, 5, self.rad, self.rad] = 1.0

                # Action mask by walls/obstacles
                def valid(xx, yy):
                    return 0 <= xx < H and 0 <= yy < W and obstacles[xx, yy] == 0.0
                if not valid(x-1, y): amasks[i, 1] = 0.0  # N
                if not valid(x+1, y): amasks[i, 2] = 0.0  # S
                if not valid(x, y-1): amasks[i, 3] = 0.0  # W
                if not valid(x, y+1): amasks[i, 4] = 0.0  # E

            # Entities for critic: [x_norm, y_norm, carrying, was_blocked(0)]
            ents = []
            for rid in rids:
                x, y, c = agent_xyc[rid]
                ents.append([x/(H-1), y/(W-1), 1.0 if c else 0.0, 0.0])
            ents = np.asarray(ents, dtype=np.float32)

            return {
                "rids": np.array(rids),
                "crops": crops,               # [A, 6, 11, 11]
                "amasks": amasks,             # [A, 5]
                "global_map": global_map,     # [5, H, W]
                "entities": ents              # [A, 4]
            }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ENV wrapper
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Actor and critic classes
    """)
    return


@app.cell
def _(layer_init, nn, torch):
    class SharedActor(nn.Module):
        """Per-agent policy over 11x11 crops, shared across agents."""
        def __init__(self, c_in=6, n_actions=5, hidden=128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
            )
            self.head = nn.Sequential(
                nn.Linear(64*11*11, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
            for m in self.modules():
                if isinstance(m, nn.Linear): layer_init(m)

        def forward(self, crops, amask=None):
            x = self.conv(crops)
            x = x.reshape(x.size(0), -1)
            logits = self.head(x)
            if amask is not None:
                logits = logits + (amask==0).float()*-1e9
            return logits

    class CentralCritic(nn.Module):
        """Size- & count-invariant: CNN+AdaptiveAvgPool over map, mean over entities."""
        def __init__(self, c_map=5, f_agent=4, d=128):
            super().__init__()
            self.map = nn.Sequential(
                nn.Conv2d(c_map, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.ent  = nn.Sequential(nn.Linear(f_agent, d), nn.ReLU(),
                                      nn.Linear(d, d), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(64 + d, 128), nn.ReLU(),
                                      nn.Linear(128, 1))
            for m in self.modules():
                if isinstance(m, nn.Linear): layer_init(m)

        def forward(self, global_map, entities, agent_mask=None):
            M = self.map(global_map.unsqueeze(0))      # [1,64,H,W]
            M = self.pool(M).view(64)                   # [64]
            E = self.ent(entities)                      # [A,d]
            if agent_mask is None:
                Ep = E.mean(0)
            else:
                mask = agent_mask.view(-1,1).float()
                Ep = (E*mask).sum(0) / (mask.sum()+1e-6)
            x = torch.cat([M, Ep], dim=-1)
            return self.head(x).squeeze(-1)            # scalar V(s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Miscancellous checks
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
