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
    from fast_env import FastWarehouseInstance, FastWarehousePickPlaceMultiEnv


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DictToListWrapper,
        FastWarehouseInstance,
        RecordEpisodeStatistics,
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
    return (layer_init,)


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
    return create_env, test_env


@app.cell
def _(DictToListWrapper, test_env):
    print(f"Observation space: {test_env.observation_space}")

    print(f"Action space: {test_env.action_space}")
    print(test_env.render())

    test_dict_env = DictToListWrapper(test_env)

    test_dict_env.get_action_description()
    test_dict_env.get_state_description()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment creation (fast version)
    See the full definition in `./fast_env.py`

    This used caching with numpy
    """)
    return


@app.cell
def _(FastWarehouseInstance, RecordEpisodeStatistics, create_env):
    def create_fast_env():
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
        inst = FastWarehouseInstance.from_dict(inst_dict)

        env = inst.make_env(render_mode="ansi", seed=0)
        env = RecordEpisodeStatistics(env)
        return env

    test_fast_env = create_env().env
    return


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
def _():
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
