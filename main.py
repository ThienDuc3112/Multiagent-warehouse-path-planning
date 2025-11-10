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
    import torch.optim as optim

    from pyRDDLGym.core.env import RDDLEnv

    from gymnasium.wrappers import RecordEpisodeStatistics

    from utils import DictToListWrapper, build_action_mask, BufferRollout, build_global_map5, build_entities, build_actor_crop6

    from env import WarehouseInstance, WarehousePickPlaceMultiEnv
    from fast_env import FastWarehouseInstance, FastWarehousePickPlaceMultiEnv


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        BufferRollout,
        DictToListWrapper,
        FastWarehouseInstance,
        RecordEpisodeStatistics,
        WarehouseInstance,
        build_action_mask,
        build_actor_crop6,
        build_entities,
        build_global_map5,
        nn,
        np,
        optim,
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
    ## Environment creation (Slow version)
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
def _(FastWarehouseInstance, RecordEpisodeStatistics):
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

    env = create_fast_env().env
    return (env,)


@app.cell
def _(env):
    print(f"Observation space: {env.observation_space}")

    print(f"Action space: {env.action_space}")
    print(env.render())
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
            if global_map.dim() == 3:          # (5,H,W) -> (1,5,H,W)
                global_map = global_map.unsqueeze(0)
            elif global_map.dim() != 4:
                raise ValueError(f"global_map must be (5,H,W) or (B,5,H,W), got {tuple(global_map.shape)}")

            B = global_map.size(0)
            M = self.map(global_map)           # [B,64,H,W]
            M = self.pool(M).view(B, 64)       # [B,64]

            # ---- Entities to [B,d] via mean over agents (masked if provided) ----
            if entities.dim() == 2:            # (A,f)
                E = self.ent(entities)         # [A,d]
                if agent_mask is not None and agent_mask.dim() == 1:  # (A,)
                    mask = agent_mask.view(1, -1, 1).float()          # [1,A,1]
                    Ep = (E.unsqueeze(0) * mask).sum(1) / (mask.sum(1) + 1e-6)  # [1,d]
                else:
                    Ep = E.mean(0, keepdim=True)                      # [1,d]
                Ep = Ep.expand(B, -1)                                 # [B,d]

            elif entities.dim() == 3:          # (B,A,f)  (or (1,A,f) broadcastable)
                if entities.size(0) not in (1, B):
                    raise ValueError(f"entities batch {entities.size(0)} not 1 or {B}")
                if entities.size(0) == 1 and B > 1:
                    entities = entities.expand(B, -1, -1)             # broadcast across batch

                B2, A, F = entities.shape
                E = self.ent(entities.view(-1, F))                    # [(B*A), d]
                d = E.size(-1)
                E = E.view(B2, A, d)                                  # [B,A,d]

                if agent_mask is not None:
                    if agent_mask.dim() == 1:                         # (A,)
                        mask = agent_mask.view(1, A, 1).float().expand(B2, -1, -1)
                    elif agent_mask.dim() == 2:                       # (B,A)
                        if agent_mask.size(0) == 1 and B2 > 1:
                            agent_mask = agent_mask.expand(B2, -1)
                        mask = agent_mask.view(B2, A, 1).float()
                    else:
                        raise ValueError(f"agent_mask must be (A,) or (B,A), got {agent_mask.shape}")
                    Ep = (E * mask).sum(1) / (mask.sum(1) + 1e-6)     # [B,d]
                else:
                    Ep = E.mean(1)                                    # [B,d]
            else:
                raise ValueError(f"entities must be (A,f) or (B,A,f), got {tuple(entities.shape)}")

            x = torch.cat([M, Ep], dim=1)                             # [B,64+d]
            return self.head(x).squeeze(-1)                           # [B]
    return CentralCritic, SharedActor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trainer class
    """)
    return


@app.cell
def _(
    BufferRollout,
    build_action_mask,
    build_actor_crop6,
    build_entities,
    build_global_map5,
    nn,
    optim,
    torch,
):
    class Trainer:
        def __init__(
            self,
            env,
            actor: nn.Module,           # your SharedActor(c_in=6)
            critic: nn.Module,          # your CentralCritic(c_map=5, f_agent=4)
            device: str = "cpu",
            steps_per_rollout: int = 1024,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_eps: float = 0.2,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            lr: float = 3e-4,
            max_grad_norm: float = 0.5,
            crop: int = 11,
        ):
            self.env = env
            self.actor = actor.to(device)
            self.critic = critic.to(device)
            self.device = torch.device(device)

            self.robots = env.robots
            self.A = len(self.robots)
            self.T = steps_per_rollout
            self.gamma, self.lam = gamma, lam
            self.clip_eps = clip_eps
            self.ent_coef, self.vf_coef = ent_coef, vf_coef
            self.max_grad_norm = max_grad_norm
            self.crop = crop

            self.buf = BufferRollout(T=self.T, A=self.A, crop=crop, device=self.device)
            self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

            # ensure crop is as actor expects
            self.env.set_crop(crop)

        @torch.no_grad()
        def _act(self):
            """Build inputs, run actor+critic once, sample actions for all agents."""
            # Central critic inputs
            gmap = build_global_map5(self.env).to(self.device)           # (5,H,W)
            ents = build_entities(self.env).to(self.device)              # (A,4)
            V = self.critic(gmap, ents)                                  # scalar

            # Actor inputs for all agents
            crops = []
            amasks = []
            for rid in self.robots:
                crops.append(build_actor_crop6(self.env, rid, self.crop))
                amasks.append(build_action_mask(self.env, rid))
            crops = torch.stack(crops, dim=0).to(self.device)            # (A,6,11,11)
            amasks = torch.stack(amasks, dim=0).to(self.device)          # (A,5)

            logits = self.actor(crops, amask=amasks)                     # (A,5)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()                                       # (A,)
            logps = dist.log_prob(actions)                                # (A,)

            return gmap, ents, V, crops, amasks, actions, logps

        def rollout(self):
            self.buf.ptr = 0
            obs, info = self.env.reset()
            done = False
            while not self.buf.full():
                gmap, ents, V, crops, amasks, actions, logps = self._act()

                action_dict = {rid: int(a.item()) for rid, a in zip(self.robots, actions)}
                _, reward, terminated, truncated, _ = self.env.step(action_dict)
                done = terminated or truncated

                # store
                self.buf.add_global(gmap, ents, V, reward, float(done))
                self.buf.add_local(crops, amasks, actions, logps)

                if done:
                    obs, info = self.env.reset()

        def _gae(self):
            """Compute GAE(λ) on the scalar central value stream."""
            V = self.buf.values
            R = self.buf.rewards
            D = self.buf.dones
            T = self.T

            adv = torch.zeros(T, dtype=torch.float32, device=self.device)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                nonterminal = 1.0 - D[t]
                nextv = V[t+1] if t < T-1 else V[t]
                delta = R[t] + self.gamma * nextv * nonterminal - V[t]
                lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
                adv[t] = lastgaelam
            ret = adv + V
            # normalize advantages
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            return adv, ret

        def update(self):
            adv, ret = self._gae()

            # Flatten over (T * A)
            T, A = self.T, self.A
            crops = self.buf.crops.reshape(T*A, 6, self.crop, self.crop)
            amask = self.buf.amask.reshape(T*A, 5)
            actions = self.buf.actions.reshape(T*A)
            old_logps = self.buf.logps.reshape(T*A).detach()

            # Repeat central inputs per agent (each agent shares same V/adv at that step)
            gmap = self.buf.gmap.unsqueeze(1).repeat(1, A, 1, 1, 1).reshape(T*A, *self.buf.gmap.shape[1:])
            ents = self.buf.entities.unsqueeze(1).repeat(1, A, 1, 1).reshape(T*A, *self.buf.entities.shape[1:])
            values = self.critic(gmap, ents)                              # (T*A,)

            adv_flat = adv.unsqueeze(1).repeat(1, A).reshape(T*A)
            ret_flat = ret.unsqueeze(1).repeat(1, A).reshape(T*A)

            logits = self.actor(crops, amask=amask)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO losses
            ratio = (new_logps - old_logps).exp()
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (values - ret_flat).pow(2).mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.opt.step()

            return {
                "loss": float(loss.item()),
                "pi_loss": float(policy_loss.item()),
                "v_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "adv_mean": float(adv.mean().item()),
                "ret_mean": float(ret.mean().item()),
            }

        def train_step(self):
            self.rollout()
            return self.update()
    return (Trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training loop
    """)
    return


@app.cell
def _(CentralCritic, SharedActor, Trainer, env, torch):
    actor = SharedActor(c_in=6, n_actions=5, hidden=128)
    critic = CentralCritic(c_map=5, f_agent=4, d=128)

    # 3) trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(env, actor, critic, device=device, steps_per_rollout=512)

    # 4) loop
    for it in range(200):
        stats = trainer.train_step()
        if (it+1) % 5 == 0:
            print(f"[{it+1:04d}] loss={stats['loss']:.3f} pi={stats['pi_loss']:.3f} "
                  f"v={stats['v_loss']:.3f} H={stats['entropy']:.3f} "
                  f"adv={stats['adv_mean']:.3f} ret={stats['ret_mean']:.3f}")

    return


if __name__ == "__main__":
    app.run()
