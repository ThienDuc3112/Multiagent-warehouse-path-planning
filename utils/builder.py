import numpy as np
import torch
from fast_env import FastWarehousePickPlaceMultiEnv


def build_action_mask(env: FastWarehousePickPlaceMultiEnv, rid: str):
    """Binary mask [WAIT,N,S,W,E]; 1=valid, 0=invalid (bounds/obstacles only)."""
    H, W = env.H, env.W
    x, y = env._x[rid], env._y[rid]
    obs = env.get_global_map_fast()[0]  # obstacles channel
    m = np.ones(5, dtype=np.int8)
    # NORTH
    if not (x > 0 and obs[x - 1, y] == 0):
        m[1] = 0
    # SOUTH
    if not (x < H - 1 and obs[x + 1, y] == 0):
        m[2] = 0
    # WEST
    if not (y > 0 and obs[x, y - 1] == 0):
        m[3] = 0
    # EAST
    if not (y < W - 1 and obs[x, y + 1] == 0):
        m[4] = 0
    # WAIT is always valid
    return torch.from_numpy(m)


def build_actor_crop6(env, rid: str, crop: int = 11):
    """
    6-channel egocentric crop (11x11 by default) to match your SharedActor(c_in=6):
      0: obstacles
      1: other robots (self-excluded)
      2: A-target for this agent
      3: B-target for this agent
      4: current target (A if !carry else B)
      5: time-fraction constant plane (t/horizon); carry is inferable via ch4 vs 2&3
    """
    assert crop % 2 == 1
    env.set_crop(crop)  # no-op if already same; keeps pads consistent

    # Base 3 channels from the env's fast egocentric getter
    #   [0]=obstacles, [1]=other robots, [2]=current target (A or B depending on carry)
    local3 = env.get_local_map_fast(rid, crop)  # (3, crop, crop), int8
    C, Hc, Wc = local3.shape

    # Build A/B target crops without touching env internals:
    rx, ry = env._x[rid], env._y[rid]
    ax, ay = env.A[rid]
    bx, by = env.B[rid]
    r = crop // 2

    A_crop = np.zeros((Hc, Wc), dtype=np.int8)
    dxA, dyA = ax - rx, ay - ry
    if abs(dxA) <= r and abs(dyA) <= r:
        A_crop[r + dxA, r + dyA] = 1

    B_crop = np.zeros((Hc, Wc), dtype=np.int8)
    dxB, dyB = bx - rx, by - ry
    if abs(dxB) <= r and abs(dyB) <= r:
        B_crop[r + dxB, r + dyB] = 1

    cur_tgt = local3[2]  # current target plane

    tfrac = env._t / float(env.horizon)
    T_plane = np.full((Hc, Wc), tfrac, dtype=np.float32)

    # Stack into 6 channels
    # Cast to float32 for the network; keep cur_tgt as float too.
    chs = [
        local3[0].astype(np.float32),
        local3[1].astype(np.float32),
        A_crop.astype(np.float32),
        B_crop.astype(np.float32),
        cur_tgt.astype(np.float32),
        T_plane.astype(np.float32),
    ]
    return torch.from_numpy(np.stack(chs, axis=0))  # (6, crop, crop) float32


def build_global_map5(env):
    """
    Compress env global map (C=1+3N) into 5 channels your critic expects:
      0: obstacles
      1: any-robot occupancy (clip >0 to 1)
      2: any A-target (clip)
      3: any B-target (clip)
      4: carrying-robot occupancy (positions of robots with carry=True)
    """
    g = env.get_global_map_fast()  # (1+3N, H, W) int8
    N = len(env.robots)
    obs = g[0]
    robots_any = (g[1:1 + N].sum(axis=0) > 0).astype(np.float32)
    As = (g[1 + N:1 + 2 * N].sum(axis=0) > 0).astype(np.float32)
    Bs = (g[1 + 2 * N:1 + 3 * N].sum(axis=0) > 0).astype(np.float32)

    # carrying occupancy
    carry_occ = np.zeros_like(obs, dtype=np.float32)
    for i, rid in enumerate(env.robots):
        if env._carry[rid]:
            carry_occ = np.maximum(carry_occ, g[1 + i].astype(np.float32))

    out = np.stack([
        obs.astype(np.float32),
        robots_any,
        As,
        Bs,
        carry_occ
    ], axis=0)  # (5,H,W)
    return torch.from_numpy(out)


def build_entities(env):
    """Per-agent features [x/H, y/W, carry, delivered] -> tensor (A,4) float32."""
    Hm1 = max(1, env.H - 1)
    Wm1 = max(1, env.W - 1)
    rows = []
    for rid in env.robots:
        x, y = env._x[rid], env._y[rid]
        c = 1.0 if env._carry[rid] else 0.0
        d = 1.0 if env._delivered[rid] else 0.0
        rows.append([x / Hm1, y / Wm1, c, d])
    return torch.tensor(rows, dtype=torch.float32)


def current_goal_xy(env, rid):
    carrying = env._carry[rid]  # or env.state['carry'][rid]
    if carrying:
        gx, gy = env.B[rid]
    else:
        gx, gy = env.A[rid]
    return gx, gy, float(carrying)


def build_goal_vec(env, rid, H, W, device):
    rx, ry = env._x[rid], env._y[rid]   # (x,y) of agent
    gx, gy, phase = current_goal_xy(env, rid)
    dx = 2.0 * (gx - rx) / max(W - 1, 1)   # [-1,1]
    dy = 2.0 * (gy - ry) / max(H - 1, 1)   # [-1,1]
    dist = ((gx - rx)**2 + (gy - ry)**2) ** 0.5
    dist = dist / ((H * H + W * W) ** 0.5)  # [0,1]
    return torch.tensor([dx, dy, dist, phase], dtype=torch.float32, device=device)
