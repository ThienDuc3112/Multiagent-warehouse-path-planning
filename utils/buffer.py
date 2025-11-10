from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class BufferRollout:
    T: int
    A: int
    crop: int
    device: torch.device

    def __post_init__(self):
        self.crops = torch.zeros(self.T, self.A, 6, self.crop, self.crop, dtype=torch.float32, device=self.device)
        self.amask = torch.zeros(self.T, self.A, 5, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.T, self.A, dtype=torch.long, device=self.device)
        self.logps = torch.zeros(self.T, self.A, dtype=torch.float32, device=self.device)

        self.gmap = None          # (T, 5, H, W) float32
        self.entities = None      # (T, A, 4) float32
        self.values = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add_global(self, gmap_t, ents_t, value_t, reward_t, done_t):
        if self.gmap is None:
            self.gmap = torch.zeros(self.T, *gmap_t.shape, dtype=torch.float32, device=self.device)
            self.entities = torch.zeros(self.T, *ents_t.shape, dtype=torch.float32, device=self.device)
        i = self.ptr
        self.gmap[i].copy_(gmap_t)
        self.entities[i].copy_(ents_t)
        self.values[i] = value_t
        self.rewards[i] = reward_t
        self.dones[i] = done_t

    def add_local(self, crops_t, amask_t, actions_t, logps_t):
        i = self.ptr
        self.crops[i].copy_(crops_t)
        self.amask[i].copy_(amask_t)
        self.actions[i].copy_(actions_t)
        self.logps[i].copy_(logps_t)
        self.ptr += 1

    def full(self):
        return self.ptr >= self.T
