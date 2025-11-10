from IPython.display import clear_output
import matplotlib.pyplot as plt

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class DictToListWrapper(gym.Wrapper):
    """
    The wrapper to convert Dict observation space to List observation space.
    """

    def __init__(self, env):
        super(DictToListWrapper, self).__init__(env)
        # assert isinstance(env.observation_space,
        #                   gym.spaces.Dict), "The observation space must be of type gym.spaces.Dict"
        assert isinstance(env.action_space, gym.spaces.Dict), "The action space must be of type gym.spaces.Dict"

        # transform the observation space to a Box space
        self.env_features = list(env.observation_space.spaces.keys())
        self.observation_space = Box(
            low=-float('inf'), high=float('inf'), shape=(len(self.env_features),), dtype=np.float32
        )
        # transform the action space to a Discrete space,
        # as each action dim is a Discrete space, the size of the Discrete space is the sum of the sizes of each action dim
        # e.g., if action space is {'a': Discrete(2), 'b': Discrete(2)}, then the new action space is Discrete(4),
        # means, 0 -> {'a': 0}, 1 -> {'a': 1}, 2 -> {'b': 0}, 3 -> {'b': 1}
        action_mapping = {}
        action_id = 0
        for key, value in env.action_space.spaces.items():
            for i in range(value.n):
                action_mapping[action_id] = {key: np.int64(i)}
                action_id += 1

        self.action_space = Discrete(action_id)
        self.action_mapping = action_mapping

    def reset(self, **kwargs):
        state, info = super(DictToListWrapper, self).reset(**kwargs)
        state = self.convert_state_dict2list(state)
        return state, info

    def step(self, action):
        env_action = self.convert_action_id2dict(action)
        state, reward, done, truncated, info = super(DictToListWrapper, self).step(env_action)
        state = self.convert_state_dict2list(state)
        return state, reward, done, truncated, info

    def convert_state_dict2list(self, state_dict):
        out = []
        for key in self.env_features:
            v = state_dict.get(key, 0)
            if isinstance(v, (bool, np.bool_)):
                out.append(int(v))
            elif isinstance(v, (int, np.integer)):
                out.append(int(v))
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(0.0)
        return np.array(out, dtype=np.float32)

    def convert_action_id2dict(self, action):
        return self.action_mapping[action]

    def get_state_description(self):
        print("State description:")
        for f in range(len(self.env_features)):
            print(f"state dim {f}: {self.env_features[f]}")

    def get_action_description(self):
        print("Action description:")
        for k, v in self.action_mapping.items():
            print(f"Action {k}: {v}")

# ---


def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(data_dict, save_pdf=False, output_file='training_curves.pdf'):
    """Plot the live graph with multiple subplots."""

    plt.style.use('ggplot')
    n_plots = len(data_dict)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(7, 4 * n_plots), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    plt.ion()
    clear_output(wait=True)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
        ax.clear()
        ax.plot(data, label=label, color="yellow", linestyle='--')
        # Compute and plot moving average for total reward
        if len(data) > 0:
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(range(ma_idx_start, len(data)), ma, label="Smoothed Value",
                    linestyle="-", color="purple", linewidth=2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='upper left')

    if save_pdf is False:
        plt.show()
    else:
        plt.savefig(output_file)
