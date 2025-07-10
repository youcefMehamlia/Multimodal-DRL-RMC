import gymnasium as gym
from gymnasium import spaces
import numpy as np

import os
from csv import DictWriter


class CustomEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, custom_env):
        super(CustomEnvWrapper, self).__init__()

        self.custom_env = custom_env

        self.mode = self.custom_env.mode
        self.player = self.custom_env.player

        self.steps = 0
        self.total_reward = 0.

        action_space_n = self.custom_env.action_space_n
        observation_space_n = (self.custom_env.observation_space_n,) \
            if isinstance(self.custom_env.observation_space_n, int) else self.custom_env.observation_space_n

        self.action_space = spaces.Discrete(action_space_n)
        self.observation_space = spaces.Box(low=0., high=1., shape=observation_space_n, dtype=np.float32)

        self.log_info_buffer = []

    def get_env(self):
        return self.custom_env

    def _obs(self):
        obs = self.custom_env.obs()

        if isinstance(obs, np.ndarray):
            if obs.dtype == np.float32:
                return obs
            else:
                return obs.astype('float32')
        else:
            return np.array(obs, dtype=np.float32)

    def _rew(self):
        rew = self.custom_env.rew()

        self.total_reward += rew
        return rew

    def _done(self):
        return self.custom_env.done()

    def _info(self):
        info = {
            "l": self.steps,
            "r": self.total_reward
        }
        # When not training, we get more detailed info from the underlying env
        if not self.mode["train"]:
            # Ensure custom_env.info() returns a dictionary
            detailed_info = self.custom_env.info()
            if detailed_info:
                info.update(detailed_info)
        return info

    # --- THIS IS THE METHOD TO CHANGE ---
    def reset(self, *, seed=None, options=None):
        # The new gymnasium API passes seed and options. We should accept them.
        super().reset(seed=seed) # This handles seeding in the parent class

        self.steps = 0
        self.total_reward = 0.

        # This will call DqnEnv.reset() -> RLController.reset() / Baseline.reset()
        self.custom_env.reset()

        if not self.mode["train"]:
            self.reset_render()

        # The initial observation is created by _obs()
        initial_obs = self._obs()
        
        # The initial info dictionary is created by _info()
        # At reset, steps=0 and total_reward=0, which is correct.
        initial_info = self._info()
        
        # --- RETURN TWO VALUES AS REQUIRED ---
        return initial_obs, initial_info

    # --- ALSO MODIFY THE STEP METHOD TO RETURN 5 VALUES ---
    def step(self, action):
        self.custom_env.step(action)

        if not self.mode["train"]:
            self.step_render()

        self.steps += 1
        
        # In Gymnasium, `done` is split into `terminated` and `truncated`.
        # For simplicity here, we'll treat them as the same.
        # A more robust implementation would have self.custom_env distinguish them.
        terminated = self._done()
        truncated = False # Assuming TimeLimit wrapper will handle truncation

        # The new API requires 5 return values
        return self._obs(), self._rew(), terminated, truncated, self._info()

    def reset_render(self):
        self.custom_env.reset_render()

    def step_render(self):
        self.custom_env.step_render()

    def render(self, mode='human'):
        pass

    def log_info_writer(self, info, done, log, log_step, log_path):
        if log and (done or (log_step > 0 and info["l"] % log_step == 0)):
            if "TimeLimit.truncated" not in info:
                info["TimeLimit.truncated"] = False
            info["done"] = done

            self.log_info_buffer.append(info)

            if done:
                # The log_path from evaluate.py already has the full name, no need to add .csv
                file_exists = os.path.isfile(log_path)

                with open(log_path, 'a', newline='') as f: # Added newline='' for better csv handling
                    # Make sure all keys are present in all info dicts, or handle missing keys
                    fieldnames = sorted(list(info.keys()))
                    csv_writer = DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
                    
                    if not file_exists:
                        csv_writer.writeheader()
                    
                    for log_info_row in self.log_info_buffer:
                        # Ensure all rows have all keys, fill with None if missing
                        row_to_write = {key: log_info_row.get(key) for key in fieldnames}
                        csv_writer.writerow(row_to_write)
                    
                self.log_info_buffer = []
                
    def close(self):
        """Calls the close method of the wrapped environment."""
        # --- FIX IS HERE ---
        # Call close() on self.custom_env, not self.env
        if hasattr(self.custom_env, 'close'):
            self.custom_env.close()