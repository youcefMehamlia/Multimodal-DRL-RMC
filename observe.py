from env import HYPER_PARAMS, network_config, CustomEnv, View
from dqn import CustomEnvWrapper, make_env, Networks

import os
import argparse
import numpy as np

from torch import device, cuda


class Observe(View):
    def __init__(self, args):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        super(Observe, self).__init__(type(self).__name__.upper(),
                                      make_env(
                                          env=CustomEnvWrapper(CustomEnv(type(self).__name__.lower())),
                                          max_episode_steps=args.max_s)
                                      )

        model_pack = args.d.split('/')[-1].split('_model.pack')[0]

        self.network = getattr(Networks, {
            "DQNAgent": "DeepQNetwork",
            "DoubleDQNAgent": "DeepQNetwork",
            "DuelingDoubleDQNAgent": "DuelingDeepQNetwork",
            "PerDuelingDoubleDQNAgent": "DuelingDeepQNetwork"
        }[model_pack.split('_lr')[0]])(
            device(("cuda:" + args.gpu) if cuda.is_available() else "cpu"),
            float(model_pack.split('_lr')[1].split('_')[0]),
            network_config,
            self.env.observation_space,
            self.env.action_space.n
        )

        self.network.load(args.d)

        self.obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)

        self.repeat = 0
        self.action = 0
        self.ep = 0

        print()
        print("OBSERVE")
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_episodes = args.max_e
        self.log = (args.log, args.log_s, args.log_dir + model_pack)

    def setup(self):
        self.obs = self.env.reset()
        
        
        
    def close(self):
        """Closes the environment."""
        self.env.close()

    def loop(self):
        if self.repeat % (HYPER_PARAMS['repeat'] or 1) == 0:
            self.action = self.network.actions([self.obs.tolist()])[0]

        self.repeat += 1

        self.obs, _, done, info = self.env.step(self.action)
        self.env.log_info_writer(info, done, *self.log)


        #to fix additional episode at the end
        if done:
            self.repeat = 0
            self.ep += 1

            print()
            print("Episode :", self.ep)
            # You can still print the info from the episode that just finished
            [print(k, ":", info[k]) for k in info]

            # Check if we should exit BEFORE resetting the environment
            if bool(self.max_episodes) and self.ep >= self.max_episodes:
                # The environment (and SUMO) from the completed run is still open.
                # We can now safely close it and exit the script.
                self.env.close() # Gracefully close the TraCI connection
                exit()
            
            # If we are NOT exiting, only then do we reset for the next episode.
            self.setup()
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSERVE")
    str2bool = (lambda v: v.lower() in ("yes", "y", "true", "t", "1"))
    parser.add_argument('-d', type=str, default='', help='Directory', required=True)
    parser.add_argument('-gpu', type=str, default='0', help='GPU #')
    parser.add_argument('-max_s', type=int, default=0, help='Max steps per episode if > 0, else inf')
    parser.add_argument('-max_e', type=int, default=0, help='Max episodes if > 0, else inf')
    parser.add_argument('-log', type=str2bool, default=False, help='Log csv to ./logs/test/')
    parser.add_argument('-log_s', type=int, default=0, help='Log step if > 0, else episode')
    parser.add_argument('-log_dir', type=str, default="./logs/test/", help='Log directory')

    Observe(parser.parse_args()).run()
