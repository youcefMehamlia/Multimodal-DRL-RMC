## Overview 
- This framework is derived from DQN-FrameworkQ of Romain.
    - Link: https://github.com/romainducrocq/DQN-frameworQ 

- Important change made with DQN-FrameworkQ of romain 
    - change dictorary definition for config values to dataclass

    - Add comments.
    - Add assert statements to check the input types.
    - Remove the msgpack_numpy.py file and install the msgpack-numpy package.
    - Replace the hyperparameter dictionary with a dataclass.
    - Reorganize the placement of some modules.
    - Remove the command-line argument parser.


## Requirements 
- NumPy
- Gymnasium
- Torch
- TensorBoard
- msgpack_numpy


## How to Use It:
1. Copy `dqn` and `rl_env` folders into your project
2. Place your custom environment and its additional resources in the `custom_env` folder.
3. in `custom_env -> __init__.py` import your custom environment as `CustomEnv`.
4. Your custom environment class should include the following methods:
    - step(self, action) ->(obs, reward, done, info:dict)
        - info should contain two keys:
            - "l": episode length
            - "r": episode total reward
    - reset(self)
5. Your custom environment class should include these two instance attributes:
    - self.action_space_n: number of possible actions
    - self.state_space_n: number of features in a single state

## Project structure 
<pre>
<span style="font-size: 18px">DQN_Framework</span>
  |
  |__<span style="color:#DEB887">dqn</span>
  | |____init__.py
  | |__agent.py
  | |   <span style="color:#87CEFA;font-size: 14px"> - This module implements different Deep Q-Network (DQN) algorithms </span>
  | |
  | |__dqn_config.py
  | |   <span style="color:#87CEFA;font-size: 14px">- Network Configuration: NN structure, activation functions, optimizer, etc.</span>
  | |   <span style="color:#87CEFA;font-size: 14px">- DQN Algorithm Hyperparameters: Learning rate, discount factor, etc.</span>
  | |
  | |__network.py
  | |   <span style="color:#87CEFA;font-size: 14px"> - Implements the neural networks for online and target Q-networks, with both simple and dueling architectures,</span>
  | |
  | |__replay_memory.py
  | |   <span style="color:#87CEFA;font-size: 14px"> - Implements the replay memory data buffer, with both uniform and prioritized transition sampling</span>
  | |
  | |__<span style="color:#DEB887">utils</span>
  |    |__ __init__.py
  |    |__custom_abc_meta.py
  |    |    <span style="color:#87CEFA;font-size: 14px"> - Source: https://stackoverflow.com/questions/23831510/abstract-attribute-not-property</span>
  |    |    <span style="color:#87CEFA;font-size: 14px"> - Define a decorator indicating abstract attribute</span>
  |    |    <span style="color:#87CEFA;font-size: 14px"> - Defines a custom metaclass, that extends ABCMeta to support abstract attributes</span>
  |    |
  |    |__sum_tree.py  
  |         <span style="color:#87CEFA;font-size: 14px"> - Define a data structure for efficient prioritized sampling.</span>
  |         <span style="color:#87CEFA;font-size: 14px"> - Source: https://pylessons.com/CartPole-PER</span>  
  |  
  |__<span style="color:#DEB887">logs</span> : This folder will be created automatically after the training begins.
  |  |__<span style="color:#DEB887">test</span>: Store the CSV file of the DQN test results. 
  |  |__<span style="color:#DEB887">train</span>: Store the TensorBoard log files of the DQN training.
  |
  |__<span style="color:#DEB887">rl_env</span>
  |  |__<span style="color:#DEB887">custom_env</span>
  |  |  |__ __init__.py
  |  |  |__  <span style="color:#FF4D4D">.... add your custom environment class and its additional resources here.</span> 
  |  |  
  |  |__<span style="color:#DEB887">env_tools</span>
  |     |__<span style="color:#DEB887">vector_env</span>
  |     |  |__ __init__.py
  |     |  |__dummy_vec_env.py
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Source: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/dummy_vec_env.py </span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Define the DummyVecEnv class for vectorized environment that runs all environments Sequentialy in a single process  </span>
  |     |  |
  |     |  |__LICENSE
  |     |  |
  |     |  |__monitor.py
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Source: https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py </span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - A monitor wrapper for Gym environments, used for tracking and recording episode statistics : episode reward, length, time and other data.</span>
  |     |  |
  |     |  |__subproc_vec_env.py
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Source: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py</span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Define the SubprocVecEnv class for vectorized environment that runs all environments in parallel using separate subprocesses</span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Each environment (or group of environments , see `in_series` parameter) runs in its own Python subprocess</span>
  |     |  |
  |     |  |__util.py
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Source: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/util.py</span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Contains helper functions for working with vectorized environments </span>
  |     |  |  <span style="color:#87CEFA;font-size: 14px"> - Includes functions for deep-copying observation dictionaries, converting observations, and extracting information from observation spaces</span>
  |     |  |
  |     |  |__vec_env.py
  |     |     <span style="color:#87CEFA;font-size: 14px"> - Source: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_env.py</span>
  |     |     <span style="color:#87CEFA;font-size: 14px"> - Provides abstract base classes and utility functions for vectorized environment that handle multiple environments simultaneously </span>
  |     |  
  |     |__<span style="color:#DEB887">wrappers</span>
  |     |  |__ __init__.py
  |     |  |
  |     |  |__max_epd_step_wrapper.py
  |     |  |   <span style="color:#87CEFA;font-size: 14px"> - define wrappers for limiting episode steps. </span>
  |     |  |
  |     |  |__repeat_action_wrappers.py
  |     |     <span style="color:#87CEFA;font-size: 14px"> - define wrappers for repeating actions over multiple steps. </span>
  |     |
  |     |__ __init__.py
  |     |  
  |     |__env_make.py
  |     |   <span style="color:#87CEFA;font-size: 14px"> - This module creates a vectorized environment with optional action repetition and maximum episode step limits.</span>
  |     |
  |     |__env_wrap.py
  |        <span style="color:#87CEFA;font-size: 14px"> - Wraps the customized environment in OpenAi Gym, </span>
  |
  |__<span style="color:#DEB887">save</span>: Store the trained models. This folder will be created automatically after the training begins 
  |  
  |__test_dqn.py
  |  <span style="color:#87CEFA;font-size: 14px"> - test  a trained DQN agent </span> 
  |
  |__train_dqn.py
     <span style="color:#87CEFA;font-size: 14px"> - Train a DQN agent </span>
  
