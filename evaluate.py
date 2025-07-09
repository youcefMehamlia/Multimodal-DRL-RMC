import os
import sys
import argparse
import random
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

# Ensure the project root is in the python path
# This allows us to import from 'env' and 'dqn'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from env import CustomEnv, View, SUMO_PARAMS
from dqn import CustomEnvWrapper, make_env
from evaluation.parsers import parse_tripinfo_for_episode_stats, parse_sumo_log, parse_framework_log
from play import Play
from observe import Observe

# Define strategies and their corresponding class constructors
STRATEGIES = {
    "DQNAgent": Observe,
    "AlwaysGreenBaseline": Play,
    "FixedCycleBaseline": Play,
    "AlineaDsBaseline": Play,
    "PiAlineaDsBaseline": Play
}

def run_single_episode(env_instance):
    """Runs one full episode of the simulation."""
    obs = env_instance.env.reset()
    done = False
    while not done:
        # For Play (baselines), the action is determined internally
        if isinstance(env_instance, Play):
            action = env_instance.get_play_action()
        # For Observe (agent), we get the action from the network
        else:
            action = env_instance.network.actions([obs.tolist()])[0]
        
        obs, _, done, info = env_instance.env.step(action)
        env_instance.env.log_info_writer(info, done, *env_instance.log)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark for ramp metering strategies.")
    parser.add_argument('-s', '--strategy', type=str, required=True, choices=list(STRATEGIES.keys()), help='The control strategy to evaluate.')
    parser.add_argument('-n', '--num-episodes', type=int, default=10, help='Number of episodes to run for the evaluation.')
    parser.add_argument('--master-seed', type=int, default=42, help='The master seed for reproducibility.')
    parser.add_argument('-d', '--model-path', type=str, default=None, help='Path to the trained DRL agent model (.pack file), required for DQNAgent.')
    parser.add_argument('-o', '--output-dir', type=str, default="./evaluation/results/", help='Directory to save the final results CSV.')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU to use for the agent.')
    args = parser.parse_args()

    if args.strategy not in STRATEGIES:
        print(f"Error: Strategy '{args.strategy}' not found.")
        return

    # --- Setup Paths & Initial Config ---
    os.makedirs(args.output_dir, exist_ok=True)
    strategy_class = STRATEGIES[args.strategy]
    
    # These paths are based on your project structure
    base_data_path = f"./env/custom_env/data/{SUMO_PARAMS['config']}/"
    tripinfo_xml_path = os.path.join(base_data_path, "tripinfo.xml")
    sumo_log_path = os.path.join(args.output_dir, "temp_sumo.log")
    framework_log_path = os.path.join(args.output_dir, "temp_framework_log.csv")

    all_episode_metrics = []

    print(f"{Fore.CYAN}--- Starting Evaluation for: {args.strategy} ---{Style.RESET_ALL}")
    
    # --- Main Episode Loop ---
    for episode in tqdm(range(args.num_episodes), desc=f"Evaluating {args.strategy}"):
        current_seed = args.master_seed + episode
        
        # Set environment variables for the SUMO instance to use
        os.environ['SUMO_EVAL_SEED'] = str(current_seed)
        os.environ['SUMO_EVAL_LOG_FILE'] = sumo_log_path
        # Set python's random seed. Crucial for route generation.
        random.seed(current_seed)
        
        # --- Instantiate the correct class (Play or Observe) ---
        # Create a mock 'args' object to pass to the constructors
        if strategy_class == Play:
            mock_args = argparse.Namespace(
                player=args.strategy, max_s=0, max_e=1, log=True, 
                log_s=0, log_dir=os.path.dirname(framework_log_path) + "/"
            )
            # Override the log file name for play.py
            mock_args.log_dir += "temp_framework_log" 
        else: # It's Observe
            if not args.model_path:
                print(f"{Fore.RED}Error: --model-path is required for DQNAgent.{Style.RESET_ALL}")
                return
            mock_args = argparse.Namespace(
                d=args.model_path, gpu=args.gpu, max_s=0, max_e=1, log=True, 
                log_s=0, log_dir=os.path.dirname(framework_log_path) + "/"
            )
            # Override the log file name for observe.py
            model_pack = args.model_path.split('/')[-1].split('_model.pack')[0]
            mock_args.log_dir += f"temp_framework_log_{model_pack}"
            framework_log_path = os.path.join(os.path.dirname(framework_log_path), f"temp_framework_log_{model_pack}.csv")


        env_instance = strategy_class(mock_args)
        
        # --- Run one full episode ---
        run_single_episode(env_instance)
        
        # --- Get Scenario Info & Close SUMO (which finalizes logs) ---
        scenario_info = env_instance.env.get_env().get_scenario_info()
        env_instance.close()

        # --- Parse all log files for this episode ---
        trip_stats = parse_tripinfo_for_episode_stats(tripinfo_xml_path)
        sumo_stats = parse_sumo_log(sumo_log_path)
        framework_stats = parse_framework_log(framework_log_path)

        # --- Combine and Store Results ---
        combined_stats = {
            "episode_id": episode, "seed": current_seed,
            **scenario_info, **trip_stats, **sumo_stats, **framework_stats
        }
        all_episode_metrics.append(combined_stats)

        # --- Cleanup ---
        if os.path.exists(sumo_log_path): os.remove(sumo_log_path)
        if os.path.exists(framework_log_path): os.remove(framework_log_path)

    # --- Save Final DataFrame ---
    results_df = pd.DataFrame(all_episode_metrics)
    final_csv_path = os.path.join(args.output_dir, f"results_{args.strategy}.csv")
    results_df.to_csv(final_csv_path, index=False)
    
    print(f"\n{Fore.GREEN}--- Evaluation Complete: {args.strategy} ---{Style.RESET_ALL}")
    print(f"Results for {args.num_episodes} episodes saved to: {final_csv_path}")


if __name__ == "__main__":
    main()