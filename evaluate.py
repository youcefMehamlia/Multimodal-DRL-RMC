# evaluate.py
import os
import sys
import argparse
import random
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from env import CustomEnv, View, SUMO_PARAMS
from dqn import CustomEnvWrapper, make_env
# --- The only parser we need for XML/logs ---
from evaluation.parsers import parse_tripinfo_for_episode_stats, parse_sumo_log, parse_framework_log
from play import Play
from observe import Observe

STRATEGIES = {
    "DQNAgent": Observe, "AlwaysGreenBaseline": Play, "FixedCycleBaseline": Play,
    "AlineaDsBaseline": Play, "PiAlineaDsBaseline": Play
}

def run_single_episode(env_instance):
    obs, info = env_instance.env.reset()
    done = truncated = False
    while not (done or truncated):
        action = env_instance.get_play_action() if isinstance(env_instance, Play) else env_instance.network.actions([obs.tolist()])[0]
        obs, _, terminated, truncated, info = env_instance.env.step(action)
        done = terminated
        env_instance.env.log_info_writer(info, done or truncated, *env_instance.log)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark for ramp metering strategies.")
    # ... (all arguments are the same as before) ...
    parser.add_argument('-s', '--strategy', type=str, required=True, choices=list(STRATEGIES.keys()), help='The control strategy to evaluate.')
    parser.add_argument('-n', '--num-episodes', type=int, default=10, help='Number of episodes to run for the evaluation.')
    parser.add_argument('--master-seed', type=int, default=42, help='The master seed for reproducibility.')
    parser.add_argument('-d', '--model-path', type=str, default=None, help='Path to the trained DRL agent model (.pack file), required for DQNAgent.')
    parser.add_argument('-o', '--output-dir', type=str, default="./evaluation/results/", help='Directory to save the final results CSV.')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU to use for the agent.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    strategy_class = STRATEGIES[args.strategy]
    
    # --- Paths are simpler now ---
    base_data_path = f"./env/custom_env/data/{SUMO_PARAMS['config']}/"
    tripinfo_xml_path = os.path.join(base_data_path, "tripinfo.xml")
    temp_sumo_log_path = os.path.join(args.output_dir, f"temp_sumo_log_{args.strategy}.log")
    
    all_episode_metrics = []
    print(f"{Fore.CYAN}--- Starting Evaluation for: {Style.BRIGHT}{args.strategy}{Style.RESET_ALL} ---")
    
    for episode in tqdm(range(args.num_episodes), desc=f"Evaluating {args.strategy}", unit="episode"):
        current_seed = args.master_seed + episode
        os.environ['SUMO_EVAL_SEED'] = str(current_seed)
        os.environ['SUMO_EVAL_LOG_FILE'] = temp_sumo_log_path
        random.seed(current_seed)
        
        mock_args_dict = {'max_s': 0, 'max_e': 1, 'log': True, 'log_s': 1, 'log_dir': args.output_dir}
        
        if strategy_class == Play:
            mock_args_dict['player'] = args.strategy
            temp_framework_log_path = os.path.join(args.output_dir, args.strategy)
        else:
            if not args.model_path:
                print(f"{Fore.RED}\nError: --model-path is required for DQNAgent.{Style.RESET_ALL}"); return
            mock_args_dict.update({'d': args.model_path, 'gpu': args.gpu})
            model_pack_name = args.model_path.split('/')[-1].split('_model.pack')[0]
            temp_framework_log_path = os.path.join(args.output_dir, model_pack_name)

        mock_args = argparse.Namespace(**mock_args_dict)
        env_instance = strategy_class(mock_args)
        
        run_single_episode(env_instance)
        
        scenario_info = env_instance.env.get_env().get_scenario_info()
        env_instance.close()

        # --- Parsing is now simpler ---
        trip_and_emission_stats = parse_tripinfo_for_episode_stats(tripinfo_xml_path)
        sumo_stats = parse_sumo_log(temp_sumo_log_path)
        framework_stats = parse_framework_log(temp_framework_log_path, spillback_threshold=20)
        
        combined_stats = {
            "episode_id": episode, "seed": current_seed,
            **scenario_info, **trip_and_emission_stats, **sumo_stats, **framework_stats
        }
        all_episode_metrics.append(combined_stats)

        # --- Cleanup is simpler ---
        if os.path.exists(temp_sumo_log_path): os.remove(temp_sumo_log_path)
        if os.path.exists(temp_framework_log_path): os.remove(temp_framework_log_path)

    if all_episode_metrics:
        results_df = pd.DataFrame(all_episode_metrics)
        final_csv_path = os.path.join(args.output_dir, f"results_{args.strategy}.csv")
        results_df.to_csv(final_csv_path, index=False, float_format='%.4f')
        print(f"\n{Fore.GREEN}--- Evaluation Complete: {args.strategy} ---{Style.RESET_ALL}")
        print(f"Results for {args.num_episodes} episodes saved to: {final_csv_path}")
    else:
        print(f"\n{Fore.YELLOW}Warning: No metrics were collected. Evaluation may have failed.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()