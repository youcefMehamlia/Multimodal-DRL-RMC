#!/bin/bash

# This script runs the full evaluation campaign.

# --- CONFIGURATION ---
NUM_EPISODES=3 # Number of episodes to run per strategy
SEED=42
OUTPUT_DIR="./evaluation_results"
# IMPORTANT: Update this path to your best trained model
MODEL_PATH="./save/1ramp_1x3/DuelingDoubleDQNAgent_lr0.0001_model.pack" 

# --- EXECUTION ---
echo "Starting evaluation campaign..."

# The > sumo_log.txt part redirects the standard output of the python script
# (which includes SUMO's output) to the file sumo_log.txt.
# This allows our script to parse it after each episode.
python3 evaluate.py \
    --num_episodes $NUM_EPISODES \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --model_path "$MODEL_PATH" \
    --sumo_log_path "./sumo_log.txt" \
    > ./sumo_log.txt 2>&1

echo "Evaluation campaign finished. Results are in $OUTPUT_DIR"