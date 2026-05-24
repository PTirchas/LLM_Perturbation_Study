# Copyright (C) 2026 Panagiotis Tirchas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
LLM Perturbation Stability Experiment - 3-Tier Benchmark Edition

Implements the experimental protocols outlined in "Review and Improve LLMs":
  - Tier 1: Core benchmark (6 families, 8 instances, 5 replicates)
  - Tier 2: Deep boundary analysis (dense sweeps on subset of patients)
  - Tier 3: Cross-dataset extension (Breast cancer dataset)

# Tier 1: The Core Benchmark
python main.py --tier 1
# Tier 2: Deeper Boundary Sweeps
python main.py --tier 2
# Tier 3: Cross-Dataset Extensions
python main.py --tier 3

python main.py --tier 1 --run-full
python main.py --tier 1 --compact
python main.py --tier 1 --compact --run-full
python main.py --tier 2 --run-full

"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

from utils.config import load_config, print_config
from utils.data import load_data
from utils.experiment import run_experiment
from utils.llm import initialize_client
from utils.perturbations import PERTURBATIONS
from utils.results import summarize_results, save_results
from utils.tier_selection import select_tier2_patients

test = -1
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Perturbation Stability Benchmark")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1,
                        help="Which experimental tier to run (1=Core, 2=Boundary, 3=Cross-dataset)")
    parser.add_argument("--compact", action="store_true",
                        help="Run the compact version of Tier 1 (4 families, fewer instances)")
    parser.add_argument("--test-mode", action="store_true", default=True,
                        help="Run an extremely minimal slice to avoid massive LLM billing costs. On by default via this script for safety.")
    parser.add_argument("--run-full", action="store_true",
                        help="Acknowledge the costs and run the full scale experiment. Overrides --test-mode.")
    return parser.parse_args()


def get_tier_settings(args, config):
    """Adjusts config and selects families/dataset based on the Tier."""
    
    # Defaults
    dataset_path = "data/test_ready.csv"
    families_to_run = list(PERTURBATIONS.keys())
    
    if args.tier == 1:
        print("--- Configuring Tier 1: Core Benchmark ---")
        if args.compact:
            # 4 families, 6 perts, 3 reps, 2 retrieval conditions (compact)
            config.num_perturbations = 6
            config.num_stochastic_replicates = 7
            families_to_run = ["A_order", "B_instruction", "C_json", "D_numeric_precision"]
        else:
            # 6 families, 8 perts, 5 reps
            config.num_perturbations = 8
            config.num_stochastic_replicates = 5
            families_to_run = ["A_order", "B_instruction", "C_json", "D_numeric_precision", "F_retrieval", "Composite"]
            
    elif args.tier == 2:
        print("--- Configuring Tier 2: Boundary Analysis ---")
        config.num_perturbations = 30  # Dense sweeps
        config.num_stochastic_replicates = 10
        families_to_run = ["A_order", "B_instruction", "C_json", "D_numeric_precision"]#, "Composite"]
        
    elif args.tier == 3:
        print("--- Configuring Tier 3: Cross-Dataset Extension ---")
        dataset_path = "data/breast_cancer_ready.csv"
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset {dataset_path} not found.")
            print("Please ensure you generated the secondary dataset using scikit-learn.")
            sys.exit(1)
        config.num_perturbations = 5
        config.num_stochastic_replicates = 3
        
    # Safety mechanism to prevent $100+ bills accidentally
    test_mode = args.test_mode and not args.run_full
    if test_mode:
        print("\n!!! WARNING: RUNNING IN TEST MODE !!!")
        print("  Caps execution to 2 patients, 2 perts, 1 replicate to save costs.")
        print("  Pass --run-full to run the actual massive benchmark.\n")
        config.num_perturbations = 2
        config.num_stochastic_replicates = 1
        
    config.tier = args.tier
    return config, dataset_path, families_to_run


def main():
    args = parse_args()

    # Load base configuration
    config = load_config()
    
    # Override settings for specific tiers
    config, dataset_path, families_to_run = get_tier_settings(args, config)
    print_config(config)

    # Initialize LLM client
    print(f"Initializing {config.llm_provider.value}...")
    client = initialize_client(config.llm_provider, config.azure_api_version)

    # Load patient data
    print(f"Loading patient data from {dataset_path}...")
    predict_df = load_data(dataset_path)
    if test > 0:
        predict_df = predict_df.iloc[0:test]
    if args.tier == 2:
        # For tier 2, select patients with highest uncertainty from tier 1 results
        # Load from the correct tier 1 file (with or without context) based on retrieval setting
        if config.use_retrieval:
            tier1_files = sorted(glob.glob('outputs/output_tier_1_with_context*.json'))
        else:
            tier1_files = sorted(glob.glob('outputs/output_tier_1_without_context*.json'))
        
        if tier1_files:
            latest_tier1_file = tier1_files[-1]
            retrieval_label = "with context" if config.use_retrieval else "without context"
            print(f"Found tier 1 results ({retrieval_label}): {latest_tier1_file}")
            print("Selecting patients with highest semantic uncertainty for tier 2...\n")
            
            # Get the top 20 most unstable patients
            tier2_patient_indices = select_tier2_patients(latest_tier1_file, num_patients=20)
            
            # Filter to only those patients
            predict_df = predict_df.loc[predict_df.index.isin(tier2_patient_indices)]
            print(f"Selected {len(predict_df)} patients for tier 2 analysis")
        else:
            print("Warning: No tier 1 results found. Using default slice (patients 10-25)")
            predict_df = predict_df.iloc[10:25] 
    
    # Apply test mode cap
    if args.test_mode and not args.run_full:
        predict_df = predict_df.iloc[0:2]

    # Get feature names (exclude non-feature columns)
    exclude_cols = {"patient_diagnosis", "Patient", "sample_id"}
    feature_names = [c for c in predict_df.columns if c not in exclude_cols]

    print(f"Loaded {len(predict_df)} patients to test with {len(feature_names)} features\n")

    # Filter PERTURBATIONS to only run the requested ones
    global PERTURBATIONS
    active_perts = {k: v for k, v in PERTURBATIONS.items() if k in families_to_run}
    
    # Note: We must temporarily override the global dict because utils/experiment.py imports it directly. 
    # A cleaner long-term refactor would pass this dict as an argument.
    import utils.experiment
    utils.experiment.PERTURBATIONS = active_perts

    # Load context data for retrieval perturbations (Family F) if enabled
    context_df = None
    if config.use_retrieval and args.tier != 3:
        context_data_path = "data/context_ready.csv"
        print(f"Loading context data from {context_data_path}...")
        context_df = load_data(context_data_path)
        print(f"Loaded {len(context_df)} context records\n")
    elif args.tier == 3:
        print("Skipping retrieval perturbations for cross-dataset (requires separate context bank).")
        config.use_retrieval = False
    else:
        print("Retrieval perturbations (Family F): DISABLED via .env config.")

    # Run experiment
    print("Running experiments...")
    results = run_experiment(client, config, predict_df, feature_names, context_df)

    # Print results
    summarize_results(results)

    # Save results
    save_results(results, config)


if __name__ == "__main__":
    main()
