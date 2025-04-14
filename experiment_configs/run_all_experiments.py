#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import argparse
import time
from datetime import datetime

def run_experiment_configs(config_file, output_dir, modes=None, episodes=None):
    """
    Run a batch of experiments from a JSON configuration file
    
    Parameters:
        config_file: Path to JSON file containing array of configuration objects
        output_dir: Directory to save output
        modes: List of modes to run each configuration with, or None to use default
        episodes: Override number of episodes in the configs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration file
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    if not configs:
        print(f"No configurations found in {config_file}")
        return
    
    # Default modes to run if not specified
    if modes is None:
        modes = ["full", "random_all", "random_ua", "random_rec", "random_ca"]
    
    # Log file
    log_path = os.path.join(output_dir, f"{os.path.basename(config_file).split('.')[0]}_log.txt")
    
    with open(log_path, 'w') as log_file:
        log_file.write(f"Experiment batch started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Configuration file: {config_file}\n")
        log_file.write(f"Number of configurations: {len(configs)}\n")
        log_file.write(f"Modes: {', '.join(modes)}\n\n")
        
        # Run each configuration with each mode
        for i, config in enumerate(configs):
            # Extract description for naming
            description = config.get("description", f"config_{i}")
            safe_desc = description.replace(":", "_").replace(" ", "_")
            
            # Override episodes if specified
            if episodes is not None:
                config["episodes"] = episodes
            
            # Save the individual config
            config_path = os.path.join(output_dir, f"{safe_desc}.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            log_file.write(f"[{i+1}/{len(configs)}] Running: {description}\n")
            
            # Run with each mode
            for mode in modes:
                start_time = time.time()
                log_file.write(f"  - Mode: {mode}\n")
                
                # Build command
                cmd = f"python train.py --config {config_path} --mode {mode}"
                log_file.write(f"  - Command: {cmd}\n")
                log_file.flush()
                
                try:
                    print(f"Running: {cmd}")
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    
                    # Save stdout and stderr
                    stdout_path = os.path.join(output_dir, f"{safe_desc}_{mode}_stdout.txt")
                    stderr_path = os.path.join(output_dir, f"{safe_desc}_{mode}_stderr.txt")
                    
                    with open(stdout_path, 'wb') as f:
                        f.write(stdout)
                    with open(stderr_path, 'wb') as f:
                        f.write(stderr)
                    
                    elapsed_time = time.time() - start_time
                    log_file.write(f"  - Status: Completed in {elapsed_time:.2f} seconds\n")
                    print(f"Completed: {description} - {mode} in {elapsed_time:.2f} seconds")
                except Exception as e:
                    log_file.write(f"  - Status: Failed - {str(e)}\n")
                    print(f"Failed: {description} - {mode} - {str(e)}")
            
            log_file.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Run experiments based on configuration files")
    parser.add_argument("--config_files", nargs="+", default=None,
                        help="Configuration files to run (defaults to all in experiment_configs)")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Modes to run (defaults to all 5 modes)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes for faster testing")
    parser.add_argument("--figure", type=int, default=None,
                        help="Run configuration for specific figure (4-9)")
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Determine which config files to run
    config_files = []
    
    if args.figure is not None:
        if args.figure < 4 or args.figure > 9:
            print("Figure number must be between 4 and 9")
            return
        config_files = [f"experiment_configs/fig{args.figure}_{'cache_capacity' if args.figure == 4 else 'user_count' if args.figure == 5 else 'video_count' if args.figure == 6 else 'bandwidth' if args.figure == 7 else 'recommendation_size' if args.figure == 8 else 'deviation_tolerance'}.json"]
    elif args.config_files:
        config_files = args.config_files
    else:
        # Find all JSON files in experiment_configs directory
        for file in os.listdir("experiment_configs"):
            if file.endswith(".json"):
                config_files.append(os.path.join("experiment_configs", file))
    
    if not config_files:
        print("No configuration files found")
        return
    
    print(f"Running experiments with the following configurations:")
    for cf in config_files:
        print(f"  - {cf}")
    
    # Run each configuration file
    for config_file in config_files:
        file_output_dir = os.path.join(output_dir, os.path.basename(config_file).split('.')[0])
        run_experiment_configs(config_file, file_output_dir, args.modes, args.episodes)
    
    print(f"\nExperiments completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 