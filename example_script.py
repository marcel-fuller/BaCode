#!/usr/bin/env python
"""
Example script for running parameter studies.

This script demonstrates how to use the parameter study framework
to investigate how different parameters affect causal discovery
and effect estimation.
"""

import json
import sys
import numpy as np
from parameter_study_pipeline import ParameterStudy, run_study_from_config

# Available configurations
CONFIGS = {
    'quick': {
        'N': 3,               # Number of variables (reduced for speed)
        'T': 300,             # Time series length (reduced for speed)
        'n_datasets': 2,      # Fewer datasets for quick testing
        'n_boot': 20,         # Fewer bootstrap samples for quick testing
        'auto_coeffs': [0.5, 0.9],
        'cross_coeffs': [0.2, 0.6],
        'noise_levels': [0.3, 1.0]
    },
    'standard': {
        'N': 4,               # Number of variables
        'T': 500,             # Time series length
        'n_datasets': 5,      # More datasets for better statistics
        'n_boot': 50,         # Standard bootstrap size
        'auto_coeffs': [0.2, 0.6, 0.95],
        'cross_coeffs': [0.1, 0.4, 0.7],
        'noise_levels': [0.2, 0.8, 1.5]
    },
    'full': {
        'N': 5,               # More variables for more complex interactions
        'T': 1000,            # Longer time series for better stability
        'n_datasets': 10,     # Many datasets for robust statistics
        'n_boot': 100,        # More bootstrap samples for better CI estimation
        'auto_coeffs': [0.2, 0.4, 0.6, 0.8, 0.95],
        'cross_coeffs': [0.1, 0.25, 0.4, 0.55, 0.7],
        'noise_levels': [0.2, 0.5, 0.8, 1.1, 1.5]
    }
}

def run_from_preset(preset='standard', param_to_study='all'):
    """
    Run a parameter study using a preset configuration.
    
    Parameters
    ----------
    preset : str
        Configuration preset ('quick', 'standard', or 'full')
    param_to_study : str
        Parameter to study ('auto', 'cross', 'noise', or 'all')
        
    Returns
    -------
    dict
        Study results
    """
    if preset not in CONFIGS:
        print(f"Error: Unknown preset '{preset}'. Available presets: {list(CONFIGS.keys())}")
        sys.exit(1)
    
    # Get configuration
    config = CONFIGS[preset].copy()
    config['parameter_to_study'] = param_to_study
    
    # Create temporary config file
    with open(f'config_{preset}_{param_to_study}.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run study
    print(f"Running {param_to_study} study with '{preset}' preset...")
    results = run_study_from_config(f'config_{preset}_{param_to_study}.json')
    
    return results

def run_custom_study():
    """
    Run a parameter study with custom configuration.
    
    Returns
    -------
    dict
        Study results
    """
    # Create custom study to compare bootstrap sizes
    print("Running custom study comparing bootstrap sizes...")
    
    # Create parameter study object
    study = ParameterStudy(
        N=4,
        T=500,
        n_datasets=3,
        n_boot=50,  # Default bootstrap size
        base_auto_coeff=0.8,
        base_cross_coeff=0.5,
        base_noise_sigma=0.5
    )
    
    # Run autocorrelation study with different bootstrap sizes
    bootstrap_sizes = [10, 30, 50, 100]
    results = {}
    
    for n_boot in bootstrap_sizes:
        print(f"\nTesting bootstrap size: {n_boot}")
        study.n_boot = n_boot
        results[n_boot] = study.run_auto_study([0.5, 0.9])
    
    return results

def run_timing_comparison():
    """
    Run a focused timing comparison of the three approaches.
    
    Returns
    -------
    dict
        Study results
    """
    # Create parameter study with standard settings
    study = ParameterStudy(
        N=4,
        T=500,
        n_datasets=3,
        n_boot=50,
        base_auto_coeff=0.8,
        base_cross_coeff=0.4,
        base_noise_sigma=0.5
    )
    
    # Run a single parameter study
    results = study.run_auto_study([0.8])
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter studies with different configurations.')
    parser.add_argument('--preset', type=str, choices=['quick', 'standard', 'full'], 
                        default='quick', help='Configuration preset')
    parser.add_argument('--param', type=str, choices=['auto', 'cross', 'noise', 'all'], 
                        default='auto', help='Parameter to study')
    parser.add_argument('--custom', action='store_true', 
                        help='Run custom bootstrap size comparison')
    parser.add_argument('--timing', action='store_true',
                        help='Run focused timing comparison')
    
    args = parser.parse_args()
    
    if args.custom:
        results = run_custom_study()
    elif args.timing:
        results = run_timing_comparison()
    else:
        results = run_from_preset(args.preset, args.param)
    
    print("Study complete!")
