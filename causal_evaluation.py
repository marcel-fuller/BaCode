"""
Evaluation and pipeline functionality for causal discovery and effect estimation.

This module provides functions for evaluating causal discovery and effect
estimation results, loading/saving data, and running full pipelines.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.cluster import hierarchy

from causal_core import (
    create_causal_model,
    create_stable_model,
    generate_dataset,
    run_causal_discovery,
    estimate_all_effects,
    estimate_all_effects_with_confidence
)


# =====================================================================
# Data Loading Functions
# =====================================================================

def load_dataset(file_path):
    """
    Load a dataset from file.
    
    Parameters
    ----------
    file_path : str
        Path to dataset file
        
    Returns
    -------
    tuple
        (data, metadata) - Loaded data array and metadata dictionary
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load NumPy array data
    if file_path.endswith('_data.npy'):
        data = np.load(file_path)
        metadata_path = file_path.replace('_data.npy', '_metadata.json')
    else:
        # Assume file is the base path
        data = np.load(f"{file_path}_data.npy")
        metadata_path = f"{file_path}_metadata.json"
    
    # Load metadata if available
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Create default metadata
        metadata = {
            'var_names': [f'X{i+1}' for i in range(data.shape[1])],
            'T': data.shape[0]
        }
    
    return data, metadata


def load_discovery_results(base_path):
    """
    Load causal discovery results from files.
    
    Parameters
    ----------
    base_path : str
        Base path to discovery result files
        
    Returns
    -------
    dict
        Discovery results
    """
    results = {}
    
    # Load PCMCI results
    pcmci_graph_path = f"{base_path}_pcmci_graph.npy"
    if os.path.exists(pcmci_graph_path):
        graph = np.load(pcmci_graph_path)
        val_matrix_path = f"{base_path}_pcmci_val_matrix.npy"
        val_matrix = np.load(val_matrix_path) if os.path.exists(val_matrix_path) else None
        
        # Look for metadata
        metadata_path = f"{base_path}_pcmci_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        results['pcmci'] = {
            'graph': graph, 
            'val_matrix': val_matrix,
            'metadata': metadata
        }
    
    # Load bootstrap results
    bootstrap_path = f"{base_path}_bootstrap_bagged_graph.npy"
    if os.path.exists(bootstrap_path):
        bagged_graph = np.load(bootstrap_path)
        
        # Look for boot graphs
        boot_graphs_path = f"{base_path}_bootstrap_boot_graphs.npy"
        if os.path.exists(boot_graphs_path):
            boot_graphs = np.load(boot_graphs_path)
            
            # Look for metadata
            metadata_path = f"{base_path}_bootstrap_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            results['bootstrap'] = {
                'boot_results': {'graph': boot_graphs},
                'summary_results': {'most_frequent_links': bagged_graph},
                'metadata': metadata
            }
            
            results['bagged_graph'] = bagged_graph
    
    return results


def load_effect_results(base_path):
    """
    Load causal effect estimation results from files.
    
    Parameters
    ----------
    base_path : str
        Base path to effect estimation result files
        
    Returns
    -------
    dict
        Effect estimation results
    """
    # Look for overall results file
    all_path = f"{base_path}_all.json"
    if False:#os.path.exists(all_path):
        with open(all_path, 'r') as f:
            return json.load(f)
    
    # If overall file doesn't exist, try to load individual files
    results = {}
    
    # Load True_graph effect
    true_graph_path = f"{base_path}_true_graph.json"
    if os.path.exists(true_graph_path):
        with open(true_graph_path, 'r') as f:
            true_graph_data = json.load(f)
            results['true_graph_effect'] = true_graph_data.get('effect')
    
    # Load PCMCI effect
    pcmci_path = f"{base_path}_pcmci.json"
    if os.path.exists(pcmci_path):
        with open(pcmci_path, 'r') as f:
            pcmci_data = json.load(f)
            results['pcmci_effect'] = pcmci_data.get('effect')
    
    # Load bagged effect
    bagged_path = f"{base_path}_bagged.json"
    if os.path.exists(bagged_path):
        with open(bagged_path, 'r') as f:
            bagged_data = json.load(f)
            results['bagged_effect'] = bagged_data.get('effect')
    
    # Load bootstrap effects
    bootstrap_path = f"{base_path}_bootstrap.json"
    if os.path.exists(bootstrap_path):
        with open(bootstrap_path, 'r') as f:
            bootstrap_data = json.load(f)
            results['bootstrap_effects'] = bootstrap_data.get('bootstrap_effects')
            results['bootstrap_stats'] = bootstrap_data.get('stats')
    
    return results


# Modified version of load_experiment_results that tracks file paths
def modified_load_experiment_results(directory, param_name=None, param_value=None):
    """
    Load experiment results from a directory, including file paths.
    
    Parameters
    ----------
    directory : str
        Directory containing experiment results
    param_name : str, optional
        Filter by parameter name
    param_value : float, optional
        Filter by parameter value
        
    Returns
    -------
    dict
        Dictionary with experiment results
    """
    # Define path pattern based on filters
    if param_name and param_value:
        pattern = os.path.join(directory, f"{param_name}_{param_value}", "*_all.json")
    elif param_name:
        pattern = os.path.join(directory, f"{param_name}_*", "*_all.json")
    else:
        pattern = os.path.join(directory, "*", "*_all.json")
    
    # Find all matching result files
    results_files = glob.glob(pattern)
    
    # Load data from each file
    experiment_results = {}
    for file_path in results_files:
        # Extract parameter info from directory path
        parts = file_path.split(os.sep)
        param_dir = parts[-2]  # Directory name contains parameter info
        
        param_parts = param_dir.split('_')
        if len(param_parts) >= 2:
            current_param_name = param_parts[0]
            try:
                current_param_value = float(param_parts[1])
            except ValueError:
                # Skip if parameter value is not a number
                continue
        else:
            # Skip if directory doesn't follow the expected pattern
            continue
        
        # Skip if doesn't match filters
        if param_name and current_param_name != param_name:
            continue
        if param_value is not None and current_param_value != param_value:
            continue
        
        # Read results
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Add file path to results
        results['_file_path'] = file_path
        
        # Look for corresponding bootstrap file
        bootstrap_file = file_path.replace('_all.json', '_bootstrap.json')
        if os.path.exists(bootstrap_file):
            try:
                with open(bootstrap_file, 'r') as f:
                    bootstrap_data = json.load(f)
                    # Add bootstrap effects directly to results
                    if 'bootstrap_effects' in bootstrap_data:
                        results['bootstrap_effects'] = bootstrap_data['bootstrap_effects']
            except Exception as e:
                print(f"Error loading bootstrap file {bootstrap_file}: {e}")
        
        # Extract effect pair from filename
        effect_key = os.path.basename(file_path).split('_all.json')[0]
        if effect_key.startswith('effects_'):
            effect_key = effect_key[8:]  # Remove 'effects_' prefix
        
        # Store results
        if current_param_value not in experiment_results:
            experiment_results[current_param_value] = {}
        
        experiment_results[current_param_value][effect_key] = results
    
    return experiment_results

def load_experiment_results(directory, param_name=None, param_value=None):
    """
    Load experiment results from a directory.
    
    Parameters
    ----------
    directory : str
        Directory containing experiment results
    param_name : str, optional
        Filter by parameter name
    param_value : float, optional
        Filter by parameter value
        
    Returns
    -------
    dict
        Dictionary with experiment results
    """
    # Define path pattern based on filters
    if param_name and param_value:
        pattern = os.path.join(directory, f"{param_name}_{param_value}", "*_all.json")
    elif param_name:
        pattern = os.path.join(directory, f"{param_name}_*", "*_all.json")
    else:
        pattern = os.path.join(directory, "*", "*_all.json")
    
    # Find all matching result files
    results_files = glob.glob(pattern)
    
    # Load data from each file
    experiment_results = {}
    for file_path in results_files:
        # Extract parameter info from directory path
        parts = file_path.split(os.sep)
        param_dir = parts[-2]  # Directory name contains parameter info
        
        param_parts = param_dir.split('_')
        current_param_name = param_parts[0]
        current_param_value = float(param_parts[1])
        
        # Skip if doesn't match filters
        if param_name and current_param_name != param_name:
            continue
        if param_value and current_param_value != param_value:
            continue
        
        # Read results
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Extract effect pair from filename
        effect_key = os.path.basename(file_path).split('_all.json')[0]
        if effect_key.startswith('effects_'):
            effect_key = effect_key[8:]  # Remove 'effects_' prefix
        
        # Store results
        if current_param_value not in experiment_results:
            experiment_results[current_param_value] = {}
        
        experiment_results[current_param_value][effect_key] = results
    
    return experiment_results


def load_all_results_to_dataframe(directory):
    """
    Load all experiment results from a directory into a DataFrame.
    
    Parameters
    ----------
    directory : str
        Directory containing experiment results
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all experiment results
    """
    # Find all experiment results
    all_results_files = glob.glob(os.path.join(directory, "*", "effects_*_all.json"))
    
    # Extract data from each file
    data = []
    for file_path in all_results_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Extract parameter values from directory path
        parts = file_path.split(os.sep)
        param_dir = parts[-2]  # Directory name contains parameter info
        
        param_parts = param_dir.split('_')
        param_name = param_parts[0]
        param_value = float(param_parts[1])
        
        # Extract effect pair from filename
        effect_str = os.path.basename(file_path).split('_all.json')[0]
        if effect_str.startswith('effects_'):
            effect_str = effect_str[8:]  # Remove 'effects_' prefix
        
        # Extract X and Y from string
        try:
            x_part, y_part = effect_str.split('_to_')
            X_var, X_lag = map(int, x_part.split('_'))
            Y_var, Y_lag = map(int, y_part.split('_'))
            
            # Format X and Y
            X = (X_var, X_lag)
            Y = (Y_var, Y_lag)
        except:
            # If parsing fails, use the original string
            X = results.get('X')
            Y = results.get('Y')
        
        # Create row
        row = {
            'param_name': param_name,
            'param_value': param_value,
            'X': X,
            'Y': Y,
            'effect_key': effect_str,
            'true_effect': results.get('true_effect'),
            'true_graph_effect': results.get('true_graph_effect'),
            'pcmci_effect': results.get('pcmci_effect'),
            'bagged_effect': results.get('bagged_effect')
        }
        
        # Add bootstrap statistics if available
        bootstrap_stats = results.get('bootstrap_stats', {})
        if bootstrap_stats:
            row.update({
                'bootstrap_mean': bootstrap_stats.get('mean'),
                'bootstrap_std': bootstrap_stats.get('std'),
                'bootstrap_ci_lower': bootstrap_stats.get('ci_lower'),
                'bootstrap_ci_upper': bootstrap_stats.get('ci_upper'),
                'bootstrap_success_rate': bootstrap_stats.get('estimation_success_rate')
            })
        
        # Add timing information if available
        timings = results.get('timings', {})
        if timings:
            for method, time_val in timings.items():
                row[f'{method}_time'] = time_val
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add error columns
    for method in ['pcmci', 'bagged', 'bootstrap']:
        if method == 'bootstrap':
            effect_col = 'bootstrap_mean'
        else:
            effect_col = f'{method}_effect'
        
        # Skip if column doesn't exist
        if effect_col not in df.columns:
            continue
        
        # Calculate errors
        df[f'{method}_error'] = df[effect_col] - df['true_effect']
        df[f'{method}_abs_error'] = np.abs(df[f'{method}_error'])
        df[f'{method}_squared_error'] = df[f'{method}_error'] ** 2
        
        # Flag if true effect is within CI (for bootstrap)
        if method == 'bootstrap' and 'bootstrap_ci_lower' in df.columns:
            df['ci_covers_true'] = (df['true_effect'] >= df['bootstrap_ci_lower']) & \
                                  (df['true_effect'] <= df['bootstrap_ci_upper'])
    
    return df


# =====================================================================
# Evaluation Functions
# =====================================================================

def calculate_metrics(results_df, group_by=None):
    """
    Calculate evaluation metrics from results DataFrame.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results
    group_by : str or list, optional
        Column(s) to group by
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated metrics
    """
    # Define metrics to calculate
    metrics = {}
    for method in ['true_graph','pcmci', 'bagged', 'bootstrap']:
        abs_error_col = f'{method}_abs_error'
        squared_error_col = f'{method}_squared_error'
        error_col = f'{method}_error'
        
        # Skip if column doesn't exist
        if abs_error_col not in results_df.columns:
            continue
        
        metrics[f'{method}_mae'] = pd.NamedAgg(column=abs_error_col, aggfunc='mean')
        metrics[f'{method}_rmse'] = pd.NamedAgg(column=squared_error_col, aggfunc=lambda x: np.sqrt(np.mean(x)))
        metrics[f'{method}_bias'] = pd.NamedAgg(column=error_col, aggfunc='mean')
    
    # Add CI coverage for bootstrap
    if 'ci_covers_true' in results_df.columns:
        metrics['bootstrap_ci_coverage'] = pd.NamedAgg(column='ci_covers_true', aggfunc='mean')

    # Group if specified
    if group_by:
        return results_df.groupby(group_by).agg(**metrics).reset_index()
    else:
        # Calculate overall metrics
        result = pd.DataFrame([results_df.agg(metrics)])
        return result


def evaluate_estimation_performance(all_effects):
    """
    Evaluate causal effect estimation performance across datasets.
    
    Parameters
    ----------
    all_effects : dict
        Dictionary of effect estimation results
        
    Returns
    -------
    dict
        Dictionary of evaluation results by effect pair and method
    """
    results = {}
    
    # Collect all effect pairs
    effect_pairs = set()
    for dataset_effects in all_effects.values():
        effect_pairs.update(dataset_effects.keys())
    
    for effect_pair in effect_pairs:
        pair_results = {
            'pcmci': {'values': [], 'errors': []},
            'bagged': {'values': [], 'errors': []},
            'true_graph': {'values': [], 'errors': []},
            'bootstrap': {'values': [], 'errors': [], 'ci_coverage': []}
        }
        
        # Collect values and errors for each dataset
        for dataset_id, dataset_effects in all_effects.items():
            if effect_pair in dataset_effects:
                effect_data = dataset_effects[effect_pair]
                
                true_effect = effect_data.get('true_effect')
                if true_effect is None:
                    continue
                
                true_graph_effect = effect_data.get('true_graph_effect')
                if true_graph_effect is not None and not np.isnan(true_graph_effect):
                    pair_results['true_graph']['values'].append(true_graph_effect)
                    pair_results['true_graph']['errors'].append(true_graph_effect - true_effect)
                
                # PCMCI
                pcmci_effect = effect_data.get('pcmci_effect')
                if pcmci_effect is not None and not np.isnan(pcmci_effect):
                    pair_results['pcmci']['values'].append(pcmci_effect)
                    pair_results['pcmci']['errors'].append(pcmci_effect - true_effect)
                
                # Bagged
                bagged_effect = effect_data.get('bagged_effect')
                if bagged_effect is not None and not np.isnan(bagged_effect):
                    pair_results['bagged']['values'].append(bagged_effect)
                    pair_results['bagged']['errors'].append(bagged_effect - true_effect)
                
                # Bootstrap
                bootstrap_stats = effect_data.get('bootstrap_stats', {})
                bootstrap_mean = bootstrap_stats.get('mean')
                if bootstrap_mean is not None and not np.isnan(bootstrap_mean):
                    pair_results['bootstrap']['values'].append(bootstrap_mean)
                    pair_results['bootstrap']['errors'].append(bootstrap_mean - true_effect)
                    
                    # Check CI coverage
                    ci_lower = bootstrap_stats.get('ci_lower')
                    ci_upper = bootstrap_stats.get('ci_upper')
                    if ci_lower is not None and ci_upper is not None:
                        covers = (true_effect >= ci_lower) and (true_effect <= ci_upper)
                        pair_results['bootstrap']['ci_coverage'].append(int(covers))
        
        # Calculate metrics for each method
        pair_metrics = {}
        for method in ['pcmci', 'bagged', 'bootstrap','true_graph']:
            method_metrics = {}
            
            errors = np.array(pair_results[method]['errors'])
            if len(errors) > 0:
                method_metrics['mae'] = np.mean(np.abs(errors))
                method_metrics['rmse'] = np.sqrt(np.mean(errors**2))
                method_metrics['bias'] = np.mean(errors)
            
            if method == 'bootstrap' and len(pair_results[method]['ci_coverage']) > 0:
                method_metrics['ci_coverage'] = np.mean(pair_results[method]['ci_coverage'])
            
            pair_metrics[method] = method_metrics
        
        results[effect_pair] = pair_metrics
    
    return results


# =====================================================================
# Pipeline Functions
# =====================================================================

def run_pipeline(auto_coeff=0.8, cross_coeff=0.5, noise_sigma=0.5, n_vars=4, T=500, 
                pc_alpha=0.05, tau_max=5, n_boot=100, effect_pairs=None, 
                output_dir=None, seed=None):
    """
    Run complete causal discovery and effect estimation pipeline with adjustment set tracking.
    
    Parameters
    ----------
    auto_coeff : float
        Coefficient for auto-regressive links
    cross_coeff : float
        Coefficient for cross-variable links
    noise_sigma : float
        Standard deviation of noise
    n_vars : int
        Number of variables
    T : int
        Length of time series
    pc_alpha : float
        Significance level for PCMCI
    tau_max : int
        Maximum time lag
    n_boot : int
        Number of bootstrap samples
    effect_pairs : list, optional
        List of (X, Y) pairs to estimate effects for
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Pipeline results
    """
    start_time = time.time()
    
    # Set default effect pairs if not provided
    if effect_pairs is None:
        effect_pairs = [((0, -3), (3, 0))]  # X1(t-3) → X4(t)
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameters
        params = {
            'auto_coeff': auto_coeff,
            'cross_coeff': cross_coeff,
            'noise_sigma': noise_sigma,
            'n_vars': n_vars,
            'T': T,
            'pc_alpha': pc_alpha,
            'tau_max': tau_max,
            'n_boot': n_boot,
            'seed': seed,
            'effect_pairs': effect_pairs,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
            json.dump(params, f, indent=4)
    
    # Create causal model
    links, noises = create_causal_model(auto_coeff=auto_coeff, cross_coeff=cross_coeff, noise_sigma=noise_sigma)
    
    # Generate dataset
    dataset, data_time = generate_dataset(
        links, T, noises, seed, 
        save_path=os.path.join(output_dir, 'dataset') if output_dir else None
    )
    
    # Run causal discovery
    discovery_results, discovery_times = run_causal_discovery(
        dataset, pc_alpha, tau_max, n_boot, 
        save_path=os.path.join(output_dir, 'discovery') if output_dir else None
    )
    
    # Estimate causal effects
    effect_results = {}
    adjustment_sets = {}
    confidence_intervals = {}
    effect_times = {}
    
    for X, Y in effect_pairs:
        # Format effect pair as string
        effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
        
        # Estimate effects with confidence intervals
        effects, adj_sets, cis, times = estimate_all_effects_with_confidence(
            discovery_results, dataset, X, Y, links, noises,
            save_path=os.path.join(output_dir, f'effects_{effect_key}') if output_dir else None
        )
        
        effect_results[effect_key] = effects
        adjustment_sets[effect_key] = adj_sets
        confidence_intervals[effect_key] = cis
        effect_times[effect_key] = times
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Save overall timing information
    if output_dir:
        timing_summary = {
            'total_time': total_time,
            'data_generation_time': data_time,
            'discovery_times': discovery_times,
            'effect_times': effect_times
        }
        
        with open(os.path.join(output_dir, 'timing_summary.json'), 'w') as f:
            json.dump(timing_summary, f, indent=4)
    
    # Return all results
    return {
        'links': links,
        'dataset': dataset,
        'discovery': discovery_results,
        'effects': effect_results,
        'adjustment_sets': adjustment_sets,
        'confidence_intervals': confidence_intervals,
        'timings': {
            'total': total_time,
            'data_generation': data_time,
            'discovery': discovery_times,
            'effects': effect_times
        }
    }


def run_parameter_study(param_name, param_values, base_params=None, output_dir=None, seed=None):
    """
    Run parameter study varying one parameter.
    
    Parameters
    ----------
    param_name : str
        Name of parameter to vary ('auto', 'cross', or 'noise')
    param_values : list
        List of parameter values to test
    base_params : dict, optional
        Base parameters for the study
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Base random seed
        
    Returns
    -------
    dict
        Study results
    """
    # Set default base parameters if not provided
    if base_params is None:
        base_params = {
            'auto_coeff': 0.5,
            'cross_coeff': 0.5,
            'noise_sigma': 0.5,
            'n_vars': 4,
            'T': 500,
            'pc_alpha': 0.05,
            'tau_max': 5,
            'n_boot': 100,
            'effect_pairs': [((0, -3), (3, 0))]  # X1(t-3) → X4(t)
        }
    
    # Create timestamp for output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f"parameter_study_{param_name}_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save study parameters
    study_params = {
        'param_name': param_name,
        'param_values': param_values,
        'base_params': base_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    with open(os.path.join(output_dir, 'study_parameters.json'), 'w') as f:
        json.dump(study_params, f, indent=4)
    
    # Run pipeline for each parameter value
    results = {}
    overall_timing = {}
    
    # Track run indices for each parameter value
    run_indices = {}
    
    for i, param_value in enumerate(param_values):
        print(f"\nRunning {param_name}={param_value} ({i+1}/{len(param_values)})")
        
        # Set parameters for this run
        params = base_params.copy()
        
        if param_name == 'auto':
            params['auto_coeff'] = param_value
        elif param_name == 'cross':
            params['cross_coeff'] = param_value
        elif param_name == 'noise':
            params['noise_sigma'] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        # Track run index for this parameter value
        if param_value not in run_indices:
            run_indices[param_value] = 0
        else:
            run_indices[param_value] += 1
        run_idx = run_indices[param_value]
        
        # Create parameter-specific output directory with run index
        param_dir = os.path.join(output_dir, f"{param_name}_run{run_idx}_{param_value}")
        os.makedirs(param_dir, exist_ok=True)
        
        
        # Set seed based on parameter value and base seed
        if seed is not None:
            run_seed = seed + i
        else:
            run_seed = None
        
        # Run pipeline
        start_time = time.time()
        result = run_pipeline(
            auto_coeff=params['auto_coeff'],
            cross_coeff=params['cross_coeff'],
            noise_sigma=params['noise_sigma'],
            n_vars=params['n_vars'],
            T=params['T'],
            pc_alpha=params['pc_alpha'],
            tau_max=params['tau_max'],
            n_boot=params['n_boot'],
            effect_pairs=params['effect_pairs'],
            output_dir=param_dir,
            seed=run_seed
        )
        
        # Record total time
        total_time = time.time() - start_time
        overall_timing[str(param_value)] = total_time
        
        results[param_value] = result
        
        print(f"  Completed in {total_time:.2f} seconds")
    
    # Save overall timing
    with open(os.path.join(output_dir, 'overall_timing.json'), 'w') as f:
        json.dump(overall_timing, f, indent=4)
    
    return results


def run_full_study(auto_values=None, cross_values=None, noise_values=None, 
                  base_params=None, output_dir=None, seed=None):
    """
    Run comprehensive parameter study for autocorrelation, cross-link strength, and noise level.
    
    Parameters
    ----------
    auto_values : list, optional
        List of autocorrelation values to test
    cross_values : list, optional
        List of cross-link values to test
    noise_values : list, optional
        List of noise values to test
    base_params : dict, optional
        Base parameters for all studies
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Base random seed
        
    Returns
    -------
    dict
        Complete study results
    """
    # Set default values if not provided
    if auto_values is None:
        auto_values = [0.2, 0.5, 0.8, 0.95]
    if cross_values is None:
        cross_values = [0.1, 0.3, 0.5, 0.7]
    if noise_values is None:
        noise_values = [0.2, 0.5, 1.0, 1.5]
    
    # Create timestamp for output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f"full_parameter_study_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save study parameters
    study_params = {
        'auto_values': auto_values,
        'cross_values': cross_values,
        'noise_values': noise_values,
        'base_params': base_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    print(type(study_params))
    with open(os.path.join(output_dir, 'full_study_parameters.json'), 'w') as f:
        json.dump(study_params, f, indent=4)
    
    # Run each parameter study
    results = {}
    overall_timing = {}
    
    # Study 1: Autocorrelation
    print("\n== AUTOCORRELATION STUDY ==")
    start_time = time.time()
    results['auto'] = run_parameter_study(
        'auto', auto_values, base_params, 
        os.path.join(output_dir, 'auto_study'),
        seed=seed if seed is not None else None
    )
    auto_time = time.time() - start_time
    overall_timing['auto_study'] = auto_time
    print(f"Autocorrelation study completed in {auto_time:.2f} seconds")
    
    # Study 2: Cross-link strength
    print("\n== CROSS-LINK STUDY ==")
    start_time = time.time()
    results['cross'] = run_parameter_study(
        'cross', cross_values, base_params, 
        os.path.join(output_dir, 'cross_study'),
        seed=seed+1000 if seed is not None else None
    )
    cross_time = time.time() - start_time
    overall_timing['cross_study'] = cross_time
    print(f"Cross-link study completed in {cross_time:.2f} seconds")
    
    # Study 3: Noise level
    print("\n== NOISE LEVEL STUDY ==")
    start_time = time.time()
    results['noise'] = run_parameter_study(
        'noise', noise_values, base_params, 
        os.path.join(output_dir, 'noise_study'),
        seed=seed+2000 if seed is not None else None
    )
    noise_time = time.time() - start_time
    overall_timing['noise_study'] = noise_time
    print(f"Noise level study completed in {noise_time:.2f} seconds")
    
    # Save overall timing
    total_time = auto_time + cross_time + noise_time
    overall_timing['total_time'] = total_time
    
    with open(os.path.join(output_dir, 'full_study_timing.json'), 'w') as f:
        json.dump(overall_timing, f, indent=4)
    
    print(f"\nFull parameter study completed in {total_time:.2f} seconds")
    
    return results

def run_extended_parameter_study(param_name, param_values, base_params=None, output_dir=None, seed=None):
    """
    Run parameter study varying one parameter with extended parameter options.
    
    Parameters
    ----------
    param_name : str
        Name of parameter to vary ('auto', 'cross', 'noise', 'T', 'n_boot')
    param_values : list
        List of parameter values to test
    base_params : dict, optional
        Base parameters for the study
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Base random seed
        
    Returns
    -------
    dict
        Study results
    """
    # Set default base parameters if not provided
    if base_params is None:
        base_params = {
            'auto_coeff': 0.5,
            'cross_coeff': 0.5,
            'noise_sigma': 0.5,
            'n_vars': 4,
            'T': 500,
            'pc_alpha': 0.05,
            'tau_max': 5,
            'n_boot': 100,
            'effect_pairs': [((0, -3), (3, 0))]  # X1(t-3) → X4(t)
        }
    
    # Create timestamp for output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f"parameter_study_{param_name}_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save study parameters
    study_params = {
        'param_name': param_name,
        'param_values': param_values,
        'base_params': base_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    with open(os.path.join(output_dir, 'study_parameters.json'), 'w') as f:
        json.dump(study_params, f, indent=4)
    
    # Run pipeline for each parameter value
    results = {}
    overall_timing = {}
    
    # Track run indices for each parameter value
    run_indices = {}
   
    for i, param_value in enumerate(param_values):
        print(f"\nRunning {param_name}={param_value} ({i+1}/{len(param_values)})")
        
        # Set parameters for this run
        params = base_params.copy()
        
        if param_name == 'auto':
            params['auto_coeff'] = param_value
        elif param_name == 'cross':
            params['cross_coeff'] = param_value
        elif param_name == 'noise':
            params['noise_sigma'] = param_value
        elif param_name == 'T':
            params['T'] = param_value
        elif param_name == 'n_boot':
            params['n_boot'] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        # Track run index for this parameter value
        if param_value not in run_indices:
            run_indices[param_value] = 0
        else:
            run_indices[param_value] += 1
        run_idx = run_indices[param_value]
        
        # Create parameter-specific output directory with run index
        param_dir = os.path.join(output_dir, f"{param_name}_run{run_idx}_{param_value}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Set seed based on parameter value and base seed
        if seed is not None:
            run_seed = seed + i
        else:
            run_seed = None
        
        # Run pipeline
        start_time = time.time()
        result = run_pipeline(
            auto_coeff=params['auto_coeff'],
            cross_coeff=params['cross_coeff'],
            noise_sigma=params['noise_sigma'],
            n_vars=params['n_vars'],
            T=params['T'],
            pc_alpha=params['pc_alpha'],
            tau_max=params['tau_max'],
            n_boot=params['n_boot'],
            effect_pairs=params['effect_pairs'],
            output_dir=param_dir,
            seed=run_seed
        )
        
        # Record total time
        total_time = time.time() - start_time
        overall_timing[str(param_value)] = total_time
        
        results[param_value] = result
        
        print(f"  Completed in {total_time:.2f} seconds")
    
    # Save overall timing
    with open(os.path.join(output_dir, 'overall_timing.json'), 'w') as f:
        json.dump(overall_timing, f, indent=4)
    
    return results

def run_full_extended_study(auto_values=None, cross_values=None, noise_values=None, 
                           T_values=None, n_boot_values=None,
                           base_params=None, output_dir=None, seed=None):
    """
    Run comprehensive parameter study with extended parameter options.
    
    Parameters
    ----------
    auto_values : list, optional
        List of autocorrelation values to test
    cross_values : list, optional
        List of cross-link values to test
    noise_values : list, optional
        List of noise values to test
    T_values : list, optional
        List of time series lengths to test
    n_boot_values : list, optional
        List of bootstrap replicas to test
    base_params : dict, optional
        Base parameters for all studies
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Base random seed
        
    Returns
    -------
    dict
        Complete study results
    """
    # Set default values if not provided
    if auto_values is None:
        auto_values = [0.2, 0.5, 0.8, 0.95]
    if cross_values is None:
        cross_values = [0.1, 0.3, 0.5, 0.7]
    if noise_values is None:
        noise_values = [0.2, 0.5, 1.0, 1.5]
    if T_values is None:
        T_values = [200, 500, 1000, 2000]
    if n_boot_values is None:
        n_boot_values = [25, 50, 100, 200]
    
    # Create timestamp for output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f"full_extended_study_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save study parameters
    study_params = {
        'auto_values': auto_values,
        'cross_values': cross_values,
        'noise_values': noise_values,
        'T_values': T_values,
        'n_boot_values': n_boot_values,
        'base_params': base_params,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    with open(os.path.join(output_dir, 'full_study_parameters.json'), 'w') as f:
        json.dump(study_params, f, indent=4)
    
    # Run each parameter study
    results = {}
    overall_timing = {}
    
    # Study 1: Autocorrelation
    print("\n== AUTOCORRELATION STUDY ==")
    start_time = time.time()
    results['auto'] = run_extended_parameter_study(
        'auto', auto_values, base_params, 
        os.path.join(output_dir, 'auto_study'),
        seed=seed if seed is not None else None
    )
    auto_time = time.time() - start_time
    overall_timing['auto_study'] = auto_time
    print(f"Autocorrelation study completed in {auto_time:.2f} seconds")
    
    # Study 2: Cross-link strength
    print("\n== CROSS-LINK STUDY ==")
    start_time = time.time()
    results['cross'] = run_extended_parameter_study(
        'cross', cross_values, base_params, 
        os.path.join(output_dir, 'cross_study'),
        seed=seed+1000 if seed is not None else None
    )
    cross_time = time.time() - start_time
    overall_timing['cross_study'] = cross_time
    print(f"Cross-link study completed in {cross_time:.2f} seconds")
    
    # Study 3: Noise level
    print("\n== NOISE LEVEL STUDY ==")
    start_time = time.time()
    results['noise'] = run_extended_parameter_study(
        'noise', noise_values, base_params, 
        os.path.join(output_dir, 'noise_study'),
        seed=seed+2000 if seed is not None else None
    )
    noise_time = time.time() - start_time
    overall_timing['noise_study'] = noise_time
    print(f"Noise level study completed in {noise_time:.2f} seconds")
    
    # Study 4: Time series length
    print("\n== TIME SERIES LENGTH STUDY ==")
    start_time = time.time()
    results['T'] = run_extended_parameter_study(
        'T', T_values, base_params, 
        os.path.join(output_dir, 'T_study'),
        seed=seed+3000 if seed is not None else None
    )
    T_time = time.time() - start_time
    overall_timing['T_study'] = T_time
    print(f"Time series length study completed in {T_time:.2f} seconds")
    
    # Study 5: Bootstrap replicas
    print("\n== BOOTSTRAP REPLICAS STUDY ==")
    start_time = time.time()
    results['n_boot'] = run_extended_parameter_study(
        'n_boot', n_boot_values, base_params, 
        os.path.join(output_dir, 'n_boot_study'),
        seed=seed+4000 if seed is not None else None
    )
    n_boot_time = time.time() - start_time
    overall_timing['n_boot_study'] = n_boot_time
    print(f"Bootstrap replicas study completed in {n_boot_time:.2f} seconds")
    
    # Save overall timing
    total_time = auto_time + cross_time + noise_time + T_time + n_boot_time
    overall_timing['total_time'] = total_time
    
    with open(os.path.join(output_dir, 'full_study_timing.json'), 'w') as f:
        json.dump(overall_timing, f, indent=4)
    
    print(f"\nFull extended parameter study completed in {total_time:.2f} seconds")
    
    return results





def analyze_adjustment_sets(results_df):
    """
    Analyze adjustment sets in relation to effect estimation errors.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results including adjustment sets
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with analysis results
    """
    # Create new DataFrame for analysis
    analysis_df = pd.DataFrame()
    
    # Extract adjustment set sizes for different methods
    methods = ['true_graph', 'pcmci', 'bagged', 'bootstrap']
    
    for method in methods:
        if method == 'bootstrap':
            if 'bootstrap_adjustment_set_stats' in results_df.columns:
                analysis_df[f'{method}_adj_size'] = results_df['bootstrap_adjustment_set_stats'].apply(
                    lambda x: x.get('mean_size', np.nan) if isinstance(x, dict) else np.nan
                )
                analysis_df[f'{method}_adj_std'] = results_df['bootstrap_adjustment_set_stats'].apply(
                    lambda x: x.get('std_size', np.nan) if isinstance(x, dict) else np.nan
                )
        else:
            # Extract adjustment set size from each method
            adj_col = f'{method}_adjustment_set'
            if adj_col in results_df.columns:
                analysis_df[f'{method}_adj_size'] = results_df[adj_col].apply(
                    lambda x: len(x) if isinstance(x, list) else np.nan
                )
    
    # Add error metrics for comparison
    for method in methods:
        if method == 'bootstrap':
            error_col = 'bootstrap_error'
        else:
            error_col = f'{method}_error'
        
        if error_col in results_df.columns:
            analysis_df[error_col] = results_df[error_col]
            analysis_df[f'{method}_abs_error'] = results_df[f'{method}_abs_error']
    
    # Add parameter information
    if 'param_name' in results_df.columns:
        analysis_df['param_name'] = results_df['param_name']
        analysis_df['param_value'] = results_df['param_value']
    
    # Add effect key
    if 'effect_key' in results_df.columns:
        analysis_df['effect_key'] = results_df['effect_key']
    
    return analysis_df

def analyze_linreg_ci_statistics(df, save_path=None):
    """
    Perform statistical analysis of linear regression confidence interval properties.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with CI information
    save_path : str, optional
        Path to save the analysis results
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with detailed CI statistics
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import os
    
    # Define the methods to analyze
    methods = ['true_graph', 'pcmci', 'bagged']
    method_labels = ['True Graph', 'PCMCI', 'Bagged PCMCI']
    
    # Filter to only include rows with the necessary data
    valid_df = df.dropna(subset=['true_effect']).copy()
    
    if len(valid_df) == 0:
        print("No valid data with true effects found for CI analysis")
        return pd.DataFrame()
    
    # Prepare results container
    results = []
    
    # Loop through parameter types
    param_names = valid_df['param_name'].unique()
    
    for param_name in param_names:
        param_df = valid_df[valid_df['param_name'] == param_name]
        param_values = sorted(param_df['param_value'].unique())
        
        for param_value in param_values:
            subset = param_df[param_df['param_value'] == param_value]
            
            if len(subset) == 0:
                continue
            
            # Analyze each method
            for method, label in zip(methods, method_labels):
                lower_col = f'{method}_ci_lower'
                upper_col = f'{method}_ci_upper'
                effect_col = f'{method}_effect'
                
                if not all(col in subset.columns for col in [lower_col, upper_col, effect_col]):
                    continue
                
                # Filter rows with valid data for this method
                method_rows = subset.dropna(subset=[lower_col, upper_col, effect_col, 'true_effect'])
                
                if len(method_rows) == 0:
                    continue
                
                # Calculate CI properties
                ci_widths = method_rows[upper_col] - method_rows[lower_col]
                errors = np.abs(method_rows[effect_col] - method_rows['true_effect'])
                
                # Calculate coverage
                covers = ((method_rows['true_effect'] >= method_rows[lower_col]) & 
                        (method_rows['true_effect'] <= method_rows[upper_col]))
                
                coverage_rate = covers.mean() * 100
                
                # Calculate statistic to error ratio (ideal is around 2 for 95% CI)
                width_error_ratios = ci_widths / errors.replace(0, np.nan)
                
                # Calculate deviation of CI center from true effect
                ci_centers = (method_rows[upper_col] + method_rows[lower_col]) / 2
                center_deviations = ci_centers - method_rows['true_effect']
                
                # Collect statistics
                stats_row = {
                    'param_name': param_name,
                    'param_value': param_value,
                    'method': label,
                    'sample_size': len(method_rows),
                    
                    # CI width statistics
                    'mean_ci_width': ci_widths.mean(),
                    'median_ci_width': ci_widths.median(),
                    'std_ci_width': ci_widths.std(),
                    'min_ci_width': ci_widths.min(),
                    'max_ci_width': ci_widths.max(),
                    
                    # Error statistics
                    'mean_error': errors.mean(),
                    'median_error': errors.median(),
                    'std_error': errors.std(),
                    
                    # Width to error ratio statistics
                    'mean_width_error_ratio': width_error_ratios.mean(),
                    'median_width_error_ratio': width_error_ratios.median(),
                    
                    # Coverage statistics
                    'coverage_rate': coverage_rate,
                    'coverage_std_error': np.sqrt((coverage_rate/100) * (1 - coverage_rate/100) / len(method_rows)) * 100,
                    'coverage_95ci_lower': max(0, coverage_rate - 1.96 * np.sqrt((coverage_rate/100) * (1 - coverage_rate/100) / len(method_rows)) * 100),
                    'coverage_95ci_upper': min(100, coverage_rate + 1.96 * np.sqrt((coverage_rate/100) * (1 - coverage_rate/100) / len(method_rows)) * 100),
                    
                    # Center deviation statistics
                    'mean_center_deviation': center_deviations.mean(),
                    'mean_abs_center_deviation': np.abs(center_deviations).mean(),
                    
                    # CI reliability metrics
                    'calibration_score': np.abs(coverage_rate - 95),  # Lower is better (closer to 95%)
                    'efficiency_score': ci_widths.mean() / errors.mean(),  # Should be close to 2 for 95% CI
                }
                
                # Test if coverage is significantly different from 95%
                p_value = stats.binom_test(
                    x=int(covers.sum()), 
                    n=len(covers), 
                    p=0.95, 
                    alternative='two-sided'
                )
                stats_row['coverage_differs_from_95_pvalue'] = p_value
                stats_row['coverage_is_valid'] = p_value > 0.05  # Not significantly different from 95%
                
                results.append(stats_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save if path is provided
    if save_path and len(results_df) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as CSV
        results_df.to_csv(save_path, index=False)
        
        # Also save a summary grouped by method
        summary_df = results_df.groupby('method').agg({
            'sample_size': 'sum',
            'mean_ci_width': 'mean',
            'mean_error': 'mean',
            'mean_width_error_ratio': 'mean',
            'coverage_rate': 'mean',
            'coverage_is_valid': lambda x: (x == True).mean() * 100,
            'calibration_score': 'mean',
            'efficiency_score': 'mean'
        }).reset_index()
        
        summary_df.rename(columns={'coverage_is_valid': 'percent_valid_coverage'}, inplace=True)
        
        summary_df.to_csv(save_path.replace('.csv', '_summary.csv'), index=False)
    
    return results_df


# Convert to DataFrames for easier analysis
def results_to_dataframe(results_dict, param_name):
    rows = []
    
    for param_value, effect_dict in results_dict.items():
        for effect_key, effect_data in effect_dict.items():
            # Extract data
            row = {
                'param_name': param_name,
                'param_value': param_value,
                'effect_key': effect_key,
                'true_effect': effect_data.get('true_effect'),
                'true_graph_effect': effect_data.get('true_graph_effect'),
                'pcmci_effect': effect_data.get('pcmci_effect'),
                'bagged_effect': effect_data.get('bagged_effect')
            }
            
            # Add bootstrap statistics if available
            bootstrap_stats = effect_data.get('bootstrap_stats', {})
            if bootstrap_stats:
                row.update({
                    'bootstrap_mean': bootstrap_stats.get('mean'),
                    'bootstrap_std': bootstrap_stats.get('std'),
                    'bootstrap_ci_lower': bootstrap_stats.get('ci_lower'),
                    'bootstrap_ci_upper': bootstrap_stats.get('ci_upper'),
                    'bootstrap_success_rate': bootstrap_stats.get('estimation_success_rate')
                })
            
            # Add bootstrap effects if available
            if 'bootstrap_effects' in effect_data:
                row['bootstrap_effects'] = effect_data['bootstrap_effects']
            
            # Add timing information if available
            timings = effect_data.get('timings', {})
            if timings:
                for method, time_val in timings.items():
                    row[f'{method}_time'] = time_val
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add error columns
    for method in ['true_graph','pcmci', 'bagged', 'bootstrap']:
        if method == 'bootstrap':
            effect_col = 'bootstrap_mean'
        else:
            effect_col = f'{method}_effect'
        
        # Skip if column doesn't exist
        if effect_col not in df.columns:
            continue
        
        # Calculate errors - handle NaN values safely
        df[f'{method}_error'] = df[effect_col].subtract(df['true_effect'], fill_value=np.nan)
        df[f'{method}_abs_error'] = df[f'{method}_error'].abs()
        df[f'{method}_squared_error'] = df[f'{method}_error'].pow(2)
        
        # Flag if true effect is within CI (for bootstrap)
        if method == 'bootstrap' and 'bootstrap_ci_lower' in df.columns and 'bootstrap_ci_upper' in df.columns:
            df['ci_covers_true'] = (df['true_effect'] >= df['bootstrap_ci_lower']) & \
                                  (df['true_effect'] <= df['bootstrap_ci_upper'])
    
    return df


def load_study_results(study_folder, output_folder=None):
    """
    Load all results from a study folder structure into a comprehensive DataFrame.
    
    Parameters
    ----------
    study_folder : str
        Path to the main study folder containing substudy folders
    output_folder : str, optional
        Path to save the DataFrame as CSV
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all study results
    """

    
    print(f"Loading study results from {study_folder}...")
    
    # Find all substudy folders
    substudy_folders = [f for f in glob.glob(os.path.join(study_folder, "*")) 
                        if os.path.isdir(f) and not f.endswith("_results")]
    
    if not substudy_folders:
        raise ValueError(f"No substudy folders found in {study_folder}")
    
    print(f"Found {len(substudy_folders)} substudy folders")
    
    # Initialize list to collect all rows
    all_rows = []
    
    # Process each substudy folder
    for substudy_folder in substudy_folders:
        substudy_name = os.path.basename(substudy_folder)
        print(f"Processing substudy: {substudy_name}")
        
        # Extract parameter name (e.g., 'auto', 'cross', 'noise')
        param_name = substudy_name.split('_')[0] if '_' in substudy_name else substudy_name
        
        # Find all parameter value folders
        param_folders = [f for f in glob.glob(os.path.join(substudy_folder, "*")) 
                         if os.path.isdir(f)]
        
        # Process each parameter value folder
        for param_folder in param_folders:
            folder_name = os.path.basename(param_folder)
            
            # Extract parameter value from folder name
            param_parts = folder_name.split('_')
            if len(param_parts) >= 3 and param_parts[-2].startswith('run'):
                try:
                    # Extract parameter value from "param_name_runX_value" format
                    param_value = float(param_parts[-1])
                except ValueError:
                    print(f"  Skipping folder {folder_name} - cannot parse parameter value")
                    continue
            elif len(param_parts) >= 2:
                try:
                    # Handle old format "param_name_value"
                    param_value = float(param_parts[-1])
                except ValueError:
                    print(f"  Skipping folder {folder_name} - cannot parse parameter value")
                    continue
            else:
                print(f"  Skipping folder {folder_name} - unexpected naming format")
                continue
            
            
            
            
            print(f"  Processing parameter value: {param_value}")
            
            # Load parameters.json
            params_file = os.path.join(param_folder, "parameters.json")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    parameters = json.load(f)
            else:
                parameters = {}
                print(f"  Warning: parameters.json not found in {param_folder}")
            
            # Load timing_summary.json if available
            timing_summary_file = os.path.join(param_folder, "timing_summary.json")
            if os.path.exists(timing_summary_file):
                with open(timing_summary_file, 'r') as f:
                    timing_summary = json.load(f)
            else:
                timing_summary = {}
                print(f"  Warning: timing_summary.json not found in {param_folder}")
            
            # Find all effect result files
            effect_files = glob.glob(os.path.join(param_folder, "effects_*_all.json"))
            
            # Process each effect file
            for effect_file in effect_files:
                effect_basename = os.path.basename(effect_file)
                effect_key = effect_basename.replace("effects_", "").replace("_all.json", "")
                
                # Parse X and Y from effect key
                try:
                    x_part, y_part = effect_key.split("_to_")
                    X_var, X_lag = map(int, x_part.split('_'))
                    Y_var, Y_lag = map(int, y_part.split('_'))
                    X = (X_var, X_lag)
                    Y = (Y_var, Y_lag)
                except:
                    # If parsing fails, use placeholders
                    X = None
                    Y = None
                    print(f"  Warning: Could not parse X and Y from effect key: {effect_key}")
                
                # Load effect results
                with open(effect_file, 'r') as f:
                    effect_data = json.load(f)
                
                # Create base row with parameter and effect information
                row = {
                    'substudy': substudy_name,
                    'param_name': param_name,
                    'param_value': param_value,
                    'effect_key': effect_key,
                    'X': str(X),
                    'Y': str(Y),
                    'X_var': X_var if X is not None else None,
                    'X_lag': X_lag if X is not None else None,
                    'Y_var': Y_var if Y is not None else None,
                    'Y_lag': Y_lag if Y is not None else None,
                }
                
                # Add parameters from parameters.json
                for key, value in parameters.items():
                    if key not in ['effect_pairs']:  # Skip complex objects
                        row[f'param_{key}'] = value
                
                # Add effect estimation results
                row.update({
                    'true_effect': effect_data.get('true_effect'),
                    'true_graph_effect': effect_data.get('true_graph_effect'),
                    'pcmci_effect': effect_data.get('pcmci_effect'),
                    'bagged_effect': effect_data.get('bagged_effect')
                })
                
                # Add linear regression confidence intervals for each method
                if 'confidence_intervals' in effect_data:
                    ci_data = effect_data.get('confidence_intervals', {})
                    
                    # Add true_graph CI
                    if 'true_graph' in ci_data and len(ci_data['true_graph']) == 2:
                        row['true_graph_linreg_ci_lower'] = ci_data['true_graph'][0]
                        row['true_graph_linreg_ci_upper'] = ci_data['true_graph'][1]
                    
                    # Add pcmci CI
                    if 'pcmci' in ci_data and len(ci_data['pcmci']) == 2:
                        row['pcmci_linreg_ci_lower'] = ci_data['pcmci'][0]
                        row['pcmci_linreg_ci_upper'] = ci_data['pcmci'][1]
                    
                    # Add bagged CI
                    if 'bagged' in ci_data and len(ci_data['bagged']) == 2:
                        row['bagged_linreg_ci_lower'] = ci_data['bagged'][0]
                        row['bagged_linreg_ci_upper'] = ci_data['bagged'][1]
                        
                    # Add bootstrap linreg CI summary statistics
                    if 'bootstrap_summary' in ci_data:
                        bootstrap_ci_summary = ci_data['bootstrap_summary']
                        row.update({
                            'bootstrap_linreg_ci_mean_width': bootstrap_ci_summary.get('mean_width'),
                            'bootstrap_linreg_ci_std_width': bootstrap_ci_summary.get('std_width'),
                            'bootstrap_linreg_ci_min_width': bootstrap_ci_summary.get('min_width'),
                            'bootstrap_linreg_ci_max_width': bootstrap_ci_summary.get('max_width')
                        })
                
                # Add timing information from timing_summary.json
                if timing_summary:
                    # Add overall timing
                    if 'total_time' in timing_summary:
                        row['total_time'] = timing_summary['total_time']
                    
                    # Add data generation time
                    if 'data_generation_time' in timing_summary:
                        row['data_generation_time'] = timing_summary['data_generation_time']
                    
                    # Add discovery times
                    if 'discovery_times' in timing_summary:
                        discovery_times = timing_summary['discovery_times']
                        for method, time_value in discovery_times.items():
                            row[f'discovery_{method}_time'] = time_value
                    
                    # Add effect estimation times per pair (if available)
                    if 'effect_times' in timing_summary and effect_key in timing_summary['effect_times']:
                        effect_times = timing_summary['effect_times'][effect_key]
                        for method, time_value in effect_times.items():
                            row[f'effect_{method}_time'] = time_value
                
                # Add timing information from effect file
                if 'timings' in effect_data:
                    for method, time_value in effect_data['timings'].items():
                        row[f'{method}_time'] = time_value
                
                # Load bootstrap results to get individual estimates and CIs
                bootstrap_file = effect_file.replace("_all.json", "_bootstrap.json")
                if os.path.exists(bootstrap_file):
                    with open(bootstrap_file, 'r') as f:
                        bootstrap_data = json.load(f)
                    
                    # Add individual bootstrap estimates
                    row['bootstrap_effects'] = bootstrap_data.get('bootstrap_effects', [])
                    
                    # Add individual bootstrap linreg confidence intervals
                    row['bootstrap_replica_ci_lower'] = [ci[0] if ci[0] is not None else None for ci in bootstrap_data.get('confidence_intervals', [])]
                    row['bootstrap_replica_ci_upper'] = [ci[1] if ci[1] is not None else None for ci in bootstrap_data.get('confidence_intervals', [])]
                    
                    # Combine with estimates to create a list of (estimate, ci_lower, ci_upper) tuples
                    effects = bootstrap_data.get('bootstrap_effects', [])
                    ci_lower = row['bootstrap_replica_ci_lower']
                    ci_upper = row['bootstrap_replica_ci_upper']
                    
                    # Make sure the lists have the same length
                    min_len = min(len(effects), len(ci_lower), len(ci_upper))
                    valid_estimates_with_ci = [
                        (effects[i], ci_lower[i], ci_upper[i]) 
                        for i in range(min_len) 
                        if effects[i] is not None and ci_lower[i] is not None and ci_upper[i] is not None
                    ]
                    row['bootstrap_replica_estimate_with_ci'] = valid_estimates_with_ci
                    
                    # Calculate combined CI from all valid replica CIs
                    if valid_estimates_with_ci:
                        # Extract valid CIs
                        valid_ci_lower = [item[1] for item in valid_estimates_with_ci]
                        valid_ci_upper = [item[2] for item in valid_estimates_with_ci]
                        
                        # Calculate mean of lower and upper bounds
                        combined_ci_lower = sum(valid_ci_lower) / len(valid_ci_lower)
                        combined_ci_upper = sum(valid_ci_upper) / len(valid_ci_upper)
                        
                        # Calculate the most conservative CI (minimum of lower bounds, maximum of upper bounds)
                        conservative_ci_lower = min(valid_ci_lower)
                        conservative_ci_upper = max(valid_ci_upper)
                        
                        # Add to row
                        row['bootstrap_combined_ci_lower'] = combined_ci_lower
                        row['bootstrap_combined_ci_upper'] = combined_ci_upper
                        row['bootstrap_combined_ci_width'] = combined_ci_upper - combined_ci_lower
                        
                        row['bootstrap_conservative_ci_lower'] = conservative_ci_lower
                        row['bootstrap_conservative_ci_upper'] = conservative_ci_upper
                        row['bootstrap_conservative_ci_width'] = conservative_ci_upper - conservative_ci_lower
                        
                        # Check if true effect is within the combined CIs if true_effect is available
                        if 'true_effect' in row and row['true_effect'] is not None:
                            row['bootstrap_combined_ci_covers_true'] = (
                                row['true_effect'] >= combined_ci_lower and 
                                row['true_effect'] <= combined_ci_upper
                            )
                            row['bootstrap_conservative_ci_covers_true'] = (
                                row['true_effect'] >= conservative_ci_lower and 
                                row['true_effect'] <= conservative_ci_upper
                            )
                    
                    # Add bootstrap statistics for bootstrap-derived confidence intervals
                    bootstrap_stats = bootstrap_data.get('stats', {})
                    if bootstrap_stats:
                        row.update({
                            'bootstrap_mean': bootstrap_stats.get('mean'),
                            'bootstrap_std': bootstrap_stats.get('std'),
                            'bootstrap_ci_lower': bootstrap_stats.get('ci_lower'),
                            'bootstrap_ci_upper': bootstrap_stats.get('ci_upper'),
                            'bootstrap_success_rate': bootstrap_stats.get('estimation_success_rate'),
                            'bootstrap_n_total': bootstrap_stats.get('n_total'),
                            'bootstrap_n_nan': bootstrap_stats.get('n_nan')
                        })
                    
                    # Add bootstrap adjustment set statistics
                    if 'adjustment_set_stats' in bootstrap_data:
                        adj_stats = bootstrap_data['adjustment_set_stats']
                        row.update({
                            'bootstrap_adj_mean_size': adj_stats.get('mean_size'),
                            'bootstrap_adj_std_size': adj_stats.get('std_size'),
                            'bootstrap_adj_min_size': adj_stats.get('min_size'),
                            'bootstrap_adj_max_size': adj_stats.get('max_size'),
                            'bootstrap_unique_graphs': adj_stats.get('unique_graphs')
                        })
                    
                    # Add bootstrap estimation time
                    if 'estimation_time' in bootstrap_data:
                        row['bootstrap_estimation_time'] = bootstrap_data['estimation_time']
                
                # Load additional timing information from individual method files
                for method in ['pcmci', 'bagged', 'true_graph']:
                    method_file = effect_file.replace("_all.json", f"_{method}.json")
                    if os.path.exists(method_file):
                        with open(method_file, 'r') as f:
                            method_data = json.load(f)
                        
                        # Add method-specific estimation time
                        if 'estimation_time' in method_data:
                            row[f'{method}_estimation_time'] = method_data['estimation_time']
                        
                        # Add method-specific adjustment set size
                        if 'adjustment_set_size' in method_data:
                            row[f'{method}_adjustment_set_size'] = method_data['adjustment_set_size']
                
                # Calculate errors and coverage
                if 'true_effect' in row and row['true_effect'] is not None:

                    if 'true_graph_effect' in row and row['true_graph_effect'] is not None:
                        row['true_graph_error'] = row['true_graph_effect'] - row['true_effect']
                        row['true_graph_abs_error'] = abs(row['true_graph_error'])
                        row['true_graph_squared_error'] = row['true_graph_error'] ** 2

                    # Calculate PCMCI errors
                    if 'pcmci_effect' in row and row['pcmci_effect'] is not None:
                        row['pcmci_error'] = row['pcmci_effect'] - row['true_effect']
                        row['pcmci_abs_error'] = abs(row['pcmci_error'])
                        
                    # Calculate bagged errors
                    if 'bagged_effect' in row and row['bagged_effect'] is not None:
                        row['bagged_error'] = row['bagged_effect'] - row['true_effect']
                        row['bagged_abs_error'] = abs(row['bagged_error'])
                        row['bagged_squared_error'] = row['bagged_error'] ** 2
                    
                    # Calculate bootstrap errors
                    if 'bootstrap_mean' in row and row['bootstrap_mean'] is not None:
                        row['bootstrap_error'] = row['bootstrap_mean'] - row['true_effect']
                        row['bootstrap_abs_error'] = abs(row['bootstrap_error'])
                        row['bootstrap_squared_error'] = row['bootstrap_error'] ** 2

                   # Calculate relative errors for all methods
                    if row['true_effect'] != 0:
                        if 'pcmci_error' in row:
                            row['pcmci_relative_error'] = row['pcmci_error'] / row['true_effect']
                        if 'true_graph_error' in row:
                            row['true_graph_relative_error'] = row['true_graph_error'] / row['true_effect']
                        if 'bagged_error' in row:
                            row['bagged_relative_error'] = row['bagged_error'] / row['true_effect']
                        if 'bootstrap_error' in row:
                            row['bootstrap_relative_error'] = row['bootstrap_error'] / row['true_effect']
                    

                    # Check if bootstrap CI covers true value
                    if ('bootstrap_ci_lower' in row and 'bootstrap_ci_upper' in row and
                        row['bootstrap_ci_lower'] is not None and row['bootstrap_ci_upper'] is not None):
                        row['bootstrap_ci_covers_true'] = (row['true_effect'] >= row['bootstrap_ci_lower'] and 
                                                         row['true_effect'] <= row['bootstrap_ci_upper'])
                    
                    # Check if linreg CIs cover true value
                    for method in ['true_graph', 'pcmci', 'bagged']:
                        if (f'{method}_linreg_ci_lower' in row and f'{method}_linreg_ci_upper' in row and
                            row[f'{method}_linreg_ci_lower'] is not None and row[f'{method}_linreg_ci_upper'] is not None):
                            row[f'{method}_linreg_ci_covers_true'] = (row['true_effect'] >= row[f'{method}_linreg_ci_lower'] and 
                                                                    row['true_effect'] <= row[f'{method}_linreg_ci_upper'])
                
                # Add row to collection
                all_rows.append(row)
    
    # Create DataFrame from all rows
    results_df = pd.DataFrame(all_rows)
    
    # Mapping dictionary for parameter values
    rename_dict = {
        'auto': 'a',
        'cross': 'c', 
        'n': 'B',
        'noise': 'sigmaN',
        'T': 'T'
    }
    
    # Replace values in the 'param_name' column
    results_df['param_name'] = results_df['param_name'].replace(rename_dict)
    
    # Save to CSV if output folder is provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_path = os.path.join(output_folder, f"study_results_{timestamp}.csv")
        
        # Create a version of the DataFrame without complex list columns for CSV export
        export_df = results_df.copy()
        list_columns = [col for col in export_df.columns if 
                        isinstance(export_df[col].iloc[0] if len(export_df) > 0 and not export_df[col].isnull().all() else None, list)]
        
        for col in list_columns:
            export_df[f'{col}_count'] = export_df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            export_df.drop(col, axis=1, inplace=True)
        
        export_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save full DataFrame with all columns as pickle for later use
        pickle_path = os.path.join(output_folder, f"study_results_{timestamp}.pkl")
        results_df.to_pickle(pickle_path)
        print(f"Full results (including list columns) saved to {pickle_path}")
    
    print(f"Loaded {len(results_df)} data points")
    
    return results_df
        
def load_study_results_with_graphs(study_folder, output_folder=None, include_graphs=True):
    """
    Load all results from a study folder structure into a comprehensive DataFrame,
    including the actual graph structures from .npy files.
    
    Parameters
    ----------
    study_folder : str
        Path to the main study folder containing substudy folders
    output_folder : str, optional
        Path to save the DataFrame as CSV and pickle
    include_graphs : bool, default=True
        Whether to include the actual graph structures in the DataFrame
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with all study results and graph structures
    """
    import os
    import glob
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    print(f"Loading study results from {study_folder}...")
    
    # Find all substudy folders
    substudy_folders = [f for f in glob.glob(os.path.join(study_folder, "*")) 
                        if os.path.isdir(f) and not f.endswith("_results")]
    
    if not substudy_folders:
        raise ValueError(f"No substudy folders found in {study_folder}")
    
    print(f"Found {len(substudy_folders)} substudy folders")
    
    # Initialize list to collect all rows
    all_rows = []
    
    # Process each substudy folder
    for substudy_folder in substudy_folders:
        substudy_name = os.path.basename(substudy_folder)
        print(f"Processing substudy: {substudy_name}")
        
        # Extract parameter name (e.g., 'auto', 'cross', 'noise')
        param_name = substudy_name.split('_')[0] if '_' in substudy_name else substudy_name
        
        # Find all parameter value folders
        param_folders = [f for f in glob.glob(os.path.join(substudy_folder, "*")) 
                         if os.path.isdir(f)]
        
        # Process each parameter value folder
        for param_folder in param_folders:
            folder_name = os.path.basename(param_folder)
            
            # Extract parameter value from folder name
            param_parts = folder_name.split('_')
            if len(param_parts) >= 3 and param_parts[-2].startswith('run'):
                try:
                    # Extract parameter value from "param_name_runX_value" format
                    param_value = float(param_parts[-1])
                except ValueError:
                    print(f"  Skipping folder {folder_name} - cannot parse parameter value")
                    continue
            elif len(param_parts) >= 2:
                try:
                    # Handle old format "param_name_value"
                    param_value = float(param_parts[-1])
                except ValueError:
                    print(f"  Skipping folder {folder_name} - cannot parse parameter value")
                    continue
            else:
                print(f"  Skipping folder {folder_name} - unexpected naming format")
                continue
            
            print(f"  Processing parameter value: {param_value}")
            
            # Load parameters.json
            params_file = os.path.join(param_folder, "parameters.json")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    parameters = json.load(f)
            else:
                parameters = {}
                print(f"  Warning: parameters.json not found in {param_folder}")
            
            # Load timing_summary.json if available
            timing_summary_file = os.path.join(param_folder, "timing_summary.json")
            if os.path.exists(timing_summary_file):
                with open(timing_summary_file, 'r') as f:
                    timing_summary = json.load(f)
            else:
                timing_summary = {}
                print(f"  Warning: timing_summary.json not found in {param_folder}")
            
            # Find all effect result files
            effect_files = glob.glob(os.path.join(param_folder, "effects_*_all.json"))
            
            # Load graph structures if include_graphs is True
            graph_structures = {}
            if include_graphs:
                # Load PCMCI graph
                pcmci_graph_path = os.path.join(param_folder, "discovery_pcmci_graph.npy")
                if os.path.exists(pcmci_graph_path):
                    try:
                        graph_structures['pcmci_graph'] = np.load(pcmci_graph_path)
                    except:
                        print(f"  Warning: Failed to load PCMCI graph from {pcmci_graph_path}")
                
                # Load bagged graph
                bagged_graph_path = os.path.join(param_folder, "discovery_bootstrap_bagged_graph.npy")
                if os.path.exists(bagged_graph_path):
                    try:
                        graph_structures['bagged_graph'] = np.load(bagged_graph_path)
                    except:
                        print(f"  Warning: Failed to load bagged graph from {bagged_graph_path}")
                
                # Load bootstrap graphs
                boot_graphs_path = os.path.join(param_folder, "discovery_bootstrap_boot_graphs.npy")
                if os.path.exists(boot_graphs_path):
                    try:
                        graph_structures['boot_graphs'] = np.load(boot_graphs_path)
                    except:
                        print(f"  Warning: Failed to load bootstrap graphs from {boot_graphs_path}")
            
            # Process each effect file
            for effect_file in effect_files:
                effect_basename = os.path.basename(effect_file)
                effect_key = effect_basename.replace("effects_", "").replace("_all.json", "")
                
                # Parse X and Y from effect key
                try:
                    x_part, y_part = effect_key.split("_to_")
                    X_var, X_lag = map(int, x_part.split('_'))
                    Y_var, Y_lag = map(int, y_part.split('_'))
                    X = (X_var, X_lag)
                    Y = (Y_var, Y_lag)
                except:
                    # If parsing fails, use placeholders
                    X = None
                    Y = None
                    print(f"  Warning: Could not parse X and Y from effect key: {effect_key}")
                
                # Load effect results
                with open(effect_file, 'r') as f:
                    effect_data = json.load(f)
                
                # Create base row with parameter and effect information
                row = {
                    'substudy': substudy_name,
                    'param_name': param_name,
                    'param_value': param_value,
                    'effect_key': effect_key,
                    'X': str(X),
                    'Y': str(Y),
                    'X_var': X_var if X is not None else None,
                    'X_lag': X_lag if X is not None else None,
                    'Y_var': Y_var if Y is not None else None,
                    'Y_lag': Y_lag if Y is not None else None,
                }
                
                # Add parameters from parameters.json
                for key, value in parameters.items():
                    if key not in ['effect_pairs']:  # Skip complex objects
                        row[f'param_{key}'] = value
                
                # Add effect estimation results
                row.update({
                    'true_effect': effect_data.get('true_effect'),
                    'true_graph_effect': effect_data.get('true_graph_effect'),
                    'pcmci_effect': effect_data.get('pcmci_effect'),
                    'bagged_effect': effect_data.get('bagged_effect')
                })
                
                # Add graph structures if available
                if include_graphs:
                    for graph_name, graph in graph_structures.items():
                        row[graph_name] = graph
                
                # Add linear regression confidence intervals for each method
                if 'confidence_intervals' in effect_data:
                    ci_data = effect_data.get('confidence_intervals', {})
                    
                    # Add true_graph CI
                    if 'true_graph' in ci_data and len(ci_data['true_graph']) == 2:
                        row['true_graph_linreg_ci_lower'] = ci_data['true_graph'][0]
                        row['true_graph_linreg_ci_upper'] = ci_data['true_graph'][1]
                    
                    # Add pcmci CI
                    if 'pcmci' in ci_data and len(ci_data['pcmci']) == 2:
                        row['pcmci_linreg_ci_lower'] = ci_data['pcmci'][0]
                        row['pcmci_linreg_ci_upper'] = ci_data['pcmci'][1]
                    
                    # Add bagged CI
                    if 'bagged' in ci_data and len(ci_data['bagged']) == 2:
                        row['bagged_linreg_ci_lower'] = ci_data['bagged'][0]
                        row['bagged_linreg_ci_upper'] = ci_data['bagged'][1]
                        
                    # Add bootstrap linreg CI summary statistics
                    if 'bootstrap_summary' in ci_data:
                        bootstrap_ci_summary = ci_data['bootstrap_summary']
                        row.update({
                            'bootstrap_linreg_ci_mean_width': bootstrap_ci_summary.get('mean_width'),
                            'bootstrap_linreg_ci_std_width': bootstrap_ci_summary.get('std_width'),
                            'bootstrap_linreg_ci_min_width': bootstrap_ci_summary.get('min_width'),
                            'bootstrap_linreg_ci_max_width': bootstrap_ci_summary.get('max_width')
                        })
                
                # Add timing information from timing_summary.json
                if timing_summary:
                    # Add overall timing
                    if 'total_time' in timing_summary:
                        row['total_time'] = timing_summary['total_time']
                    
                    # Add data generation time
                    if 'data_generation_time' in timing_summary:
                        row['data_generation_time'] = timing_summary['data_generation_time']
                    
                    # Add discovery times
                    if 'discovery_times' in timing_summary:
                        discovery_times = timing_summary['discovery_times']
                        for method, time_value in discovery_times.items():
                            row[f'discovery_{method}_time'] = time_value
                    
                    # Add effect estimation times per pair (if available)
                    if 'effect_times' in timing_summary and effect_key in timing_summary['effect_times']:
                        effect_times = timing_summary['effect_times'][effect_key]
                        for method, time_value in effect_times.items():
                            row[f'effect_{method}_time'] = time_value
                
                # Add timing information from effect file
                if 'timings' in effect_data:
                    for method, time_value in effect_data['timings'].items():
                        row[f'{method}_time'] = time_value
                
                # Load bootstrap results to get individual estimates and CIs
                bootstrap_file = effect_file.replace("_all.json", "_bootstrap.json")
                if os.path.exists(bootstrap_file):
                    with open(bootstrap_file, 'r') as f:
                        bootstrap_data = json.load(f)
                    
                    # Add individual bootstrap estimates
                    row['bootstrap_effects'] = bootstrap_data.get('bootstrap_effects', [])
                    
                    # Add individual bootstrap linreg confidence intervals
                    if 'confidence_intervals' in bootstrap_data:
                        row['bootstrap_replica_ci_lower'] = [ci[0] if ci[0] is not None else None for ci in bootstrap_data.get('confidence_intervals', [])]
                        row['bootstrap_replica_ci_upper'] = [ci[1] if ci[1] is not None else None for ci in bootstrap_data.get('confidence_intervals', [])]
                    
                    # Add bootstrap statistics
                    bootstrap_stats = bootstrap_data.get('stats', {})
                    if bootstrap_stats:
                        row.update({
                            'bootstrap_mean': bootstrap_stats.get('mean'),
                            'bootstrap_std': bootstrap_stats.get('std'),
                            'bootstrap_ci_lower': bootstrap_stats.get('ci_lower'),
                            'bootstrap_ci_upper': bootstrap_stats.get('ci_upper'),
                            'bootstrap_success_rate': bootstrap_stats.get('estimation_success_rate'),
                            'bootstrap_n_total': bootstrap_stats.get('n_total'),
                            'bootstrap_n_nan': bootstrap_stats.get('n_nan')
                        })
                    
                    # Add bootstrap adjustment set statistics
                    if 'adjustment_set_stats' in bootstrap_data:
                        adj_stats = bootstrap_data['adjustment_set_stats']
                        row.update({
                            'bootstrap_adj_mean_size': adj_stats.get('mean_size'),
                            'bootstrap_adj_std_size': adj_stats.get('std_size'),
                            'bootstrap_adj_min_size': adj_stats.get('min_size'),
                            'bootstrap_adj_max_size': adj_stats.get('max_size'),
                            'bootstrap_unique_graphs': adj_stats.get('unique_graphs')
                        })
                
                # Load adjustment set information
                for method in ['pcmci', 'bagged', 'true_graph']:
                    method_file = effect_file.replace("_all.json", f"_{method}.json")
                    if os.path.exists(method_file):
                        with open(method_file, 'r') as f:
                            method_data = json.load(f)
                        
                        # Add method-specific adjustment set if available
                        if 'adjustment_set' in method_data:
                            row[f'{method}_adjustment_set'] = method_data['adjustment_set']
                        
                        # Add method-specific adjustment set size
                        if 'adjustment_set_size' in method_data:
                            row[f'{method}_adjustment_set_size'] = method_data['adjustment_set_size']
                
                # Calculate errors and coverage
                if 'true_effect' in row and row['true_effect'] is not None:
                    # Calculate PCMCI errors
                    if 'pcmci_effect' in row and row['pcmci_effect'] is not None:
                        row['pcmci_error'] = row['pcmci_effect'] - row['true_effect']
                        row['pcmci_abs_error'] = abs(row['pcmci_error'])
                        row['pcmci_squared_error'] = row['pcmci_error'] ** 2
                    
                    # Calculate bagged errors
                    if 'bagged_effect' in row and row['bagged_effect'] is not None:
                        row['bagged_error'] = row['bagged_effect'] - row['true_effect']
                        row['bagged_abs_error'] = abs(row['bagged_error'])
                        row['bagged_squared_error'] = row['bagged_error'] ** 2
                    
                    # Calculate bootstrap errors
                    if 'bootstrap_mean' in row and row['bootstrap_mean'] is not None:
                        row['bootstrap_error'] = row['bootstrap_mean'] - row['true_effect']
                        row['bootstrap_abs_error'] = abs(row['bootstrap_error'])
                        row['bootstrap_squared_error'] = row['bootstrap_error'] ** 2
                    
                    # Check if bootstrap CI covers true value
                    if ('bootstrap_ci_lower' in row and 'bootstrap_ci_upper' in row and
                        row['bootstrap_ci_lower'] is not None and row['bootstrap_ci_upper'] is not None):
                        row['bootstrap_ci_covers_true'] = (row['true_effect'] >= row['bootstrap_ci_lower'] and 
                                                         row['true_effect'] <= row['bootstrap_ci_upper'])
                    
                    # Check if linreg CIs cover true value
                    for method in ['true_graph', 'pcmci', 'bagged']:
                        if (f'{method}_linreg_ci_lower' in row and f'{method}_linreg_ci_upper' in row and
                            row[f'{method}_linreg_ci_lower'] is not None and row[f'{method}_linreg_ci_upper'] is not None):
                            row[f'{method}_linreg_ci_covers_true'] = (row['true_effect'] >= row[f'{method}_linreg_ci_lower'] and 
                                                                    row['true_effect'] <= row[f'{method}_linreg_ci_upper'])
                
                # Add row to collection
                all_rows.append(row)
    
    # Create DataFrame from all rows
    results_df = pd.DataFrame(all_rows)
    
    # Mapping dictionary for parameter values
    rename_dict = {
        'auto': 'a',
        'cross': 'c', 
        'n': 'B',
        'noise': 'sigmaN',
        'T': 'T'
    }
    
    # Replace values in the 'param_name' column
    results_df['param_name'] = results_df['param_name'].replace(rename_dict)
    
    # Save to files if output folder is provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create a version of the DataFrame without complex list columns and array columns for CSV export
        export_df = results_df.copy()
        
        # Identify columns with complex types (lists, arrays, etc.)
        complex_columns = []
        for col in export_df.columns:
            if len(export_df) > 0 and not export_df[col].isnull().all():
                sample_value = export_df[col].dropna().iloc[0]
                if isinstance(sample_value, (list, np.ndarray)) or col in ['pcmci_graph', 'bagged_graph', 'boot_graphs']:
                    complex_columns.append(col)
        
        # Process complex columns
        for col in complex_columns:
            if col.endswith('_effects') or col.endswith('_ci_lower') or col.endswith('_ci_upper'):
                # For list columns, add a count column
                export_df[f'{col}_count'] = export_df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            # Remove the complex column from the export DataFrame
            export_df.drop(col, axis=1, inplace=True)
        
        # Save simplified DataFrame to CSV
        csv_path = os.path.join(output_folder, f"study_results_{timestamp}.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Simplified results (without complex columns) saved to {csv_path}")
        
        # Save full DataFrame with all columns as pickle for later use
        pickle_path = os.path.join(output_folder, f"study_results_{timestamp}.pkl")
        results_df.to_pickle(pickle_path)
        print(f"Full results (including graph structures) saved to {pickle_path}")
    
    print(f"Loaded {len(results_df)} data points")
    
    return results_df

# Example usage
if __name__ == "__main__":
    
    if 1:
        ### Full Study ###

        ## from causal_evaluation import run_full_study
        # Define parameter values to test
        auto_values = np.repeat(np.linspace(0.01,.99,10),5).tolist()
        cross_values = np.repeat(np.linspace(0.01,.99,10),5).tolist()
        noise_values = np.repeat(np.linspace(0.1,3.,10),5).tolist()
        T_values = np.repeat([200,300,400,500,600,700,800,900,1000,1500,2000,3000,5000],5).tolist()
        n_boot_values = np.repeat([10,50,100,200,300],5).tolist()
        # Set base parameters
        base_params = {
            'auto_coeff': 0.5,
            'cross_coeff': 0.5,
            'noise_sigma': 1,
            'n_vars': 4,
            'T': 500,  # Smaller dataset for faster execution
            'pc_alpha': 0.05,
            'tau_max': 4,
            'n_boot': 100,  # Fewer bootstrap samples for faster execution
            'effect_pairs': [((0, -2), (3, 0))]  # Focus on one effect pair
        }
        
        full_results = run_full_extended_study(    
            auto_values=auto_values,
            cross_values=cross_values,
            noise_values=noise_values,
            T_values=T_values, 
            n_boot_values=n_boot_values,
            base_params=base_params,
            output_dir="full_study_results_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()),
            seed=None)

    
    
    if 0:
        # Set the main study folder path
        main_study_folder = "full_study_results_2025-04-01_18-05-50"
        
        # Create a results folder for the analysis
        results_folder = f"analysis_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"Loading results from {main_study_folder}...")
        
        # Load parameter studies using the modified function
        print("Loading autocorrelation study...")
        auto_results = modified_load_experiment_results(os.path.join(main_study_folder, "auto_study"), "auto")
        
        # Convert to DataFrame
        auto_df = results_to_dataframe(auto_results, "auto")
        
        # Check if bootstrap effects were loaded
        print(f"Bootstrap effects loaded for {sum('bootstrap_effects' in row for _, row in auto_df.iterrows())} rows")
        
        # Print first row with bootstrap effects to verify
        bootstrap_rows = auto_df[auto_df['bootstrap_effects'].notna()]
        if not bootstrap_rows.empty:
            row = bootstrap_rows.iloc[0]
            print(f"Sample bootstrap effects for {row['effect_key']} at {row['param_name']}={row['param_value']}:")
            print(f"Number of bootstrap samples: {len(row['bootstrap_effects'])}")
            print(f"Sample values: {row['bootstrap_effects'][:3]}...")
