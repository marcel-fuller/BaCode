"""
Core functionality for causal discovery and effect estimation.

This module provides the essential functions for creating causal models,
generating data, running causal discovery, and estimating causal effects.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.causal_effects import CausalEffects
from sklearn.linear_model import LinearRegression


class Timer:
    """Context manager for timing operations with optional logging."""
    
    def __init__(self, operation_name=None, save_path=None):
        """
        Initialize timer.
        
        Parameters
        ----------
        operation_name : str, optional
            Name of the operation being timed
        save_path : str, optional
            Path to save timing data
        """
        self.operation_name = operation_name
        self.save_path = save_path
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        
        if self.save_path and self.operation_name:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Load existing timing data if available
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    timing_data = json.load(f)
            else:
                timing_data = {}
            
            # Update timing data
            if self.operation_name not in timing_data:
                timing_data[self.operation_name] = []
            
            timing_data[self.operation_name].append(self.elapsed)
            
            # Save timing data
            with open(self.save_path, 'w') as f:
                json.dump(timing_data, f, indent=4)


# =====================================================================
# Model Creation Functions
# =====================================================================

def create_causal_model(graph_structure=None, n_vars=4, auto_coeff=0.8, cross_coeff=0.5, noise_sigma=0.5):
    """
    Create a full causal model from a graph structure or with default connectivity.
    
    Parameters
    ----------
    graph_structure : dict or ndarray, optional
        Graph structure specifying connections between variables.
        If None, a default structure is created.
    n_vars : int
        Number of variables (used if graph_structure is None)
    auto_coeff : float
        Coefficient for auto-regressive links
    cross_coeff : float
        Coefficient for cross-variable links
    noise_sigma : float
        Standard deviation of noise
        
    Returns
    -------
    tuple
        (links, noises) - Structural causal model components
    """
    # Define linear function
    linear_func = lambda x: x
    
    # Create links based on input or default structure
    if graph_structure is None:
        # Create default structure with configurable strengths
        links = {
            0: [((0, -1), auto_coeff, linear_func), 
                ((2, -3), cross_coeff * 0.7, linear_func)],
            1: [((1, -1), auto_coeff, linear_func), 
                ((0, -1), cross_coeff, linear_func)],
            2: [((2, -1), auto_coeff, linear_func), 
                ((1, -1), cross_coeff, linear_func)],       
            3: [((3, -1), auto_coeff, linear_func),
                ((0, -3), cross_coeff, linear_func), 
                ((1, -2), cross_coeff, linear_func), 
                ((2, -3), cross_coeff, linear_func)]
        }
    elif isinstance(graph_structure, np.ndarray):
        n_vars = graph_structure.shape[0]
        links = {}
        
        # Create links from graph structure
        for i in range(n_vars):
            links[i] = []
            for j in range(n_vars):
                for tau in range(graph_structure.shape[2]):
                    if graph_structure[i, j, tau] != 0:
                        # Add link with appropriate coefficient
                        if i == j:  # Auto-link
                            links[i].append(((i, -tau-1), auto_coeff, linear_func))
                        else:  # Cross-link
                            links[i].append(((j, -tau-1), cross_coeff, linear_func))
    else:
        # Assume graph_structure is already a links dictionary
        links = graph_structure
        n_vars = max(links.keys()) + 1
    
    # Create noise distributions
    noises = [lambda size: np.random.normal(0, noise_sigma, size) for _ in range(n_vars)]
    
    return links, noises


def create_stable_model(n_vars=4, auto_coeff=0.8, cross_coeff=0.5, noise_sigma=0.5):
    """
    Create a stable causal model with carefully scaled coefficients to avoid non-stationarity.
    
    Parameters
    ----------
    n_vars : int
        Number of variables
    auto_coeff : float
        Base coefficient for auto-regressive links
    cross_coeff : float
        Base coefficient for cross-variable links
    noise_sigma : float
        Standard deviation of noise
        
    Returns
    -------
    tuple
        (links, noises) - Structural causal model components
    """
    linear_func = lambda x: x
    
    # Scale coefficients to maintain stability
    auto_coeffs = []
    for i in range(n_vars):
        # Scale autocorrelation based on number of parents
        if i == 0:  # First variable has fewer cross-links
            scaled_auto = auto_coeff * 0.7
        elif i == n_vars - 1:  # Last variable has more cross-links
            scaled_auto = auto_coeff * 0.8
        else:  # Middle variables have medium number of cross-links
            scaled_auto = auto_coeff * 0.9
        auto_coeffs.append(scaled_auto)
    
    # Calculate cross-coefficients to ensure stability
    cross_coeffs = []
    for i in range(n_vars):
        max_cross = (0.95 - auto_coeffs[i]) / 2.0
        scaled_cross = min(cross_coeff, max_cross)
        cross_coeffs.append(scaled_cross)
    
    # Create links with fixed structure but configurable strengths
    links = {
        0: [((0, -1), auto_coeffs[0], linear_func), 
            ((2, -3), cross_coeffs[0] * 0.7, linear_func)],
        1: [((1, -1), auto_coeffs[1], linear_func), 
            ((0, -1), cross_coeffs[1], linear_func)],
        2: [((2, -1), auto_coeffs[2], linear_func), 
            ((1, -1), cross_coeffs[2], linear_func)],       
        3: [((3, -1), auto_coeffs[3], linear_func),
            ((0, -3), cross_coeffs[3], linear_func), 
            ((1, -2), cross_coeffs[3], linear_func), 
            ((2, -3), cross_coeffs[3], linear_func)]
    }
    
    # Create noise distributions
    noises = [lambda size: np.random.normal(0, noise_sigma, size) for _ in range(n_vars)]
    
    return links, noises


# =====================================================================
# Data Generation Functions
# =====================================================================

def generate_dataset(links, T=500, noises=None, seed=None, save_path=None):
    """
    Generate a dataset from a causal model.
    
    Parameters
    ----------
    links : dict
        Links dictionary specifying the causal model
    T : int
        Length of time series
    noises : list, optional
        List of noise generation functions
    seed : int, optional
        Random seed
    save_path : str, optional
        Path to save the dataset
        
    Returns
    -------
    tuple
        (dataframe, elapsed_time) - Generated time series data and generation time
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Measure time
    with Timer("data_generation", save_path) as timer:
        # Generate data
        data, nonstat = toys.structural_causal_process(links=links, T=T, noises=noises)
    
    # Create variable names
    n_vars = data.shape[1]
    var_names = [f'X{i+1}' for i in range(n_vars)]
    
    # Convert to DataFrame
    df = pp.DataFrame(data, var_names=var_names)
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save data and metadata
        np.save(f"{save_path}_data.npy", data)
        
        metadata = {
            'var_names': var_names,
            'T': T,
            'seed': seed,
            'nonstat': bool(nonstat),
            'generation_time': timer.elapsed
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    return df, timer.elapsed


# =====================================================================
# Causal Discovery Functions
# =====================================================================

def run_pcmci(dataset, pc_alpha=0.05, tau_max=5, cond_ind_test=None, save_path=None):
    """
    Run standard PCMCI causal discovery.
    
    Parameters
    ----------
    dataset : DataFrame
        Time series data
    pc_alpha : float
        Significance level for conditional independence tests
    tau_max : int
        Maximum time lag
    cond_ind_test : object, optional
        Conditional independence test (defaults to ParCorr)
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (results, pcmci, elapsed_time) - PCMCI results, object, and execution time
    """
    if cond_ind_test is None:
        cond_ind_test = ParCorr()
    
    with Timer("pcmci_discovery", save_path) as timer:
        pcmci = PCMCI(dataframe=dataset, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save graph and results
        np.save(f"{save_path}_graph.npy", results['graph'])
        np.save(f"{save_path}_val_matrix.npy", results['val_matrix'])
        
        # Save metadata
        metadata = {
            'pc_alpha': pc_alpha,
            'tau_max': tau_max,
            'test_type': cond_ind_test.__class__.__name__,
            'discovery_time': timer.elapsed
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    return results, pcmci, timer.elapsed


def run_bootstrap_pcmci(pcmci, pc_alpha=0.05, tau_max=5, n_boot=100, boot_blocklength=10, save_path=None):
    """
    Run bootstrapped PCMCI causal discovery.
    
    Parameters
    ----------
    pcmci : PCMCI
        Initialized PCMCI object
    pc_alpha : float
        Significance level
    tau_max : int
        Maximum time lag
    n_boot : int
        Number of bootstrap samples
    boot_blocklength : int
        Block length for bootstrap
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (results, bagged_graph, elapsed_time) - Bootstrap results, bagged graph, and execution time
    """
    # Run bootstrap
    with Timer("bootstrap_discovery", save_path) as timer:
        results = pcmci.run_bootstrap_of(
            method='run_pcmci', 
            method_args={'tau_max': tau_max, 'pc_alpha': pc_alpha}, 
            boot_samples=n_boot,
            boot_blocklength=boot_blocklength
        )
    
    # Create bagged graph
    bagged_graph = results['summary_results']['most_frequent_links']
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save graphs and results
        np.save(f"{save_path}_bagged_graph.npy", bagged_graph)
        
        # Save boot graphs separately
        boot_graphs = results['boot_results']['graph']
        np.save(f"{save_path}_boot_graphs.npy", boot_graphs)
        
        # Save metadata
        metadata = {
            'pc_alpha': pc_alpha,
            'tau_max': tau_max,
            'n_boot': n_boot,
            'boot_blocklength': boot_blocklength,
            'bootstrap_time': timer.elapsed
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    return results, bagged_graph, timer.elapsed


def run_causal_discovery(dataset, pc_alpha=0.05, tau_max=5, n_boot=100, 
                        cond_ind_test=None, save_path=None):
    """
    Run complete causal discovery with standard and bootstrapped PCMCI.
    
    Parameters
    ----------
    dataset : DataFrame
        Time series data
    pc_alpha : float
        Significance level
    tau_max : int
        Maximum time lag
    n_boot : int
        Number of bootstrap samples
    cond_ind_test : object, optional
        Conditional independence test
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (discovery_results, timings) - Complete discovery results and timing information
    """
    timings = {}
    
    # Run standard PCMCI
    pcmci_results, pcmci, pcmci_time = run_pcmci(
        dataset, pc_alpha, tau_max, cond_ind_test, 
        save_path=f"{save_path}_pcmci" if save_path else None
    )
    timings['pcmci'] = pcmci_time
    
    # Run bootstrapped PCMCI
    bootstrap_results, bagged_graph, bootstrap_time = run_bootstrap_pcmci(
        pcmci, pc_alpha, tau_max, n_boot, 
        save_path=f"{save_path}_bootstrap" if save_path else None
    )
    timings['bootstrap'] = bootstrap_time
    
    # Return combined results
    discovery_results = {
        'pcmci': pcmci_results,
        'bootstrap': bootstrap_results,
        'bagged_graph': bagged_graph
    }
    
    return discovery_results, timings


# =====================================================================
# Causal Effect Estimation Functions
# =====================================================================

def get_groundtruth(links, X, Y, noises=None, seed=1, T_int=100):
    """
    Calculate ground truth causal effect through intervention.
    
    Parameters
    ----------
    links : dict
        Links dictionary for structural causal model
    X : tuple
        Variable to intervene on, in format (var_idx, time_idx) 
    Y : tuple
        Outcome variable to measure, in format (var_idx, time_idx)
    noises : dict, optional
        Noise distributions for each variable
    seed : int, optional
        Random seed for ensemble generation
    T_int : int, optional
        Length of time series
        
    Returns
    -------
    float
        Mean difference in outcome between interventions
    """
    with Timer("groundtruth_calculation") as timer:
        # Set up intervention values for treatment
        intervention1 = np.ones(T_int)
        intervention1[:] = np.nan
        intervention1[T_int-1+X[1]] = 1

        # Generate ensemble with intervention=1
        intervention_data1, nonstat = toys.structural_causal_process_ensemble(
            realizations=20, 
            ensemble_seed=seed,
            links=links, 
            T=T_int, 
            noises=noises,
            intervention={X[0]: intervention1}, 
            intervention_type='hard',
        )

        # Set up intervention values for control condition
        intervention2 = np.ones(T_int)
        intervention2[:] = np.nan
        intervention2[T_int-1+X[1]] = 0

        # Generate ensemble with intervention=0
        intervention_data2, nonstat = toys.structural_causal_process_ensemble(
            realizations=20, 
            ensemble_seed=seed,
            links=links, 
            T=T_int, 
            noises=noises,
            intervention={X[0]: intervention2}, 
            intervention_type='hard',
        )

        # Calculate average treatment effect
        effect = (intervention_data1 - intervention_data2)[:, -1, Y[0]].mean(axis=0)
    
    return effect, timer.elapsed


def estimate_causal_effect(graph, dataframe, X, Y, save_path=None):
    """
    Estimate causal effect from X to Y using the given graph.
    
    Parameters
    ----------
    graph : numpy.ndarray
        Causal graph from PCMCI
    dataframe : DataFrame
        Time series data
    X : tuple
        Variable to intervene on (var_idx, time_idx)
    Y : tuple
        Outcome variable (var_idx, time_idx)
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (effect, elapsed_time) - Estimated causal effect and execution time
    """
    with Timer("effect_estimation", save_path) as timer:
        try:
            # Initialize CausalEffects object
            causal_effects = CausalEffects(
                graph, 
                graph_type='stationary_admg', 
                X=[X], Y=[Y], 
                S=None, 
                hidden_variables=None, 
                verbosity=0
            )

            # Check if there is a valid causal path
            if not causal_effects.check_XYS_paths()[0]:
                effect = np.nan
            else:
                # Fit the causal effect model
                causal_effects.fit_total_effect(
                    dataframe=dataframe, 
                    estimator=None,  # Use default linear regression
                    adjustment_set='optimal',
                    conditional_estimator=None,  
                    data_transform=None,
                    mask_type=None,
                )
                
                # Predict intervention outcomes
                intervention_data = np.ones((1, 1))  # X=1
                y1 = causal_effects.predict_total_effect(intervention_data=intervention_data)
                    
                intervention_data = np.zeros((1, 1))  # X=0
                y2 = causal_effects.predict_total_effect(intervention_data=intervention_data)
                
                # Calculate causal effect as the difference
                effect = (y1 - y2)[0]
        except ValueError as e:
            # Handle invalid graph edge or other ValueError
            effect = np.nan
            print(f"Warning: {str(e)} Setting effect to NaN.")
        except Exception as e:
            # Handle any other exceptions
            effect = np.nan
            print(f"Warning: Unexpected error in effect estimation: {str(e)}. Setting effect to NaN.")
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save effect estimate
        effect_data = {
            'X': X,
            'Y': Y,
            'effect': float(effect) if not np.isnan(effect) else None,
            'estimation_time': timer.elapsed
        }
        
        with open(save_path, 'w') as f:
            json.dump(effect_data, f, indent=4)
    
    return effect, timer.elapsed

def estimate_bootstrap_effects(bootstrap_graphs, dataframe, X, Y, save_path=None):
    """
    Estimate causal effects using bootstrap graphs.
    
    Parameters
    ----------
    bootstrap_graphs : list
        List of bootstrap graphs
    dataframe : DataFrame
        Time series data
    X : tuple
        Variable to intervene on (var_idx, time_idx)
    Y : tuple
        Outcome variable (var_idx, time_idx)
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (result_dict, elapsed_time) - Dictionary with bootstrap effects and statistics, and execution time
    """
    with Timer("bootstrap_effect_estimation", save_path) as timer:
        bootstrap_effects = []
        
        # Estimate effect for each bootstrap graph
        for i, b_graph in enumerate(bootstrap_graphs):
            try:
                effect, _ = estimate_causal_effect(b_graph, dataframe, X, Y)
                bootstrap_effects.append(effect)
            except Exception as e:
                print(f"Warning: Error in bootstrap sample {i}: {str(e)}")
                bootstrap_effects.append(np.nan)
        
        # Calculate statistics
        effects = np.array(bootstrap_effects)
        
        # Handle NaN values
        nan_mask = np.isnan(effects)
        n_nan = np.sum(nan_mask)
        
        # If all values are NaN, return special case
        if n_nan == len(effects):
            stats = {
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_total': len(effects),
                'n_nan': n_nan,
                'estimation_success_rate': 0.0
            }
        else:
            # Filter out NaN values
            valid_effects = effects[~nan_mask]
            
            # Calculate statistics
            mean = np.mean(valid_effects)
            std = np.std(valid_effects)
            
            # Calculate confidence interval
            n_valid = len(valid_effects)
            se = std / np.sqrt(n_valid)
            ci = stats.t.interval(0.95, df=n_valid-1, loc=mean, scale=se)
            
            stats = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_total': int(len(effects)),
                'n_nan': int(n_nan),
                'estimation_success_rate': float(n_valid / len(effects))
            }
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save effects and statistics
        result = {
            'X': X,
            'Y': Y,
            'bootstrap_effects': [float(e) if not np.isnan(e) else None for e in bootstrap_effects],
            'stats': stats,
            'estimation_time': timer.elapsed
        }
        
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
    
    result_dict = {
        'effects': bootstrap_effects,
        'stats': stats
    }
    
    return result_dict, timer.elapsed


def estimate_all_effects(discovery_results, dataframe, X, Y, links=None, noises=None, save_path=None):
    """
    Estimate causal effects using all methods.
    
    Parameters
    ----------
    discovery_results : dict
        Results from causal discovery
    dataframe : DataFrame
        Time series data
    X : tuple
        Variable to intervene on (var_idx, time_idx)
    Y : tuple
        Outcome variable (var_idx, time_idx)
    links : dict, optional
        Links dictionary for ground truth
    noises : dict, optional
        Noise distributions for ground truth
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    tuple
        (results, timings) - Dictionary with all effect estimates and timing information
    """
    results = {}
    timings = {}
    
    # Get ground truth if links are provided
    if links is not None:
        true_effect, true_time = get_groundtruth(links, X, Y, noises)
        results['true_effect'] = true_effect
        timings['true_effect'] = true_time
    
    # Estimate with PCMCI graph
    pcmci_graph = discovery_results['pcmci']['graph']
    pcmci_effect, pcmci_time = estimate_causal_effect(
        pcmci_graph, dataframe, X, Y, 
        save_path=f"{save_path}_pcmci.json" if save_path else None
    )
    results['pcmci_effect'] = pcmci_effect
    timings['pcmci_effect'] = pcmci_time
    
    # Estimate with bagged graph
    bagged_graph = discovery_results['bagged_graph']
    bagged_effect, bagged_time = estimate_causal_effect(
        bagged_graph, dataframe, X, Y, 
        save_path=f"{save_path}_bagged.json" if save_path else None
    )
    results['bagged_effect'] = bagged_effect
    timings['bagged_effect'] = bagged_time
    
    # Estimate with bootstrap graphs
    bootstrap_graphs = discovery_results['bootstrap']['boot_results']['graph']
    bootstrap_results, bootstrap_time = estimate_bootstrap_effects(
        bootstrap_graphs, dataframe, X, Y, 
        save_path=f"{save_path}_bootstrap.json" if save_path else None
    )
    results['bootstrap_effects'] = bootstrap_results['effects']
    results['bootstrap_stats'] = bootstrap_results['stats']
    timings['bootstrap_effects'] = bootstrap_time
    
    # Save overall results if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create serializable version
        serializable_results = {
            'X': X,
            'Y': Y,
            'true_effect': float(results['true_effect']) if 'true_effect' in results else None,
            'pcmci_effect': float(pcmci_effect) if not np.isnan(pcmci_effect) else None,
            'bagged_effect': float(bagged_effect) if not np.isnan(bagged_effect) else None,
            'bootstrap_stats': results['bootstrap_stats'] if 'bootstrap_stats' in results else None,
            'timings': {k: float(v) for k, v in timings.items()}
        }
        
        with open(f"{save_path}_all.json", 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    return results, timings
