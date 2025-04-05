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
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

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
                

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


# =====================================================================
# Model Creation Functions
# =====================================================================

def create_consistent_noise(seed, sigma, size):
    np.random.seed(seed)
    return np.random.normal(0, sigma, size)





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
                ((2, -1), cross_coeff, linear_func)],
            1: [((1, -1), auto_coeff, linear_func), 
                ((0, -1), cross_coeff, linear_func)],
            2: [((2, -1), auto_coeff, linear_func)],       
            3: [((3, -1), auto_coeff, linear_func),
                ((0, -1), cross_coeff, linear_func), 
                ((1, -1), cross_coeff, linear_func), 
                ((2, -2), cross_coeff, linear_func)]
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
    # Usage
    # Generate unique seeds for each noise generator
    unique_seeds = np.random.SeedSequence(42).spawn(n_vars)
    noises = [lambda size, seed=seed.generate_state(1)[0], sigma=noise_sigma: 
          create_consistent_noise(seed, sigma, size) 
          for seed in unique_seeds]
    
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
            ((2, -3), cross_coeffs[0] , linear_func)],
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
    if seed is None:
        seed = np.random.randint(0,2**31-1)
    
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
        
        linkString =""
        for var, parents in links.items():
            linkString += f"Variable X{var+1} has parents:" +"\n"
            for parent in parents:
                parent_var, parent_lag = parent[0]
                coeff = parent[1]
                func = parent[2].__name__ if hasattr(parent[2], '__name__') else str(parent[2])
                linkString += f"  X{parent_var+1}(t{parent_lag}) with coefficient {coeff:.4f} and function {func}" +"\n"
        
        metadata = {
            'var_names': var_names,
            'links': linkString,
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
    bagged_val_matrix = results['summary_results']['val_matrix_mean']
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save graphs and results
        np.save(f"{save_path}_bagged_graph.npy", bagged_graph)
        np.save(f"{save_path}_val_matrix.npy", bagged_val_matrix)
        
        # Save boot graphs separately
        boot_graphs = results['boot_results']['graph']
        boot_val_matrix = results['boot_results']['val_matrix']
        np.save(f"{save_path}_boot_graphs.npy", boot_graphs)
        np.save(f"{save_path}_val_matrix.npy", boot_val_matrix)
        
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

def get_groundtruth(links, X, Y, noises=None, seed=None, T_int=500):
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
    if seed is None:
        #smaller max-seed since its multiplied in the ensemble function
        seed = np.random.randint(0,2**16-1)
    
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

# Function to get the optimal adjustment set for a given graph
def get_optimal_adjustment_set(graph, X, Y, var_names=None):
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
            return None, "No valid causal path found"
        
        # Get optimal adjustment set
        optimal_set = causal_effects.get_optimal_set()
        
        # Convert node indices to variable names if provided
        if var_names is not None:
            named_optimal_set = []
            for node in optimal_set:
                var_idx, lag = node
                var_name = var_names[var_idx]
                named_optimal_set.append(f"{var_name}(t{lag if lag < 0 else ''})")
            return optimal_set, named_optimal_set
        else:
            return optimal_set, None
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# Run the analysis for each graph
def compare_adjustment_sets(discovery_results, X, Y, true_graph=None, var_names=None):
    results = {}
    
    # Extract graphs
    pcmci_graph = discovery_results['pcmci']['graph']
    bagged_graph = discovery_results['bagged_graph']
    
    # Define graph dictionary
    graphs = {
        "PCMCI Graph": pcmci_graph,
        "Bagged Graph": bagged_graph
    }
    
    # Add true graph if provided
    if true_graph is not None:
        graphs["True Graph"] = true_graph
    
    # Get adjustment sets for each graph
    for name, graph in graphs.items():
        optimal_set, readable_set = get_optimal_adjustment_set(graph, X, Y, var_names)
        results[name] = {
            "optimal_set": optimal_set,
            "readable_set": readable_set
        }
    
    return results

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
                    estimator=LinearRegression(),  # Use default linear regression
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
                'mean': None,
                'std': None,
                'ci_lower': None,
                'ci_upper': None,
                'n_total': int(len(effects)),
                'n_nan': None,
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
            ci = scipy_stats.t.interval(0.95, df=n_valid-1, loc=mean, scale=se)
            
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
    
    # Estimate with True Graph
    if links is not None:
        true_graph = PCMCI.get_graph_from_dict(links)
        true_graph_effect, true_graph_time = estimate_causal_effect(
            true_graph, dataframe, X, Y, 
            save_path=f"{save_path}_true_graph.json" if save_path else None
        )
        results['true_graph_effect'] = true_graph_effect
        timings['true_graph_effect'] = true_graph_time
    
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
            'true_graph_effect': float(results['true_graph_effect']) if 'true_effect' in results else None,
            'pcmci_effect': float(pcmci_effect) if not np.isnan(pcmci_effect) else None,
            'bagged_effect': float(bagged_effect) if not np.isnan(bagged_effect) else None,
            'bootstrap_stats': results['bootstrap_stats'] if 'bootstrap_stats' in results else None,
            'timings': {k: float(v) for k, v in timings.items()}
        }
        
        with open(f"{save_path}_all.json", 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    return results, timings


def estimate_causal_effect_with_confidence(graph, dataframe, X, Y, save_path=None):
    """
    Estimate causal effect from X to Y with confidence intervals using tigramite.
    
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
        (effect, adjustment_set, confidence_interval, elapsed_time)
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
                adjustment_set = None
                confidence_interval = (np.nan, np.nan)
            else:
                # Get optimal adjustment set before fitting
                adjustment_set = causal_effects.get_optimal_set()
                
                # Fit the causal effect model using standard LinearRegression
                causal_effects.fit_total_effect(
                    dataframe=dataframe, 
                    estimator=LinearRegression(),
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
                
                # Now calculate confidence intervals using statsmodels
                # Extract data used for the model to refit with statsmodels
                # First get the internal data from causal_effects
                try:
                    # Calculate the causal effect using statsmodels directly
                    if adjustment_set is not None:
                        # Get the data for our variables
                        X_var, X_lag = X
                        Y_var, Y_lag = Y
                        
                        # Get the full time series data
                        data = dataframe.values[0]
                        
                        # Select predictor and outcome time points accounting for lags
                        max_lag = abs(min(X_lag, Y_lag, 0))
                        T = data.shape[0]
                        time_points = np.arange(max_lag, T)
                        
                        # Create design matrix with the treatment variable
                        X_vals = data[time_points + X_lag, X_var].reshape(-1, 1)
                        X_design = X_vals  # Start with just the treatment
                        
                        # Add adjustment variables if needed
                        for adj_var, adj_lag in adjustment_set:
                            adj_vals = data[time_points + adj_lag, adj_var].reshape(-1, 1)
                            X_design = np.hstack([X_design, adj_vals])
                        
                        # Get the outcome variable
                        Y_vals = data[time_points + Y_lag, Y_var]
                        
                        # Add constant
                        X_design = sm.add_constant(X_design)
                        
                        # Fit OLS model
                        sm_model = sm.OLS(Y_vals, X_design)
                        sm_results = sm_model.fit()
                        
                        # Get coefficient for treatment (X) - should be at index 1
                        effect = sm_results.params[1]
                        
                        # Get standard error
                        se = sm_results.bse[1]
                        
                        # Calculate confidence interval
                        ci_lower = effect - 1.96 * se
                        ci_upper = effect + 1.96 * se
                        confidence_interval = (ci_lower, ci_upper)
                    else:
                        confidence_interval = (np.nan, np.nan)
                except:
                    # If extraction fails, return NaN confidence interval
                    confidence_interval = (np.nan, np.nan)
        except ValueError as e:
            # Handle invalid graph edge or other ValueError
            effect = np.nan
            adjustment_set = None
            confidence_interval = (np.nan, np.nan)
            print(f"Warning: {str(e)} Setting effect to NaN.")
        except Exception as e:
            # Handle any other exceptions
            effect = np.nan
            adjustment_set = None
            confidence_interval = (np.nan, np.nan)
            print(f"Warning: Unexpected error in effect estimation: {str(e)}. Setting effect to NaN.")
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print (type(adjustment_set), adjustment_set)
        # Save effect estimate with adjustment set and confidence interval
        effect_data = {
            'X': X,
            'Y': Y,
            'effect': float(effect) if not np.isnan(effect) else None,
            'adjustment_set': adjustment_set if adjustment_set is not None else None,
            'adjustment_set_size': len(adjustment_set) if adjustment_set is not None else 0,
            'ci_lower': float(confidence_interval[0]) if not np.isnan(confidence_interval[0]) else None,
            'ci_upper': float(confidence_interval[1]) if not np.isnan(confidence_interval[1]) else None,
            'estimation_time': timer.elapsed
        }
        
        with open(save_path, 'w') as f:
            json.dump(effect_data, f, indent=4,cls=NumpyEncoder)
    
    return effect, adjustment_set, confidence_interval, timer.elapsed


def estimate_bootstrap_effects_with_confidence(bootstrap_graphs, dataframe, X, Y, save_path=None):
    """
    Estimate causal effects using bootstrap graphs with confidence intervals.
    
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
        (result_dict, elapsed_time) - Dictionary with bootstrap effects, confidence intervals, and statistics
    """
    with Timer("bootstrap_effect_estimation", save_path) as timer:
        bootstrap_effects = []
        adjustment_sets = []
        adjustment_set_sizes = []
        confidence_intervals = []
        
        # Track unique graphs and their effects to optimize computation
        unique_graphs = {}
        
        # Estimate effect for each bootstrap graph
        for i, b_graph in enumerate(bootstrap_graphs):
            # Convert graph to hashable format for dictionary key
            graph_key = str(b_graph)
            
            # Check if we've already computed for this exact graph
            if graph_key in unique_graphs:
                effect, adj_set, ci = unique_graphs[graph_key]
                bootstrap_effects.append(effect)
                adjustment_sets.append(adj_set)
                confidence_intervals.append(ci)
                if adj_set is not None:
                    adjustment_set_sizes.append(len(adj_set))
                else:
                    adjustment_set_sizes.append(0)
            else:
                try:
                    effect, adj_set, ci, _ = estimate_causal_effect_with_confidence(b_graph, dataframe, X, Y)
                    bootstrap_effects.append(effect)
                    adjustment_sets.append(adj_set)
                    confidence_intervals.append(ci)
                    if adj_set is not None:
                        adjustment_set_sizes.append(len(adj_set))
                    else:
                        adjustment_set_sizes.append(0)
                    
                    # Store for potential reuse
                    unique_graphs[graph_key] = (effect, adj_set, ci)
                except Exception as e:
                    print(f"Warning: Error in bootstrap sample {i}: {str(e)}")
                    bootstrap_effects.append(np.nan)
                    adjustment_sets.append(None)
                    confidence_intervals.append((np.nan, np.nan))
                    adjustment_set_sizes.append(0)
        
        # Calculate statistics for effects
        effects = np.array(bootstrap_effects)
        
        # Handle NaN values
        nan_mask = np.isnan(effects)
        n_nan = np.sum(nan_mask)
        
        # Calculate statistics for adjustment set sizes
        adj_sizes = np.array(adjustment_set_sizes)
        
        # Extract confidence interval metrics
        ci_widths = []
        for ci in confidence_intervals:
            if ci[0] is not np.nan and ci[1] is not np.nan:
                ci_widths.append(ci[1] - ci[0])
            else:
                ci_widths.append(np.nan)
        
        # If all values are NaN, return special case for effects
        if n_nan == len(effects):
            effect_stats = {
                'mean': None,
                'std': None,
                'ci_lower': None,
                'ci_upper': None,
                'n_total': int(len(effects)),
                'n_nan': int(n_nan),
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
            ci = scipy_stats.t.interval(0.95, df=n_valid-1, loc=mean, scale=se)
            
            effect_stats = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_total': int(len(effects)),
                'n_nan': int(n_nan),
                'estimation_success_rate': float(n_valid / len(effects))
            }
        
        # Statistics for adjustment sets
        adjustment_set_stats = {
            'mean_size': float(np.mean(adj_sizes)),
            'std_size': float(np.std(adj_sizes)),
            'min_size': int(np.min(adj_sizes)),
            'max_size': int(np.max(adj_sizes)),
            'unique_graphs': len(unique_graphs)
        }
        
        # Statistics for confidence intervals
        valid_ci_widths = np.array([w for w in ci_widths if not np.isnan(w)])
        if len(valid_ci_widths) > 0:
            ci_stats = {
                'mean_width': float(np.mean(valid_ci_widths)),
                'std_width': float(np.std(valid_ci_widths)),
                'min_width': float(np.min(valid_ci_widths)),
                'max_width': float(np.max(valid_ci_widths))
            }
        else:
            ci_stats = {
                'mean_width': None,
                'std_width': None,
                'min_width': None,
                'max_width': None
            }
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save effects, adjustment sets, confidence intervals and statistics
        result = {
            'X': X,
            'Y': Y,
            'bootstrap_effects': [float(e) if not np.isnan(e) else None for e in bootstrap_effects],
            'adjustment_set_sizes': [int(s) for s in adjustment_set_sizes],
            'confidence_intervals': [
                [float(ci[0]) if not np.isnan(ci[0]) else None, 
                 float(ci[1]) if not np.isnan(ci[1]) else None] 
                for ci in confidence_intervals
            ],
            'stats': effect_stats,
            'adjustment_set_stats': adjustment_set_stats,
            'confidence_interval_stats': ci_stats,
            'unique_graphs': len(unique_graphs),
            'estimation_time': timer.elapsed
        }
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
    
    result_dict = {
        'effects': bootstrap_effects,
        'adjustment_sets': adjustment_sets,
        'adjustment_set_sizes': adjustment_set_sizes,
        'confidence_intervals': confidence_intervals,
        'stats': effect_stats,
        'adjustment_set_stats': adjustment_set_stats,
        'confidence_interval_stats': ci_stats,
        'unique_graphs': len(unique_graphs)
    }
    
    return result_dict, timer.elapsed


def estimate_all_effects_with_confidence(discovery_results, dataframe, X, Y, links=None, noises=None, save_path=None):
    """
    Estimate causal effects using all methods with confidence intervals.
    
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
        (results, adjustment_sets, timings) - Dictionaries with all effect estimates and additional information
    """
    results = {}
    timings = {}
    adjustment_sets = {}
    confidence_intervals = {}
    
    # Get ground truth if links are provided
    if links is not None:
        true_effect, true_time = get_groundtruth(links, X, Y, noises)
        results['true_effect'] = true_effect
        timings['true_effect'] = true_time
    
    # Estimate with True Graph
    if links is not None:
        true_graph = PCMCI.get_graph_from_dict(links)
        true_graph_effect, true_adj_set, true_ci, true_graph_time = estimate_causal_effect_with_confidence(
            true_graph, dataframe, X, Y, 
            save_path=f"{save_path}_true_graph.json" if save_path else None
        )
        results['true_graph_effect'] = true_graph_effect
        adjustment_sets['true_graph'] = true_adj_set
        confidence_intervals['true_graph'] = true_ci
        timings['true_graph_effect'] = true_graph_time
    
    # Estimate with PCMCI graph
    pcmci_graph = discovery_results['pcmci']['graph']
    pcmci_effect, pcmci_adj_set, pcmci_ci, pcmci_time = estimate_causal_effect_with_confidence(
        pcmci_graph, dataframe, X, Y, 
        save_path=f"{save_path}_pcmci.json" if save_path else None
    )
    results['pcmci_effect'] = pcmci_effect
    adjustment_sets['pcmci'] = pcmci_adj_set
    confidence_intervals['pcmci'] = pcmci_ci
    timings['pcmci_effect'] = pcmci_time
    
    # Estimate with bagged graph
    bagged_graph = discovery_results['bagged_graph']
    bagged_effect, bagged_adj_set, bagged_ci, bagged_time = estimate_causal_effect_with_confidence(
        bagged_graph, dataframe, X, Y, 
        save_path=f"{save_path}_bagged.json" if save_path else None
    )
    results['bagged_effect'] = bagged_effect
    adjustment_sets['bagged'] = bagged_adj_set
    confidence_intervals['bagged'] = bagged_ci
    timings['bagged_effect'] = bagged_time
    
    # Estimate with bootstrap graphs
    bootstrap_graphs = discovery_results['bootstrap']['boot_results']['graph']
    bootstrap_results, bootstrap_time = estimate_bootstrap_effects_with_confidence(
        bootstrap_graphs, dataframe, X, Y, 
        save_path=f"{save_path}_bootstrap.json" if save_path else None
    )
    results['bootstrap_effects'] = bootstrap_results['effects']
    results['bootstrap_stats'] = bootstrap_results['stats']
    results['bootstrap_confidence_intervals'] = bootstrap_results['confidence_intervals']
    timings['bootstrap_effects'] = bootstrap_time
    
    # Save overall results if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create serializable version
        serializable_results = {
            'X': X,
            'Y': Y,
            'true_effect': float(results['true_effect']) if 'true_effect' in results else None,
            'true_graph_effect': float(results['true_graph_effect']) if 'true_graph_effect' in results else None,
            'pcmci_effect': float(pcmci_effect) if not np.isnan(pcmci_effect) else None,
            'bagged_effect': float(bagged_effect) if not np.isnan(bagged_effect) else None,
            'bootstrap_stats': results['bootstrap_stats'] if 'bootstrap_stats' in results else None,
            'adjustment_sets': {
                'true_graph': adjustment_sets.get('true_graph', []),
                'pcmci': adjustment_sets.get('pcmci', []),
                'bagged': adjustment_sets.get('bagged', []),
                'bootstrap_summary': bootstrap_results.get('adjustment_set_stats', {})
            },
            'confidence_intervals': {
                'true_graph': [float(ci) if not np.isnan(ci) else None for ci in confidence_intervals.get('true_graph', (np.nan, np.nan))],
                'pcmci': [float(ci) if not np.isnan(ci) else None for ci in confidence_intervals.get('pcmci', (np.nan, np.nan))],
                'bagged': [float(ci) if not np.isnan(ci) else None for ci in confidence_intervals.get('bagged', (np.nan, np.nan))],
                'bootstrap_summary': bootstrap_results.get('confidence_interval_stats', {})
            },
            'timings': {k: float(v) for k, v in timings.items()}
        }
        
        with open(f"{save_path}_all.json", 'w') as f:
            json.dump(serializable_results, f, indent=4,cls=NumpyEncoder)
    
    return results, adjustment_sets, confidence_intervals, timings


if __name__ == "__main__":

    auto_coeff=0.8
    cross_coeff=0.5
    noise_sigma=0.5
    n_vars=4
    T=500
    pc_alpha=0.05
    tau_max=5
    n_boot=10
    effect_pairs=None
    output_dir=None
    seed=None    

    effect_pairs = [((0, -3), (3, 0))]  # X1(t-3) → X4(t)
    

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
    