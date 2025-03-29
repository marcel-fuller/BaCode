"""
Core utilities for causal discovery and effect estimation.

This module provides essential functions for structural causal models,
effect estimation, and result evaluation. Works with tigramite
for time series causal analysis.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.causal_effects import CausalEffects
from sklearn.linear_model import LinearRegression

# =====================================================================
# SCM Generation
# =====================================================================

def create_simpler_scm(
    N=5, max_a=0.95, L=None, max_lag=5, 
    model_type='linear', noise_type='gaussian', noise_sigmas=None
):
    """
    Create a simple structural causal model with specific characteristics.
    
    Parameters
    ----------
    N : int
        Number of variables
    max_a : float
        Maximum autoregressive coefficient
    L : int, optional
        Number of cross-links, defaults to 1.5*N
    max_lag : int
        Maximum time lag
    model_type : str
        Type of model ('linear' or 'nonlinear')
    noise_type : str
        Type of noise ('gaussian' or 'mixed')
    noise_sigmas : list, optional
        List of noise standard deviations
        
    Returns
    -------
    tuple
        (links, noises)
    """
    if L is None:
        L = int(1.5 * N)
        
    # Choose transfer function based on model type
    if model_type == 'linear':
        f = lambda x: x  # Linear function
    elif model_type == 'nonlinear':
        f = lambda x: x + 5*x**2 * np.exp(-x**2/20)  # Nonlinear function
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate links and noises using tigramite's function
    links, noises = toys.generate_structural_causal_process(
        N=N, L=L, max_lag=max_lag, 
        auto_coeffs=[max_a], 
        noise_sigmas=[0.5] if noise_sigmas is None else noise_sigmas
    )
    
    return links, noises


def create_configurable_links(auto_coeff, cross_coeff, n_vars=4):
    """
    Build links dictionary with custom coefficients.
    
    Creates a fixed structure with auto-regressive links and 
    cross-variable links, but with configurable strengths.
    
    Parameters
    ----------
    auto_coeff : float
        Coefficient for auto-regressive links
    cross_coeff : float
        Coefficient for cross-variable links
    n_vars : int
        Number of variables
        
    Returns
    -------
    dict
        Links dictionary for use with tigramite
    """
    # Define linear function
    linear_func = lambda x: x
    
    # Create links with fixed structure but configurable strengths
    links = {
        0: [((0, -1), auto_coeff, linear_func), ((3, -2), cross_coeff, linear_func), ((2, -3), cross_coeff, linear_func)],  # X1
        1: [((1, -1), auto_coeff, linear_func), ((0, -2), cross_coeff, linear_func)],  # X2
        2: [((2, -1), auto_coeff, linear_func), ((1, -2), cross_coeff, linear_func)],  # X3        
        3: [((3, -1), auto_coeff, linear_func), ((2, -1), cross_coeff, linear_func)]   # X4
    }
    return links

def create_stable_links(auto_coeff, cross_coeff, n_vars=4):
    """Create links with more stable parameters to reduce non-stationarity."""
    linear_func = lambda x: x
    
    # Scale coefficients to maintain stability
    auto_coeffs = []
    for i in range(n_vars):
        # Scale autocorrelation based on number of parents
        if i == 0:  # First variable has two cross-links
            scaled_auto = auto_coeff * 0.7
        elif i == n_vars - 1:  # Last variable has one cross-link
            scaled_auto = auto_coeff * 0.8
        else:  # Middle variables have one cross-link
            scaled_auto = auto_coeff * 0.9
        auto_coeffs.append(scaled_auto)
    
    # Calculate cross-coefficients to ensure sum < 1
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
    return links


def create_noise_distributions(noise_sigma, n_vars=4):
    """
    Create noise distribution generators.
    
    Parameters
    ----------
    noise_sigma : float
        Noise standard deviation
    n_vars : int
        Number of variables
        
    Returns
    -------
    list
        List of noise generation functions
    """
    return [lambda size: np.random.normal(0, noise_sigma, size) for _ in range(n_vars)]

# =====================================================================
# Effect Estimation
# =====================================================================

def get_groundtruth(links, X, Y, noises=None, seed=1, T_int=100):
    """
    Calculate intervention effect between two interventions on X measuring outcome on Y.
    
    Uses tigramite's SCM simulation to compute the ground truth effect.
    
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
    # Set up intervention values for treatment
    intervention1 = np.ones(T_int)
    intervention1[:] = np.nan
    intervention1[T_int-1+X[0][1]] = 1

    # Generate ensemble with intervention=1
    intervention_data1, nonstat = toys.structural_causal_process_ensemble(
        realizations=20, 
        ensemble_seed=seed,
        links=links, 
        T=T_int, 
        noises=noises,
        intervention={X[0][0]: intervention1}, 
        intervention_type='hard',
    )

    # Set up intervention values for control condition
    intervention2 = np.ones(T_int)
    intervention2[:] = np.nan
    intervention2[T_int-1+X[0][1]] = 0

    # Generate ensemble with intervention=0
    intervention_data2, nonstat = toys.structural_causal_process_ensemble(
        realizations=20, 
        ensemble_seed=seed,
        links=links, 
        T=T_int, 
        noises=noises,
        intervention={X[0][0]: intervention2}, 
        intervention_type='hard',
    )

    # Calculate average treatment effect
    return (intervention_data1 - intervention_data2)[:, -1, Y[0][0]].mean(axis=0)


def get_estimate(graph, dataframe, X, Y, verbosity=0):
    """
    Estimate causal effect from X to Y using the given graph.
    
    Uses tigramite's CausalEffects to compute the effect.
    
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
    verbosity : int, optional
        Verbosity level
        
    Returns
    -------
    float
        Estimated causal effect
    """
    # Initialize CausalEffects object
    causal_effects = CausalEffects(
        graph, 
        graph_type='stationary_admg', 
        X=[X], Y=[Y], 
        S=None, 
        hidden_variables=None, 
        verbosity=verbosity
    )

    # Check if there is a valid causal path
    if not causal_effects.check_XYS_paths()[0]:
        if verbosity > 0:
            print(f"No causal path from {X} to {Y}")
        return np.nan
    
    # Fit the causal effect model
    causal_effects.fit_total_effect(
        dataframe=dataframe, 
        estimator=LinearRegression(),
        adjustment_set='optimal',
        conditional_estimator=None,  
        data_transform=None,
        mask_type=None,
    )
    
    # Predict intervention outcomes
    intervention_data = 1.0 * np.ones((1, 1))
    y1 = causal_effects.predict_total_effect(
        intervention_data=intervention_data,
    )
        
    intervention_data = 0.0 * np.ones((1, 1))
    y2 = causal_effects.predict_total_effect( 
        intervention_data=intervention_data,
    )
    
    # Calculate causal effect as the difference
    beta = (y1 - y2)
    return beta[0]


def calculate_bootstrap_stats(bootstrap_effects):
    """
    Calculate statistics from bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list
        List of effect estimates from bootstrap graphs
        
    Returns
    -------
    dict
        Dictionary with statistical measures
    """
    effects = np.array(bootstrap_effects)
    
    # Handle NaN values
    nan_mask = np.isnan(effects)
    n_nan = np.sum(nan_mask)
    
    # If all values are NaN, return special case
    if n_nan == len(effects):
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_total': len(effects),
            'n_nan': n_nan,
            'estimation_success_rate': 0.0
        }
    
    # Filter out NaN values
    valid_effects = effects[~nan_mask]
    
    # Calculate statistics
    mean = np.mean(valid_effects)
    std = np.std(valid_effects)
    
    # Calculate confidence interval
    n_valid = len(valid_effects)
    se = std / np.sqrt(n_valid)
    ci = stats.t.interval(0.95, df=n_valid-1, loc=mean, scale=se)
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_total': len(effects),
        'n_nan': n_nan,
        'estimation_success_rate': n_valid / len(effects)
    }

# =====================================================================
# Evaluation Functions
# =====================================================================

def evaluate_estimation_performance(all_effects, metrics=None):
    """
    Evaluate causal effect estimation performance across datasets.
    
    Parameters
    ----------
    all_effects : dict
        Dictionary of effect estimation results
    metrics : list, optional
        List of metrics to calculate
        
    Returns
    -------
    dict
        Dictionary of evaluation results
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'bias', 'ci_coverage']
    
    results = {}
    
    # Collect all effect pairs
    effect_pairs = set()
    for dataset_effects in all_effects.values():
        effect_pairs.update(dataset_effects.keys())
    
    for effect_pair in effect_pairs:
        pair_results = {
            'pcmci': {'values': [], 'errors': []},
            'bagged': {'values': [], 'errors': []},
            'bootstrap': {'values': [], 'errors': [], 'ci_coverage': []}
        }
        
        # Collect values and errors for each dataset
        for dataset_id, dataset_effects in all_effects.items():
            if effect_pair in dataset_effects:
                effect_data = dataset_effects[effect_pair]
                
                true_effect = effect_data.get('true_effect')
                if true_effect is None:
                    continue
                
                # PCMCI
                pcmci_effect = effect_data.get('pcmci_effect')
                if not np.isnan(pcmci_effect):
                    pair_results['pcmci']['values'].append(pcmci_effect)
                    pair_results['pcmci']['errors'].append(pcmci_effect - true_effect)
                
                # Bagged
                bagged_effect = effect_data.get('bagged_effect')
                if not np.isnan(bagged_effect):
                    pair_results['bagged']['values'].append(bagged_effect)
                    pair_results['bagged']['errors'].append(bagged_effect - true_effect)
                
                # Bootstrap
                bootstrap_stats = effect_data.get('bootstrap_stats', {})
                bootstrap_mean = bootstrap_stats.get('mean')
                if not np.isnan(bootstrap_mean):
                    pair_results['bootstrap']['values'].append(bootstrap_mean)
                    pair_results['bootstrap']['errors'].append(bootstrap_mean - true_effect)
                    
                    # Check CI coverage
                    ci_lower = bootstrap_stats.get('ci_lower')
                    ci_upper = bootstrap_stats.get('ci_upper')
                    if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                        covers = (true_effect >= ci_lower) and (true_effect <= ci_upper)
                        pair_results['bootstrap']['ci_coverage'].append(int(covers))
        
        # Calculate metrics for each method
        pair_metrics = {}
        for method in ['pcmci', 'bagged', 'bootstrap']:
            method_metrics = {}
            
            errors = np.array(pair_results[method]['errors'])
            if len(errors) > 0:
                if 'mae' in metrics:
                    method_metrics['mae'] = np.mean(np.abs(errors))
                
                if 'rmse' in metrics:
                    method_metrics['rmse'] = np.sqrt(np.mean(errors**2))
                
                if 'bias' in metrics:
                    method_metrics['bias'] = np.mean(errors)
            
            if method == 'bootstrap' and 'ci_coverage' in metrics:
                ci_coverage = pair_results[method]['ci_coverage']
                if len(ci_coverage) > 0:
                    method_metrics['ci_coverage'] = np.mean(ci_coverage)
            
            pair_metrics[method] = method_metrics
        
        results[effect_pair] = pair_metrics
    
    return results


def plot_effect_comparison(all_effects, effect_pair, method='bootstrap'):
    """
    Plot comparison of estimated vs true effects.
    
    Parameters
    ----------
    all_effects : dict
        Dictionary of effect estimation results
    effect_pair : str
        Effect pair to plot
    method : str
        Estimation method to plot ('pcmci', 'bagged', or 'bootstrap')
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    true_effects = []
    estimated_effects = []
    confidence_intervals = []
    
    for dataset_id, dataset_effects in all_effects.items():
        if effect_pair in dataset_effects:
            effect_data = dataset_effects[effect_pair]
            
            true_effect = effect_data.get('true_effect')
            if true_effect is None:
                continue
            
            if method == 'pcmci':
                estimated_effect = effect_data.get('pcmci_effect')
                ci = None
            elif method == 'bagged':
                estimated_effect = effect_data.get('bagged_effect')
                ci = None
            elif method == 'bootstrap':
                bootstrap_stats = effect_data.get('bootstrap_stats', {})
                estimated_effect = bootstrap_stats.get('mean')
                ci_lower = bootstrap_stats.get('ci_lower')
                ci_upper = bootstrap_stats.get('ci_upper')
                ci = (ci_lower, ci_upper) if not np.isnan(ci_lower) and not np.isnan(ci_upper) else None
            
            if not np.isnan(estimated_effect):
                true_effects.append(true_effect)
                estimated_effects.append(estimated_effect)
                confidence_intervals.append(ci)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot identity line
    min_val = min(min(true_effects), min(estimated_effects))
    max_val = max(max(true_effects), max(estimated_effects))
    margin = 0.1 * (max_val - min_val)
    ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.7)
    
    # Plot estimates
    for i, (true, est, ci) in enumerate(zip(true_effects, estimated_effects, confidence_intervals)):
        ax.scatter(true, est, color='blue', s=40, alpha=0.7)
        
        if ci is not None:
            ax.plot([true, true], ci, color='blue', alpha=0.4)
    
    # Add labels and title
    ax.set_xlabel('True Effect')
    ax.set_ylabel(f'Estimated Effect ({method})')
    ax.set_title(f'True vs Estimated Effects for {effect_pair}')
    
    # Calculate performance metrics
    errors = np.array(estimated_effects) - np.array(true_effects)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Add metrics to plot
    textstr = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig




def analyze_bootstrap_distribution(bootstrap_effects):
    """
    Analyze the distribution of bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list
        List of bootstrap effect estimates
        
    Returns
    -------
    dict
        Dictionary of distribution characteristics
    """
    import numpy as np
    from scipy import stats
    from scipy.signal import find_peaks
    
    # Filter out NaN values
    valid_effects = np.array([e for e in bootstrap_effects if not np.isnan(e)])
    
    if len(valid_effects) < 5:
        return {
            'status': 'insufficient_samples',
            'n_valid': len(valid_effects),
            'n_total': len(bootstrap_effects)
        }
    
    # Calculate basic statistics
    mean = np.mean(valid_effects)
    median = np.median(valid_effects)
    std = np.std(valid_effects)
    
    # Analyze distribution shape
    skewness = stats.skew(valid_effects)
    kurtosis = stats.kurtosis(valid_effects)
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(valid_effects)
    is_normal = shapiro_p > 0.05
    
    # Check for multimodality using KDE
    try:
        from scipy.stats import gaussian_kde
        
        # Estimate density with appropriate bandwidth
        kde = gaussian_kde(valid_effects)
        x = np.linspace(min(valid_effects) - 0.1*std, max(valid_effects) + 0.1*std, 1000)
        density = kde(x)
        
        # Find peaks
        peaks, _ = find_peaks(density, height=0.05*np.max(density), distance=len(x)//20)
        
        is_multimodal = len(peaks) > 1
        n_modes = len(peaks)
        
        mode_positions = []
        if peaks.size > 0:
            mode_positions = x[peaks].tolist()
    except:
        # Fall back to simple mode counting
        is_multimodal = False
        n_modes = 1
        mode_positions = []
    
    # Calculate success rate
    success_rate = len(valid_effects) / len(bootstrap_effects)
    
    return {
        'status': 'analyzed',
        'n_valid': len(valid_effects),
        'n_total': len(bootstrap_effects),
        'success_rate': success_rate,
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'is_normal': is_normal,
        'is_multimodal': is_multimodal,
        'n_modes': n_modes,
        'mode_positions': mode_positions
    }

def extended_evaluate_estimation_performance(all_effects, metrics=None):
    """
    Enhanced evaluation of causal effect estimation performance with bootstrap distribution analysis.
    
    Parameters
    ----------
    all_effects : dict
        Dictionary of effect estimation results
    metrics : list, optional
        List of metrics to calculate
        
    Returns
    -------
    dict
        Dictionary of extended evaluation results
    """
    import numpy as np
    
    if metrics is None:
        metrics = ['mae', 'rmse', 'bias', 'ci_coverage', 'success_rate', 'multimodality']
    
    # Initialize results with structure from original evaluate_estimation_performance
    results = {}
    
    # Initialize extended results
    extended_results = {
        'overall_metrics': {
            'pcmci': {'success_count': 0, 'attempt_count': 0},
            'bagged': {'success_count': 0, 'attempt_count': 0},
            'bootstrap': {'success_count': 0, 'attempt_count': 0, 'multimodal_count': 0}
        }
    }
    
    # Collect all effect pairs
    effect_pairs = set()
    for dataset_effects in all_effects.values():
        effect_pairs.update(dataset_effects.keys())
    
    for effect_pair in effect_pairs:
        pair_results = {
            'pcmci': {'values': [], 'errors': []},
            'bagged': {'values': [], 'errors': []},
            'bootstrap': {'values': [], 'errors': [], 'ci_coverage': [], 'distributions': []}
        }
        
        # Collect values and errors for each dataset
        for dataset_id, dataset_effects in all_effects.items():
            if effect_pair in dataset_effects:
                effect_data = dataset_effects[effect_pair]
                
                true_effect = effect_data.get('true_effect')
                if true_effect is None:
                    continue
                
                # PCMCI
                extended_results['overall_metrics']['pcmci']['attempt_count'] += 1
                pcmci_effect = effect_data.get('pcmci_effect')
                if not np.isnan(pcmci_effect):
                    extended_results['overall_metrics']['pcmci']['success_count'] += 1
                    pair_results['pcmci']['values'].append(pcmci_effect)
                    pair_results['pcmci']['errors'].append(pcmci_effect - true_effect)
                
                # Bagged
                extended_results['overall_metrics']['bagged']['attempt_count'] += 1
                bagged_effect = effect_data.get('bagged_effect')
                if not np.isnan(bagged_effect):
                    extended_results['overall_metrics']['bagged']['success_count'] += 1
                    pair_results['bagged']['values'].append(bagged_effect)
                    pair_results['bagged']['errors'].append(bagged_effect - true_effect)
                
                # Bootstrap
                extended_results['overall_metrics']['bootstrap']['attempt_count'] += 1
                bootstrap_stats = effect_data.get('bootstrap_stats', {})
                bootstrap_mean = bootstrap_stats.get('mean')
                if not np.isnan(bootstrap_mean):
                    extended_results['overall_metrics']['bootstrap']['success_count'] += 1
                    pair_results['bootstrap']['values'].append(bootstrap_mean)
                    pair_results['bootstrap']['errors'].append(bootstrap_mean - true_effect)
                    
                    # Check CI coverage
                    ci_lower = bootstrap_stats.get('ci_lower')
                    ci_upper = bootstrap_stats.get('ci_upper')
                    if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                        covers = (true_effect >= ci_lower) and (true_effect <= ci_upper)
                        pair_results['bootstrap']['ci_coverage'].append(int(covers))
                
                # Analyze bootstrap distribution if available
                bootstrap_effects = effect_data.get('bootstrap_effects', [])
                if bootstrap_effects:
                    dist_analysis = analyze_bootstrap_distribution(bootstrap_effects)
                    pair_results['bootstrap']['distributions'].append(dist_analysis)
                    
                    # Count multimodal distributions
                    if dist_analysis.get('is_multimodal', False):
                        extended_results['overall_metrics']['bootstrap']['multimodal_count'] += 1
        
        # Calculate metrics for each method as in original function
        pair_metrics = {}
        for method in ['pcmci', 'bagged', 'bootstrap']:
            method_metrics = {}
            
            errors = np.array(pair_results[method]['errors'])
            if len(errors) > 0:
                if 'mae' in metrics:
                    method_metrics['mae'] = np.mean(np.abs(errors))
                
                if 'rmse' in metrics:
                    method_metrics['rmse'] = np.sqrt(np.mean(errors**2))
                
                if 'bias' in metrics:
                    method_metrics['bias'] = np.mean(errors)
                
                # Add success rate
                if 'success_rate' in metrics:
                    attempt_count = extended_results['overall_metrics'][method]['attempt_count']
                    if attempt_count > 0:
                        method_metrics['success_rate'] = len(errors) / attempt_count
            
            if method == 'bootstrap' and 'ci_coverage' in metrics:
                ci_coverage = pair_results[method]['ci_coverage']
                if len(ci_coverage) > 0:
                    method_metrics['ci_coverage'] = np.mean(ci_coverage)
            
            # Add distribution analysis for bootstrap
            if method == 'bootstrap' and 'multimodality' in metrics and pair_results[method]['distributions']:
                # Aggregate distribution metrics
                multimodal_count = sum(1 for d in pair_results[method]['distributions'] 
                                     if d.get('is_multimodal', False))
                total_count = len(pair_results[method]['distributions'])
                
                if total_count > 0:
                    method_metrics['multimodal_rate'] = multimodal_count / total_count
                    method_metrics['multimodal_count'] = multimodal_count
                    method_metrics['distribution_count'] = total_count
            
            pair_metrics[method] = method_metrics
        
        results[effect_pair] = pair_metrics
    
    # Calculate overall success rates and multimodal rates
    for method in ['pcmci', 'bagged', 'bootstrap']:
        attempt_count = extended_results['overall_metrics'][method]['attempt_count']
        if attempt_count > 0:
            extended_results['overall_metrics'][method]['success_rate'] = (
                extended_results['overall_metrics'][method]['success_count'] / attempt_count
            )
    
    bootstrap_attempt_count = extended_results['overall_metrics']['bootstrap']['attempt_count']
    if bootstrap_attempt_count > 0 and 'multimodal_count' in extended_results['overall_metrics']['bootstrap']:
        multimodal_count = extended_results['overall_metrics']['bootstrap']['multimodal_count']
        extended_results['overall_metrics']['bootstrap']['multimodal_rate'] = multimodal_count / bootstrap_attempt_count
    
    # Combine original and extended results
    extended_results['metrics_by_effect'] = results
    
    return extended_results

def plot_bootstrap_distribution(bootstrap_effects, true_effect=None, other_estimates=None, title=None, output_file=None):
    """
    Plot the distribution of bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list
        List of bootstrap effect estimates
    true_effect : float, optional
        True effect value for reference
    other_estimates : dict, optional
        Dictionary of other estimates to show (e.g., {'pcmci': value, 'bagged': value})
    title : str, optional
        Title for the plot
    output_file : str, optional
        Path to save the plot to
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    
    # Filter out NaN values
    valid_effects = [e for e in bootstrap_effects if not np.isnan(e)]
    
    if len(valid_effects) < 5:
        return None
    
    # Analyze the distribution
    dist_analysis = analyze_bootstrap_distribution(valid_effects)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(valid_effects, kde=True, ax=ax, color='blue', alpha=0.6)
    
    # Mark true effect if provided
    if true_effect is not None:
        ax.axvline(true_effect, color='red', linestyle='-', 
                 label=f'True effect: {true_effect:.4f}')
    
    # Mark other estimates if provided
    if other_estimates:
        colors = {'pcmci': 'green', 'bagged': 'orange', 'naive': 'purple'}
        styles = {'pcmci': '--', 'bagged': '-.', 'naive': ':'}
        
        for method, value in other_estimates.items():
            if not np.isnan(value):
                ax.axvline(value, color=colors.get(method, 'gray'), 
                         linestyle=styles.get(method, '-'), 
                         label=f'{method} estimate: {value:.4f}')
    
    # Mark modes if multimodal
    if dist_analysis.get('is_multimodal', False) and 'mode_positions' in dist_analysis:
        for i, pos in enumerate(dist_analysis['mode_positions']):
            ax.axvline(pos, color='purple', linestyle=':', alpha=0.7,
                     label=f'Mode {i+1}: {pos:.4f}' if i == 0 else None)
    
    # Add metadata text box
    textstr = (
        f"N: {dist_analysis['n_valid']}/{dist_analysis['n_total']}\n"
        f"Mean: {dist_analysis['mean']:.4f}\n"
        f"Median: {dist_analysis['median']:.4f}\n"
        f"Std: {dist_analysis['std']:.4f}\n"
        f"Multimodal: {'Yes' if dist_analysis.get('is_multimodal', False) else 'No'}\n"
        f"Success rate: {dist_analysis['success_rate']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
          verticalalignment='top', bbox=props)
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Bootstrap Effect Distribution')
    
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    ax.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig