"""
Visualization functions for causal discovery and effect estimation.

This module provides functions for visualizing causal graphs, bootstrap
distributions, parameter studies, and performance comparisons.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI


# Set Seaborn style
sns.set_theme(style="whitegrid")


# =====================================================================
# Graph Visualization
# =====================================================================

def plot_causal_graph(graph, val_matrix=None, var_names=None, save_path=None, title=None, fig_ax = None):
    """
    Plot a causal graph.
    
    Parameters
    ----------
    graph : numpy.ndarray
        Causal graph from PCMCI
    val_matrix : numpy.ndarray, optional
        Value matrix for edge colors
    var_names : list, optional
        Variable names
    save_path : str, optional
        Path to save the plot
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default variable names if not provided
    if var_names is None:
        var_names = [f'X{i+1}' for i in range(graph.shape[0])]
    
    # Let tigramite create the figure and plot the graph
    if fig_ax is None:
        fig_ax = plt.subplots(figsize=(10, 8))
    tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_colorbar_label='MCI test strength' if val_matrix is not None else None,
        node_colorbar_label='Auto-MCI' if val_matrix is not None else None,
        fig_ax=fig_ax,
        figsize=None  # Don't set figsize since we already created the figure
    )
    
    # Set title
    #if title:
        #plt.title(title, fontsize=14)
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig_ax

def plot_ts_graph(graph, val_matrix=None, var_names=None, save_path=None, title=None, fig_ax = None):
    """
    Plot a causal graph.
    
    Parameters
    ----------
    graph : numpy.ndarray
        Causal graph from PCMCI
    val_matrix : numpy.ndarray, optional
        Value matrix for edge colors
    var_names : list, optional
        Variable names
    save_path : str, optional
        Path to save the plot
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default variable names if not provided
    if var_names is None:
        var_names = [f'X{i+1}' for i in range(graph.shape[0])]
    
    # Let tigramite create the figure and plot the graph
    if fig_ax is None:
        fig_ax = plt.subplots(figsize=(10, 8))
        
    tp.plot_time_series_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_colorbar_label='MCI test strength' if val_matrix is not None else None,
        fig_ax=fig_ax,
        figsize=None  # Don't set figsize since we already created the figure
    )
    
    # Set title
    #if title:
        #plt.title(title, fontsize=14)
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig_ax

def plot_graph_comparison(graphs_dict, var_names=None, save_path=None):
    """
    Plot multiple causal graphs for comparison.
    
    Parameters
    ----------
    graphs_dict : dict
        Dictionary of graphs with names as keys
    var_names : list, optional
        Variable names
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate subplot layout
    n_graphs = len(graphs_dict)
    cols = min(3, n_graphs)
    rows = (n_graphs + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    
    # Set default variable names if not provided
    if var_names is None:
        for name, graph_data in graphs_dict.items():
            if isinstance(graph_data, tuple):
                graph = graph_data[0]
            else:
                graph = graph_data
            var_names = [f'X{i+1}' for i in range(graph.shape[0])]
            break
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each graph
    for i, (name, graph_data) in enumerate(graphs_dict.items()):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Extract graph and val_matrix if provided as tuple
        if isinstance(graph_data, tuple):
            graph, val_matrix = graph_data
        else:
            graph = graph_data
            val_matrix = None
        
        # Plot graph
        tp.plot_graph(
            val_matrix=val_matrix,
            graph=graph,
            var_names=var_names,
            fig_ax=(fig,ax)
        )
        
        ax.set_title(name, fontsize=12)
    
    # Hide empty subplots
    for i in range(n_graphs, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =====================================================================
# Effect Visualization
# =====================================================================

def plot_effect_histogram(bootstrap_effects, true_effect=None, save_path=None):
    """
    Plot histogram of bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list or numpy.ndarray
        List of bootstrap effect estimates
    true_effect : float, optional
        True effect for reference
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array and filter NaNs
    effects = np.array(bootstrap_effects)
    effects = effects[~np.isnan(effects)]
    
    if len(effects) == 0:
        ax.text(0.5, 0.5, "No valid bootstrap estimates", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Calculate statistics
    mean = np.mean(effects)
    std = np.std(effects)
    
    # Plot histogram
    sns.histplot(effects, kde=True, ax=ax)
    
    # Add vertical lines for mean and true effect
    ax.axvline(mean, color='blue', linestyle='--', label=f'Bootstrap Mean: {mean:.4f}')
    
    if true_effect is not None:
        ax.axvline(true_effect, color='red', linestyle='-', label=f'True Effect: {true_effect:.4f}')
    
    # Add 95% confidence interval
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    # Add labels and title
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bootstrap Effect Estimates')
    ax.legend()
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_effect_density(bootstrap_effects, true_effect=None, save_path=None):
    """
    Plot kernel density estimate of bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list or numpy.ndarray
        List of bootstrap effect estimates
    true_effect : float, optional
        True effect for reference
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array and filter NaNs
    effects = np.array(bootstrap_effects)
    effects = effects[~np.isnan(effects)]
    
    if len(effects) == 0:
        ax.text(0.5, 0.5, "No valid bootstrap estimates", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Calculate statistics
    mean = np.mean(effects)
    std = np.std(effects)
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std
    
    # Compute kernel density estimate
    try:
        kde = gaussian_kde(effects)
        x = np.linspace(np.min(effects) - std, np.max(effects) + std, 1000)
        y = kde(x)
        
        # Plot density
        ax.plot(x, y, label='KDE')
        ax.fill_between(x, 0, y, alpha=0.3)
        
        # Add markers for key points
        mode_x = x[np.argmax(y)]
        ax.axvline(mode_x, color='green', linestyle=':', label=f'Mode: {mode_x:.4f}')
    except:
        # Fall back to histogram if KDE fails
        sns.histplot(effects, kde=True, ax=ax)
    
    # Add vertical lines for mean and true effect
    ax.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.4f}')
    
    if true_effect is not None:
        ax.axvline(true_effect, color='red', linestyle='-', label=f'True Effect: {true_effect:.4f}')
    
    # Add 95% confidence interval
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    # Add labels and title
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    ax.set_title('Density of Bootstrap Effect Estimates')
    ax.legend()
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multimodality_test(bootstrap_effects, save_path=None):
    """
    Test and visualize potential multimodality in bootstrap effect estimates.
    
    Parameters
    ----------
    bootstrap_effects : list or numpy.ndarray
        List of bootstrap effect estimates
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    tuple
        (Figure, is_multimodal)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array and filter NaNs
    effects = np.array(bootstrap_effects)
    effects = effects[~np.isnan(effects)]
    
    if len(effects) < 10:
        ax.text(0.5, 0.5, "Insufficient data for multimodality test", 
                ha='center', va='center', fontsize=14)
        return fig, False
    
    # Compute kernel density estimate
    try:
        kde = gaussian_kde(effects)
        x = np.linspace(np.min(effects) - 0.5, np.max(effects) + 0.5, 1000)
        y = kde(x)
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(x) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peaks.append((x[i], y[i]))
        
        # Plot density
        ax.plot(x, y, label='KDE')
        ax.fill_between(x, 0, y, alpha=0.3)
        
        # Plot peaks
        for i, (px, py) in enumerate(peaks):
            ax.plot(px, py, 'ro')
            ax.text(px, py + max(y) * 0.05, f'Peak {i+1}: {px:.4f}', ha='center')
        
        # Determine if multimodal (more than one significant peak)
        is_multimodal = len(peaks) > 1
        
        # Add label for multimodality
        if is_multimodal:
            ax.set_title('Multimodal Distribution Detected')
        else:
            ax.set_title('No Multimodality Detected')
    except:
        # Fall back to histogram if KDE fails
        sns.histplot(effects, kde=True, ax=ax)
        is_multimodal = False
        ax.set_title('Multimodality Test Failed - Using Histogram')
    
    # Add labels
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, is_multimodal


def plot_effect_comparison(results_df, param_name=None, effect_key=None, 
                          methods=None, save_path=None):
    """
    Plot comparison of effect estimates across methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results
    param_name : str, optional
        Parameter name for x-axis
    effect_key : str, optional
        Effect pair to plot, if None plots average across all pairs
    methods : list, optional
        List of methods to include
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default methods if not provided
    if methods is None:
        methods = ['true_graph','pcmci', 'bagged', 'bootstrap']
    
    # Filter DataFrame if effect_key is provided
    if effect_key is not None:
        plot_df = results_df[results_df['effect_key'] == effect_key].copy()
    else:
        plot_df = results_df.copy()
    
    # Check if DataFrame is empty
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for selected parameters", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x variable based on param_name
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name].copy()
        x_var = 'param_value'
        x_label = f'{param_name.capitalize()} Value'
    else:
        # If no param_name, use effect_key as categorical variable
        x_var = 'effect_key'
        x_label = 'Effect Pair'
    
    # Create a melted DataFrame for plotting
    plot_cols = []
    method_labels = []
    
    for method in methods:
        if method == 'bootstrap':
            col = 'bootstrap_mean'
        else:
            col = f'{method}_effect'
        
        if col in plot_df.columns:
            plot_cols.append(col)
            method_labels.append(method.capitalize())
    
    # Add true effect if available
    if 'true_effect' in plot_df.columns:
        plot_cols.append('true_effect')
        method_labels.append('True Effect')
    
    # Check if we have any valid columns
    if not plot_cols:
        ax.text(0.5, 0.5, "No effect estimates available for selected methods", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Melt DataFrame for plotting
    melt_df = pd.melt(plot_df, id_vars=[x_var], value_vars=plot_cols, 
                      var_name='Method', value_name='Effect')
    
    # Map method names to labels
    method_map = dict(zip(plot_cols, method_labels))
    melt_df['Method'] = melt_df['Method'].map(method_map)
    
    # Create the plot
    if x_var == 'effect_key':
        # Categorical plot
        sns.barplot(data=melt_df, x=x_var, y='Effect', hue='Method', ax=ax)
        plt.xticks(rotation=45)
    else:
        # Line plot for parameter study
        sns.lineplot(data=melt_df, x=x_var, y='Effect', hue='Method', 
                    marker='o', ax=ax)
    
    # Add confidence intervals for bootstrap if available
    if 'bootstrap_mean' in plot_cols and param_name is not None:
        if 'bootstrap_ci_lower' in plot_df.columns and 'bootstrap_ci_upper' in plot_df.columns:
            # Sort to ensure correct order
            ci_df = plot_df.sort_values(x_var)
            ax.fill_between(ci_df[x_var], ci_df['bootstrap_ci_lower'], ci_df['bootstrap_ci_upper'],
                          alpha=0.2, color='green', label='Bootstrap 95% CI')
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Effect Size')
    if effect_key is not None:
        ax.set_title(f'Effect Comparison for {effect_key}')
    else:
        ax.set_title('Effect Comparison Across Methods')
    
    ax.legend(title='Method')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =====================================================================
# Error Visualization
# =====================================================================

def plot_error_comparison(results_df, param_name=None, effect_key=None, 
                         methods=None, error_type='abs', save_path=None):
    """
    Plot comparison of estimation errors across methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results
    param_name : str, optional
        Parameter name for x-axis
    effect_key : str, optional
        Effect pair to plot, if None plots average across all pairs
    methods : list, optional
        List of methods to include
    error_type : str
        Type of error to plot ('abs', 'squared', or 'raw')
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default methods if not provided
    if methods is None:
        methods = ['true_graph','pcmci', 'bagged', 'bootstrap']
    
    # Map error type to column suffix
    error_col_map = {
        'abs': '_abs_error',
        'squared': '_squared_error',
        'raw': '_error'
    }
    error_suffix = error_col_map.get(error_type, '_abs_error')
    
    # Set error label based on type
    error_label = {
        'abs': 'Absolute Error',
        'squared': 'Squared Error',
        'raw': 'Error'
    }.get(error_type, 'Absolute Error')
    
    # Filter DataFrame if effect_key is provided
    if effect_key is not None:
        plot_df = results_df[results_df['effect_key'] == effect_key].copy()
    else:
        plot_df = results_df.copy()
    
    # Check if DataFrame is empty
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for selected parameters", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x variable based on param_name
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name].copy()
        x_var = 'param_value'
        x_label = f'{param_name.capitalize()} Value'
    else:
        # If no param_name, use effect_key as categorical variable
        x_var = 'effect_key'
        x_label = 'Effect Pair'
    
    # Create a melted DataFrame for plotting
    plot_cols = []
    method_labels = []
    
    for method in methods:
        col = f'{method}{error_suffix}'
        if col in plot_df.columns:
            plot_cols.append(col)
            method_labels.append(method.capitalize())
    
    # Check if we have any valid columns
    if not plot_cols:
        ax.text(0.5, 0.5, "No error estimates available for selected methods", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Melt DataFrame for plotting
    melt_df = pd.melt(plot_df, id_vars=[x_var], value_vars=plot_cols, 
                      var_name='Method', value_name=error_label)
    
    # Map method names to labels
    method_map = dict(zip(plot_cols, method_labels))
    melt_df['Method'] = melt_df['Method'].map(lambda x: method_map.get(x, x))
    
    # Create the plot
    if x_var == 'effect_key':
        # Categorical plot
        sns.barplot(data=melt_df, x=x_var, y=error_label, hue='Method', ax=ax)
        plt.xticks(rotation=45)
    else:
        # Line plot for parameter study
        sns.lineplot(data=melt_df, x=x_var, y=error_label, hue='Method', 
                    marker='o', ax=ax)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(error_label)
    if effect_key is not None:
        ax.set_title(f'{error_label} Comparison for {effect_key}')
    else:
        ax.set_title(f'{error_label} Comparison Across Methods')
    
    ax.legend(title='Method')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(results_df, param_name=None, metric='mae', 
                          methods=None, save_path=None):
    """
    Plot comparison of performance metrics across methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with metrics (from calculate_metrics)
    param_name : str, optional
        Parameter name for x-axis (must be a column in results_df)
    metric : str
        Metric to plot ('mae', 'rmse', or 'bias')
    methods : list, optional
        List of methods to include
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Set default methods if not provided
    if methods is None:
        methods = ['true_graph','pcmci', 'bagged', 'bootstrap']
    
    # Create plot DataFrame
    plot_df = results_df.copy()
    
    # Set x variable based on param_name
    if param_name is not None:
        if param_name not in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Parameter '{param_name}' not found in DataFrame", 
                    ha='center', va='center', fontsize=14)
            return fig
        x_var = param_name
        x_label = f'{param_name.capitalize()} Value'
    else:
        # If no param_name, use index as x-axis
        plot_df = plot_df.reset_index(drop=True)
        x_var = plot_df.index
        x_label = 'Configuration'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a melted DataFrame for plotting
    plot_cols = []
    method_labels = []
    
    for method in methods:
        col = f'{method}_{metric}'
        if col in plot_df.columns:
            plot_cols.append(col)
            method_labels.append(method.capitalize())
    
    # Check if we have any valid columns
    if not plot_cols:
        ax.text(0.5, 0.5, f"No '{metric}' metric available for selected methods", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Melt DataFrame for plotting
    melt_df = pd.melt(plot_df, id_vars=[x_var], value_vars=plot_cols, 
                      var_name='Method', value_name=metric.upper())
    
    # Map method names to labels
    method_map = dict(zip(plot_cols, method_labels))
    melt_df['Method'] = melt_df['Method'].map(lambda x: method_map.get(x, x))
    
    # Create the plot
    if isinstance(x_var, pd.Index):
        # Categorical plot
        sns.barplot(data=melt_df, x=melt_df.index, y=metric.upper(), hue='Method', ax=ax)
    else:
        # Line plot
        sns.lineplot(data=melt_df, x=x_var, y=metric.upper(), hue='Method', 
                    marker='o', ax=ax)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    metric_labels = {'mae': 'Mean Absolute Error', 
                    'rmse': 'Root Mean Squared Error', 
                    'bias': 'Bias (Mean Error)'}
    y_label = metric_labels.get(metric, metric.upper())
    ax.set_ylabel(y_label)
    ax.set_title(f'{y_label} Comparison Across Methods')
    
    ax.legend(title='Method')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_linreg_ci_coverage(df, param_name=None, save_path=None):
    """
    Plot coverage rate of linear regression confidence intervals across estimation methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with confidence interval information
    param_name : str, optional
        Name of the parameter to group by (e.g., 'auto', 'cross', 'noise', 'T')
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the methods to track
    methods = ['true_graph', 'pcmci', 'bagged']
    method_labels = ['True Graph', 'PCMCI', 'Bagged PCMCI']
    
    # Filter to only include rows with the necessary data
    required_cols = ['true_effect']
    for method in methods:
        required_cols.extend([f'{method}_ci_lower', f'{method}_ci_upper'])
    
    valid_df = df.dropna(subset=['true_effect']).copy()
    
    # Exit if no valid data
    if len(valid_df) == 0:
        ax.text(0.5, 0.5, "No valid data with confidence intervals found", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Calculate coverage for each method
    for method in methods:
        if f'{method}_ci_lower' in valid_df.columns and f'{method}_ci_upper' in valid_df.columns:
            # Calculate whether true effect is within the CI
            valid_df[f'{method}_ci_covers'] = (
                (valid_df['true_effect'] >= valid_df[f'{method}_ci_lower']) & 
                (valid_df['true_effect'] <= valid_df[f'{method}_ci_upper'])
            )
    
    # Plotting logic depends on whether we're grouping by parameter
    if param_name and param_name in valid_df.columns:
        # Group by parameter value
        param_values = sorted(valid_df['param_value'].unique())
        
        # Calculate coverage rate for each parameter value and method
        coverage_data = []
        
        for param_val in param_values:
            subset = valid_df[valid_df['param_value'] == param_val]
            
            for method, label in zip(methods, method_labels):
                if f'{method}_ci_covers' in subset.columns:
                    coverage = subset[f'{method}_ci_covers'].mean() * 100
                    coverage_data.append({
                        'param_value': param_val,
                        'Method': label,
                        'Coverage Rate (%)': coverage
                    })
        
        # Convert to DataFrame for plotting
        coverage_df = pd.DataFrame(coverage_data)
        
        # Create line plot
        sns.lineplot(
            data=coverage_df,
            x='param_value',
            y='Coverage Rate (%)',
            hue='Method',
            marker='o',
            ax=ax
        )
        
        # Add the 95% reference line
        ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
        
        # Improve axis labels
        ax.set_xlabel(f'{param_name.capitalize()} Value')
        
    else:
        # Calculate overall coverage rates
        coverage_data = []
        
        for method, label in zip(methods, method_labels):
            if f'{method}_ci_covers' in valid_df.columns:
                coverage = valid_df[f'{method}_ci_covers'].mean() * 100
                coverage_data.append({
                    'Method': label,
                    'Coverage Rate (%)': coverage
                })
        
        # Convert to DataFrame and sort by coverage rate
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df = coverage_df.sort_values('Coverage Rate (%)', ascending=False)
        
        # Create bar plot
        bars = ax.bar(
            coverage_df['Method'],
            coverage_df['Coverage Rate (%)'],
            alpha=0.7
        )
        
        # Add the 95% reference line
        ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom'
            )
        
        # Improve axis
        ax.set_ylim(0, max(max(coverage_df['Coverage Rate (%)']), 95) * 1.1)
        ax.set_xlabel('')
    
    # Add counts to legend
    if param_name:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for method, label in zip(methods, method_labels):
            if f'{method}_ci_covers' in valid_df.columns:
                count = valid_df[f'{method}_ci_covers'].count()
                new_labels.append(f"{label} (n={count})")
            else:
                new_labels.append(label)
        
        # Replace legend with counts
        if len(handles) >= len(methods):
            ax.legend(handles[:len(methods)], new_labels, title="Method")
    
    # Add title and y-label
    ax.set_title(f'Linear Regression CI Coverage by {"Method" if not param_name else param_name.capitalize()}')
    ax.set_ylabel('Coverage Rate (%)')
    
    # Add reference to the plot
    fig.text(
        0.01, 0.01, 
        "Coverage rate = % of cases where true effect is within the linear regression CI",
        fontsize=8, 
        style='italic'
    )
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# =====================================================================
# Timing Visualization
# =====================================================================


def analyze_linreg_ci_properties(df, param_name=None, save_path=None):
    """
    Analyze properties of linear regression confidence intervals across methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with CI information
    param_name : str, optional
        Name of the parameter to group by
    save_path : str, optional
        Path to save the plots
        
    Returns
    -------
    tuple
        (width_fig, coverage_fig, deviation_fig) - Generated figures
    """
    from matplotlib.gridspec import GridSpec
    
    # Define the methods to analyze
    methods = ['true_graph', 'pcmci', 'bagged']
    method_labels = ['True Graph', 'PCMCI', 'Bagged PCMCI']
    
    # Filter to only include rows with the necessary data
    valid_df = df.dropna(subset=['true_effect']).copy()
    
    # Create figures
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # CI Width
    ax2 = fig.add_subplot(gs[0, 1])  # Coverage Rate
    ax3 = fig.add_subplot(gs[1, :])  # True effect vs CI
    
    # Exit if no valid data
    if len(valid_df) == 0:
        for ax in [ax1, ax2, ax3]:
            ax.text(0.5, 0.5, "No valid data with confidence intervals found", 
                    ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, None, None
    
    # Calculate CI properties for each method
    for method in methods:
        lower_col = f'{method}_ci_lower'
        upper_col = f'{method}_ci_upper'
        
        if lower_col in valid_df.columns and upper_col in valid_df.columns:
            # Calculate CI width
            valid_df[f'{method}_ci_width'] = valid_df[upper_col] - valid_df[lower_col]
            
            # Calculate coverage
            valid_df[f'{method}_ci_covers'] = (
                (valid_df['true_effect'] >= valid_df[lower_col]) & 
                (valid_df['true_effect'] <= valid_df[upper_col])
            )
            
            # Calculate deviation from center
            valid_df[f'{method}_center'] = (valid_df[lower_col] + valid_df[upper_col]) / 2
            valid_df[f'{method}_deviation'] = valid_df[f'{method}_center'] - valid_df['true_effect']
    
    #---------------------------------------
    # PLOT 1: CI Width by Method/Parameter
    #---------------------------------------
    if param_name and param_name in valid_df.columns:
        # Group by parameter value
        param_values = sorted(valid_df['param_value'].unique())
        
        # Prepare data for width plot
        width_data = []
        for param_val in param_values:
            subset = valid_df[valid_df['param_value'] == param_val]
            
            for method, label in zip(methods, method_labels):
                width_col = f'{method}_ci_width'
                if width_col in subset.columns:
                    width_data.append({
                        'param_value': param_val,
                        'Method': label,
                        'CI Width': subset[width_col].mean()
                    })
        
        # Convert to DataFrame for plotting
        width_df = pd.DataFrame(width_data)
        
        # Create line plot for CI width
        sns.lineplot(
            data=width_df,
            x='param_value',
            y='CI Width',
            hue='Method',
            marker='o',
            ax=ax1
        )
        
        ax1.set_xlabel(f'{param_name.capitalize()} Value')
        
    else:
        # Calculate average CI width by method
        width_data = []
        for method, label in zip(methods, method_labels):
            width_col = f'{method}_ci_width'
            if width_col in valid_df.columns:
                width_data.append({
                    'Method': label,
                    'CI Width': valid_df[width_col].mean()
                })
        
        # Convert to DataFrame
        width_df = pd.DataFrame(width_data)
        
        # Create bar plot for CI width
        sns.barplot(
            data=width_df,
            x='Method',
            y='CI Width',
            ax=ax1
        )
        
        # Add value labels
        for i, p in enumerate(ax1.patches):
            ax1.annotate(
                f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
    
    ax1.set_title('Average Linear Regression CI Width')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    #---------------------------------------
    # PLOT 2: Coverage Rate
    #---------------------------------------
    # Prepare data for coverage plot
    if param_name and param_name in valid_df.columns:
        coverage_data = []
        for param_val in param_values:
            subset = valid_df[valid_df['param_value'] == param_val]
            
            for method, label in zip(methods, method_labels):
                covers_col = f'{method}_ci_covers'
                if covers_col in subset.columns:
                    coverage_data.append({
                        'param_value': param_val,
                        'Method': label,
                        'Coverage Rate (%)': subset[covers_col].mean() * 100
                    })
        
        # Convert to DataFrame
        coverage_df = pd.DataFrame(coverage_data)
        
        # Create line plot for coverage
        sns.lineplot(
            data=coverage_df,
            x='param_value',
            y='Coverage Rate (%)',
            hue='Method',
            marker='o',
            ax=ax2
        )
        
        ax2.set_xlabel(f'{param_name.capitalize()} Value')
        
    else:
        # Calculate overall coverage by method
        coverage_data = []
        for method, label in zip(methods, method_labels):
            covers_col = f'{method}_ci_covers'
            if covers_col in valid_df.columns:
                coverage_data.append({
                    'Method': label,
                    'Coverage Rate (%)': valid_df[covers_col].mean() * 100
                })
        
        # Convert to DataFrame
        coverage_df = pd.DataFrame(coverage_data)
        
        # Create bar plot for coverage
        sns.barplot(
            data=coverage_df,
            x='Method',
            y='Coverage Rate (%)',
            ax=ax2
        )
        
        # Add value labels
        for i, p in enumerate(ax2.patches):
            ax2.annotate(
                f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
    
    # Add the 95% reference line
    ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
    ax2.set_title('Linear Regression CI Coverage Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    #---------------------------------------
    # PLOT 3: True Effect vs CI
    #---------------------------------------
    # Select a subset of points to visualize if param_name is specified
    if param_name and param_name in valid_df.columns:
        # Choose representative points from different parameter values
        sample_points = []
        for param_val in param_values:
            subset = valid_df[valid_df['param_value'] == param_val]
            if len(subset) > 0:
                # Take one sample from each parameter value
                sample_points.append(subset.iloc[0])
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(sample_points)
    else:
        # Randomly sample points for visualization
        sample_size = min(20, len(valid_df))
        sample_df = valid_df.sample(n=sample_size)
    
    # Prepare data for visualization
    visualization_data = []
    
    for i, row in sample_df.iterrows():
        true_effect = row['true_effect']
        
        for method, label in zip(methods, method_labels):
            lower_col = f'{method}_ci_lower'
            upper_col = f'{method}_ci_upper'
            
            if lower_col in row and upper_col in row:
                ci_lower = row[lower_col]
                ci_upper = row[upper_col]
                
                if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                    visualization_data.append({
                        'Method': label,
                        'True Effect': true_effect,
                        'CI Lower': ci_lower,
                        'CI Upper': ci_upper,
                        'Covers': (true_effect >= ci_lower) and (true_effect <= ci_upper)
                    })
    
    # Convert to DataFrame
    vis_df = pd.DataFrame(visualization_data)
    
    if len(vis_df) > 0:
        # Add index for x-axis
        methods_count = len(methods)
        vis_df['Sample'] = np.repeat(range(len(vis_df) // methods_count), methods_count)
        
        # Sort for better visualization
        vis_df = vis_df.sort_values(['Sample', 'Method'])
        
        # Jitter points by method
        method_offset = {method: i * 0.2 - 0.2 for i, method in enumerate(method_labels)}
        vis_df['x_pos'] = vis_df.apply(lambda row: row['Sample'] + method_offset[row['Method']], axis=1)
        
        # Plot true effects
        samples = vis_df['Sample'].unique()
        for sample in samples:
            sample_true = vis_df[vis_df['Sample'] == sample]['True Effect'].iloc[0]
            ax3.plot([sample - 0.3, sample + 0.3], [sample_true, sample_true], 'k-', alpha=0.5)
        
        # Plot CIs for each method
        for method, label in zip(method_labels, ['b', 'g', 'r']):
            method_df = vis_df[vis_df['Method'] == method]
            
            # Plot CIs
            for i, row in method_df.iterrows():
                color = label if row['Covers'] else 'gray'
                ax3.plot([row['x_pos'], row['x_pos']], [row['CI Lower'], row['CI Upper']], 
                         color=color, marker='_', linestyle='-', alpha=0.7,
                         label=method if i == method_df.index[0] else "")
        
        # Improve plot appearance
        ax3.set_xticks(samples)
        if param_name and param_name in valid_df.columns:
            ax3.set_xticklabels([f"{param_name}={val}" for val in param_values[:len(samples)]])
        else:
            ax3.set_xticklabels([f"Sample {i+1}" for i in range(len(samples))])
        
        ax3.set_title('Confidence Intervals Around True Effects')
        ax3.set_ylabel('Effect Value')
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        handles, labels = [], []
        for i, (method, color) in enumerate(zip(method_labels, ['b', 'g', 'r'])):
            handles.append(plt.Line2D([0], [0], color=color, marker='_', linestyle='-', alpha=0.7))
            labels.append(method)
        
        # Add line for true effect
        handles.append(plt.Line2D([0], [0], color='k', linestyle='-', alpha=0.5))
        labels.append('True Effect')
        
        ax3.legend(handles, labels, loc='upper right')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for CI visualization", 
                ha='center', va='center', fontsize=14)
    
    # Add descriptive text
    fig.suptitle('Linear Regression Confidence Interval Analysis', fontsize=16)
    fig.text(
        0.01, 0.01, 
        "CI = Confidence Interval   |   Coverage Rate = % of cases where true effect is within CI",
        fontsize=8, 
        style='italic'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_error_vs_ci_width(df, param_name=None, save_path=None):
    """
    Plot relationship between estimation error and CI width across methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with error and CI information
    param_name : str, optional
        Name of the parameter to color by
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """

    from matplotlib.colors import LinearSegmentedColormap
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define the methods to analyze
    methods = ['true_graph', 'pcmci', 'bagged']
    method_labels = ['True Graph', 'PCMCI', 'Bagged PCMCI']
    
    # Filter to only include rows with the necessary data
    valid_df = df.dropna(subset=['true_effect']).copy()
    
    # Exit if no valid data
    if len(valid_df) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No valid data with confidence intervals found", 
                    ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Calculate CI properties for each method
    for method in methods:
        lower_col = f'{method}_ci_lower'
        upper_col = f'{method}_ci_upper'
        effect_col = f'{method}_effect'
        
        if (lower_col in valid_df.columns and upper_col in valid_df.columns 
            and effect_col in valid_df.columns):
            # Calculate CI width
            valid_df[f'{method}_ci_width'] = valid_df[upper_col] - valid_df[lower_col]
            
            # Calculate absolute error
            valid_df[f'{method}_error'] = np.abs(valid_df[effect_col] - valid_df['true_effect'])
            
            # Calculate coverage
            valid_df[f'{method}_ci_covers'] = (
                (valid_df['true_effect'] >= valid_df[lower_col]) & 
                (valid_df['true_effect'] <= valid_df[upper_col])
            )
    
    # Create custom colormap for coverage (red for not covered, green for covered)
    cmap = LinearSegmentedColormap.from_list('coverage', ['#ff6b6b', '#88e986'])
    
    # Plot for each method
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        ax = axes[i]
        
        width_col = f'{method}_ci_width'
        error_col = f'{method}_error'
        covers_col = f'{method}_ci_covers'
        
        if all(col in valid_df.columns for col in [width_col, error_col, covers_col]):
            # Create dataframe for this method with non-NaN values
            method_df = valid_df.dropna(subset=[width_col, error_col, covers_col])
            
            if len(method_df) > 0:
                # Plot error vs width, colored by coverage
                if param_name and param_name in method_df.columns:
                    # Color by parameter value
                    scatter = ax.scatter(
                        method_df[width_col],
                        method_df[error_col],
                        c=method_df['param_value'],
                        cmap='viridis',
                        alpha=0.7,
                        s=50,
                        edgecolors='w',
                        linewidths=0.5
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(f'{param_name.capitalize()} Value')
                    
                    # Add markers for coverage
                    covered = method_df[method_df[covers_col] == True]
                    not_covered = method_df[method_df[covers_col] == False]
                    
                    if len(covered) > 0:
                        ax.scatter(
                            covered[width_col],
                            covered[error_col],
                            s=80,
                            facecolors='none',
                            edgecolors='g',
                            linewidths=1.5,
                            label='Covered'
                        )
                    
                    if len(not_covered) > 0:
                        ax.scatter(
                            not_covered[width_col],
                            not_covered[error_col],
                            s=80,
                            facecolors='none',
                            edgecolors='r',
                            linewidths=1.5,
                            label='Not Covered'
                        )
                    
                else:
                    # Color by coverage
                    scatter = ax.scatter(
                        method_df[width_col],
                        method_df[error_col],
                        c=method_df[covers_col],
                        cmap=cmap,
                        alpha=0.7,
                        s=50
                    )
                    
                    # Add legend for coverage
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', 
                                   markersize=10, label='Not Covered'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#88e986', 
                                   markersize=10, label='Covered')
                    ]
                    ax.legend(handles=legend_elements)
                
                # Add diagonal line (width = 2*error)
                max_val = max(method_df[width_col].max(), method_df[error_col].max() * 2)
                diag_x = np.linspace(0, max_val, 100)
                ax.plot(diag_x, diag_x/2, 'k--', alpha=0.5, label='Width = 2×Error')
                
                # Add reference line for coverage
                mean_width = method_df[width_col].mean()
                ax.axvline(mean_width, color='gray', linestyle=':', alpha=0.7, 
                          label=f'Mean Width: {mean_width:.4f}')
                
                # Calculate correlation
                corr = method_df[width_col].corr(method_df[error_col])
                coverage_rate = method_df[covers_col].mean() * 100
                
                # Add correlation and coverage info
                ax.text(
                    0.05, 0.95,
                    f"Correlation: {corr:.2f}\nCoverage: {coverage_rate:.1f}%",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                
                # Set axis labels
                ax.set_xlabel('Confidence Interval Width')
                ax.set_ylabel('Absolute Error')
                ax.set_title(f'{label}')
                
                # Make axes start at 0
                ax.set_xlim(0, None)
                ax.set_ylim(0, None)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, f"No valid data for {label}", 
                        ha='center', va='center', fontsize=14)
        else:
            ax.text(0.5, 0.5, f"Missing data for {label}", 
                    ha='center', va='center', fontsize=14)
    
    # Add overall title
    fig.suptitle('Relationship Between Estimation Error and CI Width', fontsize=16)
    
    # Add descriptive text
    fig.text(
        0.01, 0.01, 
        "Ideally, CI width should be approximately 2× the error for proper coverage. Points below the diagonal line have wider CIs than needed, "
        "points above have narrower CIs than needed for error magnitude.",
        fontsize=8, 
        style='italic'
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# =====================================================================
# Timing Visualization
# =====================================================================

def plot_timing_breakdown(results_df, param_name=None, param_value=None, save_path=None):
    """
    Plot breakdown of execution times for different operations.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with timing information
    param_name : str, optional
        Parameter name to filter by
    param_value : float, optional
        Parameter value to filter by
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create a copy of the DataFrame for plotting
    plot_df = results_df.copy()
    
    # Filter by parameter if specified
    if param_name is not None:
        if param_name not in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Parameter '{param_name}' not found in DataFrame", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        plot_df = plot_df[plot_df['param_name'] == param_name]
        
        if param_value is not None:
            plot_df = plot_df[plot_df['param_value'] == param_value]
    
    # Check if DataFrame is empty
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for selected parameters", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Select timing columns
    time_cols = [col for col in plot_df.columns if col.endswith('_time')]
    
    # Group timing columns by operation
    operation_cols = {}
    for col in time_cols:
        operation = col.replace('_time', '')
        operation_cols[operation] = col
    
    # Calculate total time
    if 'total_time' in operation_cols:
        total_time = plot_df[operation_cols['total_time']].mean()
    else:
        total_time = sum(plot_df[col].mean() for col in operation_cols.values())
    
    # Create figure with two subplots: pie chart and bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create data for plots
    operations = []
    times = []
    for operation, col in operation_cols.items():
        if operation == 'total':
            continue
        operations.append(operation)
        times.append(plot_df[col].mean())
    
    # Pie chart
    ax1.pie(times, labels=operations, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Proportion of Execution Time by Operation')
    
    # Bar chart
    colors = plt.cm.tab10(np.arange(len(operations)) % 10)
    bars = ax2.bar(operations, times, color=colors)
    
    # Add total time annotation
    ax2.text(0.5, 0.95, f'Total Time: {total_time:.2f} s', 
             ha='center', va='center', transform=ax2.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Execution Time by Operation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_pipeline_timing_by_T(df, save_path=None):
    """
    Plot full computation time comparison between pipelines based on time series length T.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with timing information
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Filter data for T study
    T_df = df[df['param_name'] == 'T'].copy()
    
    if len(T_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for T parameter study", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Sort by T value
    T_values = sorted(T_df['param_value'].unique())
    
    # Calculate total pipeline times
    pipeline_times = []
    
    for T_val in T_values:
        subset = T_df[T_df['param_value'] == T_val]
        
        # Calculate average times for each pipeline
        # Pipeline 1: PCMCI → Effect estimation
        pcmci_time = subset['discovery_pcmci_time'].mean() + subset['effect_pcmci_effect_time'].mean()
        
        # Pipeline 2: Bagged PCMCI → Effect estimation
        bagged_time = subset['discovery_pcmci_time'].mean() + subset['discovery_bootstrap_time'].mean() + subset['effect_bagged_effect_time'].mean()
        
        # Pipeline 3: Bootstrap PCMCI → Effect estimation for all replicas
        bootstrap_time = subset['discovery_pcmci_time'].mean() + subset['discovery_bootstrap_time'].mean() + subset['effect_bootstrap_effects_time'].mean()
        
        pipeline_times.append({
            'T': T_val,
            'PCMCI → Effect': pcmci_time,
            'Bagged PCMCI → Effect': bagged_time,
            'Bootstrap PCMCI → Effect': bootstrap_time
        })
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(pipeline_times)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Melt DataFrame for seaborn
    melted_df = pd.melt(plot_df, id_vars=['T'], 
                        value_vars=['PCMCI → Effect', 'Bagged PCMCI → Effect', 'Bootstrap PCMCI → Effect'],
                        var_name='Pipeline', value_name='Time (seconds)')
    
    # Create plot with log scale
    sns.lineplot(data=melted_df, x='T', y='Time (seconds)', hue='Pipeline', 
                marker='o', ax=ax)
    
    # Set log scale for better visualization
    ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Time Series Length (T)')
    ax.set_ylabel('Computation Time (seconds, log scale)')
    ax.set_title('Pipeline Computation Time by Time Series Length')
    
    # Improve x-axis
    ax.set_xticks(T_values)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title='Pipeline')
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_timing_by_n_boot(df, save_path=None):
    """
    Plot computation time comparison for bootstrap-based methods by number of bootstrap samples.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with timing information
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Filter data for n_boot study
    n_boot_df = df[df['param_name'] == 'n'].copy()
    
    if len(n_boot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for n_boot parameter study", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Sort by n_boot value
    n_boot_values = sorted(n_boot_df['param_value'].unique())
    
    # Calculate bootstrap-based pipeline times
    bootstrap_times = []
    
    for n_boot_val in n_boot_values:
        subset = n_boot_df[n_boot_df['param_value'] == n_boot_val]
        
        # Pipeline 2: Bagged PCMCI → Effect estimation (excluding PCMCI time)
        bagged_time = subset['discovery_bootstrap_time'].mean() + subset['effect_bagged_effect_time'].mean()
        
        # Pipeline 3: Bootstrap PCMCI → Effect estimation for all replicas (excluding PCMCI time)
        bootstrap_time = subset['discovery_bootstrap_time'].mean() + subset['effect_bootstrap_effects_time'].mean()
        
        # Add to list
        bootstrap_times.append({
            'n_boot': n_boot_val,
            'Bagged PCMCI → Effect': bagged_time,
            'Bootstrap PCMCI → Effect': bootstrap_time
        })
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(bootstrap_times)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Melt DataFrame for seaborn
    melted_df = pd.melt(plot_df, id_vars=['n_boot'], 
                        value_vars=['Bagged PCMCI → Effect', 'Bootstrap PCMCI → Effect'],
                        var_name='Pipeline', value_name='Time (seconds)')
    
    # Create plot
    sns.lineplot(data=melted_df, x='n_boot', y='Time (seconds)', hue='Pipeline', 
                marker='o', ax=ax)
    
    # Add polynomial trend lines
    for pipeline in ['Bagged PCMCI → Effect', 'Bootstrap PCMCI → Effect']:
        pipeline_data = melted_df[melted_df['Pipeline'] == pipeline]
        x = pipeline_data['n_boot']
        y = pipeline_data['Time (seconds)']
        
        # Fit polynomial regression (degree=1 for linear relationship with n_boot)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Add trend line
        x_trend = np.linspace(min(x), max(x), 100)
        ax.plot(x_trend, p(x_trend), linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Number of Bootstrap Samples (n_boot)')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Bootstrap-Based Methods Computation Time by Number of Bootstrap Samples')
    
    # Improve x-axis
    ax.set_xticks(n_boot_values)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title='Pipeline')
    
    # Add annotation about linear relationship
    ax.text(0.05, 0.95, "Note: Dashed lines show linear trends", 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =====================================================================
# Distribution/CI Visualization
# =====================================================================

def plot_bootstrap_distribution_with_methods(bootstrap_effects, true_effect=None, 
                                            true_graph_effect=None, pcmci_effect=None, 
                                            bagged_effect=None, title=None, save_path=None):
    """
    Plot histogram of bootstrap effect estimates with other method estimates.
    
    Parameters
    ----------
    bootstrap_effects : list
        List of bootstrap effect estimates
    true_effect : float, optional
        True effect value
    true_graph_effect : float, optional
        True graph effect value
    pcmci_effect : float, optional
        PCMCI effect estimate
    bagged_effect : float, optional
        Bagged effect estimate
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    plt.figure(figsize=(12, 7))
    
    # Filter out None values and convert to numeric
    if bootstrap_effects is None:
        plt.text(0.5, 0.5, "No bootstrap effects data available", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()
    
    # Convert to list if it's not already
    if not isinstance(bootstrap_effects, list):
        bootstrap_effects = [bootstrap_effects]
    
    # Convert to numeric and filter out None/NaN
    effects = []
    for e in bootstrap_effects:
        try:
            val = float(e)
            if not np.isnan(val):
                effects.append(val)
        except (ValueError, TypeError):
            continue
    
    if len(effects) == 0:
        plt.text(0.5, 0.5, "No valid bootstrap estimates", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()
    
    # Calculate statistics
    mean = np.mean(effects)
    std = np.std(effects)
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std
    
    # Plot histogram
    plt.hist(effects, bins=20, alpha=0.5, density=True, label='Bootstrap Distribution')
    
    # Add KDE
    kde = gaussian_kde(effects)
    x = np.linspace(min(effects) - 0.1, max(effects) + 0.1, 1000)
    y = kde(x)
    plt.plot(x, y, 'k-', linewidth=1.5, label='Density Estimate')
    
    # Add vertical lines for different methods
    plt.axvline(mean, color='blue', linestyle='--', linewidth=2,
               label=f'Bootstrap Mean: {mean:.4f}')
    
    if true_effect is not None and not np.isnan(true_effect):
        plt.axvline(true_effect, color='red', linestyle='-', linewidth=2,
                   label=f'True Effect: {true_effect:.4f}')
    
    if true_graph_effect is not None and not np.isnan(true_graph_effect):
        plt.axvline(true_graph_effect, color='green', linestyle='-', linewidth=2,
                   label=f'True Graph: {true_graph_effect:.4f}')
    
    if pcmci_effect is not None and not np.isnan(pcmci_effect):
        plt.axvline(pcmci_effect, color='orange', linestyle='-', linewidth=2,
                   label=f'PCMCI: {pcmci_effect:.4f}')
    
    if bagged_effect is not None and not np.isnan(bagged_effect):
        plt.axvline(bagged_effect, color='purple', linestyle='-', linewidth=2,
                   label=f'Bagged: {bagged_effect:.4f}')
    
    # Add confidence interval
    plt.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', 
                label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    plt.xlabel('Effect Size')
    plt.ylabel('Density')
    #plt.title(title if title else 'Bootstrap Effect Distribution with Method Comparison')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def select_interesting_cases(df, param_name, min_value=None, med_value=None, max_value=None):
    """
    Select interesting parameter values for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with results
    param_name : str
        Parameter name to select values for
    min_value, med_value, max_value : float, optional
        Specific values to select. If None, will be calculated from data.
        
    Returns
    -------
    list
        List of selected parameter values
    """
    # Filter for the specific parameter
    param_df = df[df['param_name'] == param_name]
    
    if len(param_df) == 0:
        return []
    
    # Get unique parameter values
    param_values = sorted(param_df['param_value'].unique())
    
    if len(param_values) == 0:
        return []
    
    # Select values
    selected_values = []
    
    # Add min value
    if min_value is not None and min_value in param_values:
        selected_values.append(min_value)
    else:
        selected_values.append(param_values[0])
    
    # Add median value if we have at least 3 values
    if len(param_values) >= 3:
        if med_value is not None and med_value in param_values:
            selected_values.append(med_value)
        else:
            selected_values.append(param_values[len(param_values)//2])
    
    # Add max value
    if max_value is not None and max_value in param_values:
        selected_values.append(max_value)
    else:
        selected_values.append(param_values[-1])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(selected_values))

"""
CI Visualization and Analysis Approaches for Bootstrap-Based Causal Effect Estimation
"""

def plot_ci_width_comparison(df, param_name=None, effect_key=None, save_path=None):
    """
    Plot comparison of CI widths across methods, focusing on precision rather than coverage.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_name : str, optional
        Parameter name to filter by
    effect_key : str, optional
        Effect pair to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data if needed
    plot_df = df.copy()
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name]
    if effect_key is not None:
        plot_df = plot_df[plot_df['effect_key'] == effect_key]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # Calculate CI widths for all methods
    ci_data = []
    
    # Process each row
    for _, row in plot_df.iterrows():
        param_value = row['param_value']
        
        # Add standard method CIs
        for method in ['true_graph', 'pcmci', 'bagged']:
            if (f'{method}_linreg_ci_lower' in row and f'{method}_linreg_ci_upper' in row and
                row[f'{method}_linreg_ci_lower'] is not None and row[f'{method}_linreg_ci_upper'] is not None):
                width = row[f'{method}_linreg_ci_upper'] - row[f'{method}_linreg_ci_lower']
                ci_data.append({
                    'param_value': param_value,
                    'method': f'{method}_linreg',
                    'ci_width': width
                })
        
        # Add bootstrap-derived CIs
        if 'bootstrap_ci_lower' in row and 'bootstrap_ci_upper' in row and row['bootstrap_ci_lower'] is not None:
            width = row['bootstrap_ci_upper'] - row['bootstrap_ci_lower']
            ci_data.append({
                'param_value': param_value,
                'method': 'bootstrap_distribution',
                'ci_width': width
            })
        
        # Add combined bootstrap CIs
        if 'bootstrap_combined_ci_lower' in row and row['bootstrap_combined_ci_lower'] is not None:
            width = row['bootstrap_combined_ci_width']
            ci_data.append({
                'param_value': param_value,
                'method': 'bootstrap_combined',
                'ci_width': width
            })
        
        if 'bootstrap_conservative_ci_lower' in row and row['bootstrap_conservative_ci_lower'] is not None:
            width = row['bootstrap_conservative_ci_width']
            ci_data.append({
                'param_value': param_value,
                'method': 'bootstrap_conservative',
                'ci_width': width
            })
    
    if not ci_data:
        print("No CI data available for plotting.")
        return
    
    ci_df = pd.DataFrame(ci_data)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot CI widths
    if param_name is not None:
        # Line plot by parameter value
        sns.lineplot(
            data=ci_df, 
            x='param_value', 
            y='ci_width', 
            hue='method',
            marker='o',
            linewidth=2
        )
        plt.xlabel(f'{param_name.capitalize()} Value')
    else:
        # Bar plot for different methods
        sns.barplot(
            data=ci_df,
            x='method',
            y='ci_width'
        )
        plt.xticks(rotation=45)
    
    plt.ylabel('CI Width')
    title = 'Confidence Interval Width Comparison'
    if effect_key:
        title += f' for {effect_key}'
    if param_name:
        title += f' by {param_name}'
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_ci_width_to_error_ratio(df, param_name=None, effect_key=None, save_path=None):
    """
    Plot the ratio of CI width to absolute error, showing efficiency of each method.
    A lower ratio means a tighter CI relative to the actual error.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_name : str, optional
        Parameter name to filter by
    effect_key : str, optional
        Effect pair to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data if needed
    plot_df = df.copy()
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name]
    if effect_key is not None:
        plot_df = plot_df[plot_df['effect_key'] == effect_key]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # Calculate ratio of CI width to error
    ratio_data = []
    
    # Process each row
    for _, row in plot_df.iterrows():
        param_value = row['param_value']
        
        # Skip if true effect is missing
        if 'true_effect' not in row or row['true_effect'] is None:
            continue
        
        # Add standard method CI ratios
        for method in ['true_graph', 'pcmci', 'bagged']:
            if (f'{method}_linreg_ci_lower' in row and f'{method}_linreg_ci_upper' in row and
                row[f'{method}_linreg_ci_lower'] is not None and row[f'{method}_linreg_ci_upper'] is not None and
                f'{method}_abs_error' in row and row[f'{method}_abs_error'] > 0):
                
                width = row[f'{method}_linreg_ci_upper'] - row[f'{method}_linreg_ci_lower']
                error = row[f'{method}_abs_error']
                ratio = width / error if error > 0 else np.nan
                
                ratio_data.append({
                    'param_value': param_value,
                    'method': f'{method}_linreg',
                    'width_to_error_ratio': ratio
                })
        
        # Add bootstrap-derived CI ratio
        if ('bootstrap_ci_lower' in row and 'bootstrap_ci_upper' in row and 
            row['bootstrap_ci_lower'] is not None and 'bootstrap_abs_error' in row and 
            row['bootstrap_abs_error'] > 0):
            
            width = row['bootstrap_ci_upper'] - row['bootstrap_ci_lower']
            error = row['bootstrap_abs_error']
            ratio = width / error if error > 0 else np.nan
            
            ratio_data.append({
                'param_value': param_value,
                'method': 'bootstrap_distribution',
                'width_to_error_ratio': ratio
            })
        
        # Add combined bootstrap CI ratios
        if ('bootstrap_combined_ci_width' in row and row['bootstrap_combined_ci_width'] is not None and
            'bootstrap_abs_error' in row and row['bootstrap_abs_error'] > 0):
            
            width = row['bootstrap_combined_ci_width']
            error = row['bootstrap_abs_error']
            ratio = width / error if error > 0 else np.nan
            
            ratio_data.append({
                'param_value': param_value,
                'method': 'bootstrap_combined',
                'width_to_error_ratio': ratio
            })
        
        if ('bootstrap_conservative_ci_width' in row and row['bootstrap_conservative_ci_width'] is not None and
            'bootstrap_abs_error' in row and row['bootstrap_abs_error'] > 0):
            
            width = row['bootstrap_conservative_ci_width']
            error = row['bootstrap_abs_error']
            ratio = width / error if error > 0 else np.nan
            
            ratio_data.append({
                'param_value': param_value,
                'method': 'bootstrap_conservative',
                'width_to_error_ratio': ratio
            })
    
    if not ratio_data:
        print("No ratio data available for plotting.")
        return
    
    ratio_df = pd.DataFrame(ratio_data)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot width-to-error ratios
    if param_name is not None:
        # Line plot by parameter value
        sns.lineplot(
            data=ratio_df, 
            x='param_value', 
            y='width_to_error_ratio', 
            hue='method',
            marker='o',
            linewidth=2
        )
        plt.xlabel(f'{param_name.capitalize()} Value')
    else:
        # Box plot for different methods
        sns.boxplot(
            data=ratio_df,
            x='method',
            y='width_to_error_ratio'
        )
        plt.xticks(rotation=45)
    
    plt.ylabel('CI Width to Error Ratio')
    title = 'CI Width to Error Ratio Comparison'
    if effect_key:
        title += f' for {effect_key}'
    if param_name:
        title += f' by {param_name}'
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_ci_coverage_heatmap(df, param_name=None, save_path=None):
    """
    Create a heatmap showing CI coverage (binary: covers true effect or not)
    for different parameter values and methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_name : str, optional
        Parameter name to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data if needed
    plot_df = df.copy()
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # Get unique effect pairs and parameter values
    effect_keys = plot_df['effect_key'].unique()
    param_values = sorted(plot_df['param_value'].unique())
    
    # Define methods to include
    methods = [
        'true_graph_linreg_ci_covers_true',
        'pcmci_linreg_ci_covers_true',
        'bagged_linreg_ci_covers_true',
        'bootstrap_ci_covers_true',
        'bootstrap_combined_ci_covers_true',
        'bootstrap_conservative_ci_covers_true'
    ]
    
    # Create a matrix for each method
    coverage_matrices = {}
    for method in methods:
        if method in plot_df.columns:
            matrix = np.zeros((len(effect_keys), len(param_values)))
            for i, effect_key in enumerate(effect_keys):
                for j, param_value in enumerate(param_values):
                    subset = plot_df[(plot_df['effect_key'] == effect_key) & 
                                    (plot_df['param_value'] == param_value)]
                    if len(subset) > 0 and method in subset.columns:
                        # Get coverage value (1.0 for True, 0.0 for False, 0.5 for missing)
                        coverage = subset[method].mean()
                        matrix[i, j] = 1.0 if coverage > 0.5 else 0.0 if coverage >= 0 else 0.5
            
            coverage_matrices[method] = matrix
    
    # Create plot
    n_methods = len(coverage_matrices)
    if n_methods == 0:
        print("No coverage data available for plotting.")
        return
    
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3*n_methods), squeeze=False)
    
    # Plot each method
    for i, (method, matrix) in enumerate(coverage_matrices.items()):
        ax = axes[i, 0]
        
        # Define color map (green for coverage, red for no coverage, gray for missing)
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        
        # Set labels
        ax.set_yticks(np.arange(len(effect_keys)))
        ax.set_yticklabels(effect_keys)
        ax.set_xticks(np.arange(len(param_values)))
        ax.set_xticklabels(param_values)
        
        # Add title
        method_name = method.replace('_ci_covers_true', '').replace('_', ' ').title()
        ax.set_title(f'{method_name} Coverage')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Add overall title
    title = 'CI Coverage Heatmap'
    if param_name:
        title += f' by {param_name}'
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_ci_scatter(df, param_values=None,  save_path=None,effect_key=None):
    """
    Create scatter plots showing bootstrap estimates with their CIs for selected parameter values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_values : list, optional
        List of parameter values to include
    effect_key : str, optional
        Effect pair to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data if needed
    plot_df = df.copy()
    if effect_key is not None:
        plot_df = plot_df[plot_df['effect_key'] == effect_key]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # If param_values not provided, select a few interesting values
    if param_values is None:
        param_values = sorted(plot_df['param_value'].unique())
        if len(param_values) > 3:
            # Take min, median, max
            param_values = [
                param_values[0],
                param_values[len(param_values)//2],
                param_values[-1]
            ]
    
    if type(param_values) == str:
        param_values = [param_values]
        

    # Filter to selected parameter values
    plot_df = plot_df[plot_df['param_value'].isin(param_values)]

    
    if len(plot_df) == 0:
        print("No data available for the selected parameter values.")
        return
    
    # Create figure with subplots for each parameter value
    n_params = len(param_values)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 5*n_params), squeeze=False)
    
    # Plot for each parameter value
    for i, param_value in enumerate(param_values):
        ax = axes[i, 0]
        
        # Get row for this parameter value
        row = plot_df[plot_df['param_value'] == param_value].iloc[0]
        
        # Get true effect if available
        true_effect = row.get('true_effect')
        
        # Get bootstrap estimates with CIs
        if 'bootstrap_replica_estimate_with_ci' not in row or not isinstance(row['bootstrap_replica_estimate_with_ci'], list):
            ax.text(0.5, 0.5, "No bootstrap data available", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        estimates_with_ci = row['bootstrap_replica_estimate_with_ci']
        if not estimates_with_ci:
            ax.text(0.5, 0.5, "No valid bootstrap estimates", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Extract data
        estimates = [item[0] for item in estimates_with_ci]
        ci_lower = [item[1] for item in estimates_with_ci]
        ci_upper = [item[2] for item in estimates_with_ci]
        
        # Sort by estimate value for better visualization
        indices = np.argsort(estimates)
        estimates = [estimates[idx] for idx in indices]
        ci_lower = [ci_lower[idx] for idx in indices]
        ci_upper = [ci_upper[idx] for idx in indices]
        
        # Plot each estimate with CI as a horizontal line
        x_values = np.arange(len(estimates))
        ax.scatter(estimates, x_values, color='blue', alpha=0.6, label='Bootstrap Estimate')
        
        for j, (est, lower, upper) in enumerate(zip(estimates, ci_lower, ci_upper)):
            # Get CI coverage
            covers_true = lower <= true_effect <= upper if true_effect is not None else False
            color = 'green' if covers_true else 'red'
            
            # Plot CI as horizontal line
            ax.plot([lower, upper], [x_values[j], x_values[j]], 
                   color=color, alpha=0.4, linewidth=1.5)
        
        # Add vertical line for true effect if available
        if true_effect is not None:
            ax.axvline(x=true_effect, color='black', linestyle='--', 
                      linewidth=2, label='True Effect')
        
        # Add vertical lines for different method estimates
        method_colors = {
            'true_graph_effect': 'green',
            'pcmci_effect': 'orange',
            'bagged_effect': 'purple',
            'bootstrap_mean': 'red'
        }
        
        for method, color in method_colors.items():
            if method in row and row[method] is not None:
                ax.axvline(x=row[method], color=color, linestyle='-', 
                          linewidth=1.5, label=f'{method.replace("_effect", "")}')
        
        # Add title and labels
        ax.set_title(f'Bootstrap Estimates with CIs (Param Value = {param_value})')
        ax.set_xlabel('Effect Estimate')
        ax.set_ylabel('Bootstrap Replica')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        
        # Remove y-ticks
        ax.set_yticks([])
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    title = 'Bootstrap Estimates with CIs'
    if effect_key:
        title += f' for {effect_key}'
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_ci_success_rate(df, param_name=None, save_path=None):
    """
    Plot the success rate of CI estimation (valid non-None estimates) for different methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_name : str, optional
        Parameter name to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data if needed
    plot_df = df.copy()
    if param_name is not None:
        plot_df = plot_df[plot_df['param_name'] == param_name]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # Calculate success rates
    success_data = []
    
    for _, row in plot_df.iterrows():
        param_value = row['param_value']
        
        # Add standard method success rates (always 1 if exists)
        for method in ['true_graph', 'pcmci', 'bagged']:
            if f'{method}_linreg_ci_lower' in row and f'{method}_linreg_ci_upper' in row:
                has_ci = (row[f'{method}_linreg_ci_lower'] is not None and 
                          row[f'{method}_linreg_ci_upper'] is not None)
                
                success_data.append({
                    'param_value': param_value,
                    'method': f'{method}_linreg',
                    'success': 1.0 if has_ci else 0.0
                })
        
        # Add bootstrap success rate
        if 'bootstrap_success_rate' in row and row['bootstrap_success_rate'] is not None:
            success_data.append({
                'param_value': param_value,
                'method': 'bootstrap',
                'success': row['bootstrap_success_rate']
            })
    
    if not success_data:
        print("No success rate data available for plotting.")
        return
    
    success_df = pd.DataFrame(success_data)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot success rates
    if param_name is not None and len(plot_df['param_value'].unique()) > 1:
        # Line plot by parameter value
        sns.lineplot(
            data=success_df, 
            x='param_value', 
            y='success', 
            hue='method',
            marker='o',
            linewidth=2
        )
        plt.xlabel(f'{param_name.capitalize()} Value')
    else:
        # Bar plot for different methods
        sns.barplot(
            data=success_df,
            x='method',
            y='success'
        )
        plt.xticks(rotation=45)
    
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.05)
    title = 'CI Estimation Success Rate'
    if param_name:
        title += f' by {param_name}'
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_ci_distribution_comparison(df, param_value, effect_key=None, save_path=None):
    """
    Create a density plot comparing the distribution of different bootstrap-based CI methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with study results
    param_value : float
        Parameter value to filter by
    effect_key : str, optional
        Effect pair to filter by
    save_path : str, optional
        Path to save the plot
    """
    # Filter data
    plot_df = df[(df['param_value'] == param_value)]
    if effect_key is not None:
        plot_df = plot_df[plot_df['effect_key'] == effect_key]
    
    if len(plot_df) == 0:
        print("No data available for the selected filters.")
        return
    
    # Get the first row that matches
    row = plot_df.iloc[0]
    
    # Check if bootstrap data is available
    if ('bootstrap_replica_estimate_with_ci' not in row or 
        not isinstance(row['bootstrap_replica_estimate_with_ci'], list) or 
        not row['bootstrap_replica_estimate_with_ci']):
        print("No bootstrap data available for plotting.")
        return
    
    # Get bootstrap data
    estimates_with_ci = row['bootstrap_replica_estimate_with_ci']
    estimates = [item[0] for item in estimates_with_ci]
    ci_lower = [item[1] for item in estimates_with_ci]
    ci_upper = [item[2] for item in estimates_with_ci]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot distribution of bootstrap estimates
    sns.kdeplot(estimates, ax=ax, color='blue', label='Bootstrap Estimates')
    
    # Calculate and plot mean of bootstrap estimates
    bootstrap_mean = np.mean(estimates)
    ax.axvline(x=bootstrap_mean, color='blue', linestyle='--', 
               linewidth=2, label='Bootstrap Mean')
    
    # Get bootstrap CI
    if 'bootstrap_ci_lower' in row and 'bootstrap_ci_upper' in row:
        bootstrap_ci_lower = row['bootstrap_ci_lower']
        bootstrap_ci_upper = row['bootstrap_ci_upper']
        ax.axvspan(bootstrap_ci_lower, bootstrap_ci_upper, color='blue', alpha=0.2,
                  label='Bootstrap CI (from Distribution)')
    
    # Get combined CIs
    if 'bootstrap_combined_ci_lower' in row and 'bootstrap_combined_ci_upper' in row:
        combined_ci_lower = row['bootstrap_combined_ci_lower']
        combined_ci_upper = row['bootstrap_combined_ci_upper']
        ax.axvspan(combined_ci_lower, combined_ci_upper, color='green', alpha=0.2,
                  label='Combined Bootstrap CI (Mean of CIs)')
    
    if 'bootstrap_conservative_ci_lower' in row and 'bootstrap_conservative_ci_upper' in row:
        conserv_ci_lower = row['bootstrap_conservative_ci_lower']
        conserv_ci_upper = row['bootstrap_conservative_ci_upper']
        ax.axvspan(conserv_ci_lower, conserv_ci_upper, color='red', alpha=0.1,
                  label='Conservative Bootstrap CI (Union of CIs)')
    
    # Add other method estimates and CIs
    methods = ['true_graph', 'pcmci', 'bagged']
    colors = ['green', 'orange', 'purple']
    
    for method, color in zip(methods, colors):
        # Add effect estimate
        effect_col = f'{method}_effect'
        if effect_col in row and row[effect_col] is not None:
            ax.axvline(x=row[effect_col], color=color, linewidth=2, 
                      label=f'{method.replace("_", " ").title()} Estimate')
        
        # Add CI
        lower_col = f'{method}_linreg_ci_lower'
        upper_col = f'{method}_linreg_ci_upper'
        if lower_col in row and upper_col in row and row[lower_col] is not None:
            ax.axvspan(row[lower_col], row[upper_col], color=color, alpha=0.15,
                      label=f'{method.replace("_", " ").title()} CI')
    
    # Add true effect
    if 'true_effect' in row and row['true_effect'] is not None:
        ax.axvline(x=row['true_effect'], color='black', linestyle='-', 
                  linewidth=2, label='True Effect')
    
    # Add title and labels
    title = f'Comparison of Bootstrap CIs'
    if effect_key:
        title += f' for {effect_key}'
    title += f' (Param Value = {param_value})'
    ax.set_title(title)
    ax.set_xlabel('Effect Estimate')
    ax.set_ylabel('Density')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    #plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



# =====================================================================
# Adjustment Visualization
# =====================================================================
def plot_adjustment_sets(df, param_name, save_path=None):
    """
    Plot adjustment set sizes for different methods across parameter values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing adjustment set information
    param_name : str
        Name of the parameter to plot on x-axis
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract relevant columns
    adj_cols = [col for col in df.columns if 'adjustment_set_size' in col]
    if not adj_cols:
        plt.text(0.5, 0.5, "No adjustment set data available", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    param_values = sorted(df['param_value'].unique())
    
    # Group by parameter value and calculate mean and std of adjustment set sizes
    methods = []
    for col in adj_cols:
        method = col.split('_adjustment_set_size')[0]
        methods.append(method)
        
        means = []
        stds = []
        
        for param_val in param_values:
            subset = df[df['param_value'] == param_val]
            adjustment_sizes = subset[col].dropna()
            
            if len(adjustment_sizes) == 0:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            
            means.append(np.mean(adjustment_sizes))
            stds.append(np.std(adjustment_sizes) if len(adjustment_sizes) > 1 else 0)
        
        # Plot with error bars
        ax.errorbar(param_values, means, yerr=stds, 
                   marker='o', label=method.capitalize(), linewidth=2, capsize=5)
    
    ax.set_xlabel(f'{param_name.capitalize()} Value')
    ax.set_ylabel('Adjustment Set Size')
    ax.set_title(f'Adjustment Set Size Comparison - {param_name.capitalize()}')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def plot_adjustment_size_vs_error(analysis_df, method='bagged', param_name=None, save_path=None):
    """
    Plot relationship between adjustment set size and error.
    
    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame with adjustment set analysis
    method : str
        Method to plot ('pcmci', 'bagged', or 'bootstrap')
    param_name : str, optional
        Parameter name to color points by
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get columns for plotting
    size_col = f'{method}_adj_size'
    error_col = f'{method}_abs_error'
    
    # Check if columns exist
    if size_col not in analysis_df.columns or error_col not in analysis_df.columns:
        ax.text(0.5, 0.5, f"Data for {method} not available", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Create plot DataFrame by filtering NaNs
    plot_df = analysis_df[[size_col, error_col]].dropna()
    
    # Check if DataFrame is empty
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, "No valid data points", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Add parameter value for coloring if specified
    if param_name and 'param_name' in analysis_df.columns and 'param_value' in analysis_df.columns:
        param_df = analysis_df[analysis_df['param_name'] == param_name]
        
        if len(param_df) > 0:
            plot_df = pd.concat([
                plot_df, 
                param_df[['param_value']].reset_index(drop=True)
            ], axis=1)
            
            # Create scatter plot with color by parameter
            scatter = ax.scatter(plot_df[size_col], plot_df[error_col], 
                               c=plot_df['param_value'], cmap='viridis', 
                               alpha=0.7, s=50)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{param_name.capitalize()} Value')
        else:
            # If no parameter data, create simple scatter
            ax.scatter(plot_df[size_col], plot_df[error_col], alpha=0.7, s=50)
    else:
        # Create simple scatter plot
        ax.scatter(plot_df[size_col], plot_df[error_col], alpha=0.7, s=50)
    
    # Add trendline
    try:
        sns.regplot(x=size_col, y=error_col, data=plot_df, scatter=False, 
                   ax=ax, color='red', line_kws={'linestyle':'--'})
    except:
        # Skip trendline if it fails
        pass
    
    # Calculate correlation coefficient
    corr = plot_df[size_col].corr(plot_df[error_col])
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add labels and title
    ax.set_xlabel('Adjustment Set Size')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Adjustment Set Size vs. Error for {method.capitalize()}')
    
    # Set integer ticks on x-axis if the range is small
    if plot_df[size_col].nunique() <= 10:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_bootstrap_graph_diversity(bootstrap_graphs, save_path=None):
    """
    Analyze diversity of bootstrap graphs.
    
    Parameters
    ----------
    bootstrap_graphs : list
        List of bootstrap graphs
    save_path : str, optional
        Path to save results
        
    Returns
    -------
    dict
        Dictionary with diversity analysis results
    """
    # Convert graphs to hashable format
    graph_hashes = [str(g) for g in bootstrap_graphs]
    
    # Count unique graphs
    unique_graphs = set(graph_hashes)
    unique_count = len(unique_graphs)
    
    # Count frequency of each unique graph
    frequency = {}
    for gh in graph_hashes:
        if gh in frequency:
            frequency[gh] += 1
        else:
            frequency[gh] = 1
    
    # Sort frequencies
    sorted_freq = sorted(frequency.values(), reverse=True)
    
    # Calculate diversity metrics
    total_graphs = len(bootstrap_graphs)
    diversity_ratio = unique_count / total_graphs
    
    # Compute similarity matrix between all graphs
    similarity_matrix = np.zeros((total_graphs, total_graphs))
    
    for i in range(total_graphs):
        for j in range(i, total_graphs):
            # Simple similarity metric: proportion of matching elements
            match_ratio = np.mean(bootstrap_graphs[i] == bootstrap_graphs[j])
            similarity_matrix[i, j] = match_ratio
            similarity_matrix[j, i] = match_ratio
    
    # Compute dissimilarity matrix
    dissimilarity = 1 - similarity_matrix
    
    # Calculate average dissimilarity
    avg_dissimilarity = np.mean(dissimilarity[np.triu_indices(total_graphs, k=1)])
    
    # Compute graph statistics
    edge_density = []
    for g in bootstrap_graphs:
        n_edges = np.sum(g != 0)
        total_possible = g.shape[0] * g.shape[1] * g.shape[2]
        edge_density.append(n_edges / total_possible)
    
    avg_edge_density = np.mean(edge_density)
    std_edge_density = np.std(edge_density)
    
    # Compile results
    results = {
        'n_total_graphs': total_graphs,
        'n_unique_graphs': unique_count,
        'diversity_ratio': diversity_ratio,
        'avg_dissimilarity': float(avg_dissimilarity),
        'top_5_frequencies': sorted_freq[:5] if len(sorted_freq) >= 5 else sorted_freq,
        'avg_edge_density': float(avg_edge_density),
        'std_edge_density': float(std_edge_density)
    }
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results, dissimilarity

def plot_graph_diversity_heatmap(dissimilarity_matrix, save_path=None):
    """
    Plot heatmap of graph dissimilarities.
    
    Parameters
    ----------
    dissimilarity_matrix : numpy.ndarray
        Matrix of graph dissimilarities
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot dissimilarity heatmap
    im = ax1.imshow(dissimilarity_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Dissimilarity')
    
    # Add labels
    ax1.set_xlabel('Graph Index')
    ax1.set_ylabel('Graph Index')
    ax1.set_title('Graph Dissimilarity Matrix')
    
    # Compute linkage matrix for hierarchical clustering
    condensed_dissimilarity = squareform(dissimilarity_matrix)
    Z = hierarchy.linkage(condensed_dissimilarity, method='average')
    
    # Plot dendrogram
    dn = hierarchy.dendrogram(Z, ax=ax2)
    ax2.set_title('Hierarchical Clustering of Graphs')
    ax2.set_xlabel('Graph Index')
    ax2.set_ylabel('Distance')
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_adjustment_set_stats(results_df, param_name=None, save_path=None):
    """
    Plot adjustment set statistics across methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results
    param_name : str, optional
        Parameter name to group by
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create analysis DataFrame
    analysis_df = analyze_adjustment_sets(results_df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Identify adjustment size columns
    adj_cols = [col for col in analysis_df.columns if col.endswith('_adj_size')]
    
    # Create plot data
    if param_name and param_name in analysis_df.columns:
        # Filter to only include rows for this parameter
        param_df = analysis_df[analysis_df['param_name'] == param_name]
        
        # Check if DataFrame is empty
        if len(param_df) == 0:
            ax.text(0.5, 0.5, f"No data for parameter {param_name}", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Group by parameter value and calculate mean adjustment set size
        plot_data = param_df.groupby('param_value')[adj_cols].mean().reset_index()
        
        # Melt for plotting
        melt_df = pd.melt(plot_data, id_vars='param_value', value_vars=adj_cols,
                        var_name='Method', value_name='Adjustment Set Size')
        
        # Clean method names
        melt_df['Method'] = melt_df['Method'].str.replace('_adj_size', '')
        
        # Create the plot
        sns.lineplot(data=melt_df, x='param_value', y='Adjustment Set Size', 
                    hue='Method', marker='o', ax=ax)
        
        ax.set_xlabel(f'{param_name.capitalize()} Value')
        ax.set_title(f'Adjustment Set Size by {param_name.capitalize()} Value')
    else:
        # Calculate mean adjustment set size for each method
        means = {col.replace('_adj_size', ''): analysis_df[col].mean() for col in adj_cols}
        stds = {col.replace('_adj_size', ''): analysis_df[col].std() for col in adj_cols}
        
        # Create bar plot
        methods = list(means.keys())
        values = list(means.values())
        errors = [stds.get(m, 0) for m in methods]
        
        ax.bar(methods, values, yerr=errors, alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        ax.set_xlabel('Method')
        ax.set_title('Average Adjustment Set Size by Method')
    
    ax.set_ylabel('Adjustment Set Size')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_graph_diversity_by_parameter(results_df, param_name, save_path=None):
    """
    Plot graph diversity metrics against parameter values.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with estimation results
    param_name : str
        Parameter name for x-axis
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter DataFrame to get rows for this parameter
    param_df = results_df[results_df['param_name'] == param_name].copy()
    
    # Check if DataFrame is empty
    if len(param_df) == 0:
        ax1.text(0.5, 0.5, f"No data for parameter {param_name}", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Extract diversity metrics if they're in bootstrap_stats column
    if 'bootstrap_adjustment_set_stats' in param_df.columns:
        param_df['unique_graphs'] = param_df['bootstrap_adjustment_set_stats'].apply(
            lambda x: x.get('unique_graphs', np.nan) if isinstance(x, dict) else np.nan
        )
    
    # Create plot for unique graph count
    if 'unique_graphs' in param_df.columns:
        sns.lineplot(data=param_df, x='param_value', y='unique_graphs', 
                   marker='o', ax=ax1, color='blue')
        
        ax1.set_xlabel(f'{param_name.capitalize()} Value')
        ax1.set_ylabel('Number of Unique Graphs')
        ax1.set_title(f'Graph Diversity vs {param_name.capitalize()}')
        ax1.grid(True, linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, "No graph diversity data", 
                ha='center', va='center', fontsize=14)
    
    # Plot bootstrap CI width vs parameter value
    if 'bootstrap_ci_upper' in param_df.columns and 'bootstrap_ci_lower' in param_df.columns:
        param_df['ci_width'] = param_df['bootstrap_ci_upper'] - param_df['bootstrap_ci_lower']
        
        sns.lineplot(data=param_df, x='param_value', y='ci_width', 
                   marker='o', ax=ax2, color='green')
        
        ax2.set_xlabel(f'{param_name.capitalize()} Value')
        ax2.set_ylabel('Bootstrap CI Width')
        ax2.set_title(f'Uncertainty vs {param_name.capitalize()}')
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, "No bootstrap CI data", 
                ha='center', va='center', fontsize=14)
    
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =====================================================================
# Timing Visualization
# =====================================================================

def plot_timing_comparison(df, param_name, save_path=None):
    """
    Plot computation time comparison for different methods across parameter values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing timing information
    param_name : str
        Name of the parameter to plot on x-axis
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract relevant columns
    time_cols = [col for col in df.columns if '_time' in col]
    param_values = sorted(df['param_value'].unique())
    
    # Group by parameter value and calculate mean timing
    timing_data = {}
    for col in time_cols:
        method = col.split('_')[0]
        timing_data[method] = []
        
        for param_val in param_values:
            subset = df[df['param_value'] == param_val]
            mean_time = subset[col].mean()
            timing_data[method].append(mean_time)
    
    # Plot timing data
    markers = ['o', 's', '^', 'D', 'x']
    for i, (method, times) in enumerate(timing_data.items()):
        ax.plot(param_values, times, marker=markers[i % len(markers)], 
                label=method.capitalize(), linewidth=2)
    
    ax.set_xlabel(f'{param_name.capitalize()} Value')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title(f'Computation Time Comparison - {param_name.capitalize()}')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_discovery_estimation_ratio_by_T(df, save_path=None):
    """
    Plot ratio of discovery time to estimation time for different methods based on time series length T.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results with timing information
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Filter data for T study
    T_df = df[df['param_name'] == 'T'].copy()
    
    if len(T_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for T parameter study", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Sort by T value
    T_values = sorted(T_df['param_value'].unique())
    
    # Calculate discovery/estimation ratios for each method
    ratio_data = []
    
    for T_val in T_values:
        subset = T_df[T_df['param_value'] == T_val]
        
        # PCMCI method
        pcmci_discovery_time = subset['discovery_pcmci_time'].mean()
        pcmci_estimation_time = subset['effect_pcmci_effect_time'].mean()
        pcmci_ratio = pcmci_discovery_time / pcmci_estimation_time if pcmci_estimation_time > 0 else np.nan
        
        # Bagged PCMCI method
        bagged_discovery_time = subset['discovery_pcmci_time'].mean() + subset['discovery_bootstrap_time'].mean()
        bagged_estimation_time = subset['effect_bagged_effect_time'].mean()
        bagged_ratio = bagged_discovery_time / bagged_estimation_time if bagged_estimation_time > 0 else np.nan
        
        # Bootstrap PCMCI method
        bootstrap_discovery_time = subset['discovery_pcmci_time'].mean() + subset['discovery_bootstrap_time'].mean()
        bootstrap_estimation_time = subset['effect_bootstrap_effects_time'].mean()
        bootstrap_ratio = bootstrap_discovery_time / bootstrap_estimation_time if bootstrap_estimation_time > 0 else np.nan
        
        # Add absolute times as well for reference
        ratio_data.append({
            'T': T_val,
            'PCMCI Ratio': pcmci_ratio,
            'Bagged PCMCI Ratio': bagged_ratio,
            'Bootstrap PCMCI Ratio': bootstrap_ratio,
            'PCMCI Discovery': pcmci_discovery_time,
            'PCMCI Estimation': pcmci_estimation_time,
            'Bagged Discovery': bagged_discovery_time,
            'Bagged Estimation': bagged_estimation_time,
            'Bootstrap Discovery': bootstrap_discovery_time,
            'Bootstrap Estimation': bootstrap_estimation_time
        })
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(ratio_data)
    
    # Create figure with two subplots: one for ratios, one for absolute times
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # PLOT 1: Ratio of discovery to estimation time
    # Melt DataFrame for seaborn plotting
    ratio_melted = pd.melt(plot_df, id_vars=['T'], 
                          value_vars=['PCMCI Ratio', 'Bagged PCMCI Ratio', 'Bootstrap PCMCI Ratio'],
                          var_name='Method', value_name='Discovery/Estimation Ratio')
    
    # Create ratio plot
    sns.lineplot(data=ratio_melted, x='T', y='Discovery/Estimation Ratio', hue='Method', 
                marker='o', ax=ax1)
    
    # Add horizontal line at ratio=1 (equal time)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels
    ax1.set_xlabel('Time Series Length (T)')
    ax1.set_ylabel('Discovery/Estimation Time Ratio')
    ax1.set_title('Ratio of Discovery Time to Estimation Time by Method')
    
    # Improve x-axis
    ax1.set_xticks(T_values)
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend(title='Method')
    
    # Annotate the equal time line
    ax1.text(T_values[-1], 1.1, "Equal Time", ha='right', va='bottom', 
             color='gray', fontsize=9, fontstyle='italic')
    
    # PLOT 2: Absolute times (stacked bar chart)
    # Create a copy and reshape for stacked bar chart
    stack_data = []
    
    for idx, row in plot_df.iterrows():
        T_val = row['T']
        
        # PCMCI
        stack_data.append({
            'T': T_val,
            'Method': 'PCMCI',
            'Stage': 'Discovery',
            'Time': row['PCMCI Discovery']
        })
        stack_data.append({
            'T': T_val,
            'Method': 'PCMCI',
            'Stage': 'Estimation',
            'Time': row['PCMCI Estimation']
        })
        
        # Bagged PCMCI
        stack_data.append({
            'T': T_val,
            'Method': 'Bagged PCMCI',
            'Stage': 'Discovery',
            'Time': row['Bagged Discovery']
        })
        stack_data.append({
            'T': T_val,
            'Method': 'Bagged PCMCI',
            'Stage': 'Estimation',
            'Time': row['Bagged Estimation']
        })
        
        # Bootstrap PCMCI
        stack_data.append({
            'T': T_val,
            'Method': 'Bootstrap PCMCI',
            'Stage': 'Discovery',
            'Time': row['Bootstrap Discovery']
        })
        stack_data.append({
            'T': T_val,
            'Method': 'Bootstrap PCMCI',
            'Stage': 'Estimation',
            'Time': row['Bootstrap Estimation']
        })
    
    stack_df = pd.DataFrame(stack_data)
    
    # Create stacked bar chart
    bar_plot = sns.barplot(data=stack_df, x='T', y='Time', hue='Stage', 
                         ax=ax2, dodge=False, hue_order=['Discovery', 'Estimation'])
    
    # Add labels
    ax2.set_xlabel('Time Series Length (T)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Absolute Time for Discovery and Estimation Stages')
    
    # Create nested groups for methods
    prev_T = None
    method_positions = []
    
    # Get positions of bars
    for i, bar in enumerate(ax2.patches):
        T_idx = i // 6  # 6 bars per T value (2 stages × 3 methods)
        method_idx = (i % 6) // 2  # Which method within each T
        
        # Store positions for method labels
        if i % 6 == 0:  # First bar of each T group
            method_positions.append(bar.get_x())
        
        # Adjust bar colors for methods
        if method_idx == 0:  # PCMCI
            if i % 2 == 0:  # Discovery
                bar.set_color('C0')
            else:  # Estimation
                bar.set_color('C0')
                bar.set_alpha(0.5)
        elif method_idx == 1:  # Bagged
            if i % 2 == 0:  # Discovery
                bar.set_color('C1')
            else:  # Estimation
                bar.set_color('C1')
                bar.set_alpha(0.5)
        else:  # Bootstrap
            if i % 2 == 0:  # Discovery
                bar.set_color('C2')
            else:  # Estimation
                bar.set_color('C2')
                bar.set_alpha(0.5)
    
    # Remove legend since it's confusing with the custom coloring
    ax2.legend_.remove()
    
    # Create custom legend
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='C0', label='PCMCI - Discovery'),
        Patch(facecolor='C0', alpha=0.5, label='PCMCI - Estimation'),
        Patch(facecolor='C1', label='Bagged - Discovery'),
        Patch(facecolor='C1', alpha=0.5, label='Bagged - Estimation'),
        Patch(facecolor='C2', label='Bootstrap - Discovery'),
        Patch(facecolor='C2', alpha=0.5, label='Bootstrap - Estimation')
    ]
    
    ax2.legend(handles=legend_elements, loc='upper left', ncol=2)
    
    # Adjust layout
    #plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def generate_all_plots(study_folder, results_folder, studies=None):
    """
    Generate all plots for the studies in the main folder.
    
    Parameters
    ----------
    study_folder : str
        Path to the main study folder
    results_folder : str
        Path to save results
    studies : list, optional
        List of study names to process. If None, will detect automatically.
    
    Returns
    -------
    dict
        Dictionary with information about generated plots
    """
    from causal_evaluation import load_study_results
    
    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Detect studies if not provided
    if studies is None:
        print("Detecting studies in folder...")
        studies = [d for d in os.listdir(study_folder) 
                  if os.path.isdir(os.path.join(study_folder, d)) and d.endswith('_study')]
        print(f"Found studies: {studies}")
    
    if not studies:
        print("No studies found!")
        return {}
    
    # Load all study results
    all_results = load_study_results(study_folder)
    all_results.to_csv(os.path.join(results_folder, "all_results.csv"), index=False)
    
    # Split by study type
    study_data = {}
    for study in studies:
        study_df = all_results[all_results['substudy'] == study]
        
        if len(study_df) > 0:
            param_name = all_results[all_results['substudy'] == study].iloc[0]['param_name']
            study_data[param_name] = study_df
    
    # Generate plots for each study
    plot_info = {
        'Pipeline': {}, 'Ratio': {}, 'Replica': {}, 'error': {}, 'error_ci': {}, 
        'bootstrap': {}, 'adjustment': {}, 'bootstrap_linreg_ci': {},
        'linreg_ci_coverage': {}, 'linreg_ci_analysis': {}, 'error_vs_ci_width': {}
    }
    
    # 1. Timing plots
    print(f"  Generating timing plots...")
    timing_path = os.path.join(results_folder, f"Pipeline_timing.png")
    plot_pipeline_timing_by_T(all_results, save_path=timing_path)
    plot_info['Pipeline']['T'] = timing_path
    
    timingRatio_path = os.path.join(results_folder, f"Ratio_timing.png")
    plot_discovery_estimation_ratio_by_T(all_results, save_path=timingRatio_path)
    plot_info['Ratio']['T'] = timingRatio_path
    
    timingBoot_path = os.path.join(results_folder, f"Replica_timing.png")
    plot_bootstrap_timing_by_n_boot(all_results, save_path=timingBoot_path)
    plot_info['Replica']['n_boot'] = timing_path
    
    """
    # 2. Overall CI Coverage Analysis
    print(f"  Generating overall CI coverage analysis...")
    overall_coverage_path = os.path.join(results_folder, f"overall_linreg_ci_coverage.png")
    plot_linreg_ci_coverage(all_results, save_path=overall_coverage_path)
    plot_info['linreg_ci_coverage']['overall'] = overall_coverage_path
    
    overall_ci_analysis_path = os.path.join(results_folder, f"overall_linreg_ci_analysis.png")
    analyze_linreg_ci_properties(all_results, save_path=overall_ci_analysis_path)
    plot_info['linreg_ci_analysis']['overall'] = overall_ci_analysis_path
    """
    
    overall_error_width_path = os.path.join(results_folder, f"overall_error_vs_ci_width.png")
    plot_error_vs_ci_width(all_results, save_path=overall_error_width_path)
    plot_info['error_vs_ci_width']['overall'] = overall_error_width_path
    
    for param_name, df in study_data.items():
        print(f"Generating plots for {param_name} study...")
        
        # 3. Error plots without CI
        print(f"  Generating error plots without CI...")
        # Absolute error
        abs_error_path = os.path.join(results_folder, f"{param_name}_abs_error.png")
        plot_error_comparison(df, param_name, error_type='abs', save_path=abs_error_path)
        # Bias (signed error)
        bias_path = os.path.join(results_folder, f"{param_name}_bias.png")
        plot_error_comparison(df, param_name, error_type='raw', save_path=bias_path)
        plot_info['error'][param_name] = {'abs': abs_error_path, 'bias': bias_path}
        
        
        # 4. Effect comparison
        print(f"  Generating effect comparison plots...")
        effect_path = os.path.join(results_folder, f"{param_name}_effect.png")
        plot_effect_comparison(df, param_name, save_path=effect_path)
        
        
        # 5. CI Comparison
        print(f"  Generating CI comparison plots...")
        ci_path = os.path.join(results_folder, f"{param_name}_CIs.png")
        plot_ci_width_comparison(df, param_name, save_path=ci_path)
        
        ci_error_path = os.path.join(results_folder, f"{param_name}_CIvsErrors.png")
        plot_ci_width_to_error_ratio(df, param_name, save_path=ci_error_path)
        
        bs_scatter_path = os.path.join(results_folder, f"{param_name}_bsScatter.png")
        plot_bootstrap_ci_scatter(df, param_values=None, save_path=bs_scatter_path)
        
        # 6. Bootstrap distribution with method comparisons for interesting cases
        print(f"  Generating bootstrap distribution plots...")
        # Select interesting parameter values
        interesting_values = select_interesting_cases(df, param_name)
        bootstrap_paths = {}
        
        for param_val in interesting_values:
            print(f"    Processing param_val = {param_val}")
            row = df[(df['param_name'] == param_name) & (df['param_value'] == param_val)].iloc[0]
            
            if 'bootstrap_effects' in row and sum(not pd.isna(x) for x in row['bootstrap_effects']) >= 2:
                bs_path = os.path.join(results_folder, f"{param_name}_{param_val}_bootstrap.png")
                plot_bootstrap_distribution_with_methods(
                    row['bootstrap_effects'],
                    true_effect=row.get('true_effect'),
                    true_graph_effect=row.get('true_graph_effect'),
                    pcmci_effect=row.get('pcmci_effect'),
                    bagged_effect=row.get('bagged_effect'),
                    title=f"{param_name.capitalize()} = {param_val}",
                    save_path=bs_path
                )
                bootstrap_paths[param_val] = bs_path
        
        plot_info['bootstrap'][param_name] = bootstrap_paths
        
        # 7. Adjustment set plots
        print(f"  Generating adjustment set plots...")
        if any('adjustment_set_size' in col for col in df.columns):
            adj_path = os.path.join(results_folder, f"{param_name}_adjustment_sets.png")
            plot_adjustment_sets(df, param_name, save_path=adj_path)
            plot_info['adjustment'][param_name] = adj_path
        
        """
        # 8. New: CI coverage by parameter
        print(f"  Generating CI coverage plots...")
        coverage_path = os.path.join(results_folder, f"{param_name}_linreg_ci_coverage.png")
        plot_linreg_ci_coverage(df, param_name, save_path=coverage_path)
        plot_info['linreg_ci_coverage'][param_name] = coverage_path
        
        # 9. New: CI properties analysis
        print(f"  Generating CI properties analysis...")
        ci_analysis_path = os.path.join(results_folder, f"{param_name}_linreg_ci_analysis.png")
        analyze_linreg_ci_properties(df, param_name, save_path=ci_analysis_path)
        plot_info['linreg_ci_analysis'][param_name] = ci_analysis_path
        
        # 10. New: Error vs CI width
        print(f"  Generating error vs CI width analysis...")
        error_width_path = os.path.join(results_folder, f"{param_name}_error_vs_ci_width.png")
        plot_error_vs_ci_width(df, param_name, save_path=error_width_path)
        plot_info['error_vs_ci_width'][param_name] = error_width_path
        
        
        # 11. Bootstrap linear regression CI distribution for interesting cases
        print(f"  Generating bootstrap linear regression CI distribution plots...")
        linreg_ci_paths = {}
        if 'bootstrap_replica_ci_lower' in df.columns:
            for param_val in interesting_values:
                lr_ci_path = os.path.join(results_folder, f"{param_name}_{param_val}_linreg_ci.png")
                plot_bootstrap_linreg_ci_distribution(df, param_name, param_val, save_path=lr_ci_path)
                linreg_ci_paths[param_val] = lr_ci_path
            
            plot_info['bootstrap_linreg_ci'][param_name] = linreg_ci_paths
        """
    
    # Generate combined plots across all studies
    print("Generating combined plots across all studies...")
    
    # Combined error comparison across studies
    if len(study_data) > 1:
        # Combine error metrics for all methods
        for method in ['pcmci', 'bagged', 'bootstrap']:
            error_metrics = []
            for param_name, df in study_data.items():
                # Group by parameter value and calculate mean error
                param_errors = df.groupby('param_value')[f'{method}_abs_error'].mean().reset_index()
                param_errors['param_name'] = param_name
                error_metrics.append(param_errors)
            
            if error_metrics:
                combined_errors = pd.concat(error_metrics)
                
                # Plot combined errors
                plt.figure(figsize=(12, 8))
                for param_name in study_data.keys():
                    subset = combined_errors[combined_errors['param_name'] == param_name]
                    plt.plot(subset['param_value'], subset[f'{method}_abs_error'], 
                            marker='o', label=param_name.capitalize())
                
                plt.xlabel('Parameter Value')
                plt.ylabel('Mean Absolute Error')
                #plt.title(f'{method.capitalize()} Mean Absolute Error Across Studies')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(results_folder, f"combined_{method}_error.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        """        
        # Combined CI coverage comparison
        coverage_metrics = []
        for param_name, df in study_data.items():
            for method in ['true_graph', 'pcmci', 'bagged']:
                if f'{method}_ci_covers' in df.columns:
                    # Group by parameter value and calculate coverage
                    param_coverage = df.groupby('param_value')[f'{method}_ci_covers'].mean().reset_index()
                    param_coverage['param_name'] = param_name
                    param_coverage['method'] = method
                    coverage_metrics.append(param_coverage)
        
        if coverage_metrics:
            combined_coverage = pd.concat(coverage_metrics)
            
            # Plot combined coverage by method
            for method in ['true_graph', 'pcmci', 'bagged']:
                method_coverage = combined_coverage[combined_coverage['method'] == method]
                
                if len(method_coverage) > 0:
                    plt.figure(figsize=(12, 8))
                    for param_name in study_data.keys():
                        subset = method_coverage[method_coverage['param_name'] == param_name]
                        if len(subset) > 0:
                            plt.plot(subset['param_value'], subset[f'{method}_ci_covers'] * 100, 
                                    marker='o', label=param_name.capitalize())
                    
                    plt.axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
                    plt.xlabel('Parameter Value')
                    plt.ylabel('Coverage Rate (%)')
                    #plt.title(f'{method.capitalize()} CI Coverage Across Studies')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(results_folder, f"combined_{method}_coverage.png"), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        """
    
    print(f"All plots generated and saved to {results_folder}")
    
    return plot_info