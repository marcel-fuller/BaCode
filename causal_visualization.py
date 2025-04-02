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
    if title:
        plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
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
    if title:
        plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
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
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =====================================================================
# Timing Visualization
# =====================================================================

def plot_timing_comparison(results_df, param_name=None, operation=None, 
                         methods=None, save_path=None):
    """
    Plot comparison of execution times across methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with timing information
    param_name : str, optional
        Parameter name for x-axis
    operation : str, optional
        Operation to plot times for (e.g., 'effect_estimation')
    methods : list, optional
        List of methods to include
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Create a copy of the DataFrame for plotting
    plot_df = results_df.copy()
    
    # Select timing columns
    time_cols = [col for col in plot_df.columns if col.endswith('_time')]
    
    # Filter timing columns by operation if specified
    if operation:
        time_cols = [col for col in time_cols if operation in col]
    
    # Filter timing columns by methods if specified
    if methods:
        time_cols = [col for col in time_cols if any(method in col for method in methods)]
    
    # Check if we have any valid columns
    if not time_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No timing data available for selected parameters", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x variable based on param_name
    if param_name is not None:
        if param_name not in plot_df.columns:
            ax.text(0.5, 0.5, f"Parameter '{param_name}' not found in DataFrame", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        plot_df = plot_df[plot_df['param_name'] == param_name].copy()
        x_var = 'param_value'
        x_label = f'{param_name.capitalize()} Value'
    else:
        # If no param_name, use index as x-axis
        plot_df = plot_df.reset_index(drop=True)
        x_var = plot_df.index
        x_label = 'Configuration'
    
    # Melt DataFrame for plotting
    melt_df = pd.melt(plot_df, id_vars=[x_var], value_vars=time_cols, 
                      var_name='Operation', value_name='Time (s)')
    
    # Clean up operation names
    melt_df['Operation'] = melt_df['Operation'].str.replace('_time', '')
    
    # Create the plot
    if isinstance(x_var, pd.Index):
        # Categorical plot
        sns.barplot(data=melt_df, x=melt_df.index, y='Time (s)', hue='Operation', ax=ax)
    else:
        # Line plot
        sns.lineplot(data=melt_df, x=x_var, y='Time (s)', hue='Operation', 
                    marker='o', ax=ax)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Time (seconds)')
    
    if operation:
        ax.set_title(f'Execution Time Comparison for {operation}')
    else:
        ax.set_title('Execution Time Comparison')
    
    ax.legend(title='Operation')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


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
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig