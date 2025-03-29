"""
Timing utilities for causal discovery and effect estimation.

This module provides tools for timing and profiling causal discovery and
effect estimation workflows, with functions to collect, analyze, and visualize
timing data for different approaches.
"""

import time
from contextlib import contextmanager
from collections import defaultdict
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TimingCollector:
    """Collect and analyze timing information for causal discovery and effect estimation."""
    
    def __init__(self, name="timing"):
        """
        Initialize a timing collector.
        
        Parameters
        ----------
        name : str
            Name of this timing collector for identification
        """
        self.name = name
        self.timings = defaultdict(list)
        self.current_timers = {}
        
    @contextmanager
    def timer(self, operation):
        """
        Context manager for timing an operation.
        
        Parameters
        ----------
        operation : str
            Name of the operation being timed
        
        Yields
        ------
        None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[operation].append(elapsed)
            
    def start_timer(self, operation):
        """
        Start timing an operation manually.
        
        Parameters
        ----------
        operation : str
            Name of the operation to start timing
        """
        self.current_timers[operation] = time.time()
        
    def stop_timer(self, operation):
        """
        Stop timing an operation and record it.
        
        Parameters
        ----------
        operation : str
            Name of the operation to stop timing
            
        Returns
        -------
        float or None
            Elapsed time in seconds, or None if timer wasn't started
        """
        if operation in self.current_timers:
            elapsed = time.time() - self.current_timers[operation]
            self.timings[operation].append(elapsed)
            del self.current_timers[operation]
            return elapsed
        return None
    
    def get_summary(self):
        """
        Get statistical summary of all timings.
        
        Returns
        -------
        dict
            Dictionary with statistical measures for each operation
        """
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times),
                    'total': np.sum(times)
                }
        return summary
    
    def to_dataframe(self):
        """
        Convert timings to a DataFrame for analysis.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with operation and timing information
        """
        data = []
        for operation, times in self.timings.items():
            for t in times:
                data.append({'Operation': operation, 'Time (s)': t})
        return pd.DataFrame(data)
    
    def plot_comparison(self, methods=None, figsize=(10, 6), save_path=None):
        """
        Plot timing comparison between different methods.
        
        Parameters
        ----------
        methods : list, optional
            List of method names to include in plot
        figsize : tuple, optional
            Figure dimensions
        save_path : str, optional
            If provided, saves the figure to this path
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        df = self.to_dataframe()
        if methods:
            df = df[df['Operation'].isin(methods)]
            
        plt.figure(figsize=figsize)
        sns.boxplot(x='Operation', y='Time (s)', data=df)
        plt.title(f"Timing Comparison - {self.name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def save(self, filepath):
        """
        Save timing data to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the timing data
            
        Returns
        -------
        str
            Path to the saved file
        """
        # Convert timings to serializable format
        serializable = {op: list(times) for op, times in self.timings.items()}
        
        data = {
            'name': self.name,
            'timings': serializable
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load timing data from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the timing data from
            
        Returns
        -------
        TimingCollector
            Loaded timing collector
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        collector = cls(name=data.get('name', 'loaded_timing'))
        collector.timings = defaultdict(list)
        
        for op, times in data.get('timings', {}).items():
            collector.timings[op] = times
            
        return collector
    
    def clear(self):
        """Clear all stored timings."""
        self.timings.clear()
        self.current_timers.clear()


def summarize_timing_comparison(timing_collector, approaches=None):
    """
    Generate a comprehensive summary of timing results.
    
    Parameters
    ----------
    timing_collector : TimingCollector
        Collector with timing data
    approaches : list, optional
        List of specific approaches to include in the summary
    
    Returns
    -------
    pandas.DataFrame
        Summary DataFrame
    """
    summary = timing_collector.get_summary()
    
    # Filter for specific approaches if requested
    if approaches:
        summary = {k: v for k, v in summary.items() if any(k.startswith(a) for a in approaches)}
    
    # Create summary DataFrame
    summary_data = []
    for operation, stats in summary.items():
        # For operations with "full", "discovery", or "estimation" parts, extract the approach
        if "_full" in operation or "_discovery" in operation or "_estimation" in operation:
            parts = operation.split('_')
            approach = parts[0]
            phase = parts[1]
            
            row = {
                'Approach': approach,
                'Phase': phase,
                'Mean Time (s)': stats['mean'],
                'Median Time (s)': stats['median'],
                'Std Dev (s)': stats['std'],
                'Min Time (s)': stats['min'],
                'Max Time (s)': stats['max'],
                'Count': stats['count'],
                'Total Time (s)': stats['total']
            }
            summary_data.append(row)
    
    # Create DataFrame and pivot for easier analysis
    df = pd.DataFrame(summary_data)
    if not df.empty:
        pivot = df.pivot(index='Approach', columns='Phase', values=['Mean Time (s)', 'Median Time (s)'])
        return pivot
    
    return df


def plot_timing_comparison(timing_collector, figsize=(12, 8), save_path=None):
    """
    Create comprehensive timing comparison plots.
    
    Parameters
    ----------
    timing_collector : TimingCollector
        Collector with timing data
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save the plots with this base path
    
    Returns
    -------
    tuple
        (fig1, fig2) - Two figures with different visualizations
    """
    # Extract timing data
    df = timing_collector.to_dataframe()
    
    # Filter for main approaches
    full_approaches = [op for op in df['Operation'].unique() if op.endswith('_full')]
    discovery_approaches = [op for op in df['Operation'].unique() if op.endswith('_discovery')]
    estimation_approaches = [op for op in df['Operation'].unique() if op.endswith('_estimation')]
    
    # Create full comparison plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.boxplot(x='Operation', y='Time (s)', data=df[df['Operation'].isin(full_approaches)], ax=ax1)
    ax1.set_title('Timing Comparison - Full Approaches')
    ax1.set_xlabel('Approach')
    try:
        ax1.set_xticklabels([op.replace('_full', '') for op in ax1.get_xticklabels()])
    except:
        pass  # In case of empty data
    ax1.tick_params(axis='x', rotation=45)
    
    # Create breakdown plot
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=figsize)
    
    # Discovery phase
    sns.boxplot(x='Operation', y='Time (s)', data=df[df['Operation'].isin(discovery_approaches)], ax=ax2)
    ax2.set_title('Discovery Phase')
    ax2.set_xlabel('Approach')
    try:
        ax2.set_xticklabels([op.replace('_discovery', '') for op in ax2.get_xticklabels()])
    except:
        pass  # In case of empty data
    ax2.tick_params(axis='x', rotation=45)
    
    # Estimation phase
    sns.boxplot(x='Operation', y='Time (s)', data=df[df['Operation'].isin(estimation_approaches)], ax=ax3)
    ax3.set_title('Estimation Phase')
    ax3.set_xlabel('Approach')
    try:
        ax3.set_xticklabels([op.replace('_estimation', '') for op in ax3.get_xticklabels()])
    except:
        pass  # In case of empty data
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figures if requested
    if save_path:
        fig1.savefig(f"{save_path}_full.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{save_path}_breakdown.png", dpi=300, bbox_inches='tight')
    
    return fig1, fig2
