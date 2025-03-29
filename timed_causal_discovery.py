"""
Timing-aware causal discovery and effect estimation.

This module provides functions for causal discovery and effect estimation
with integrated timing measurements. Uses tigramite for the underlying
causal discovery algorithms.
"""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from timing_utils import TimingCollector
from causal_utils import get_estimate, calculate_bootstrap_stats, get_groundtruth


class TimedCausalDiscovery:
    """
    Class that handles causal discovery with timing measurements.
    
    Wraps tigramite's PCMCI and bootstrap methods with timing
    capabilities to measure performance of different approaches.
    """
    
    def __init__(self, timing_collector=None):
        """
        Initialize timed causal discovery object.
        
        Parameters
        ----------
        timing_collector : TimingCollector, optional
            Collector for timing measurements. If None, a new one is created.
        """
        self.timing = timing_collector if timing_collector is not None else TimingCollector("timed_causal_discovery")
        
    def discover_pcmci(self, dataset, pc_alpha=0.05, tau_max=5, cond_ind_test=None, prefix=""):
        """
        Run standard PCMCI causal discovery with timing.
        
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
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        dict
            PCMCI results
        """
        if cond_ind_test is None:
            cond_ind_test = ParCorr()
            
        with self.timing.timer(f"{prefix}pcmci"):
            pcmci = PCMCI(dataframe=dataset, cond_ind_test=cond_ind_test, verbosity=0)
            results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
            
        return results, pcmci
    
    def discover_bootstrap(self, pcmci, pc_alpha=0.05, tau_max=5, n_boot=100, prefix=""):
        """
        Run bootstrapped PCMCI with timing.
        
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
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        dict
            Bootstrap results including bagged graph
        """
        with self.timing.timer(f"{prefix}bootstrap"):
            results = pcmci.run_bootstrap_of(
                method='run_pcmci', 
                method_args={'tau_max': tau_max, 'pc_alpha': pc_alpha}, 
                boot_samples=n_boot,
                boot_blocklength=10
            )
            
        with self.timing.timer(f"{prefix}create_bagged"):
            bagged_graph = results['summary_results']['most_frequent_links']
            
        return results, bagged_graph
    
    def discover_full(self, dataset, pc_alpha=0.05, tau_max=5, n_boot=100, 
                     cond_ind_test=None, prefix=""):
        """
        Run complete causal discovery process including standard and bootstrapped PCMCI.
        
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
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        dict
            Complete discovery results
        """
        # Run standard PCMCI
        pcmci_results, pcmci = self.discover_pcmci(
            dataset, pc_alpha, tau_max, cond_ind_test, prefix=f"{prefix}standard_"
        )
        
        # Run bootstrapped PCMCI
        bootstrap_results, bagged_graph = self.discover_bootstrap(
            pcmci, pc_alpha, tau_max, n_boot, prefix=f"{prefix}boostrap_"
        )
        
        # Return combined results
        return {
            'pcmci': pcmci_results,
            'bootstrap': bootstrap_results,
            'bagged_graph': bagged_graph
        }
    
    def get_timing(self):
        """
        Get the timing collector with measurements.
        
        Returns
        -------
        TimingCollector
            The timing collector object
        """
        return self.timing


class TimedCausalEffects:
    """
    Class that handles causal effect estimation with timing measurements.
    
    Wraps effect estimation methods with timing capabilities to measure
    performance of different estimation approaches.
    """
    
    def __init__(self, timing_collector=None):
        """
        Initialize timed causal effects object.
        
        Parameters
        ----------
        timing_collector : TimingCollector, optional
            Collector for timing measurements. If None, a new one is created.
        """
        self.timing = timing_collector if timing_collector is not None else TimingCollector("timed_causal_effects")
    
    def estimate_effect_pcmci(self, graph, dataset, X, Y, prefix=""):
        """
        Estimate effect using standard PCMCI graph with timing.
        
        Parameters
        ----------
        graph : ndarray
            PCMCI graph
        dataset : DataFrame
            Time series data
        X : tuple
            Variable to intervene on
        Y : tuple
            Outcome variable
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        float
            Estimated causal effect
        """
        with self.timing.timer(f"{prefix}estimate_pcmci"):
            effect = get_estimate(graph, dataset, X, Y)
            
        return effect
    
    def estimate_effect_bagged(self, bagged_graph, dataset, X, Y, prefix=""):
        """
        Estimate effect using bagged graph with timing.
        
        Parameters
        ----------
        bagged_graph : ndarray
            Bagged graph from bootstrap
        dataset : DataFrame
            Time series data
        X : tuple
            Variable to intervene on
        Y : tuple
            Outcome variable
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        float
            Estimated causal effect
        """
        with self.timing.timer(f"{prefix}estimate_bagged"):
            effect = get_estimate(bagged_graph, dataset, X, Y)
            
        return effect
    
    def estimate_effect_bootstrap(self, bootstrap_graphs, dataset, X, Y, prefix=""):
        """
        Estimate effects using all bootstrap graphs with timing.
        
        Parameters
        ----------
        bootstrap_graphs : list
            List of bootstrap graphs
        dataset : DataFrame
            Time series data
        X : tuple
            Variable to intervene on
        Y : tuple
            Outcome variable
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        tuple
            (List of effects, Statistics)
        """
        bootstrap_effects = []
        
        with self.timing.timer(f"{prefix}estimate_all_bootstrap"):
            for b_graph in bootstrap_graphs:
                effect = get_estimate(b_graph, dataset, X, Y)
                bootstrap_effects.append(effect)
                
        with self.timing.timer(f"{prefix}bootstrap_stats"):
            stats = calculate_bootstrap_stats(bootstrap_effects)
            
        return bootstrap_effects, stats
    
    def estimate_groundtruth(self, links, X, Y, noises=None, prefix=""):
        """
        Estimate ground truth effect with timing.
        
        Parameters
        ----------
        links : dict
            Links dictionary for SCM
        X : tuple
            Variable to intervene on
        Y : tuple
            Outcome variable
        noises : dict, optional
            Noise distributions
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        float
            Ground truth causal effect
        """
        with self.timing.timer(f"{prefix}groundtruth"):
            effect = get_groundtruth(links, [X], [Y], noises=noises)
            
        return effect
    
    def estimate_all_effects(self, discovery_results, dataset, X, Y, 
                           links=None, noises=None, prefix=""):
        """
        Estimate causal effects using all methods with timing.
        
        Parameters
        ----------
        discovery_results : dict
            Results from causal discovery
        dataset : DataFrame
            Time series data
        X : tuple
            Variable to intervene on
        Y : tuple
            Outcome variable
        links : dict, optional
            Links dictionary for ground truth
        noises : dict, optional
            Noise distributions for ground truth
        prefix : str
            Prefix for timing operations
            
        Returns
        -------
        dict
            Dictionary with all effect estimates
        """
        results = {}
        
        # Get ground truth if links are provided
        if links is not None:
            results['true_effect'] = self.estimate_groundtruth(
                links, X, Y, noises, prefix=f"{prefix}true_"
            )
        
        # Estimate with PCMCI graph
        pcmci_graph = discovery_results['pcmci']['graph']
        results['pcmci_effect'] = self.estimate_effect_pcmci(
            pcmci_graph, dataset, X, Y, prefix=f"{prefix}pcmci_"
        )
        
        # Estimate with bagged graph
        bagged_graph = discovery_results['bagged_graph']
        results['bagged_effect'] = self.estimate_effect_bagged(
            bagged_graph, dataset, X, Y, prefix=f"{prefix}bagged_"
        )
        
        
        # Estimate with bootstrap graphs
        bootstrap_graphs = discovery_results['bootstrap']['boot_results']['graph']
        results['bootstrap_effects'], results['bootstrap_stats'] = self.estimate_effect_bootstrap(
            bootstrap_graphs, dataset, X, Y, prefix=f"{prefix}bootstrap_"
        )
        
        return results
    
    def get_timing(self):
        """
        Get the timing collector with measurements.
        
        Returns
        -------
        TimingCollector
            The timing collector object
        """
        return self.timing


def compare_estimation_approaches(
    dataset, effect_pairs, links=None, noises=None,
    pc_alpha=0.05, tau_max=5, n_boot=100, cond_ind_test=None,
    timing_collector=None
):
    """
    Compare different causal estimation approaches with timing.
    
    Parameters
    ----------
    dataset : DataFrame
        Time series data
    effect_pairs : list
        List of (X, Y) pairs to estimate effects for
    links : dict, optional
        Links dictionary for ground truth
    noises : dict, optional
        Noise distributions for ground truth
    pc_alpha : float
        Significance level
    tau_max : int
        Maximum time lag
    n_boot : int
        Number of bootstrap samples
    cond_ind_test : object, optional
        Conditional independence test
    timing_collector : TimingCollector, optional
        Collector for timing measurements
        
    Returns
    -------
    tuple
        (Results dict, Timing collector)
    """
    # Initialize timing if not provided
    if timing_collector is None:
        timing_collector = TimingCollector("compare_approaches")
    
    # Initialize results
    results = {
        'approach1': {},  # Standard PCMCI
        'approach2': {},  # Bootstrapped PCMCI + Bagged
        'approach3': {}   # Bootstrapped PCMCI + Bootstrap graphs
    }
    
    # Approach 1: Standard PCMCI → effect estimation
    with timing_collector.timer("approach1_full"):
        # Run PCMCI
        with timing_collector.timer("approach1_discovery"):
            pcmci = PCMCI(dataframe=dataset, cond_ind_test=cond_ind_test or ParCorr(), verbosity=0)
            results_pcmci = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
        
        # Estimate effects
        approach1_effects = {}
        with timing_collector.timer("approach1_estimation"):
            for X, Y in effect_pairs:
                pcmci_graph = results_pcmci['graph']
                effect = get_estimate(pcmci_graph, dataset, X=X, Y=Y)
                
                if links is not None:
                    true_effect = get_groundtruth(links, [X], [Y], noises=noises)
                else:
                    true_effect = None
                
                effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
                approach1_effects[effect_key] = {
                    'true_effect': true_effect,
                    'effect': effect
                }
    
    # Approach 2: Bootstrapped PCMCI → effect estimation with bagged graph
    with timing_collector.timer("approach2_full"):
        # Run bootstrapped PCMCI
        with timing_collector.timer("approach2_discovery"):
            # Reuse pcmci object from approach 1
            results_boot = pcmci.run_bootstrap_of(
                method='run_pcmci', 
                method_args={'tau_max': tau_max, 'pc_alpha': pc_alpha}, 
                boot_samples=n_boot,
                boot_blocklength=10
            )
            bagged_graph = results_boot['summary_results']['most_frequent_links']
        
        # Estimate effects
        approach2_effects = {}
        with timing_collector.timer("approach2_estimation"):
            for X, Y in effect_pairs:
                effect = get_estimate(bagged_graph, dataset, X=X, Y=Y)
                
                effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
                approach2_effects[effect_key] = {
                    'true_effect': approach1_effects[effect_key]['true_effect'] if effect_key in approach1_effects else None,
                    'effect': effect
                }
    
    # Approach 3: Bootstrapped PCMCI → effect estimation with bootstrap graphs
    with timing_collector.timer("approach3_full"):
        # Use bootstrap results from approach 2
        bootstrap_graphs = results_boot['boot_results']['graph']
        
        # Estimate effects with each bootstrap graph
        approach3_effects = {}
        with timing_collector.timer("approach3_estimation"):
            for X, Y in effect_pairs:
                effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
                bootstrap_effects = []
                
                for b_graph in bootstrap_graphs:
                    b_effect = get_estimate(b_graph, dataset, X=X, Y=Y)
                    bootstrap_effects.append(b_effect)
                
                stats = calculate_bootstrap_stats(bootstrap_effects)
                approach3_effects[effect_key] = {
                    'true_effect': approach1_effects[effect_key]['true_effect'] if effect_key in approach1_effects else None,
                    'effects': bootstrap_effects,
                    'stats': stats
                }
    
    # Store results
    results['approach1'] = approach1_effects
    results['approach2'] = approach2_effects
    results['approach3'] = approach3_effects
    
    return results, timing_collector
