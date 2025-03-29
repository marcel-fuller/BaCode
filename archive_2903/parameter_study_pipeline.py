"""
Pipeline for studying parameter effects on causal discovery and effect estimation.

This script runs systematic studies of how different parameters (autocorrelation,
cross-link strength, noise levels) affect causal discovery quality and effect
estimation accuracy.
"""

import os
import json
import time
import traceback
import platform
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


from scipy import stats

from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from causal_utils import (
    create_configurable_links, 
    create_stable_links,
    create_noise_distributions,
    evaluate_estimation_performance,
    plot_effect_comparison,
    extended_evaluate_estimation_performance,
    plot_bootstrap_distribution
)

from timed_causal_discovery import (
    TimedCausalDiscovery,
    TimedCausalEffects
)

from timing_utils import TimingCollector

#Change plot backend to prevent errors
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to use the 'Agg' backend (non-interactive)


class ParameterStudy:
    """
    Class for running parameter studies on causal models.
    
    Systematically tests how varying a parameter affects causal 
    discovery and effect estimation performance.
    """
    
    def __init__(self, 
                 base_output_dir=None,
                 N=4, 
                 T=500, 
                 n_datasets=2,
                 base_auto_coeff=0.5,
                 base_cross_coeff=0.5,
                 base_noise_sigma=0.5,
                 max_lag=3,
                 pc_alpha=0.05,
                 tau_max=3,
                 n_boot=3,
                 effect_pairs=None):
        """
        Initialize parameter study with default configuration.
        
        Parameters
        ----------
        base_output_dir : str, optional
            Output directory for study results
        N : int
            Number of variables
        T : int
            Time series length
        n_datasets : int
            Number of datasets per parameter value
        base_auto_coeff : float
            Default autocorrelation coefficient
        base_cross_coeff : float
            Default cross-link coefficient
        base_noise_sigma : float
            Default noise standard deviation
        max_lag : int
            Maximum time lag in SCM
        pc_alpha : float
            Significance level for PCMCI
        tau_max : int
            Maximum time lag for causal discovery
        n_boot : int
            Number of bootstrap samples
        effect_pairs : list, optional
            List of (X, Y) pairs to estimate effects for
        """
        # Create timestamp for directory if not provided
        if base_output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.base_output_dir = f"parameter_study_{timestamp}"
        else:
            self.base_output_dir = base_output_dir
            
        # Create the directory
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Store configuration
        self.N = N
        self.T = T
        self.n_datasets = n_datasets
        self.base_auto_coeff = base_auto_coeff
        self.base_cross_coeff = base_cross_coeff 
        self.base_noise_sigma = base_noise_sigma
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.tau_max = tau_max
        self.n_boot = n_boot
        
        # Default effect pairs if none provided
        if effect_pairs is None:
            self.effect_pairs = [((0, -3), (3, 0))]  # X0_t-3 → X3_t
        else:
            self.effect_pairs = effect_pairs
            
        # Variable names
        self.var_names = [f'X{i+1}' for i in range(N)]
        
        # Save system info
        self._save_system_info()
    
    def _save_system_info(self):
        """Save system information to help interpret performance results."""
        system_info = {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        }
        
        with open(os.path.join(self.base_output_dir, 'system_info.json'), 'w') as f:
            json.dump(system_info, f, indent=4)
    
    def run_auto_study(self, auto_coeffs=None):
        """
        Run study varying autocorrelation coefficient.
        
        Parameters
        ----------
        auto_coeffs : list, optional
            List of autocorrelation values to test
            
        Returns
        -------
        dict
            Study results
        """
        if auto_coeffs is None:
            auto_coeffs = np.linspace(0.01,0.99,100)#[0.2, 0.5, 0.8]  # Default: test min and max values
            
        fixed_params = {
            'cross': self.base_cross_coeff, 
            'noise': self.base_noise_sigma
        }
        
        # Run the parameter study
        results = self.run_parameter_study('auto', auto_coeffs, fixed_params)
        
        # Generate trend plots
        self.create_parameter_trend_plots(results, self.base_output_dir)
        
        return results
    
    def run_cross_study(self, cross_coeffs=None):
        """
        Run study varying cross-link coefficient.
        
        Parameters
        ----------
        cross_coeffs : list, optional
            List of cross-link coefficient values to test
            
        Returns
        -------
        dict
            Study results
        """
        if cross_coeffs is None:
            cross_coeffs = [0.1, 0.5, 0.7]  # Default: test min and max values
            
        fixed_params = {
            'auto': self.base_auto_coeff, 
            'noise': self.base_noise_sigma
        }
        
        # Run the parameter study
        results = self.run_parameter_study('cross', cross_coeffs, fixed_params)
        
        # Generate trend plots
        self.create_parameter_trend_plots(results, self.base_output_dir)
        
        return results
    
    def run_noise_study(self, noise_levels=None):
        """
        Run study varying noise level.
        
        Parameters
        ----------
        noise_levels : list, optional
            List of noise standard deviation values to test
            
        Returns
        -------
        dict
            Study results
        """
        if noise_levels is None:
            noise_levels = [0.2, 1.5]  # Default: test min and max values
            
        fixed_params = {
            'auto': self.base_auto_coeff, 
            'cross': self.base_cross_coeff
        }
        
        # Run the parameter study
        results = self.run_parameter_study('noise', noise_levels, fixed_params)
        
        # Generate trend plots
        self.create_parameter_trend_plots(results, self.base_output_dir)
        
        return results
    
    
    def run_parameter_study(self, param_name, param_values, fixed_params, base_seed = None):
        """
        Run study for a specific parameter while keeping others fixed.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter to vary ('auto', 'cross', or 'noise')
        param_values : list
            List of values to test for the parameter
        fixed_params : dict
            Dictionary of fixed parameter values
            
        Returns
        -------
        dict
            Study results
        """
        study_results = {}
        
        # Create a master timing collector for this parameter study
        master_timing = TimingCollector(f"{param_name}_study")
        
        for param_value in param_values:
            print(f"\n{'='*80}")
            print(f"Running {param_name} parameter study with value: {param_value}")
            print(f"{'='*80}")
            
            # Create a timing collector for this specific parameter value
            param_timing = TimingCollector(f"{param_name}_{param_value}")
            
            # Set parameters based on what we're studying
            auto_coeff = param_value if param_name == 'auto' else fixed_params['auto']
            cross_coeff = param_value if param_name == 'cross' else fixed_params['cross']
            noise_sigma = param_value if param_name == 'noise' else fixed_params['noise']
            
            # Create parameter-specific output directory
            param_dir = f"{param_name}_{param_value}"
            output_dir = os.path.join(self.base_output_dir, param_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate links and noise distributions
            with param_timing.timer("setup"):
                links = create_stable_links(auto_coeff, cross_coeff, self.N)
                noises = create_noise_distributions(noise_sigma, self.N)
            
            # Save parameter configuration
            config = {
                'param_name': param_name,
                'param_value': param_value,
                'fixed_params': fixed_params,
                'N': self.N,
                'T': self.T,
                'n_datasets': self.n_datasets,
                'max_lag': self.max_lag,
                'pc_alpha': self.pc_alpha,
                'tau_max': self.tau_max,
                'n_boot': self.n_boot
            }
            
            with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            
            start_total = time.time()
            
            try:
                # Generate datasets
                print("Generating datasets...")
                with param_timing.timer("data_generation"):
                    
                    datasets = []
                    batch_dir = os.path.join(output_dir, f"{param_name}_{param_value}_batch")
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    # Create directories for intermediate results
                    samples_dir = os.path.join(batch_dir, "samples")
                    graphs_dir = os.path.join(batch_dir, "graphs")
                    effects_dir = os.path.join(batch_dir, "effects")
                    os.makedirs(samples_dir, exist_ok=True)
                    os.makedirs(graphs_dir, exist_ok=True)
                    os.makedirs(effects_dir, exist_ok=True)
                    
                    seeds = []
                    for i in range(self.n_datasets):
                        print(f"  Generating dataset {i+1}/{self.n_datasets}")
                        if base_seed is not None:
                            seed = base_seed + i
                        else:
                            seed = np.random.randint(0, 2**31 - 1)
                        
                        try:
                            with param_timing.timer(f"dataset_{i}_generation"):
                                data, nonstat = toys.structural_causal_process(
                                    links=links, T=self.T, seed=seed, noises=noises)
                                
                                # Skip if the data is non-stationary
                                if nonstat:
                                    print(f"  Warning: Dataset {i+1} is non-stationary, skipping")
                                    continue
                                
                                # Convert to DataFrame
                                df = pp.DataFrame(data, var_names=self.var_names)
                                datasets.append(df)
                                
                                # Save dataset
                                filename = f"{param_name}_{param_value}_dataset_{i}.npy"
                                np.save(os.path.join(samples_dir, filename), data)
                                
                                seeds.append(seed)
                        except Exception as e:
                            print(f"  Error generating dataset {i+1}: {e}")
                            traceback.print_exc()
                
                # Save metadata
                metadata = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "param_name": param_name,
                    "param_value": param_value,
                    "n_datasets": len(datasets),
                    "T": self.T,
                    "var_names": self.var_names,
                    "seeds": seeds,
                }
                
                with open(os.path.join(batch_dir, f"{param_name}_{param_value}_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                
                # Run causal discovery
                print("\nRunning causal discovery...")
                with param_timing.timer("causal_discovery"):
                    
                    discovery_results = {}
                    
                    # Initialize TimedCausalDiscovery with the parameter timing collector
                    tcd = TimedCausalDiscovery(param_timing)
                    
                    for i, df in enumerate(datasets):
                        print(f"  Running discovery on dataset {i+1}/{len(datasets)}")
                        
                        try:
                            # Run full discovery
                            disc_results = tcd.discover_full(
                                df,
                                pc_alpha=self.pc_alpha,
                                tau_max=self.tau_max,
                                n_boot=self.n_boot,
                                cond_ind_test=ParCorr(),
                                prefix=f"dataset_{i}_"
                            )
                            
                            # Save graphs
                            np.save(os.path.join(graphs_dir, f"pcmci_graph_dataset_{i}.npy"), 
                                   disc_results['pcmci']['graph'])
                            np.save(os.path.join(graphs_dir, f"bagged_graph_dataset_{i}.npy"), 
                                   disc_results['bagged_graph'])
                            
                            # Store results
                            discovery_results[i] = disc_results
                            
                        except Exception as e:
                            print(f"  Error in discovery for dataset {i+1}: {e}")
                            traceback.print_exc()
                
                # Estimate causal effects
                print("\nEstimating causal effects...")
                with param_timing.timer("effect_estimation"):
                    
                    effect_results = {}
                    
                    # Initialize TimedCausalEffects with the parameter timing collector
                    tce = TimedCausalEffects(param_timing)
                    
                    for i, df in enumerate(datasets):
                        if i not in discovery_results:
                            print(f"  Skipping dataset {i+1} - no discovery results available")
                            continue
                            
                        print(f"  Estimating effects for dataset {i+1}/{len(datasets)}")
                        dataset_effects = {}
                        
                        for j, (X, Y) in enumerate(self.effect_pairs):
                            print(f"    Effect pair {j+1}/{len(self.effect_pairs)}: "
                                 f"X{X[0]+1}(t{X[1]}) → X{Y[0]+1}(t{Y[1]})")
                            
                            try:
                                # Estimate all effects
                                effect_data = tce.estimate_all_effects(
                                    discovery_results[i],
                                    df,
                                    X, Y,
                                    links=links,
                                    noises=noises,
                                    prefix=f"dataset_{i}_pair_{j}_"
                                )
                                
                                # Print effect estimates
                                print(f"      Ground truth: {effect_data['true_effect']:.4f}")
                                print(f"      PCMCI estimate: {effect_data['pcmci_effect']:.4f}")
                                print(f"      Bagged estimate: {effect_data['bagged_effect']:.4f}")
                                print(f"      Bootstrap estimate: {effect_data['bootstrap_stats']['mean']:.4f} "
                                     f"(95% CI: [{effect_data['bootstrap_stats']['ci_lower']:.4f}, "
                                     f"{effect_data['bootstrap_stats']['ci_upper']:.4f}])")
                                
                                # Store results
                                effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
                                dataset_effects[effect_key] = effect_data
                                
                            except Exception as e:
                                print(f"      Error in effect estimation: {e}")
                                traceback.print_exc()
                        
                        effect_results[i] = dataset_effects
                        
                        # Save individual dataset results
                        effects_file = os.path.join(effects_dir, f"{param_name}_{param_value}_effects_dataset_{i}.json")
                        
                        # Create a serializable version of the results
                        serializable_effects = {}
                        for effect_key, effect_data in dataset_effects.items():
                            serializable_effects[effect_key] = numpy_to_python({
                                'true_effect': effect_data['true_effect'],
                                'pcmci_effect': effect_data['pcmci_effect'] if not np.isnan(effect_data['pcmci_effect']) else None,
                                'bagged_effect': effect_data['bagged_effect'] if not np.isnan(effect_data['bagged_effect']) else None,
                                'bootstrap_stats': effect_data['bootstrap_stats']
                            })
                        
                        with open(effects_file, 'w') as f:
                            json.dump(serializable_effects, f, indent=4)
                
                total_time = time.time() - start_total
                param_timing.timings['total_execution'] = [total_time]
                
                # Save timing information
                param_timing.save(os.path.join(output_dir, 'timing.json'))
                
                # Update master timing with this parameter's results
                for operation, times in param_timing.timings.items():
                    master_key = f"{param_name}_{param_value}_{operation}"
                    master_timing.timings[master_key] = times
                
                # Calculate performance metrics
                try:
                    print("\nCalculating performance metrics...")
                    # Use the enhanced evaluation function
                    metrics = extended_evaluate_estimation_performance(effect_results)
                    
                    # Save metrics
                    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                        # Convert numpy values to native Python types for JSON serialization
                        serializable_metrics = {}
                        for effect_key, methods in metrics['metrics_by_effect'].items():
                            serializable_metrics[effect_key] = {}
                            for method, metric_values in methods.items():
                                serializable_metrics[effect_key][method] = {
                                    k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                                    for k, v in metric_values.items()
                                }
                        
                        json.dump(serializable_metrics, f, indent=4)
                    
                    # Save overall metrics
                    with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
                        # Convert numpy values to native Python types
                        serializable_overall = {}
                        for method, method_metrics in metrics['overall_metrics'].items():
                            serializable_overall[method] = {
                                k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                                for k, v in method_metrics.items()
                            }
                        
                        json.dump(serializable_overall, f, indent=4)
                    
                    study_results[param_value] = {
                        'metrics': metrics,
                        'batch_dir': batch_dir,
                        'timing': param_timing.get_summary()
                    }
                    
                    # Generate and save summary figures
                    for i, (X, Y) in enumerate(self.effect_pairs):
                        effect_key = f"{X[0]}_{X[1]}_to_{Y[0]}_{Y[1]}"
                        if effect_key in metrics['metrics_by_effect']:
                            X_label = f"X{X[0]+1}(t{X[1]})"
                            Y_label = f"X{Y[0]+1}(t{Y[1]})"
                            title = f"Effect from {X_label} to {Y_label} - {param_name}={param_value}"
                            
                            # Get metrics
                            pcmci_mae = metrics['metrics_by_effect'][effect_key]['pcmci'].get('mae', np.nan)
                            bagged_mae = metrics['metrics_by_effect'][effect_key]['bagged'].get('mae', np.nan)
                            bootstrap_mae = metrics['metrics_by_effect'][effect_key]['bootstrap'].get('mae', np.nan)
                            
                            # Print summary
                            print(f"\nEffect {X_label} → {Y_label}:")
                            print(f"  PCMCI MAE: {pcmci_mae:.4f}")
                            print(f"  Bagged MAE: {bagged_mae:.4f}")
                            print(f"  Bootstrap MAE: {bootstrap_mae:.4f}")
                            
                            # Print success rates
                            pcmci_success = metrics['metrics_by_effect'][effect_key]['pcmci'].get('success_rate', np.nan)
                            bagged_success = metrics['metrics_by_effect'][effect_key]['bagged'].get('success_rate', np.nan)
                            bootstrap_success = metrics['metrics_by_effect'][effect_key]['bootstrap'].get('success_rate', np.nan)
                            
                            print(f"  PCMCI Success Rate: {pcmci_success:.2f}")
                            print(f"  Bagged Success Rate: {bagged_success:.2f}")
                            print(f"  Bootstrap Success Rate: {bootstrap_success:.2f}")
                            
                            # Report multimodality if detected
                            if 'multimodal_rate' in metrics['metrics_by_effect'][effect_key]['bootstrap']:
                                multimodal_rate = metrics['metrics_by_effect'][effect_key]['bootstrap']['multimodal_rate']
                                print(f"  Bootstrap Multimodal Rate: {multimodal_rate:.2f}")
                            
                            try:
                                # Plot comparison if we have effects
                                fig = plot_effect_comparison(effect_results, effect_key, method='bootstrap')
                                fig.suptitle(title)
                                fig.savefig(os.path.join(output_dir, f"effect_{effect_key}.png"))
                                plt.close(fig)
                                
                                # Create bootstrap distribution visualization directory
                                dist_dir = os.path.join(output_dir, 'bootstrap_distributions')
                                os.makedirs(dist_dir, exist_ok=True)
                                
                                # Find a dataset with bootstrap effects
                                for dataset_id, dataset_effects in effect_results.items():
                                    if effect_key in dataset_effects:
                                        effect_data = dataset_effects[effect_key]
                                        bootstrap_effects = effect_data.get('bootstrap_effects', [])
                                        
                                        if bootstrap_effects:
                                            # Create bootstrap distribution plot
                                            other_estimates = {
                                                'pcmci': effect_data.get('pcmci_effect'),
                                                'bagged': effect_data.get('bagged_effect')
                                            }
                                            
                                            fig = plot_bootstrap_distribution(
                                                bootstrap_effects,
                                                true_effect=effect_data.get('true_effect'),
                                                other_estimates=other_estimates,
                                                title=f"Bootstrap Distribution - {title}",
                                                output_file=os.path.join(dist_dir, f"bootstrap_dist_{effect_key}_{dataset_id}.png")
                                            )
                                            
                                            if fig:
                                                plt.close(fig)
                                
                            except Exception as e:
                                print(f"  Error creating effect visualization: {e}")
                    
                    # Generate additional visualization: Success rates comparison
                    try:
                        # Create success rates bar plot
                        plt.figure(figsize=(10, 6))
                        methods = ['pcmci', 'bagged', 'bootstrap']
                        success_rates = [metrics['overall_metrics'][m].get('success_rate', 0) for m in methods]
                        
                        bars = plt.bar(methods, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                        
                        # Add value labels
                        for bar, rate in zip(bars, success_rates):
                            plt.text(bar.get_x() + bar.get_width()/2, 
                                    bar.get_height() + 0.02, 
                                    f'{rate:.2f}', 
                                    ha='center', va='bottom')
                        
                        plt.ylim(0, 1.1)
                        plt.xlabel('Method')
                        plt.ylabel('Success Rate')
                        plt.title(f'Causal Effect Estimation Success Rate - {param_name}={param_value}')
                        plt.savefig(os.path.join(output_dir, f"success_rates_{param_name}_{param_value}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"  Error creating success rates visualization: {e}")
                
                except Exception as e:
                    print(f"Error in performance metrics calculation: {e}")
                    traceback.print_exc()
            
            except Exception as e:
                print(f"Error in parameter study for {param_name}={param_value}: {e}")
                traceback.print_exc()
                
                # Save error information
                with open(os.path.join(output_dir, 'error_log.txt'), 'w') as f:
                    f.write(f"Error in parameter study: {str(e)}\n\n")
                    f.write(traceback.format_exc())
        
        # Save master timing data
        master_timing.save(os.path.join(self.base_output_dir, f"{param_name}_study_timing.json"))
        
        # Create comparison plots for this parameter study
        try:
            # Extract parameter-specific timings for key operations
            param_data = []
            
            for param_value in param_values:
                data_gen_key = f"{param_name}_{param_value}_data_generation"
                discovery_key = f"{param_name}_{param_value}_causal_discovery"
                effect_key = f"{param_name}_{param_value}_effect_estimation"
                total_key = f"{param_name}_{param_value}_total_execution"
                
                if data_gen_key in master_timing.timings:
                    param_data.append({
                        'Parameter Value': param_value,
                        'Operation': 'Data Generation',
                        'Time (s)': np.mean(master_timing.timings[data_gen_key])
                    })
                
                if discovery_key in master_timing.timings:
                    param_data.append({
                        'Parameter Value': param_value,
                        'Operation': 'Causal Discovery',
                        'Time (s)': np.mean(master_timing.timings[discovery_key])
                    })
                
                if effect_key in master_timing.timings:
                    param_data.append({
                        'Parameter Value': param_value,
                        'Operation': 'Effect Estimation',
                        'Time (s)': np.mean(master_timing.timings[effect_key])
                    })
                
                if total_key in master_timing.timings:
                    param_data.append({
                        'Parameter Value': param_value,
                        'Operation': 'Total Execution',
                        'Time (s)': np.mean(master_timing.timings[total_key])
                    })
            
            if param_data:
                param_df = pd.DataFrame(param_data)
                
                # Create timing comparison plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Parameter Value', y='Time (s)', hue='Operation', data=param_df)
                plt.title(f'Timing Comparison for {param_name.capitalize()}')
                plt.savefig(os.path.join(self.base_output_dir, f"{param_name}_timing_comparison.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating parameter timing plots: {e}")
        
        return study_results
    
    def run_all_studies(self, auto_coeffs=None, cross_coeffs=None, noise_levels=None):
        """
        Run all parameter studies.
        
        Parameters
        ----------
        auto_coeffs : list, optional
            List of autocorrelation values to test
        cross_coeffs : list, optional
            List of cross-link coefficient values to test
        noise_levels : list, optional
            List of noise standard deviation values to test
            
        Returns
        -------
        dict
            All study results
        """
        master_study_timing = TimingCollector("overall_study")
        all_results = {}
        overall_timing = {}
        
        # Run autocorrelation study
        print("\n\nSTARTING AUTOCORRELATION STUDY")
        print("==============================")
        master_study_timing.start_timer("auto_study")
        try:
            all_results['auto'] = self.run_auto_study(auto_coeffs)
            overall_timing['auto'] = master_study_timing.stop_timer("auto_study")
            print(f"Autocorrelation study completed in {overall_timing['auto']:.2f} seconds")
        except Exception as e:
            print(f"Error in autocorrelation study: {e}")
            traceback.print_exc()
            overall_timing['auto'] = master_study_timing.stop_timer("auto_study")
            print(f"Autocorrelation study failed after {overall_timing['auto']:.2f} seconds")
        
        # Run cross-link study
        print("\n\nSTARTING CROSS-LINK COEFFICIENT STUDY")
        print("====================================")
        master_study_timing.start_timer("cross_study")
        try:
            all_results['cross'] = self.run_cross_study(cross_coeffs)
            overall_timing['cross'] = master_study_timing.stop_timer("cross_study")
            print(f"Cross-link coefficient study completed in {overall_timing['cross']:.2f} seconds")
        except Exception as e:
            print(f"Error in cross-link study: {e}")
            traceback.print_exc()
            overall_timing['cross'] = master_study_timing.stop_timer("cross_study")
            print(f"Cross-link study failed after {overall_timing['cross']:.2f} seconds")
        
        # Run noise level study
        print("\n\nSTARTING NOISE LEVEL STUDY")
        print("=========================")
        master_study_timing.start_timer("noise_study")
        try:
            all_results['noise'] = self.run_noise_study(noise_levels)
            overall_timing['noise'] = master_study_timing.stop_timer("noise_study")
            print(f"Noise level study completed in {overall_timing['noise']:.2f} seconds")
        except Exception as e:
            print(f"Error in noise study: {e}")
            traceback.print_exc()
            overall_timing['noise'] = master_study_timing.stop_timer("noise_study")
            print(f"Noise study failed after {overall_timing['noise']:.2f} seconds")
        
        # Save overall timing information
        with open(os.path.join(self.base_output_dir, 'overall_timing.json'), 'w') as f:
            json.dump(overall_timing, f, indent=4)
        
        # Save master study timing
        master_study_timing.save(os.path.join(self.base_output_dir, 'master_study_timing.json'))
        
        # Create an overall timing comparison visualization
        active_studies = list(overall_timing.keys())
        if active_studies:
            plt.figure(figsize=(10, 6))
            plt.bar(active_studies, [overall_timing[study] for study in active_studies])
            plt.title('Overall Study Execution Times')
            plt.ylabel('Time (s)')
            plt.savefig(os.path.join(self.base_output_dir, 'overall_timing_comparison.png'))
            plt.close()
        
        # Create summary plots
        self._create_summary_plots(all_results)
        
        return all_results
    
    def _create_summary_plots(self, all_results):
        
        
        """
        Create summary plots for all studies.
        
        Parameters
        ----------
        all_results : dict
            Dictionary of all study results
        """
        try:
            # Determine which parameter studies were actually run
            active_studies = [param for param in all_results if all_results[param]]
            
            if not active_studies:
                print("No parameter studies to create summary plots for.")
                return
            
            # Create a simple summary table as CSV
            summary_data = []
            
            for param_name in active_studies:
                param_results = all_results[param_name]
                for param_value, results in param_results.items():
                    if 'metrics' in results:
                        for effect_key, methods in results['metrics'].items():
                            for method_name, metrics in methods.items():
                                if 'mae' in metrics:
                                    row = {
                                        'Parameter': param_name,
                                        'Value': param_value,
                                        'Effect': effect_key,
                                        'Method': method_name,
                                        'MAE': metrics.get('mae', np.nan),
                                        'RMSE': metrics.get('rmse', np.nan),
                                        'Bias': metrics.get('bias', np.nan)
                                    }
                                    if method_name == 'bootstrap' and 'ci_coverage' in metrics:
                                        row['CI_Coverage'] = metrics['ci_coverage']
                                    
                                    summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(os.path.join(self.base_output_dir, "summary_results.csv"), index=False)
                print(f"Summary data saved to {os.path.join(self.base_output_dir, 'summary_results.csv')}")
                
                # Create a visualization of the MAE across parameters and methods
                plt.figure(figsize=(12, 8))
                pivot_df = df.pivot_table(
                    index=['Parameter', 'Value'], 
                    columns='Method', 
                    values='MAE'
                )
                
                # Plot as heatmap if enough data
                if len(pivot_df) > 1:
                    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
                    plt.title('MAE Comparison Across Parameters and Methods')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.base_output_dir, 'mae_comparison.png'))
        
        except Exception as e:
            print(f"Error creating summary plots: {e}")
            traceback.print_exc()
            
            
    def create_parameter_trend_plots(self, results, output_dir):
        """
        Create trend plots showing how metrics vary with parameter values.
        
        Parameters
        ----------
        results : dict
            Results from parameter studies
        output_dir : str
            Directory to save plots
        """
        # Create output directory
        trends_dir = os.path.join(output_dir, 'trends')
        os.makedirs(trends_dir, exist_ok=True)
        
        # Extract parameter values and metrics
        param_values = sorted(results.keys())
        metrics_to_plot = ['mae', 'rmse', 'success_rate', 'multimodal_rate']
        methods = ['pcmci', 'bagged', 'bootstrap']
        
        # Effect pairs from the first parameter value
        if not param_values or 'metrics' not in results[param_values[0]]:
            print("No metrics available for trend plots")
            return
        
        first_metrics = results[param_values[0]]['metrics']
        if 'metrics_by_effect' not in first_metrics:
            effect_keys = list(first_metrics.keys())
        else:
            effect_keys = list(first_metrics['metrics_by_effect'].keys())
        
        # For each effect key, create trend plots
        for effect_key in effect_keys:
            # For each metric, create a plot
            for metric in metrics_to_plot:
                plt.figure(figsize=(12, 6))
                
                # Data for plot
                for method in methods:
                    method_values = []
                    method_param_values = []
                    
                    # Collect values for each parameter
                    for param_value in param_values:
                        if param_value not in results or 'metrics' not in results[param_value]:
                            continue
                        
                        # Get metrics depending on format
                        metrics = results[param_value]['metrics']
                        
                        if 'metrics_by_effect' in metrics:
                            # New format
                            if effect_key in metrics['metrics_by_effect'] and method in metrics['metrics_by_effect'][effect_key]:
                                method_metrics = metrics['metrics_by_effect'][effect_key][method]
                                
                                if metric in method_metrics and not np.isnan(method_metrics[metric]):
                                    method_values.append(method_metrics[metric])
                                    method_param_values.append(param_value)
                        else:
                            # Old format
                            if effect_key in metrics and method in metrics[effect_key]:
                                method_metrics = metrics[effect_key][method]
                                
                                if metric in method_metrics and not np.isnan(method_metrics[metric]):
                                    method_values.append(method_metrics[metric])
                                    method_param_values.append(param_value)
                    
                    # Plot if we have data
                    if method_values:
                        plt.plot(method_param_values, method_values, 
                               marker='o', linestyle='-', label=method)
                
                # Set labels and title
                plt.xlabel('Parameter Value')
                plt.ylabel(metric.upper())
                plt.title(f'{metric.upper()} vs Parameter - {effect_key}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save plot
                plt.savefig(os.path.join(trends_dir, f"{metric}_{effect_key}_trend.png"))
                plt.close()
        
        # Create overall trend plots
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            # Data for plot
            for method in methods:
                method_values = []
                method_param_values = []
                
                # Collect values for each parameter
                for param_value in param_values:
                    if param_value not in results or 'metrics' not in results[param_value]:
                        continue
                    
                    metrics = results[param_value]['metrics']
                    
                    if 'overall_metrics' in metrics and method in metrics['overall_metrics']:
                        method_metrics = metrics['overall_metrics'][method]
                        
                        if metric in method_metrics and not np.isnan(method_metrics[metric]):
                            method_values.append(method_metrics[metric])
                            method_param_values.append(param_value)
                
                # Plot if we have data
                if method_values:
                    plt.plot(method_param_values, method_values, 
                           marker='o', linestyle='-', label=method)
            
            # Set labels and title
            plt.xlabel('Parameter Value')
            plt.ylabel(metric.upper())
            plt.title(f'Overall {metric.upper()} vs Parameter')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(trends_dir, f"overall_{metric}_trend.png"))
            plt.close()

def run_study_from_config(config_file=None):
    """
    Run parameter study from configuration file.
    
    Parameters
    ----------
    config_file : str, optional
        Path to JSON configuration file
        
    Returns
    -------
    dict
        Study results
    """
    # Default configuration
    config = {
        'output_dir': None,
        'N': 4,
        'T': 500,
        'n_datasets': 2,
        'base_auto_coeff': 0.95,
        'base_cross_coeff': 0.5,
        'base_noise_sigma': 0.5,
        'max_lag': 3,
        'pc_alpha': 0.05,
        'tau_max': 3,
        'n_boot': 50,
        'auto_coeffs': [0.2, 0.95],
        'cross_coeffs': [0.1, 0.7],
        'noise_levels': [0.2, 1.5],
        'parameter_to_study': 'auto'  # 'auto', 'cross', 'noise', or 'all'
    }
    
    # Load configuration from file if provided
    if config_file is not None:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Create parameter study
    study = ParameterStudy(
        base_output_dir=config['output_dir'],
        N=config['N'],
        T=config['T'],
        n_datasets=config['n_datasets'],
        base_auto_coeff=config['base_auto_coeff'],
        base_cross_coeff=config['base_cross_coeff'],
        base_noise_sigma=config['base_noise_sigma'],
        max_lag=config['max_lag'],
        pc_alpha=config['pc_alpha'],
        tau_max=config['tau_max'],
        n_boot=config['n_boot']
    )
    
    # Run study based on parameter_to_study
    if config['parameter_to_study'] == 'auto':
        results = study.run_auto_study(config['auto_coeffs'])
    elif config['parameter_to_study'] == 'cross':
        results = study.run_cross_study(config['cross_coeffs'])
    elif config['parameter_to_study'] == 'noise':
        results = study.run_noise_study(config['noise_levels'])
    elif config['parameter_to_study'] == 'all':
        results = study.run_all_studies(
            config['auto_coeffs'], 
            config['cross_coeffs'], 
            config['noise_levels']
        )
    else:
        raise ValueError(f"Unknown parameter_to_study: {config['parameter_to_study']}")
    
    return results

def numpy_to_python(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(v) for v in obj)
    else:
        return obj
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter studies for causal discovery and effect estimation.')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--param', type=str, choices=['auto', 'cross', 'noise', 'all'], 
                        default='all', help='Parameter to study')
    parser.add_argument('--fine-grained', action='store_true', 
                        help='Run with fine-grained parameter values for smoother trends')
    
    args = parser.parse_args()
    
    if args.config:
        results = run_study_from_config(args.config)
    else:
        # Use default configuration but override parameter_to_study
        config = {
            'parameter_to_study': args.param
        }
        study = ParameterStudy()
        
        # If fine-grained mode is requested, use more parameter values
        if args.fine_grained:
            auto_coeffs = np.linspace(0.5, 0.98, 20)  # Focus on higher autocorrelation
            cross_coeffs = np.linspace(0.1, 0.8, 15)  # Wide range of cross-effects
            noise_levels = np.linspace(0.1, 2.0, 15)  # Wide range of noise levels
        else:
            auto_coeffs = None  # Use defaults
            cross_coeffs = None
            noise_levels = None
        
        if args.param == 'auto':
            results = study.run_auto_study(auto_coeffs)
        elif args.param == 'cross':
            results = study.run_cross_study(cross_coeffs)
        elif args.param == 'noise':
            results = study.run_noise_study(noise_levels)
        elif args.param == 'all':
            results = study.run_all_studies(auto_coeffs, cross_coeffs, noise_levels)
        
    print(f"Parameter study complete. Results saved in {study.base_output_dir}")
