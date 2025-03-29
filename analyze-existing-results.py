#!/usr/bin/env python
"""
Script to analyze the results of existing parameter studies.

This script can be used to generate additional visualizations
and analysis from parameter studies that have already been run,
without needing to re-run the studies themselves.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
from collections import defaultdict

# Import the enhanced evaluation functions
# Make sure causal_utils.py has been updated with the new functions
from causal_utils import (
    analyze_bootstrap_distribution, 
    extended_evaluate_estimation_performance,
    plot_bootstrap_distribution
)

def load_parameter_study_results(study_dir):
    """
    Load results from a parameter study directory.
    
    Parameters
    ----------
    study_dir : str
        Directory containing parameter study results
        
    Returns
    -------
    dict
        Dictionary of results by parameter value
    """
    results = {}
    
    # Find parameter directories
    param_dirs = []
    for item in os.listdir(study_dir):
        item_path = os.path.join(study_dir, item)
        if os.path.isdir(item_path):
            # Parameter directories typically contain a config.txt file
            if os.path.exists(os.path.join(item_path, 'config.txt')):
                param_dirs.append(item_path)
    
    # Extract parameter type and value from directory names
    for param_dir in param_dirs:
        dir_name = os.path.basename(param_dir)
        parts = dir_name.split('_')
        
        if len(parts) >= 2:
            param_type = parts[0]
            try:
                param_value = float(parts[1])
            except ValueError:
                param_value = parts[1]
            
            # Load metrics if available
            metrics_file = os.path.join(param_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            
            # Find batch directory
            batch_dir = None
            for item in os.listdir(param_dir):
                if item.endswith('_batch') and os.path.isdir(os.path.join(param_dir, item)):
                    batch_dir = os.path.join(param_dir, item)
                    break
            
            # Store results
            results[param_value] = {
                'param_type': param_type,
                'param_value': param_value,
                'metrics': metrics,
                'batch_dir': batch_dir
            }
    
    return results

def load_effects_from_batch_dir(batch_dir):
    """
    Load effect data from a batch directory.
    
    Parameters
    ----------
    batch_dir : str
        Batch directory containing effect data
        
    Returns
    -------
    dict
        Dictionary of effect data by dataset and effect pair
    """
    effects_dir = os.path.join(batch_dir, 'effects')
    if not os.path.exists(effects_dir):
        return {}
    
    all_effects = {}
    
    # Load effect files
    for filename in os.listdir(effects_dir):
        if filename.endswith('.json') and 'effects_dataset' in filename:
            # Extract dataset ID
            try:
                dataset_id = int(filename.split('dataset_')[1].split('.')[0])
            except (ValueError, IndexError):
                continue
            
            # Load effects
            with open(os.path.join(effects_dir, filename), 'r') as f:
                dataset_effects = json.load(f)
            
            all_effects[dataset_id] = dataset_effects
    
    return all_effects

def reanalyze_bootstrap_distributions(results, output_dir):
    """
    Reanalyze bootstrap distributions from parameter study results.
    
    Parameters
    ----------
    results : dict
        Dictionary of results by parameter value
    output_dir : str
        Directory to save visualizations
        
    Returns
    -------
    dict
        Dictionary of analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analysis results
    analysis_results = {
        'by_parameter': {},
        'multimodal_cases': [],
        'overall': {
            'total_distributions': 0,
            'multimodal_count': 0,
            'multimodal_rate': 0
        }
    }
    
    # Process each parameter value
    for param_value, param_results in results.items():
        batch_dir = param_results.get('batch_dir')
        if not batch_dir:
            continue
        
        # Load effects from batch directory
        effect_data = load_effects_from_batch_dir(batch_dir)
        if not effect_data:
            continue
        
        # Create directory for this parameter
        param_dir = os.path.join(output_dir, f"{param_results['param_type']}_{param_value}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Create directory for bootstrap distributions
        dist_dir = os.path.join(param_dir, 'bootstrap_distributions')
        os.makedirs(dist_dir, exist_ok=True)
        
        # Initialize parameter results
        param_analysis = {
            'total_distributions': 0,
            'multimodal_count': 0,
            'effect_pairs': {}
        }
        
        # Process each effect pair
        all_effect_keys = set()
        for dataset_effects in effect_data.values():
            all_effect_keys.update(dataset_effects.keys())
        
        for effect_key in all_effect_keys:
            effect_analysis = {
                'distributions': [],
                'multimodal_count': 0
            }
            
            # Collect bootstrap effects for this effect key
            for dataset_id, dataset_effects in effect_data.items():
                if effect_key in dataset_effects:
                    effect_info = dataset_effects[effect_key]
                    bootstrap_effects = effect_info.get('bootstrap_effects', [])
                    
                    if bootstrap_effects:
                        # Analyze bootstrap distribution
                        dist_analysis = analyze_bootstrap_distribution(bootstrap_effects)
                        
                        if dist_analysis['status'] == 'analyzed':
                            effect_analysis['distributions'].append(dist_analysis)
                            param_analysis['total_distributions'] += 1
                            
                            # Count multimodal distributions
                            if dist_analysis.get('is_multimodal', False):
                                effect_analysis['multimodal_count'] += 1
                                param_analysis['multimodal_count'] += 1
                                
                                # Store as interesting case
                                analysis_results['multimodal_cases'].append({
                                    'param_type': param_results['param_type'],
                                    'param_value': param_value,
                                    'effect_key': effect_key,
                                    'dataset_id': dataset_id,
                                    'distribution': dist_analysis
                                })
                                
                                # Create visualization
                                other_estimates = {
                                    'pcmci': effect_info.get('pcmci_effect'),
                                    'bagged': effect_info.get('bagged_effect')
                                }
                                
                                # Parse effect key to create a meaningful title
                                try:
                                    parts = effect_key.split('_')
                                    X_idx, X_lag = int(parts[0]), int(parts[1])
                                    Y_idx, Y_lag = int(parts[3]), int(parts[4])
                                    title = f"Multimodal Bootstrap Distribution\nX{X_idx+1}(t{X_lag}) → X{Y_idx+1}(t{Y_lag})\n{param_results['param_type']}={param_value}"
                                except:
                                    title = f"Multimodal Bootstrap Distribution - {effect_key}\n{param_results['param_type']}={param_value}"
                                
                                plot_bootstrap_distribution(
                                    bootstrap_effects,
                                    true_effect=effect_info.get('true_effect'),
                                    other_estimates=other_estimates,
                                    title=title,
                                    output_file=os.path.join(dist_dir, f"multimodal_{effect_key}_dataset_{dataset_id}.png")
                                )
            
            # Calculate multimodal rate for this effect pair
            if effect_analysis['distributions']:
                effect_analysis['multimodal_rate'] = effect_analysis['multimodal_count'] / len(effect_analysis['distributions'])
            
            # Store effect analysis
            param_analysis['effect_pairs'][effect_key] = effect_analysis
        
        # Calculate multimodal rate for this parameter
        if param_analysis['total_distributions'] > 0:
            param_analysis['multimodal_rate'] = param_analysis['multimodal_count'] / param_analysis['total_distributions']
        
        # Store parameter analysis
        analysis_results['by_parameter'][f"{param_results['param_type']}_{param_value}"] = param_analysis
        
        # Update overall counts
        analysis_results['overall']['total_distributions'] += param_analysis['total_distributions']
        analysis_results['overall']['multimodal_count'] += param_analysis['multimodal_count']
    
    # Calculate overall multimodal rate
    if analysis_results['overall']['total_distributions'] > 0:
        analysis_results['overall']['multimodal_rate'] = (
            analysis_results['overall']['multimodal_count'] / 
            analysis_results['overall']['total_distributions']
        )
    
    # Create summary visualizations
    create_summary_visualizations(analysis_results, output_dir)
    
    # Save analysis results
    with open(os.path.join(output_dir, 'bootstrap_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    return analysis_results

def create_trend_plots(results, output_dir):
    """
    Create trend plots showing how metrics vary with parameter values.
    
    Parameters
    ----------
    results : dict
        Dictionary of results by parameter value
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    trends_dir = os.path.join(output_dir, 'trends')
    os.makedirs(trends_dir, exist_ok=True)
    
    # Determine parameter type from the first result
    if not results:
        return
    
    first_result = next(iter(results.values()))
    param_type = first_result.get('param_type', 'unknown')
    
    # Extract parameter values
    param_values = sorted(results.keys())
    
    # Check if there are enough parameter values for trend plots
    if len(param_values) < 3:
        print(f"Not enough parameter values for trend plots: {len(param_values)}")
        return
    
    # Get metrics from the first result to determine effect pairs
    if 'metrics' not in first_result:
        print("No metrics available for trend plots")
        return
    
    # Determine effect keys from metrics
    metrics = first_result['metrics']
    effect_keys = list(metrics.keys())
    
    # Metrics to plot
    metrics_to_plot = ['mae', 'rmse', 'bias', 'ci_coverage', 'success_rate', 'multimodal_rate']
    methods = ['pcmci', 'bagged', 'bootstrap']
    
    # For each effect key, create trend plots
    for effect_key in effect_keys:
        # For each metric, create a plot
        for metric in metrics_to_plot:
            # Data for plot
            metric_data = []
            
            for method in methods:
                method_values = []
                method_param_values = []
                
                # Collect values for each parameter
                for param_value in param_values:
                    if 'metrics' not in results[param_value]:
                        continue
                    
                    metrics = results[param_value]['metrics']
                    
                    if effect_key in metrics and method in metrics[effect_key]:
                        method_metrics = metrics[effect_key][method]
                        
                        if metric in method_metrics and not np.isnan(method_metrics[metric]):
                            method_values.append(method_metrics[metric])
                            method_param_values.append(param_value)
                
                # Skip if no data
                if not method_values:
                    continue
                
                # Add to plot data
                for x, y in zip(method_param_values, method_values):
                    metric_data.append({
                        'Parameter Value': x,
                        'Metric Value': y,
                        'Method': method
                    })
            
            # Skip if no data
            if not metric_data:
                continue
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            df = pd.DataFrame(metric_data)
            
            # Use seaborn for better visualization
            sns.lineplot(data=df, x='Parameter Value', y='Metric Value', hue='Method', marker='o')
            
            # Set labels and title
            plt.xlabel(f'{param_type.capitalize()} Value')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} vs {param_type.capitalize()} - {effect_key}')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(trends_dir, f"{metric}_{effect_key}_trend.png"))
            plt.close()

def create_summary_visualizations(analysis_results, output_dir):
    """
    Create summary visualizations for bootstrap distribution analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Results from bootstrap distribution analysis
    output_dir : str
        Directory to save visualizations
    """
    # Create multimodality rate by parameter plot
    param_data = []
    
    for param_key, param_analysis in analysis_results['by_parameter'].items():
        if param_analysis['total_distributions'] > 0:
            param_data.append({
                'Parameter': param_key,
                'Multimodal Rate': param_analysis['multimodal_rate'],
                'Multimodal Count': param_analysis['multimodal_count'],
                'Total Distributions': param_analysis['total_distributions']
            })
    
    if not param_data:
        return
    
    # Sort by parameter key
    param_data = sorted(param_data, key=lambda x: x['Parameter'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot multimodal rate
    bars = plt.bar([p['Parameter'] for p in param_data], 
                  [p['Multimodal Rate'] for p in param_data],
                  color='purple', alpha=0.7)
    
    # Add value labels
    for bar, p in zip(bars, param_data):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02, 
                f'{p["Multimodal Rate"]:.2f}', 
                ha='center', va='bottom')
    
    plt.ylim(0, max(1.0, max(p['Multimodal Rate'] for p in param_data) + 0.1))
    plt.xlabel('Parameter')
    plt.ylabel('Multimodal Rate')
    plt.title('Bootstrap Distribution Multimodality Rate by Parameter')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'multimodality_rate_by_parameter.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed table of multimodal cases
    if analysis_results['multimodal_cases']:
        multimodal_df = pd.DataFrame([
            {
                'Parameter Type': case['param_type'],
                'Parameter Value': case['param_value'],
                'Effect Pair': case['effect_key'],
                'Dataset ID': case['dataset_id'],
                'Number of Modes': case['distribution'].get('n_modes', 0),
                'Success Rate': case['distribution'].get('success_rate', 0)
            }
            for case in analysis_results['multimodal_cases']
        ])
        
        multimodal_df.to_csv(os.path.join(output_dir, 'multimodal_cases.csv'), index=False)

def analyze_estimation_success(results, output_dir):
    """
    Analyze success rates of causal effect estimation.
    
    Parameters
    ----------
    results : dict
        Dictionary of results by parameter value
    output_dir : str
        Directory to save visualizations
        
    Returns
    -------
    dict
        Dictionary of success rate analysis by parameter
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analysis results
    success_analysis = {
        'by_parameter': {},
        'overall': {
            'pcmci': {'attempts': 0, 'successes': 0},
            'bagged': {'attempts': 0, 'successes': 0},
            'bootstrap': {'attempts': 0, 'successes': 0}
        }
    }
    
    # Determine parameter type from the first result
    if not results:
        return success_analysis
    
    first_result = next(iter(results.values()))
    param_type = first_result.get('param_type', 'unknown')
    
    # Process each parameter value
    for param_value, param_results in results.items():
        batch_dir = param_results.get('batch_dir')
        if not batch_dir:
            continue
        
        # Load effects from batch directory
        effect_data = load_effects_from_batch_dir(batch_dir)
        if not effect_data:
            continue
        
        # Initialize parameter results
        param_success = {
            'pcmci': {'attempts': 0, 'successes': 0},
            'bagged': {'attempts': 0, 'successes': 0},
            'bootstrap': {'attempts': 0, 'successes': 0},
            'effect_pairs': {}
        }
        
        # Process each effect pair
        all_effect_keys = set()
        for dataset_effects in effect_data.values():
            all_effect_keys.update(dataset_effects.keys())
        
        for effect_key in all_effect_keys:
            effect_success = {
                'pcmci': {'attempts': 0, 'successes': 0},
                'bagged': {'attempts': 0, 'successes': 0},
                'bootstrap': {'attempts': 0, 'successes': 0}
            }
            
            # Count successes and attempts for each method
            for dataset_id, dataset_effects in effect_data.items():
                if effect_key in dataset_effects:
                    effect_info = dataset_effects[effect_key]
                    
                    # PCMCI
                    effect_success['pcmci']['attempts'] += 1
                    param_success['pcmci']['attempts'] += 1
                    success_analysis['overall']['pcmci']['attempts'] += 1
                    
                    if 'pcmci_effect' in effect_info and not np.isnan(effect_info['pcmci_effect']):
                        effect_success['pcmci']['successes'] += 1
                        param_success['pcmci']['successes'] += 1
                        success_analysis['overall']['pcmci']['successes'] += 1
                    
                    # Bagged
                    effect_success['bagged']['attempts'] += 1
                    param_success['bagged']['attempts'] += 1
                    success_analysis['overall']['bagged']['attempts'] += 1
                    
                    if 'bagged_effect' in effect_info and not np.isnan(effect_info['bagged_effect']):
                        effect_success['bagged']['successes'] += 1
                        param_success['bagged']['successes'] += 1
                        success_analysis['overall']['bagged']['successes'] += 1
                    
                    # Bootstrap
                    effect_success['bootstrap']['attempts'] += 1
                    param_success['bootstrap']['attempts'] += 1
                    success_analysis['overall']['bootstrap']['attempts'] += 1
                    
                    bootstrap_stats = effect_info.get('bootstrap_stats', {})
                    bootstrap_mean = bootstrap_stats.get('mean')
                    
                    if bootstrap_mean is not None and not np.isnan(bootstrap_mean):
                        effect_success['bootstrap']['successes'] += 1
                        param_success['bootstrap']['successes'] += 1
                        success_analysis['overall']['bootstrap']['successes'] += 1
            
            # Calculate success rates for this effect pair
            for method in ['pcmci', 'bagged', 'bootstrap']:
                attempts = effect_success[method]['attempts']
                if attempts > 0:
                    effect_success[method]['success_rate'] = effect_success[method]['successes'] / attempts
            
            # Store effect success
            param_success['effect_pairs'][effect_key] = effect_success
        
        # Calculate success rates for this parameter
        for method in ['pcmci', 'bagged', 'bootstrap']:
            attempts = param_success[method]['attempts']
            if attempts > 0:
                param_success[method]['success_rate'] = param_success[method]['successes'] / attempts
        
        # Store parameter success
        success_analysis['by_parameter'][f"{param_type}_{param_value}"] = param_success
    
    # Calculate overall success rates
    for method in ['pcmci', 'bagged', 'bootstrap']:
        attempts = success_analysis['overall'][method]['attempts']
        if attempts > 0:
            success_analysis['overall'][method]['success_rate'] = (
                success_analysis['overall'][method]['successes'] / attempts
            )
    
    # Create success rate trend plot
    if success_analysis['by_parameter']:
        create_success_rate_trend_plot(success_analysis, param_type, output_dir)
    
    # Save success analysis
    with open(os.path.join(output_dir, 'success_analysis.json'), 'w') as f:
        json.dump(success_analysis, f, indent=4)
    
    return success_analysis

def create_success_rate_trend_plot(success_analysis, param_type, output_dir):
    """
    Create trend plot for success rates.
    
    Parameters
    ----------
    success_analysis : dict
        Results from success rate analysis
    param_type : str
        Parameter type (auto, cross, noise)
    output_dir : str
        Directory to save visualizations
    """
    # Prepare data for plot
    success_data = []
    
    for param_key, param_success in success_analysis['by_parameter'].items():
        # Extract parameter value
        try:
            parts = param_key.split('_')
            param_value = float(parts[1])
        except (ValueError, IndexError):
            continue
        
        # Add success rates for each method
        for method in ['pcmci', 'bagged', 'bootstrap']:
            if 'success_rate' in param_success[method]:
                success_data.append({
                    'Parameter Value': param_value,
                    'Success Rate': param_success[method]['success_rate'],
                    'Method': method
                })
    
    if not success_data:
        return
    
    # Create dataframe
    df = pd.DataFrame(success_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(data=df, x='Parameter Value', y='Success Rate', hue='Method', marker='o')
    
    plt.xlabel(f'{param_type.capitalize()} Value')
    plt.ylabel('Success Rate')
    plt.title(f'Effect Estimation Success Rate vs {param_type.capitalize()}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'success_rate_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create overall success rate comparison
    plt.figure(figsize=(10, 6))
    
    methods = ['pcmci', 'bagged', 'bootstrap']
    success_rates = [success_analysis['overall'][m].get('success_rate', 0) for m in methods]
    
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
    plt.title(f'Overall Causal Effect Estimation Success Rate')
    
    plt.savefig(os.path.join(output_dir, 'overall_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze results from existing parameter studies.')
    parser.add_argument('study_dir', type=str, help='Directory containing parameter study results')
    parser.add_argument('--output', type=str, default=None, 
                      help='Output directory (defaults to study_dir/enhanced_analysis)')
    parser.add_argument('--bootstrap-only', action='store_true', 
                      help='Only analyze bootstrap distributions')
    parser.add_argument('--trends-only', action='store_true',
                      help='Only create trend plots')
    parser.add_argument('--success-only', action='store_true',
                      help='Only analyze estimation success rates')
    
    args = parser.parse_args()
    
    # Create output directory if not provided
    if args.output is None:
        output_dir = os.path.join(args.study_dir, 'enhanced_analysis')
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameter study results
    print(f"Loading results from {args.study_dir}...")
    results = load_parameter_study_results(args.study_dir)
    
    if not results:
        print(f"No parameter study results found in {args.study_dir}")
        return
    
    print(f"Found results for {len(results)} parameter values")
    
    # Run the requested analyses
    if args.bootstrap_only:
        print("Analyzing bootstrap distributions...")
        reanalyze_bootstrap_distributions(results, os.path.join(output_dir, 'bootstrap_analysis'))
    elif args.trends_only:
        print("Creating trend plots...")
        create_trend_plots(results, os.path.join(output_dir, 'trend_analysis'))
    elif args.success_only:
        print("Analyzing estimation success rates...")
        analyze_estimation_success(results, os.path.join(output_dir, 'success_analysis'))
    else:
        # Run all analyses
        print("Analyzing bootstrap distributions...")
        reanalyze_bootstrap_distributions(results, os.path.join(output_dir, 'bootstrap_analysis'))
        
        print("Creating trend plots...")
        create_trend_plots(results, os.path.join(output_dir, 'trend_analysis'))
        
        print("Analyzing estimation success rates...")
        analyze_estimation_success(results, os.path.join(output_dir, 'success_analysis'))
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
