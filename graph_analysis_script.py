import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr

from causal_evaluation import load_study_results_with_graphs

from causal_visualization import plot_causal_graph, plot_ts_graph


"""
Graph analysis utilities for causal discovery and effect estimation.

This module provides functions for loading, saving, and analyzing graph structures
from causal discovery studies, including:
1. Loading study results with graph structures
2. Adding graph structures to existing DataFrames
3. Analyzing graph properties across bootstrap samples
4. Comparing graphs across different parameter values
"""

def add_graphs_to_dataframe(df, study_folder):
    """
    Add graph structures to an existing DataFrame of study results.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing study results
    study_folder : str
        Path to the main study folder containing substudy folders
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added graph structures
    """
    import os
    import numpy as np
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add columns for graph structures if they don't exist
    for graph_name in ['pcmci_graph', 'bagged_graph', 'boot_graphs']:
        if graph_name not in result_df.columns:
            result_df[graph_name] = None
    
    # Process each row in the DataFrame
    for idx, row in result_df.iterrows():
        substudy = row['substudy']
        param_value = row['param_value']
        
        # Construct path to the parameter folder
        param_folder = os.path.join(study_folder, substudy, f"{row['param_name']}_{param_value}")
        
        if not os.path.exists(param_folder):
            continue
        
        # Load PCMCI graph
        pcmci_graph_path = os.path.join(param_folder, "discovery_pcmci_graph.npy")
        if os.path.exists(pcmci_graph_path):
            try:
                result_df.at[idx, 'pcmci_graph'] = np.load(pcmci_graph_path)
            except:
                print(f"Warning: Failed to load PCMCI graph from {pcmci_graph_path}")
        
        # Load bagged graph
        bagged_graph_path = os.path.join(param_folder, "discovery_bootstrap_bagged_graph.npy")
        if os.path.exists(bagged_graph_path):
            try:
                result_df.at[idx, 'bagged_graph'] = np.load(bagged_graph_path)
            except:
                print(f"Warning: Failed to load bagged graph from {bagged_graph_path}")
        
        # Load bootstrap graphs
        boot_graphs_path = os.path.join(param_folder, "discovery_bootstrap_boot_graphs.npy")
        if os.path.exists(boot_graphs_path):
            try:
                result_df.at[idx, 'boot_graphs'] = np.load(boot_graphs_path)
            except:
                print(f"Warning: Failed to load bootstrap graphs from {boot_graphs_path}")
    
    return result_df

def save_dataframe_with_graphs(df, output_path):
    """
    Save a DataFrame containing graph structures to a pickle file.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing graph structures
    output_path : str
        Path to save the pickle file
        
    Returns
    -------
    str
        Path to the saved file
    """
    import os
    import pandas as pd
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save DataFrame to pickle
    df.to_pickle(output_path)
    print(f"DataFrame with graphs saved to {output_path}")
    
    return output_path

def load_dataframe_with_graphs(pickle_path):
    """
    Load a DataFrame with graph structures from a pickle file.
    
    Parameters
    ----------
    pickle_path : str
        Path to the pickle file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with graph structures
    """
    import pandas as pd
    
    # Load DataFrame from pickle
    df = pd.read_pickle(pickle_path)
    print(f"Loaded DataFrame with graphs from {pickle_path}")
    
    return df

def get_graph_for_analysis(df, param_name, param_value, graph_type='pcmci_graph'):
    """
    Extract a specific graph from the DataFrame for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing graph structures
    param_name : str
        Name of the parameter (e.g., 'a', 'c', 'sigmaN', 'T', 'B')
    param_value : float
        Value of the parameter
    graph_type : str, optional
        Type of graph to extract ('pcmci_graph', 'bagged_graph', or 'boot_graphs')
        
    Returns
    -------
    numpy.ndarray
        The requested graph structure
    """
    # Filter DataFrame for the specified parameter
    filtered_df = df[(df['param_name'] == param_name) & (df['param_value'] == param_value)]
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for {param_name}={param_value}")
    
    # Get the first instance of the graph (assuming graphs are the same for all effect pairs with same parameters)
    graph = filtered_df.iloc[0][graph_type]
    
    if graph is None:
        raise ValueError(f"No {graph_type} available for {param_name}={param_value}")
    
    return graph

def main():
    # Set the main study folder path
    main_study_folder = "full_study_results_2025-04-03_15-39-17"
    
    # Create a results folder for the analysis
    results_folder = f"analysis_with_graphs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Option 1: Load results with graphs directly
    print("Loading study results with graphs...")
    results_df = load_study_results_with_graphs(
        main_study_folder, 
        output_folder=results_folder,
        include_graphs=True
    )
    
    # Option 2: Add graphs to an existing DataFrame
    # Uncomment the following lines if you have an existing DataFrame without graphs
    """
    from causal_evaluation import load_study_results
    
    # Load standard results first
    results_df = load_study_results(main_study_folder)
    
    # Add graphs to the DataFrame
    results_df = add_graphs_to_dataframe(results_df, main_study_folder)
    
    # Save the updated DataFrame
    save_dataframe_with_graphs(results_df, os.path.join(results_folder, "results_with_graphs.pkl"))
    """
    
    # Example: Extract a specific graph for analysis
    try:
        # Get a PCMCI graph for auto=0.8
        param_name = 'sigmaN'  # 'a' is the shortened name for 'auto'
        param_value = 0.7
        pcmci_graph = get_graph_for_analysis(results_df, param_name, param_value, 'pcmci_graph')
        
        # Plot the graph
        print(f"Plotting PCMCI graph for {param_name}={param_value}...")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_causal_graph(
            pcmci_graph, 
            var_names=[f'X{i+1}' for i in range(pcmci_graph.shape[0])],
            title=f"PCMCI graph for {param_name}={param_value}",
            save_path=os.path.join(results_folder, f"pcmci_graph_{param_name}_{param_value}.png")
        )
        
        # Get a bagged graph for the same parameter
        bagged_graph = get_graph_for_analysis(results_df, param_name, param_value, 'bagged_graph')
        
        # Plot the time series graph version
        print(f"Plotting bagged time series graph for {param_name}={param_value}...")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_ts_graph(
            bagged_graph, 
            var_names=[f'X{i+1}' for i in range(bagged_graph.shape[0])],
            title=f"Bagged graph for {param_name}={param_value}",
            save_path=os.path.join(results_folder, f"bagged_graph_ts_{param_name}_{param_value}.png")
        )
        
        # Get bootstrap graphs
        boot_graphs = get_graph_for_analysis(results_df, param_name, param_value, 'boot_graphs')
        print(f"Found {len(boot_graphs)} bootstrap graphs for {param_name}={param_value}")
        
        # Plot one of the bootstrap graphs as an example
        if len(boot_graphs) > 0:
            print("Plotting an example bootstrap graph...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_causal_graph(
                boot_graphs[0], 
                var_names=[f'X{i+1}' for i in range(boot_graphs[0].shape[0])],
                title=f"Example bootstrap graph for {param_name}={param_value}",
                save_path=os.path.join(results_folder, f"bootstrap_graph_example_{param_name}_{param_value}.png")
            )
        
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    # Example: Compare graphs across different parameter values
    print("\nComparing graphs across different parameter values...")
    
    try:
        # Select parameter values to compare
        param_name = 'sigmaN'  # autocorrelation
        param_values = [0.5, 1.5, 2.0]
        
        # Create figure for comparison
        fig, axes = plt.subplots(len(param_values), 2, figsize=(15, 5*len(param_values)))
        
        for i, value in enumerate(param_values):
            # Get graphs
            try:
                pcmci_graph = get_graph_for_analysis(results_df, param_name, value, 'pcmci_graph')
                bagged_graph = get_graph_for_analysis(results_df, param_name, value, 'bagged_graph')
                
                # Plot PCMCI graph
                plot_causal_graph(
                    pcmci_graph, 
                    var_names=[f'X{i+1}' for i in range(pcmci_graph.shape[0])],
                    title=f"PCMCI graph ({param_name}={value})",
                    fig_ax=(fig, axes[i, 0])
                )
                
                # Plot bagged graph
                plot_causal_graph(
                    bagged_graph, 
                    var_names=[f'X{i+1}' for i in range(bagged_graph.shape[0])],
                    title=f"Bagged graph ({param_name}={value})",
                    fig_ax=(fig, axes[i, 1])
                )
            except ValueError as e:
                print(f"  Skipping {param_name}={value}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"graph_comparison_{param_name}.png"), dpi=300)
        print(f"Comparison plot saved to {results_folder}/graph_comparison_{param_name}.png")
        
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
    
    print("\nAnalysis complete! Results saved to:", results_folder)

if __name__ == "__main__":
    main()