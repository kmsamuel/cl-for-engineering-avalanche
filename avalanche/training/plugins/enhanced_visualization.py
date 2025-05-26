import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
import os
from collections import Counter, defaultdict

class EnhancedVisualizationTracker:
    def __init__(self, base_dir="vis_results"):
        """
        Initialize the visualization tracker
        
        Args:
            base_dir: Directory to save visualizations
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Tracking data structures
        self.forgetting_history = {}  # {experience_id: {cluster_id: [loss values]}}
        self.cluster_selection_history = []  # List of (experience_id, selected_cluster_ids)
        self.cluster_probabilities_history = []  # List of (experience_id, {cluster_id: probability})
        self.cluster_sizes_history = []  # List of (experience_id, {cluster_id: size})
        self.sample_selection_frequency = defaultdict(int)  # {sample_id: count}
        
    def update_forgetting(self, experience_id, cluster_losses):
        """
        Update forgetting metrics
        
        Args:
            experience_id: Current experience ID
            cluster_losses: Dict mapping cluster IDs to list of losses
        """
        if experience_id not in self.forgetting_history:
            self.forgetting_history[experience_id] = {}
            
        for cluster_id, losses in cluster_losses.items():
            self.forgetting_history[experience_id][cluster_id] = losses
    
    def update_cluster_selections(self, experience_id, selected_clusters):
        """
        Track which clusters were selected
        
        Args:
            experience_id: Current experience ID
            selected_clusters: List of selected cluster IDs
        """
        self.cluster_selection_history.append((experience_id, selected_clusters))
        
        # Update frequency counter
        counter = Counter(selected_clusters)
        for cluster_id, count in counter.items():
            if cluster_id not in self.sample_selection_frequency:
                self.sample_selection_frequency[cluster_id] = 0
            self.sample_selection_frequency[cluster_id] += count
    
    def update_cluster_probabilities(self, experience_id, cluster_probs):
        """
        Track cluster selection probabilities from bandit
        
        Args:
            experience_id: Current experience ID
            cluster_probs: Dict or array of cluster probabilities
        """
        self.cluster_probabilities_history.append((experience_id, cluster_probs))
    
    def update_cluster_sizes(self, experience_id, cluster_sizes):
        """
        Track cluster sizes over time
        
        Args:
            experience_id: Current experience ID
            cluster_sizes: Dict mapping cluster IDs to sizes
        """
        self.cluster_sizes_history.append((experience_id, cluster_sizes))
    
    def plot_forgetting_dynamics_enhanced(self, save_path=None):
        """
        Plot forgetting dynamics with improved styling
        Uses iteration-level data when available for more detailed plots
        
        Args:
            save_path: Path to save the visualization
        """
        # First check if we have iteration-level data in the base directory
        iteration_files = [f for f in os.listdir(self.base_dir) 
                        if f.startswith("iteration_forgetting_log") and f.endswith("_plot_ready.csv")]
        
        if iteration_files:
            # Use iteration-level data for more detailed plotting
            plt.figure(figsize=(14, 8))
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Load and combine all iteration-level CSVs
            all_data = []
            for file in iteration_files:
                csv_path = os.path.join(self.base_dir, file)
                try:
                    df = pd.read_csv(csv_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading CSV file {file}: {e}")
            
            if not all_data:
                print("No valid iteration-level data found, falling back to experience-level data")
            else:
                # Combine all dataframes
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Get unique cluster sources
                cluster_sources = combined_df['cluster_source'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_sources)))
                color_map = {cluster: colors[i] for i, cluster in enumerate(cluster_sources)}
                
                # Plot each cluster source
                for cluster_source in cluster_sources:
                    cluster_data = combined_df[combined_df['cluster_source'] == cluster_source]
                    
                    # Sort by iteration for proper line plotting
                    cluster_data = cluster_data.sort_values(by='iteration')
                    
                    plt.plot(cluster_data['iteration'], cluster_data['mean_forgetting'], 
                            label=cluster_source, linewidth=2.5, color=color_map[cluster_source])
                    plt.fill_between(
                        cluster_data['iteration'], 
                        cluster_data['mean_forgetting'] - cluster_data['std_forgetting'],
                        cluster_data['mean_forgetting'] + cluster_data['std_forgetting'],
                        alpha=0.3,
                        color=color_map[cluster_source]
                    )
                
                # Add vertical lines to mark experience boundaries
                if 'experience_id' in combined_df.columns:
                    for exp_id in sorted(combined_df['experience_id'].unique())[1:]:
                        exp_start = combined_df[combined_df['experience_id'] == exp_id]['iteration'].min()
                        plt.axvline(x=exp_start, color='gray', linestyle='--', alpha=0.7)
                        plt.text(exp_start, plt.ylim()[1] * 0.95, f' Exp {exp_id}', 
                                fontsize=10, verticalalignment='top')
                
                plt.title('Iteration-Level Forgetting Dynamics', fontsize=16)
                plt.xlabel('Training Iteration', fontsize=14)
                plt.ylabel('Forgetting (Loss Increase from Best Performance)', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=12, loc='upper right')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Iteration-level forgetting plot saved to: {save_path}")
                else:
                    plt.show()
                plt.close()
                
                # Create separate plots for each experience to handle KMeans/TabPFN clustering better
                for exp_id in sorted(combined_df['experience_id'].unique()):
                    exp_df = combined_df[combined_df['experience_id'] == exp_id]
                    
                    # Create a new figure for each experience
                    plt.figure(figsize=(14, 8))
                    plt.style.use('seaborn-v0_8-whitegrid')
                    
                    # Get unique cluster sources for this experience
                    exp_cluster_sources = exp_df['cluster_source'].unique()
                    exp_colors = plt.cm.tab10(np.linspace(0, 1, len(exp_cluster_sources)))
                    exp_color_map = {cluster: exp_colors[i] for i, cluster in enumerate(exp_cluster_sources)}
                    
                    # Plot each cluster
                    for cluster_source in exp_cluster_sources:
                        cluster_data = exp_df[exp_df['cluster_source'] == cluster_source]
                        
                        # Sort by iteration for proper line plotting
                        cluster_data = cluster_data.sort_values(by='iteration')
                        
                        plt.plot(cluster_data['iteration'], cluster_data['mean_forgetting'], 
                                label=cluster_source, linewidth=2.5, color=exp_color_map[cluster_source])
                        plt.fill_between(
                            cluster_data['iteration'], 
                            cluster_data['mean_forgetting'] - cluster_data['std_forgetting'],
                            cluster_data['mean_forgetting'] + cluster_data['std_forgetting'],
                            alpha=0.3,
                            color=exp_color_map[cluster_source]
                        )
                    
                    plt.title(f'Forgetting Dynamics - Experience {exp_id}', fontsize=16)
                    plt.xlabel('Training Iteration', fontsize=14)
                    plt.ylabel('Forgetting (Loss Increase from Best)', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=12, loc='upper right')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save this experience's plot
                    if save_path:
                        exp_save_path = save_path.replace('.png', f'_exp{exp_id}.png')
                        os.makedirs(os.path.dirname(exp_save_path), exist_ok=True)
                        plt.savefig(exp_save_path, dpi=300, bbox_inches='tight')
                        print(f"Forgetting dynamics plot for Exp {exp_id} saved to: {exp_save_path}")
                    else:
                        plt.show()
                    plt.close()
                
                # Return early since we've created plots with iteration-level data
                return
        
        # Fall back to experience-level plotting if no iteration data available
        plt.figure(figsize=(12, 8))
        
        # Setup enhanced style
        plt.style.use('seaborn-v0_8-whitegrid')  # Updated name for the legacy style
        
        # Get unique experiences and clusters
        all_experiences = sorted(self.forgetting_history.keys())
        all_clusters = set()
        for exp_data in self.forgetting_history.values():
            all_clusters.update(exp_data.keys())
        all_clusters = sorted(all_clusters)
        
        # Generate colors for each cluster (matching your reference image style)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clusters)))
        color_map = {cluster: colors[i] for i, cluster in enumerate(all_clusters)}
        
        # Plot each cluster's forgetting
        for cluster_id in all_clusters:
            means = []
            stds = []
            for exp_id in all_experiences:
                if exp_id in self.forgetting_history and cluster_id in self.forgetting_history[exp_id]:
                    losses = self.forgetting_history[exp_id][cluster_id]
                    means.append(np.mean(losses))
                    stds.append(np.std(losses))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            # Filter out nans
            valid_indices = ~np.isnan(means)
            if not any(valid_indices):
                continue
                
            x_vals = np.array(all_experiences)[valid_indices]
            y_vals = np.array(means)[valid_indices]
            y_errs = np.array(stds)[valid_indices]
            
            # Plot with confidence bands similar to reference image
            plt.plot(x_vals, y_vals, '-', color=color_map[cluster_id], 
                    label=f'Cluster {cluster_id}', linewidth=2.5)
            plt.fill_between(x_vals, y_vals - y_errs, y_vals + y_errs, 
                            color=color_map[cluster_id], alpha=0.3)
        
        # Styling to match reference image
        plt.title('Forgetting Dynamics Across Experiences', fontsize=16)
        plt.xlabel('Experience', fontsize=14)
        plt.ylabel('Forgetting (Loss Increase from Best Performance)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to integers only
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forgetting dynamics plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_cluster_probabilities(self, save_path=None):
        """
        Plot the evolution of cluster selection probabilities from the bandit
        """
        if not self.cluster_probabilities_history:
            print("No cluster probability data available")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract data
        experiences = []
        prob_data = {}
        
        for exp_id, probs in self.cluster_probabilities_history:
            experiences.append(exp_id)
            
            # Handle both dict and array formats
            if isinstance(probs, dict):
                for cluster_id, prob in probs.items():
                    if cluster_id not in prob_data:
                        prob_data[cluster_id] = []
                    prob_data[cluster_id].append(prob)
            else:  # numpy array
                for cluster_id, prob in enumerate(probs):
                    if cluster_id not in prob_data:
                        prob_data[cluster_id] = []
                    prob_data[cluster_id].append(prob)
        
        # Plot each cluster's probability
        for cluster_id, probs in prob_data.items():
            # Ensure length matches by padding with NaN
            padded_probs = probs + [np.nan] * (len(experiences) - len(probs))
            plt.plot(experiences, padded_probs, 'o-', 
                    label=f'Cluster {cluster_id}', linewidth=2)
        
        plt.title('Bandit Cluster Selection Probabilities Over Time')
        plt.xlabel('Experience')
        plt.ylabel('Selection Probability')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to integers only
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster probabilities plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_cluster_selection_heatmap(self, save_path=None):
        """
        Create a heatmap showing how frequently each cluster was selected
        across experiences
        """
        if not self.cluster_selection_history:
            print("No cluster selection data available")
            return
            
        # Aggregate data into a matrix
        experiences = sorted(set(exp_id for exp_id, _ in self.cluster_selection_history))
        all_clusters = set()
        for _, clusters in self.cluster_selection_history:
            all_clusters.update(clusters)
        all_clusters = sorted(all_clusters)
        
        # Initialize matrix
        selection_matrix = np.zeros((len(experiences), len(all_clusters)))
        
        # Fill matrix
        for exp_id, clusters in self.cluster_selection_history:
            if exp_id in experiences:
                exp_idx = experiences.index(exp_id)
                for cluster_id in clusters:
                    if cluster_id in all_clusters:
                        cluster_idx = all_clusters.index(cluster_id)
                        selection_matrix[exp_idx, cluster_idx] += 1
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(selection_matrix, annot=True, fmt='g', cmap='viridis',
                   xticklabels=[f'C{c}' for c in all_clusters],
                   yticklabels=[f'Exp {e}' for e in experiences])
        
        plt.title('Cluster Selection Frequency by Experience')
        plt.xlabel('Cluster ID')
        plt.ylabel('Experience')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster selection heatmap saved to: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_cluster_sizes(self, save_path=None):
        """
        Plot how cluster sizes evolve over time
        """
        if not self.cluster_sizes_history:
            print("No cluster size data available")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Extract data
        experiences = []
        size_data = {}
        
        for exp_id, sizes in self.cluster_sizes_history:
            experiences.append(exp_id)
            
            for cluster_id, size in sizes.items():
                if cluster_id not in size_data:
                    size_data[cluster_id] = []
                size_data[cluster_id].append(size)
        
        # Plot each cluster's size
        for cluster_id, sizes in size_data.items():
            # Ensure length matches by padding with NaN
            padded_sizes = sizes + [np.nan] * (len(experiences) - len(sizes))
            plt.plot(experiences, padded_sizes, 'o-', 
                    label=f'Cluster {cluster_id}', linewidth=2)
        
        plt.title('Cluster Sizes Over Time')
        plt.xlabel('Experience')
        plt.ylabel('Number of Samples')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to integers only
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster sizes plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
        
    def create_forgetting_report(self, save_path=None):
        """
        Generate a comprehensive report of forgetting dynamics and selection patterns
        """
        report_data = []
        
        # Forgetting metrics
        for exp_id in sorted(self.forgetting_history.keys()):
            for cluster_id in sorted(self.forgetting_history[exp_id].keys()):
                losses = self.forgetting_history[exp_id][cluster_id]
                report_data.append({
                    'experience_id': exp_id,
                    'cluster_id': cluster_id,
                    'mean_loss': np.mean(losses),
                    'std_loss': np.std(losses),
                    'min_loss': np.min(losses),
                    'max_loss': np.max(losses),
                    'num_samples': len(losses)
                })
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Forgetting report saved to: {save_path}")
            
        return df
    
    def plot_cluster_composition(self, buffer, save_path=None):
        """
        Plot the composition of each cluster by original experience
        
        Args:
            buffer: The replay buffer containing (x, y, emb, cid, eid) tuples
            save_path: Path to save the visualization
        """
        # Count samples in each cluster by their original experience
        compositions = {}
        for x, y, emb, cid, eid in buffer:
            if cid not in compositions:
                compositions[cid] = {}
            if eid not in compositions[cid]:
                compositions[cid][eid] = 0
            compositions[cid][eid] += 1
        
        # Prepare data for plotting
        cluster_ids = sorted(compositions.keys())
        all_exp_ids = sorted(set(eid for cluster in compositions.values() for eid in cluster.keys()))
        
        # Create matrix for stacked bars
        data = np.zeros((len(cluster_ids), len(all_exp_ids)))
        for i, cid in enumerate(cluster_ids):
            for j, eid in enumerate(all_exp_ids):
                if eid in compositions[cid]:
                    data[i, j] = compositions[cid][eid]
        
        # Plot
        plt.figure(figsize=(10, 6))
        bottom = np.zeros(len(cluster_ids))
        
        # Create a colormap for experiences
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_exp_ids)))
        
        for j, eid in enumerate(all_exp_ids):
            plt.bar(cluster_ids, data[:, j], bottom=bottom, 
                    label=f'Exp {eid}', color=colors[j])
            bottom += data[:, j]
        
        plt.title('Cluster Composition by Original Experience', fontsize=14)
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(cluster_ids, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster composition plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_all_plots(self, buffer, base_filename=None):
        """
        Generate all visualization plots at once
        
        Args:
            buffer: The replay buffer containing (x, y, emb, cid, eid) tuples
            base_filename: Base filename to use for all plots
        """
        if not base_filename:
            base_filename = os.path.join(self.base_dir, "replay_analysis")
            
        self.plot_forgetting_dynamics_enhanced(f"{base_filename}_forgetting.png")
        self.plot_cluster_probabilities(f"{base_filename}_probabilities.png")
        self.plot_cluster_selection_heatmap(f"{base_filename}_selections.png")
        self.plot_cluster_sizes(f"{base_filename}_sizes.png")
        self.plot_cluster_composition(buffer, f"{base_filename}_composition.png")
        self.create_forgetting_report(f"{base_filename}_report.csv")