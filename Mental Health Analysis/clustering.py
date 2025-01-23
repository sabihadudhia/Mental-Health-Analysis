import os
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from datetime import datetime
from typing import Optional, Dict, Tuple
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get output directory from environment variable or use default
OUTPUT_DIR = 'Clustering'

class ClusterAnalyzer:
    def __init__(self, data_path: str = 'data/processed/mental_health_scores.csv', output_dir: str = 'Clustering'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        self.scaled_data: Optional[np.ndarray] = None
        
        # Setup directory structure
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.insights_dir = os.path.join(self.output_dir, 'insights')
        
        # Create directories
        for directory in [self.plots_dir, self.results_dir, self.metrics_dir, self.insights_dir]:
            os.makedirs(directory, exist_ok=True)
            
        logging.info(f"ClusterAnalyzer initialized with output directories in {self.output_dir}")

    def load_and_preprocess(self) -> None:
        """Load and preprocess the data."""
        try:
            logging.debug("Loading data...")
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded from {self.data_path} with shape {self.df.shape}")
            
            # Select only the score columns
            score_columns = [col for col in self.df.columns if 'score' in col.lower()]
            if not score_columns:
                raise ValueError("No score columns found in dataset")
            
            self.df = self.df[score_columns]
            logging.info(f"Score columns selected: {score_columns}")
            logging.info("Starting data preprocessing...")
            logging.info(f"Found {len(score_columns)} score columns: {', '.join(score_columns)}")
            logging.debug(f"Selected score columns: {score_columns}")
            
            # Scale the data
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(self.df)
            logging.info(f"Data scaled successfully: {self.scaled_data.shape}")
            logging.debug(f"Scaled data shape: {self.scaled_data.shape}")
            logging.info("Data preprocessing completed successfully")
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def find_optimal_kmeans(self, max_clusters: int = 10) -> Tuple[list, list]:
        """Find optimal number of clusters using elbow method and silhouette score."""
        logging.info(f"Finding optimal number of clusters (max_clusters={max_clusters})...")
        if self.scaled_data is None:
            raise ValueError("Data not preprocessed. Call load_and_preprocess() first.")
            
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            logging.debug(f"Evaluating KMeans for k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            logging.info(f"K={k}: inertia={kmeans.inertia_}, silhouette_score={silhouette_scores[-1]}")
            
        logging.info("Completed optimal cluster analysis")
        return inertias, silhouette_scores

    def perform_kmeans(self, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering with validation and error handling."""
        logging.info(f"Starting K-means clustering with {n_clusters} clusters...")
        if self.scaled_data is None:
            raise ValueError("Data not preprocessed. Call load_and_preprocess() first.")
            
        try:
            logging.debug(f"Initializing KMeans with n_clusters={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            logging.debug(f"K-means labels: {labels}")
            logging.info(f"K-means clustering completed successfully. Clusters formed: {len(np.unique(labels))}")
            return labels
            
        except Exception as e:
            logging.error(f"Error in K-means clustering: {str(e)}")
            raise

    def perform_clustering(self) -> Dict[str, np.ndarray]:
        """Perform multiple clustering methods and return labels."""
        logging.info("Starting multiple clustering analysis...")
        if self.scaled_data is None:
            raise ValueError("Data not preprocessed. Call load_and_preprocess() first.")
            
        try:
            # K-means clustering
            logging.debug("Performing K-means clustering")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(self.scaled_data)
            logging.info("K-means clustering performed successfully.")
            logging.info(f"K-means clusters formed: {len(np.unique(kmeans_labels))}")
            logging.debug(f"K-means labels: {kmeans_labels}")
            
            # DBSCAN clustering
            logging.debug("Performing DBSCAN clustering")
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(self.scaled_data)
            logging.info("DBSCAN clustering performed successfully.")
            logging.info(f"DBSCAN clusters formed: {len(np.unique(dbscan_labels))}")
            logging.debug(f"DBSCAN labels: {dbscan_labels}")
            
            # Hierarchical clustering
            logging.debug("Performing Hierarchical clustering")
            hierarchical = AgglomerativeClustering(n_clusters=3)
            hierarchical_labels = hierarchical.fit_predict(self.scaled_data)
            logging.info("Hierarchical clustering performed successfully.")
            logging.info(f"Hierarchical clusters formed: {len(np.unique(hierarchical_labels))}")
            logging.debug(f"Hierarchical labels: {hierarchical_labels}")
            
            logging.info("All clustering methods completed successfully")
            return {
                'kmeans': kmeans_labels,
                'dbscan': dbscan_labels,
                'hierarchical': hierarchical_labels
            }
            
        except Exception as e:
            logging.error(f"Error performing clustering: {str(e)}")
            raise

    def get_clustering_metrics(self, labels_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for all clustering results."""
        metrics = {}
        try:
            # Calculate ARI between each pair of clustering methods
            methods = list(labels_dict.keys())
            for i, method1 in enumerate(methods):
                if method1 not in metrics:
                    metrics[method1] = {}
                
                # Get existing silhouette score calculation
                if labels_dict[method1] is not None and len(np.unique(labels_dict[method1])) > 1:
                    valid_mask = labels_dict[method1] != -1
                    if np.sum(valid_mask) > 1:
                        silhouette = silhouette_score(
                            self.scaled_data[valid_mask], 
                            labels_dict[method1][valid_mask]
                        )
                        metrics[method1].update({
                            'silhouette_score': silhouette,
                            'n_clusters': len(np.unique(labels_dict[method1][valid_mask])),
                            'noise_points': np.sum(~valid_mask) if -1 in labels_dict[method1] else 0
                        })
                
                # Calculate ARI scores between this method and all others
                for j, method2 in enumerate(methods[i+1:], i+1):
                    if -1 not in labels_dict[method1] and -1 not in labels_dict[method2]:
                        ari = adjusted_rand_score(labels_dict[method1], labels_dict[method2])
                        metrics[method1][f'ari_vs_{method2}'] = ari
                        
                        if method2 not in metrics:
                            metrics[method2] = {}
                        metrics[method2][f'ari_vs_{method1}'] = ari
            
            # Log ARI scores
            for method, method_metrics in metrics.items():
                ari_scores = {k: v for k, v in method_metrics.items() if k.startswith('ari_vs_')}
                if ari_scores:
                    logging.info(f"ARI scores for {method}: {ari_scores}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating clustering metrics: {str(e)}")
            return {}

    def evaluate_clustering(self, labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate clustering results using silhouette scores."""
        logging.info("Evaluating clustering results...")
        scores = {}
        try:
            for method, cluster_labels in labels.items():
                logging.debug(f"Evaluating {method} clustering")
                if -1 not in cluster_labels and len(set(cluster_labels)) > 1:
                    score = silhouette_score(self.scaled_data, cluster_labels)
                    scores[method] = score
                    logging.info(f"{method.capitalize()} Silhouette Score: {score:.3f}")
                    logging.debug(f"{method.capitalize()} Silhouette Score: {scores[method]}")
                else:
                    scores[method] = -1
                    logging.warning(f"{method.capitalize()} clustering did not form valid clusters")
            
            return scores
        except Exception as e:
            logging.error(f"Error evaluating clusters: {str(e)}")
            raise

    def generate_insights(self, clustering_results: Dict[str, np.ndarray], evaluation_scores: Dict[str, float]) -> None:
        """Generate detailed insights for each clustering method."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for method, labels in clustering_results.items():
                # Create a DataFrame with clusters
                df_with_clusters = self.df.copy()
                df_with_clusters['cluster'] = labels
                
                # Prepare insights list
                insights = []
                insights.extend([
                    "==================================================",
                    f"CLUSTERING INSIGHTS: {method.upper()}",
                    "==================================================",
                    f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Method: {method}",
                    f"Number of Clusters: {len(np.unique(labels))}",
                    f"Silhouette Score: {evaluation_scores.get(method, 'N/A')}",
                    "\nCLUSTER STATISTICS",
                    "==================",
                ])

                # Special handling for DBSCAN
                if method == 'dbscan':
                    noise_mask = labels == -1
                    noise_points = np.sum(noise_mask)
                    insights.extend([
                        f"\nNoise Points: {noise_points} ({(noise_points/len(labels))*100:.2f}%)",
                        f"Valid Clusters: {len(np.unique(labels[labels != -1]))}"
                    ])

                # Analyze each cluster
                unique_labels = np.unique(labels)
                for cluster in unique_labels:
                    cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
                    
                    # Calculate cluster statistics
                    stats = {
                        'size': len(cluster_data),
                        'percentage': (len(cluster_data) / len(df_with_clusters)) * 100,
                        'mental_health_support_score': cluster_data['mental_health_support_score'].mean(),
                        'mental_health_risk_score': cluster_data['mental_health_risk_score'].mean(),
                        'workplace_satisfaction_score': cluster_data['workplace_satisfaction_score'].mean(),
                        'treatment_engagement_score': cluster_data['treatment_engagement_score'].mean()
                    }
                    
                    # Special handling for DBSCAN noise points
                    if method == 'dbscan' and cluster == -1:
                        cluster_label = "Noise Points"
                    else:
                        cluster_label = f"Cluster {cluster}"
                    
                    # Add cluster insights
                    insights.extend([
                        f"\n{cluster_label}:",
                        f"Size: {stats['size']} samples ({stats['percentage']:.2f}%)",
                        f"Mental Health Support Score: {stats['mental_health_support_score']:.2f}",
                        f"Mental Health Risk Score: {stats['mental_health_risk_score']:.2f}",
                        f"Workplace Satisfaction Score: {stats['workplace_satisfaction_score']:.2f}",
                        f"Treatment Engagement Score: {stats['treatment_engagement_score']:.2f}"
                    ])

                # Save insights to file
                output_path = os.path.join(self.insights_dir, f'{method}_cluster_insights.txt')
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(insights))
                
                logging.info(f"Saved insights for {method} clustering to {output_path}")
                    
        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            raise

    def evaluate_clustering_methods(self, max_clusters: int = 10) -> Dict[str, int]:
        """Evaluate optimal number of clusters using multiple methods."""
        logging.info(f"Evaluating optimal clusters using multiple methods (max_clusters={max_clusters})...")
        
        results = {}
        
        # Elbow Method
        logging.debug("Evaluating Elbow Method")
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
        
        elbow = KneeLocator(range(1, max_clusters + 1), inertias, curve='convex', direction='decreasing')
        results['elbow'] = elbow.elbow
        logging.debug(f"Elbow method suggests {elbow.elbow} clusters")
        logging.info(f"Elbow method suggests {elbow.elbow} clusters")

        # Silhouette Analysis
        logging.debug("Evaluating Silhouette Analysis")
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
        
        optimal_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        results['silhouette'] = optimal_silhouette
        logging.debug(f"Silhouette analysis suggests {optimal_silhouette} clusters")
        logging.info(f"Silhouette analysis suggests {optimal_silhouette} clusters")

        # Gaussian Mixture BIC
        logging.debug("Evaluating Gaussian Mixture BIC")
        bic_scores = []
        for k in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.scaled_data)
            bic_scores.append(gmm.bic(self.scaled_data))
        
        optimal_bic = bic_scores.index(min(bic_scores)) + 1
        results['bic'] = optimal_bic
        logging.debug(f"Gaussian BIC suggests {optimal_bic} clusters")
        logging.info(f"Gaussian BIC suggests {optimal_bic} clusters")

        # Hierarchical Clustering
        logging.debug("Evaluating Hierarchical Clustering")
        linkage_matrix = linkage(self.scaled_data, method='ward')
        last = linkage_matrix[-10:, 2]
        acceleration = np.diff(last, 2)
        optimal_hierarchical = acceleration.argmax() + 2
        results['hierarchical'] = optimal_hierarchical
        logging.debug(f"Hierarchical clustering suggests {optimal_hierarchical} clusters")
        logging.info(f"Hierarchical clustering suggests {optimal_hierarchical} clusters")

        # Determine final number of clusters (using majority voting or most conservative)
        final_k = int(np.median(list(results.values())))
        logging.info(f"Final selected number of clusters: {final_k}")
        
        return final_k

    def perform_optimal_clustering(self) -> np.ndarray:
        """Perform clustering with optimal number of clusters."""
        logging.info("Starting optimal clustering analysis...")
        
        # Find optimal number of clusters
        n_clusters = self.evaluate_clustering_methods()
        logging.debug(f"Optimal number of clusters determined: {n_clusters}")
        
        # Perform K-means with optimal clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.scaled_data)
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(self.scaled_data, labels)
        
        logging.info(f"Clustering completed with {n_clusters} clusters")
        logging.info(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return labels

    def _style_axis(self, ax, title: str, xlabel: str, ylabel: str) -> None:
        """Helper method to consistently style plot axes."""
        ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_k_selection_metrics(self, max_clusters: int = 10) -> None:
        """Create and save plots for elbow method and silhouette scores."""
        try:
            # Calculate metrics for different k values
            k_range = range(2, max_clusters + 1)
            inertias = []
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.scaled_data)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(self.scaled_data, labels))

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot Elbow Method
            ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            self._style_axis(ax1, 'Elbow Method for Optimal k',
                           'Number of Clusters (k)', 'Inertia')
            
            # Add elbow point
            elbow_locator = KneeLocator(list(k_range), inertias, 
                                      curve='convex', direction='decreasing')
            if elbow_locator.elbow:
                elbow_inertia = inertias[elbow_locator.elbow-2]  # Adjust for k starting at 2
                ax1.axvline(x=elbow_locator.elbow, color='r', linestyle='--', alpha=0.5)
                ax1.plot(elbow_locator.elbow, elbow_inertia, 'ro', markersize=10)
                ax1.annotate(f'Elbow Point (k={elbow_locator.elbow})',
                           (elbow_locator.elbow, elbow_inertia),
                           xytext=(10, -10), textcoords='offset points',
                           bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

            # Plot Silhouette Scores
            ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
            self._style_axis(ax2, 'Silhouette Score for Different k',
                           'Number of Clusters (k)', 'Silhouette Score')
            ax2.set_ylim(-0.1, 1.1)
            
            # Add optimal k based on silhouette score
            optimal_k_idx = np.argmax(silhouette_scores)
            optimal_k = k_range[optimal_k_idx]
            optimal_score = silhouette_scores[optimal_k_idx]
            ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
            ax2.plot(optimal_k, optimal_score, 'ro', markersize=10)
            ax2.annotate(f'Optimal k={optimal_k}\nScore={optimal_score:.3f}',
                        (optimal_k, optimal_score),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))

            # Add grid and improve layout
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            plt.suptitle('Cluster Number Selection Analysis', 
                        fontsize=14, y=1.05, fontweight='bold')
            plt.tight_layout()

            # Save plot to metrics directory
            output_path = os.path.join(self.metrics_dir, 'k_selection_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Also save the raw data to metrics directory
            metrics_df = pd.DataFrame({
                'k': list(k_range),
                'inertia': inertias,
                'silhouette_score': silhouette_scores
            })
            metrics_df.to_csv(os.path.join(self.metrics_dir, 'k_selection_metrics.csv'),
                            index=False)
            
            logging.info(f"K selection analysis plots saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error plotting k selection metrics: {str(e)}")
            raise

    def run_all_clustering(self) -> None:
        """Run all clustering functions and generate insights."""
        logging.info("Starting clustering analysis pipeline...")
        try:
            # Load and preprocess data
            logging.info("Step 1: Loading and preprocessing data")
            self.load_and_preprocess()
            
            # Add k selection analysis before performing clustering
            logging.info("Step 2: Analyzing optimal number of clusters")
            self.plot_k_selection_metrics()
            
            # Perform optimal clustering
            logging.info("Step 3: Performing optimal clustering")
            optimal_labels = self.perform_optimal_clustering()
            
            # Perform other clustering methods
            logging.info("Step 4: Performing multiple clustering methods")
            clustering_results = self.perform_clustering()
            
            # Combine all results
            logging.info("Step 5: Combining clustering results")
            all_clustering_results = {
                'kmeans': clustering_results['kmeans'],
                'hierarchical': clustering_results['hierarchical'],
                'optimal': optimal_labels
            }
            
            # Get evaluation metrics
            logging.info("Step 6: Calculating evaluation metrics")
            evaluation_metrics = self.get_clustering_metrics(all_clustering_results)
            
            # Get evaluation scores
            logging.info("Step 7: Calculating evaluation scores")
            evaluation_scores = self.evaluate_clustering(all_clustering_results)
            
            # Generate insights
            logging.info("Step 8: Generating insights")
            self.generate_insights(all_clustering_results, evaluation_scores)
            
            # Create visualizations
            logging.info("Step 9: Creating visualizations")
            self.visualize_clusters(all_clustering_results)
            self.plot_clustering_metrics(evaluation_metrics)
            
            logging.info("Clustering analysis pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in clustering analysis pipeline: {str(e)}")
            raise

    def visualize_clusters(self, labels_dict: Dict[str, np.ndarray]) -> None:
        """Create comprehensive cluster visualizations using dimensionality reduction."""
        try:
            for method, labels in labels_dict.items():
                # Create figure with more space for titles
                fig = plt.figure(figsize=(20, 18))  # Increased height for titles
                gs = plt.GridSpec(3, 2, height_ratios=[0.05, 1, 1.2])  # Added row for titles
                
                # Add method title at the very top
                fig.suptitle(f'{method.upper()} Clustering Analysis', 
                           fontsize=20, y=0.98, fontweight='bold')
                
                # Add subtitle row
                title_ax1 = fig.add_subplot(gs[0, 0])
                title_ax1.text(0.25, 0.25, 'PCA Projection', 
                             ha='center', va='center', fontsize=12, fontweight='bold')
                title_ax1.axis('off')
                
                title_ax2 = fig.add_subplot(gs[0, 1])
                title_ax2.text(0.25, 0.25, 't-SNE Projection',
                             ha='center', va='center', fontsize=12, fontweight='bold')
                title_ax2.axis('off')

                # 1. PCA Scatter Plot
                ax1 = fig.add_subplot(gs[1, 0])
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(self.scaled_data)
                self._plot_scatter(ax1, pca_result, labels, 
                                 '', 'PCA Component 1', 'PCA Component 2')

                # 2. t-SNE Scatter Plot
                ax2 = fig.add_subplot(gs[1, 1])
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(self.scaled_data)
                self._plot_scatter(ax2, tsne_result, labels,
                                 '', 't-SNE Component 1', 't-SNE Component 2')

                # 3. Cluster Characteristics Box Plots
                ax3 = fig.add_subplot(gs[2, :])
                self._plot_cluster_characteristics(ax3, labels, method)

                # Adjust layout
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

                # Save plot
                output_path = os.path.join(self.plots_dir, f'{method}_cluster_analysis.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                logging.info(f"Cluster analysis plots for {method} saved to {output_path}")

        except Exception as e:
            logging.error(f"Error creating cluster visualizations: {str(e)}")
            raise

    def _plot_scatter(self, ax, data: np.ndarray, labels: np.ndarray, 
                     title: str, xlabel: str, ylabel: str) -> None:
        """Helper method to create consistent scatter plots."""
        scatter = ax.scatter(data[:, 0], data[:, 1],
                           c=labels, cmap='viridis',
                           alpha=0.6, s=100, edgecolor='white')
        
        # Add cluster centers and annotations
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                mask = labels == label
                center = data[mask].mean(axis=0)
                ax.scatter(center[0], center[1], 
                          c='red', marker='x', s=200, linewidths=3,
                          label=f'Cluster {label} Center')
                ax.annotate(f'n={np.sum(mask)}', 
                          (center[0], center[1]),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        # Set labels without title
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)

    def _plot_cluster_characteristics(self, ax, labels: np.ndarray, method: str) -> None:
        """Create box plots showing score distributions for each cluster."""
        # Add title above the plot
        ax.text(0.5, 1.1, 'Score Distributions by Cluster',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                transform=ax.transAxes)
        
        # Prepare data for plotting
        plot_data = []
        cluster_names = []
        score_names = [
            'Mental Health Support\n(Employer Benefits)',
            'Mental Health Risk\n(Treatment Need)',
            'Workplace Satisfaction\n(Environment)',
            'Treatment Engagement\n(Professional Help)'
        ]

        # Get clusters excluding noise
        valid_clusters = np.unique(labels[labels != -1])
        n_clusters = len(valid_clusters)
        n_metrics = len(score_names)
        
        # Calculate total width needed
        total_positions = n_clusters * n_metrics
        positions = np.arange(total_positions)
        
        # Prepare data
        for cluster in valid_clusters:
            mask = labels == cluster
            for score_idx, score_name in enumerate([
                'mental_health_support_score',
                'mental_health_risk_score',
                'workplace_satisfaction_score',
                'treatment_engagement_score'
            ]):
                plot_data.append(self.df[score_name][mask])
                cluster_names.append(f'C{cluster}')  # Shorter cluster names

        # Create figure with increased height for labels
        fig = ax.get_figure()
        fig.set_figheight(fig.get_figheight() + 4)  # Increase height
        
        # Create violin plots
        violin_parts = ax.violinplot(plot_data, positions=positions,
                                   showmeans=True, showmedians=True)
        
        # Customize violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        # Add box plots
        bp = ax.boxplot(plot_data, positions=positions, widths=0.3,
                       patch_artist=True, showfliers=False)
        
        # Customize box plots
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(0.7)

        # Customize plot
        ax.set_title('Score Distributions by Cluster', fontsize=12, pad=20)
        ax.set_ylabel('Score Value', fontsize=10)
        
        # Create two-line labels: cluster number on top, metric name below
        labels = []
        for i in range(len(positions)):
            cluster_idx = i // n_metrics
            metric_idx = i % n_metrics
            labels.append(f'{cluster_names[i]}\n{score_names[metric_idx]}')

        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.3)  # Increase bottom margin
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add cluster separators
        for i in range(n_metrics, total_positions, n_metrics):
            ax.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add cluster group labels at the top
        y_max = ax.get_ylim()[1]
        for i, cluster in enumerate(valid_clusters):
            center_pos = i * n_metrics + (n_metrics - 1) / 2
            ax.annotate(f'Cluster {cluster}',
                       xy=(center_pos, y_max),
                       xytext=(0, 20),  # Offset above the plot
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                       fontsize=10,
                       fontweight='bold')
        
        # Add more padding for x-tick labels
        ax.tick_params(axis='x', which='major', pad=40)
        
        # Add legend for metrics
        legend_elements = [plt.Line2D([0], [0], color='w', marker='s',
                                    markerfacecolor='lightblue', markersize=10,
                                    label=score_name.split('\n')[0])
                         for score_name in score_names]
        ax.legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.15, 1.0))

        # Use tight layout but maintain bottom margin
        plt.tight_layout(rect=[0, 0.2, 1, 1])

    def plot_clustering_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot clustering metrics comparison including ARI scores."""
        try:
            logging.debug("Plotting clustering metrics comparison")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot silhouette scores
            methods = list(metrics.keys())
            silhouette_scores = [m['silhouette_score'] for m in metrics.values()]
            
            ax1.bar(methods, silhouette_scores)
            ax1.set_title('Clustering Performance - Silhouette Scores')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_ylim(0, 1)
            
            # Plot ARI scores heatmap
            ari_matrix = np.zeros((len(methods), len(methods)))
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i != j:
                        ari_key = f'ari_vs_{method2}'
                        if ari_key in metrics[method1]:
                            ari_matrix[i, j] = metrics[method1][ari_key]
            
            im = ax2.imshow(ari_matrix, cmap='coolwarm', aspect='auto')
            ax2.set_title('Adjusted Rand Index Between Methods')
            ax2.set_xticks(range(len(methods)))
            ax2.set_yticks(range(len(methods)))
            ax2.set_xticklabels(methods)
            ax2.set_yticklabels(methods)
            plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.metrics_dir, 'clustering_metrics_with_ari.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics).T
            metrics_df.to_csv(os.path.join(self.metrics_dir, 'clustering_metrics.csv'))
            
            logging.info(f"Clustering metrics with ARI saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error plotting clustering metrics: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = ClusterAnalyzer(
            data_path='data/processed/mental_health_scores.csv',
            output_dir='Clustering'
        )
        
        # Run the complete analysis pipeline
        analyzer.run_all_clustering()
        
        print(f"Analysis completed successfully. Check the '{OUTPUT_DIR}' directory for outputs.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Pipeline failed: {str(e)}")