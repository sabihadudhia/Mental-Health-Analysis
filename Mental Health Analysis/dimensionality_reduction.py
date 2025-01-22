import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DimensionalityReducer:
    def __init__(self, input_path: str = 'data/processed/mental_health_scores.csv'):
        """Initialize with the path to the scores CSV file."""
        self.input_path = input_path
        self.df: Optional[pd.DataFrame] = None
        self.scaled_data: Optional[np.ndarray] = None
        self.pca_result: Optional[np.ndarray] = None
        self.tsne_result: Optional[np.ndarray] = None
        self.umap_result: Optional[np.ndarray] = None
        self.output_dir = 'results/dimensionality_reduction'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for visualization."""
        self.df = data
        logging.info("Data set successfully for visualization")

    def load_and_prepare_data(self) -> None:
        """Load and prepare the mental health scores for dimensionality reduction."""
        try:
            # Load only the scores data
            self.df = pd.read_csv(self.input_path)
            
            # Verify we have the expected score columns
            expected_columns = [
                'mental_health_support_score',
                'mental_health_risk_score',
                'workplace_satisfaction_score',
                'treatment_engagement_score'
            ]
            
            missing_cols = [col for col in expected_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required score columns: {missing_cols}")
            
            # Scale the data
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(self.df[expected_columns])
            
            logging.info(f"Data loaded and scaled successfully: {self.scaled_data.shape}")
            
        except Exception as e:
            logging.error(f"Error loading and preparing data: {str(e)}")
            raise

    def perform_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA and return explained variance ratios."""
        if self.scaled_data is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
            
        try:
            # Initialize PCA
            pca = PCA()
            self.pca_result = pca.fit_transform(self.scaled_data)
            
            # Get explained variance ratios
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            logging.info("PCA completed successfully")
            return explained_variance, cumulative_variance
            
        except Exception as e:
            logging.error(f"Error performing PCA: {str(e)}")
            raise

    def perform_tsne(self, n_components: int = 3, perplexity: float = 30.0) -> np.ndarray:
        """Perform t-SNE dimensionality reduction."""
        if self.scaled_data is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
            
        try:
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42
            )
            self.tsne_result = tsne.fit_transform(self.scaled_data)
            logging.info("t-SNE completed successfully")
            return self.tsne_result
            
        except Exception as e:
            logging.error(f"Error performing t-SNE: {str(e)}")
            raise

    def perform_umap(self, n_components: int = 3) -> np.ndarray:
        """Perform UMAP dimensionality reduction."""
        if self.scaled_data is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
            
        try:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42
            )
            self.umap_result = reducer.fit_transform(self.scaled_data)
            logging.info("UMAP completed successfully")
            return self.umap_result
            
        except Exception as e:
            logging.error(f"Error performing UMAP: {str(e)}")
            raise

    def plot_dimensionality_reductions(self, pca_results: np.ndarray, 
                                     tsne_results: np.ndarray,
                                     umap_results: np.ndarray,
                                     original_data: pd.DataFrame) -> None:
        """Plot all dimensionality reduction results."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            methods = ['PCA', 't-SNE', 'UMAP']
            results = [pca_results, tsne_results, umap_results]
            
            for ax, method, result in zip(axes, methods, results):
                if result is not None and result.shape[1] >= 2:
                    scatter = ax.scatter(result[:, 0], result[:, 1],
                                    c=original_data['mental_health_risk_score'],
                                    cmap='viridis')
                    ax.set_title(f'{method} Projection')
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Component 2')
            
            plt.colorbar(scatter, ax=axes.ravel().tolist(),
                        label='Mental Health Risk Score')
            plt.tight_layout()
            
            # Save the comparison plot
            output_path = os.path.join(self.output_dir, 'dimensionality_reduction_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Dimensionality reduction comparison plot saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error plotting dimensionality reductions: {str(e)}")
            raise

    def plot_explained_variance(self, pca_result: Optional[np.ndarray] = None, output_dir: str = 'plots') -> None:
        """Plot PCA explained variance ratio."""
        if pca_result is None:
            raise ValueError("PCA result not provided")
            
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get variance ratios from PCA result
            explained_variance = np.var(pca_result, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Create the plot with higher quality settings
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(explained_variance_ratio) + 1),
                    cumulative_variance,
                    'bo-',
                    label='Cumulative Explained Variance',
                    linewidth=2)
            plt.bar(range(1, len(explained_variance_ratio) + 1),
                   explained_variance_ratio,
                   alpha=0.5,
                   label='Individual Explained Variance')
            plt.xlabel('Principal Components', fontsize=12)
            plt.ylabel('Explained Variance Ratio', fontsize=12)
            plt.title('PCA Explained Variance', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            # Save plot with high DPI
            plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save the actual values to CSV
            variance_df = pd.DataFrame({
                'Component': range(1, len(explained_variance_ratio) + 1),
                'Individual_Variance': explained_variance_ratio,
                'Cumulative_Variance': cumulative_variance
            })
            variance_df.to_csv(os.path.join(output_dir, 'pca_explained_variance.csv'),
                             index=False)
            
            logging.info("PCA explained variance plot and data saved successfully")
            
        except Exception as e:
            logging.error(f"Error plotting explained variance: {str(e)}")
            raise

    def plot_3d_reduction(self, reduction_type: str, reduction_result: np.ndarray, output_dir: str = 'plots') -> None:
        """Plot 3D visualization of the reduced data."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            if reduction_result is None:
                raise ValueError(f"No data available for {reduction_type}")
                
            if reduction_result.shape[1] != 3:
                raise ValueError(f"Expected 3 components, got {reduction_result.shape[1]}")
            
            title = f'{reduction_type.upper()} 3D Visualization'
            
            # Create high-quality 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(reduction_result[:, 0], 
                               reduction_result[:, 1], 
                               reduction_result[:, 2],
                               c=self.df['mental_health_risk_score'],
                               cmap='viridis',
                               s=50,
                               alpha=0.6)
            
            fig.colorbar(scatter, label='Mental Health Risk Score')
            
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.set_zlabel('Component 3', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Save plots from different angles
            for angle in [(45, 45), (0, 0), (90, 0)]:
                ax.view_init(elev=angle[0], azim=angle[1])
                plt.savefig(os.path.join(output_dir, 
                          f'{reduction_type.lower()}_3d_angle_{angle[0]}_{angle[1]}.png'),
                          dpi=300, bbox_inches='tight')
            
            plt.close()
            
            # Save the reduced coordinates
            reduced_df = pd.DataFrame(
                reduction_result,
                columns=[f'{reduction_type.lower()}_component_{i+1}' for i in range(3)]
            )
            reduced_df['mental_health_risk_score'] = self.df['mental_health_risk_score']
            reduced_df.to_csv(os.path.join(output_dir, f'{reduction_type.lower()}_coordinates.csv'),
                            index=False)
            
            logging.info(f"3D {reduction_type} plots and data saved successfully")
            
        except Exception as e:
            logging.error(f"Error plotting 3D visualization: {str(e)}")
            raise

    def create_summary_visualizations(self, output_dir: str = 'plots') -> None:
        """Create summary visualizations for the dimensionality reduction results."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot PCA explained variance
            self.plot_explained_variance(self.pca_result, output_dir)
            
            # Plot 3D t-SNE results
            self.plot_3d_reduction('tsne', self.tsne_result, output_dir)
            
            # Plot 3D UMAP results
            self.plot_3d_reduction('umap', self.umap_result, output_dir)
            
            logging.info("Summary visualizations created successfully")
            
        except Exception as e:
            logging.error(f"Error creating summary visualizations: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Initialize with the scores file
        reducer = DimensionalityReducer()
        
        # Load and prepare the data
        reducer.load_and_prepare_data()
        
        # Perform all dimensionality reductions
        pca_variance, pca_cumulative = reducer.perform_pca()
        tsne_result = reducer.perform_tsne()
        umap_result = reducer.perform_umap()
        
        # Plot dimensionality reductions
        reducer.plot_dimensionality_reductions(
            pca_results=reducer.pca_result,
            tsne_results=reducer.tsne_result,
            umap_results=reducer.umap_result,
            original_data=reducer.df
        )
        
        # Create summary visualizations
        reducer.create_summary_visualizations()
        
        print("Dimensionality reduction analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in dimensionality reduction: {str(e)}")
        print(f"Error: {str(e)}")
