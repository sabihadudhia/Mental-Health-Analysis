# Import required libraries for data analysis and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional
from matplotlib.backends.backend_pdf import PdfPages

# Set up logging configuration to track execution and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get output directory from environment variable or use default
OUTPUT_DIR = 'Data Exploration'

class DataExplorer:
    """
    A class to handle mental health data exploration and visualization.
    This class provides methods for loading, analyzing, and visualizing mental health survey data.
    """

    def __init__(self, data_path: str, output_dir: str = 'Data Exploration'):
        """
        Initialize the DataExplorer with a path to the data file.
        Args:
            data_path (str): Path to the CSV data file
            output_dir (str): Directory where output files will be saved
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        
        # Setup directory structure
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create directories
        for directory in [self.plots_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            
        logging.info(f"DataExplorer initialized with output directories in {self.output_dir}")
    
    def load_data(self) -> None:
        """Loads CSV data and logs basic dataset info"""
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Successfully loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def analyze_missing_values(self) -> None:
        """
        Analyzes and reports missing values in the dataset.
        Calculates the percentage of missing values for each column and prints the results.
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_values = self.df.isnull().sum()
        missing_percentages = (missing_values / len(self.df)) * 100
        
        print("\nMissing Values Analysis:")
        for col, missing in missing_percentages.items():
            if missing > 0:
                print(f"{col}: {missing:.2f}%")

    def summarize_data(self) -> None:
        """
        Generates comprehensive summary statistics for the dataset.
        - For numerical columns: provides count, mean, std, min, max, and quartiles
        - For categorical columns: provides value counts for top categories
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print("\nNumerical Columns Summary:")
        print(self.df[numerical_cols].describe())
        
        print("\nCategorical Columns Summary:")
        for col in categorical_cols:
            print(f"\n{col} value counts:")
            print(self.df[col].value_counts().head())

    def create_visualizations(self) -> None:
        """
        Creates and saves histograms for numerical columns in a PDF file.
        Args:
            output_dir (str): Directory where plots will be saved
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Set style for better visualizations
        sns.set_style("whitegrid")
        
        # Use absolute path for saving
        pdf_path = os.path.join(self.plots_dir, 'numerical_distributions.pdf')
        
        try:
            with PdfPages(pdf_path) as pdf:
                # Create individual plots for each numerical column
                numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                for col in numerical_cols:
                    plt.figure(figsize=(12, 6))
                    self.df[col].hist()
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
        
            logging.info(f"Saved numerical distributions to {pdf_path}")
        except Exception as e:
            logging.error(f"Error saving to {pdf_path}: {str(e)}")
            raise

    def create_summary_visualizations(self, filename: str = 'data_summary.pdf') -> None:
        """
        Creates a comprehensive PDF with distribution plots and missing values heatmap.
        Args:
            filename (str): Name of the output PDF file
        Raises:
            ValueError: If data hasn't been loaded yet
            Exception: If there's an error during plot creation
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            # Use absolute path for saving
            pdf_path = os.path.join(self.plots_dir, filename)
            
            with PdfPages(pdf_path) as pdf:
                # Distribution plots for numerical columns
                numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                for col in numerical_cols:
                    plt.figure(figsize=(14, 8))
                    plt.subplots_adjust(bottom=0.2)  # Add space for x-labels
                    sns.histplot(data=self.df, x=col, kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

                # Missing values heatmap
                plt.figure(figsize=(14, 10))
                plt.subplots_adjust(bottom=0.15)
                sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Heatmap')
                pdf.savefig(bbox_inches='tight')
                plt.close()

            logging.info(f"Saved summary visualizations to {pdf_path}")

        except Exception as e:
            logging.error(f"Error saving to {pdf_path}: {str(e)}")
            raise

    def plot_correlation_matrix(self, filename: str = 'correlation_matrix.png') -> None:
        """
        Generates and saves a correlation heatmap for numerical variables.
        Args:
            filename (str): Name of the output PNG file
        Raises:
            ValueError: If data hasn't been loaded yet
            Exception: If there's an error during plot creation
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            corr_matrix = self.df[numerical_cols].corr()

            plt.figure(figsize=(14, 12))
            plt.subplots_adjust(bottom=0.15, left=0.15)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # Use absolute path for saving
            output_path = os.path.join(self.plots_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Saved correlation matrix to {output_path}")

        except Exception as e:
            logging.error(f"Error saving to {output_path}: {str(e)}")
            raise

    def plot_score_distributions(self, filename: str = 'score_distributions.pdf') -> None:
        """
        Creates detailed visualizations of score distributions including histograms and box plots.
        Args:
            filename (str): Name of the output PDF file
        Raises:
            ValueError: If data hasn't been loaded yet
            Exception: If there's an error during plot creation
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            pdf_path = os.path.join(self.plots_dir, filename)
            with PdfPages(pdf_path) as pdf:
                # Individual distributions
                for col in self.df.columns:
                    plt.figure(figsize=(14, 8))
                    plt.subplots_adjust(bottom=0.2, left=0.15)
                    sns.histplot(data=self.df, x=col, kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

                # Box plots
                plt.figure(figsize=(16, 10))
                plt.subplots_adjust(bottom=0.2)
                self.df.boxplot()
                plt.title('Score Distributions (Box Plots)')
                plt.xticks(rotation=45)
                pdf.savefig(bbox_inches='tight')
                plt.close()

            logging.info(f"Saved score distributions to {pdf_path}")

        except Exception as e:
            logging.error(f"Error saving to {pdf_path}: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Main execution block that demonstrates the complete data exploration workflow:
    1. Initializes the DataExplorer with the data file path
    2. Loads the data from CSV
    3. Performs missing value analysis
    4. Generates statistical summaries
    5. Creates various visualizations for data analysis
    
    Note: All errors are caught and logged appropriately
    """
    try:
        # Initialize data explorer
        data_path = "data/raw/mental-heath-in-tech-2016_20161114.csv"
        explorer = DataExplorer(data_path, output_dir='Data Exploration')
        
        # Perform exploration
        explorer.load_data()
        explorer.analyze_missing_values()
        explorer.summarize_data()
        explorer.create_visualizations()
        explorer.create_summary_visualizations()
        explorer.plot_correlation_matrix()
        explorer.plot_score_distributions()
        
    except Exception as e:
        logging.error(f"Error during data exploration: {str(e)}")