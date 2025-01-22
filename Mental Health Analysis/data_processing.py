import pandas as pd
import numpy as np
import logging
import os
from typing import Optional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    def __init__(self, input_path: str, output_path: str = 'data/processed'):
        self.input_path = input_path
        self.output_path = output_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load the raw dataset with specific parsing for employee count column."""
        try:
            employee_col = 'How many employees does your company or organization have?'
            dtype_dict = {employee_col: str}
            
            def convert_employee_count(val):
                try:
                    if pd.isna(val):
                        return '25-100'
                    
                    val = str(val).strip()
                    if not val:  # Empty string check
                        return '25-100'
                    
                    # Handle existing range format
                    if '-' in val and any(x.isdigit() for x in val.split('-')):
                        parts = val.split('-')
                        # If it's already a valid range, return it
                        if all(part.strip().isdigit() for part in parts):
                            return val
                    
                    # Convert date format
                    if '-' in val:
                        parts = [p.strip().lower() for p in val.split('-')]
                        months = ['Jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                'Jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                        
                        number = None
                        # Try to find the number part
                        for part in parts:
                            if part.isdigit():
                                number = int(part)
                                break
                            # Try to extract number from string
                            digits = ''.join(filter(str.isdigit, part))
                            if digits:
                                number = int(digits)
                                break
                        
                        if number is not None:
                            # Convert to range
                            if number < 25:
                                return '1-25'
                            elif number < 100:
                                return '25-100'
                            elif number < 500:
                                return '100-500'
                            else:
                                return '500-1000'
                    
                    # Handle single numbers
                    if val.isdigit():
                        num = int(val)
                        if num < 25:
                            return '1-25'
                        elif num < 100:
                            return '25-100'
                        elif num < 500:
                            return '100-500'
                        elif num < 1000:
                            return '500-1000'
                        else:
                            return '1000+'
                    
                    return '25-100'  # Default fallback
                    
                except Exception as e:
                    print(f"Error converting value '{val}': {str(e)}")
                    return '25-100'  # Safe fallback
            
            # Read CSV with specific parsing options
            self.df = pd.read_csv(
                self.input_path,
                dtype=dtype_dict,
                converters={
                    employee_col: convert_employee_count
                }
            )
            
            # Fix encoding issue with medical coverage column
            old_col = self.df.columns[15]  # Get the 16th column (0-based index)
            new_col = "Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?"
            self.df.rename(columns={old_col: new_col}, inplace=True)
            
            # Verify the column exists and has valid data
            if employee_col not in self.df.columns:
                raise ValueError(f"Column '{employee_col}' not found in the dataset")
            
            # Print debug info
            print("\nEmployee count unique values after loading:")
            print(sorted(self.df[employee_col].unique()))
            
            logging.info(f"Data loaded successfully: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def analyze_columns(self) -> None:
        """Analyze and log column statistics before processing."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            dtype = self.df[col].dtype
            logging.info(f"Column '{col}': dtype={dtype}, null_count={null_count}, unique_values={unique_count}")

    def handle_missing_data(self) -> None:
        """Handle missing values using mode imputation and convert strings to lowercase."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.analyze_columns()
        problematic_columns = []

        logging.debug(f"Starting missing data handling for {len(self.df.columns)} columns")
        
        for col in self.df.columns:
            try:
                logging.debug(f"Processing column '{col}'...")
                logging.debug(f"Column info: dtype={self.df[col].dtype}, null_count={self.df[col].isnull().sum()}")
                
                # Skip columns with all null values
                if self.df[col].isnull().all():
                    logging.warning(f"Skipping column '{col}': All values are null")
                    problematic_columns.append(col)
                    continue

                # Get valid values for the column
                valid_data = self.df[col].dropna()
                logging.debug(f"Valid data count for '{col}': {len(valid_data)}")

                if valid_data.empty:
                    logging.warning(f"Skipping column '{col}': No valid data available")
                    problematic_columns.append(col)
                    continue

                # Safe mode calculation with detailed error handling
                if valid_data.dtype == 'object':
                    mode_values = valid_data.mode()
                    if mode_values.empty:
                        logging.warning(f"No mode found for column '{col}'")
                        problematic_columns.append(col)
                        continue
                    
                    mode_value = mode_values.iloc[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    self.df[col] = self.df[col].astype(str).str.lower()
                    logging.debug(f"Filled '{col}' with mode value: {mode_value}")
                else:
                    mode_values = valid_data.mode()
                    mode_value = mode_values.iloc[0] if not mode_values.empty else valid_data.mean()
                    self.df[col] = self.df[col].fillna(mode_value)
                    logging.debug(f"Filled '{col}' with value: {mode_value}")

                # Verify the fill operation
                if self.df[col].isnull().any():
                    logging.warning(f"Column '{col}' still has {self.df[col].isnull().sum()} null values")
                else:
                    logging.info(f"Successfully processed column '{col}'")

            except Exception as e:
                logging.error(f"Error processing column '{col}': {str(e)}", exc_info=True)
                problematic_columns.append(col)
                continue

        # Drop problematic columns
        if problematic_columns:
            self.df.drop(columns=problematic_columns, inplace=True)
            logging.info(f"Dropped {len(problematic_columns)} problematic columns: {problematic_columns}")
        
        logging.info(f"Final column count: {len(self.df.columns)}")

    def clean_employee_count(self) -> None:
        """Clean the employee count column by ensuring proper text format and handling date formats."""
        column_name = 'How many employees does your company or organization have?'
        
        if (column_name not in self.df.columns):
            logging.warning(f"Employee count column '{column_name}' not found")
            return
            
        try:
            # Manual fix for date format
            def fix_date_format(val):
                val = str(val).strip()
                # Check for full date format (YYYY/MM/DD or YYYY-MM-DD)
                import re
                if re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', val):
                    # Extract month and day
                    parts = re.split(r'[-/]', val)
                    return f"{int(parts[1])}-{int(parts[2])}"
                return val

            # Apply manual fix first
            self.df[column_name] = self.df[column_name].apply(fix_date_format)
            
            # Print raw data for debugging
            print("\nRaw unique values before cleaning:")
            raw_values = self.df[column_name].unique()
            for val in sorted(raw_values):
                print(f"  {val} (type: {type(val)})")
            
            # Convert to string and clean
            self.df[column_name] = self.df[column_name].fillna('25-100')  # Default value for NaN
            self.df[column_name] = self.df[column_name].astype(str)
            
            def clean_value(val):
                original = val
                val = str(val).strip().lower()
                
                # Print each value being processed
                print(f"\nProcessing: {original} -> {val}")
                
                # Already in correct format (e.g., "26-100")
                if '-' in val and all(part.strip().isdigit() for part in val.split('-')):
                    print(f"  Already in range format: {val}")
                    return val
                
                # Check for date-like patterns
                date_patterns = [
                    r'\d+[/-]\w+',  # e.g., "25-Jun" or "25/Jun"
                    r'\w+[/-]\d+',  # e.g., "Jun-25" or "Jun/25"
                ]
                
                import re
                is_date = any(re.search(pattern, val) for pattern in date_patterns)
                
                if is_date:
                    print(f"  Detected date format: {val}")
                    # Extract any numbers from the string
                    numbers = re.findall(r'\d+', val)
                    if numbers:
                        num = int(numbers[0])
                        if num < 25:
                            result = '1-25'
                        elif num < 100:
                            result = '25-100'
                        elif num < 500:
                            result = '100-500'
                        else:
                            result = '500-1000'
                        print(f"  Converted to range: {result}")
                        return result
                
                # Handle single numbers
                if val.isdigit():
                    num = int(val)
                    if num < 25:
                        return '1-25'
                    elif num < 100:
                        return '25-100'
                    elif num < 500:
                        return '100-500'
                    elif num < 1000:
                        return '500-1000'
                    else:
                        return '1000+'
                
                print(f"  Using default range for: {val}")
                return '25-100'
            
            # Apply cleaning
            self.df[column_name] = self.df[column_name].apply(clean_value)
            
            # Print final results
            print("\nFinal unique values after cleaning:")
            final_values = sorted(self.df[column_name].unique())
            for val in final_values:
                print(f"  {val}")
            
        except Exception as e:
            logging.error(f"Error cleaning employee count column: {str(e)}", exc_info=True)
            raise

    def calculate_scores(self) -> None:
        """Calculate mental health related scores."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            # 1. Mental Health Support Score
            self.df['mental_health_support_score'] = (
                0.4 * self.df['Does your employer provide mental health benefits as part of healthcare coverage?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                0.3 * self.df['Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                0.3 * self.df['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
            )
            
            # 2. Mental Health Risk Score
            self.df['mental_health_risk_score'] = (
                self.df['Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?'].apply(lambda x: 0 if str(x).lower() == 'yes' else 1) +
                self.df['Have you ever sought treatment for a mental health issue from a mental health professional?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                self.df['Do you feel that your employer takes mental health as seriously as physical health?'].apply(lambda x: 0 if str(x).lower() == 'yes' else 1)
            )
            
            # 3. Workplace Satisfaction Score
            self.df['workplace_satisfaction_score'] = (
                0.4 * self.df['Do you feel that your employer takes mental health as seriously as physical health?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                0.3 * self.df['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                0.3 * self.df['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].apply(lambda x: 0 if str(x).lower() == 'no' else 1)
            )
            
            # 4. Treatment Engagement Score
            self.df['treatment_engagement_score'] = (
                0.5 * self.df['Have you ever sought treatment for a mental health issue from a mental health professional?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) +
                0.3 * self.df['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'].apply(lambda x: 0 if str(x).lower() == 'no' else 1) +
                0.2 * self.df['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
            )
            
            logging.info("Successfully calculated mental health scores")
            
        except Exception as e:
            logging.error(f"Error calculating scores: {str(e)}")
            raise

    def process_data(self) -> None:
        """Clean and process the data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            logging.info(f"Initial data shape: {self.df.shape}")
            self.df = self.df.copy()
            
            # Clean employee count column first
            self.clean_employee_count()
            
            # Remove columns with all missing values
            null_columns = self.df.columns[self.df.isnull().all()].tolist()
            if null_columns:
                self.df.drop(columns=null_columns, inplace=True)
                logging.info(f"Dropped columns with all null values: {null_columns}")
            
            # Drop rows with all missing values
            initial_rows = len(self.df)
            self.df.dropna(how='all', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            dropped_rows = initial_rows - len(self.df)
            logging.info(f"Dropped {dropped_rows} rows with all null values")
            
            # Handle missing values
            self.handle_missing_data()
            
            # Calculate mental health scores
            self.calculate_scores()
            
            logging.info(f"Final data shape: {self.df.shape}")
            
        except Exception as e:
            logging.error("Error in process_data", exc_info=True)
            raise

    def save_processed_data(self) -> None:
        """Save the processed dataset and scores."""
        if self.df is None:
            raise ValueError("No data to save. Process data first.")
            
        try:
            os.makedirs(self.output_path, exist_ok=True)
            
            # Save complete processed data
            output_file = os.path.join(self.output_path, 'processed_data.csv')
            self.df.to_csv(output_file, index=False)
            logging.info(f"Complete processed data saved to {output_file}")
            
            # Define and validate score columns
            score_columns = [
                'mental_health_support_score',
                'mental_health_risk_score',
                'workplace_satisfaction_score',
                'treatment_engagement_score'
            ]
            
            if not all(col in self.df.columns for col in score_columns):
                raise ValueError("Missing score columns. Ensure calculate_scores() was called.")
            
            # Exclude the "How many employees..." column and other non-score columns
            excluded_columns = ['How many employees does your company or organization have?']
            scores_df = self.df[score_columns].copy()
            
            # Add summary statistics
            stats_df = pd.DataFrame({
                'mean': scores_df.mean(),
                'median': scores_df.median(),
                'std': scores_df.std(),
                'min': scores_df.min(),
                'max': scores_df.max()
            })
            
            # Save scores and statistics
            scores_file = os.path.join(self.output_path, 'mental_health_scores.csv')
            stats_file = os.path.join(self.output_path, 'score_statistics.csv')
            
            scores_df.to_csv(scores_file, index=False, float_format='%.3f')
            stats_df.to_csv(stats_file, float_format='%.3f')
            
            # Log summary statistics
            logging.info("\nScore Statistics Summary:")
            for col in score_columns:
                logging.info(f"\n{col}:")
                logging.info(f"  Mean: {scores_df[col].mean():.3f}")
                logging.info(f"  Median: {scores_df[col].median():.3f}")
                logging.info(f"  Std: {scores_df[col].std():.3f}")
            
            logging.info(f"Scores saved to: {scores_file}")
            logging.info(f"Statistics saved to: {stats_file}")
            
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}", exc_info=True)
            raise

    def create_summary_visualizations(self, output_dir: str = 'plots') -> None:
        """Create and save summary visualizations in a PDF."""
        try:
            if self.df is None or self.df.empty:
                logging.warning("No data available for visualization")
                return
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, 'data_summary.pdf')
            
            # Prepare data for visualization
            numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            missing_data = self.df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            # Only create PDF if we have something to visualize
            if not numerical_cols and missing_data.empty:
                logging.warning("No data to visualize")
                return
            
            with PdfPages(pdf_path) as pdf:
                # Numerical distributions
                for col in numerical_cols:
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        plt.figure(figsize=(12, 6))
                        data.hist(bins=30)
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                
                # Missing values summary
                if not missing_data.empty:
                    plt.figure(figsize=(12, 6))
                    missing_data.plot(kind='bar')
                    plt.title('Missing Values by Column')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            
            logging.info(f"Summary visualizations saved to {pdf_path}")
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}", exc_info=True)
            # Don't raise the exception, allow the program to continue

    def plot_correlation_matrix(self, output_dir: str = 'plots') -> None:
        """Plot and save the correlation matrix of the dataset."""
        try:
            if self.df is None or self.df.empty:
                logging.warning("No data available for correlation matrix")
                return
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            corr_matrix = self.df.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, 'correlation_matrix.png')
            plt.savefig(output_file)
            plt.close()
            
            logging.info(f"Correlation matrix saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error plotting correlation matrix: {str(e)}", exc_info=True)

    def plot_score_distributions(self, output_dir: str = 'plots') -> None:
        """Plot and save the distributions of the calculated scores."""
        try:
            if self.df is None or self.df.empty:
                logging.warning("No data available for score distributions")
                return
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            score_columns = [
                'mental_health_support_score',
                'mental_health_risk_score',
                'workplace_satisfaction_score',
                'treatment_engagement_score'
            ]
            
            for col in score_columns:
                if col in self.df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(self.df[col], bins=20, kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    
                    output_file = os.path.join(output_dir, f'{col}_distribution.png')
                    plt.savefig(output_file)
                    plt.close()
                    
                    logging.info(f"Distribution plot for {col} saved to {output_file}")
                    
        except Exception as e:
            logging.error(f"Error plotting score distributions: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        processor = DataProcessor("data/raw/mental-heath-in-tech-2016_20161114.csv")
        processor.load_data()
        processor.analyze_columns()  
        processor.process_data()
        processor.save_processed_data()
        
        # Try to create visualizations but don't fail if there's an error
        try:
            processor.create_summary_visualizations()
            processor.plot_correlation_matrix()
            processor.plot_score_distributions()
        except Exception as e:
            logging.error(f"Failed to create visualizations: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error during data processing: {str(e)}")