# Mental Health in Tech Analysis

This project analyzes mental health survey data from tech industry professionals using various data science techniques including exploratory data analysis, dimensionality reduction, and clustering.

## Prerequisites

1. Python 3.8+ installed
2. Git (optional, for cloning the repository)

## Setup Instructions

1. Create a new virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn sklearn scipy umap-learn kneed
```

## Project Structure

```
Mental Health Analysis/
│
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
│
├── plots/                      # Generated plots and visualizations
├── results/                    # Analysis results
├── metrics/                    # Clustering metrics
├── insights/                   # Generated insights
│
├── data_exploration.py         # Data exploration and visualization
├── data_processing.py         # Data preprocessing and feature engineering
├── dimensionality_reduction.py # Dimensionality reduction analysis
├── clustering.py              # Clustering analysis
└── run_analysis.py           # Main script to run the entire pipeline
```

## Data Setup

1. Create the required directories:
```bash
mkdir -p data/raw data/processed plots results metrics insights
```

2. Download the dataset:
   - Place the mental health survey data file (`mental-heath-in-tech-2016_20161114.csv`) in the `data/raw/` directory

## Running the Analysis

1. Single script execution:
   You can run individual scripts separately:
```bash
python data_exploration.py
python data_processing.py
python dimensionality_reduction.py
python clustering.py
```

2. Complete pipeline:
   To run the entire analysis pipeline:
```bash
python run_analysis.py
```

## Output Files

The analysis will generate several output files:

1. Data Exploration:
   - `plots/numerical_distributions.pdf`
   - `plots/data_summary.pdf`
   - `plots/correlation_matrix.png`
   - `plots/score_distributions.pdf`

2. Data Processing:
   - `data/processed/processed_data.csv`
   - `data/processed/mental_health_scores.csv`
   - `data/processed/score_statistics.csv`

3. Clustering Analysis:
   - `metrics/clustering_metrics.csv`
   - `metrics/clustering_metrics_with_ari.png`
   - `plots/cluster_comparison.png`
   - `insights/{method}_cluster_insights.txt`

## Troubleshooting

1. Missing directories:
   - Ensure all required directories exist before running the code
   - Run `python run_analysis.py` which will create necessary directories

2. Missing data file:
   - Verify the dataset is in `data/raw/mental-heath-in-tech-2016_20161114.csv`
   - Check file permissions

3. Package installation issues:
   - Try installing packages individually:
   ```bash
   pip install pandas
   pip install scikit-learn
   pip install umap-learn
   ```

4. Memory issues:
   - Close other applications
   - For large datasets, consider using a machine with more RAM

## Additional Notes

- The analysis pipeline is modular - you can run individual components separately
- All visualizations are saved automatically in their respective directories
- Check the logging output for progress and any potential errors
- The clustering analysis includes multiple methods (K-means, DBSCAN, Hierarchical)
- Results include both visualizations and detailed text insights

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
