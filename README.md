# Mental Health in Tech Analysis Pipeline

## Project Overview
This project analyzes mental health in the tech industry using various data science techniques including data processing, exploration, dimensionality reduction, and clustering analysis.

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
├── Data Exploration/           # Data exploration outputs
│   ├── plots/                  # Visualization plots
│   └── results/                # Analysis results
│
├── Data Processing/            # Data processing outputs
│   ├── plots/                  # Visualization plots
│   ├── results/                # Processing results
│   └── visualizations/         # Additional visualizations
│
├── Dimensionality Reduction/   # Dimensionality reduction outputs
│   ├── plots/                  # 2D and 3D visualizations
│   └── results/                # Reduction results
│       ├── pca/               # PCA specific results
│       ├── tsne/              # t-SNE specific results
│       └── umap/              # UMAP specific results
│
├── Clustering/                 # Clustering analysis outputs
│   ├── plots/                  # Cluster visualizations
│   ├── results/                # Clustering results
│   ├── metrics/                # Performance metrics
│   └── insights/              # Cluster insights
│
└── scripts/
    ├── run_analysis.py        # Main execution script
    ├── data_exploration.py    # Data exploration script
    ├── data_processing.py     # Data preprocessing script
    ├── dimensionality_reduction.py  # Dimensionality reduction script
    └── clustering.py          # Clustering analysis script
```
## Requirements
- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  scikit-learn
  umap-learn
  matplotlib
  seaborn
  ```

## Usage
```bash
python run_analysis.py
```

## Note
Ensure the raw data file is placed in the `data/raw/` directory before running the analysis.


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

## Key Features
- Data Processing:
  - Handles missing values
  - Standardizes data formats
  - Calculates mental health scores
  
- Dimensionality Reduction:
  - PCA with variance analysis
  - t-SNE visualization
  - UMAP reduction
  - 2D and 3D visualizations

- Clustering Analysis:
  - Multiple clustering methods (K-means, DBSCAN, Hierarchical)
  - Optimal cluster selection
  - Cluster evaluation metrics
  - Detailed cluster insights


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



