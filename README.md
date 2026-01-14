# Mental Health in Tech Analysis Pipeline

## Overview
Analyze mental health in the tech industry using data processing, exploration, dimensionality reduction, and clustering. The pipeline provides insights from survey data with visualizations and cluster analysis.

## Features
- Clean and preprocess survey data
- Exploratory Data Analysis (EDA) with visualizations
- Dimensionality reduction: PCA, t-SNE, UMAP
- Clustering analysis: K-Means, DBSCAN, Hierarchical
- Generate metrics and insights for clusters
- Modular pipeline, scripts can be run individually or together

## Technologies
- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- umap-learn, kneed

## Setup / Installation
1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
2. Install dependencies:
```bash 
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn kneed
```

3. Place the dataset (mental-heath-in-tech-2016_20161114.csv) in data/raw/

## Usage
- Run the full analysis pipeline:
```bash 
python run_analysis.py
```
- Or run individual scripts:
```bash
python data_exploration.py
python data_processing.py
python dimensionality_reduction.py
python clustering.py
```

## Project Structure
```bash
├── data/
│   ├── raw/                 # Raw dataset
│   └── processed/           # Processed data
├── Data Exploration/        # Plots and results
├── Data Processing/         # Preprocessing outputs
├── Dimensionality Reduction/
│   ├── plots/
│   └── results/
├── Clustering/
│   ├── plots/
│   ├── results/
│   ├── metrics/
│   └── insights/
├── scripts/
│   ├── run_analysis.py
│   ├── data_exploration.py
│   ├── data_processing.py
│   ├── dimensionality_reduction.py
│   └── clustering.py
└── README.md
```

## Output
- Visualizations saved in respective directories
- Metrics and cluster insights generated automatically
- Console logs show progress and key statistics
