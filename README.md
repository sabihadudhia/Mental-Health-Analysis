# Mental Health Analysis (OSMI 2016)

## Problem Overview
This project simulates an HR analytics workflow to help a tech-oriented company design a pre-emptive mental health support program. Using the OSMI Mental Health in Tech 2016 survey dataset, the objective is to process noisy and high-dimensional responses, identify participant clusters, and provide actionable insights for intervention design.

Key challenges addressed:
- High dimensionality and mixed data types.
- Missing values in multiple survey columns.
- Non-standardized text responses (for example, gender labels and free-text fields).
- Need for interpretable cluster profiles and visual evidence.

## Development Approach
The analysis is implemented as a sequential pipeline orchestrated by `run_analysis.py`:

1. Data Processing (`data_processing.py`)
- Loads raw OSMI survey data.
- Cleans and standardizes categorical text responses.
- Handles missing values and encodes features.
- Engineers four HR-facing composite scores:
  - `mental_health_support_score`
  - `mental_health_risk_score`
  - `workplace_satisfaction_score`
  - `treatment_engagement_score`
- Saves processed outputs to `data/processed/`.

2. Data Exploration (`data_exploration.py`)
- Performs missing value analysis and distribution summaries.
- Produces descriptive visualizations and score distributions.
- Exports exploration plots to `Data Exploration/plots/`.

3. Dimensionality Reduction (`dimensionality_reduction.py`)
- Scales score features and applies:
  - PCA
  - t-SNE
  - UMAP
- Saves coordinate outputs and visualizations to `Dimensionality Reduction/results/`.

4. Clustering (`clustering.py`)
- Performs K-means, Hierarchical clustering, and an Optimal clustering configuration.
- Uses elbow/silhouette/BIC/hierarchical heuristics for cluster count selection.
- Computes comparison metrics (including silhouette and ARI).
- Generates cluster insights and visualizations in `Clustering/`.

## Execution Summary
Execution outcome:
- Pipeline completed successfully end-to-end.
- Processed dataset shape: `1433 x 67`.
- Clustering input score matrix shape: `1433 x 4`.

## Summary of Results
### Score-level Findings
From `Data Processing/results/score_statistics.csv`:
- `mental_health_support_score`: mean = 0.338, median = 0.400
- `mental_health_risk_score`: mean = 1.687, median = 2.000
- `workplace_satisfaction_score`: mean = 0.456, median = 0.400
- `treatment_engagement_score`: constant at 0.300

Interpretation:
- Risk indicators are generally elevated in the population.
- Support and satisfaction are moderate overall, with substantial subgroup variability.

### Clustering Performance
From `Clustering/metrics/clustering_metrics.csv`:
- K-means: silhouette = 0.469, clusters = 3
- Hierarchical: silhouette = 0.460, clusters = 3
- Optimal: silhouette = 0.691, clusters = 9

Pairwise agreement (ARI):
- K-means vs Hierarchical: 0.584
- K-means vs Optimal: 0.459
- Hierarchical vs Optimal: 0.552

Interpretation:
- The 9-cluster optimal model provides the strongest internal separation.
- ARI scores indicate moderate agreement between alternative partitioning strategies.

### Cluster Insight Highlights (Optimal, 9 Clusters)
From `Clustering/insights/optimal_cluster_insights.txt`:
- Largest high-risk segments:
  - Cluster 0 (25.19%): high risk (2.00), moderate support (0.37), better satisfaction (0.60)
  - Cluster 5 (22.68%): high risk (2.00), very low support (0.00), low satisfaction (0.26)
- Better-outcome segment:
  - Cluster 1 (12.35%): lower risk (0.97), good support (0.50), highest satisfaction (0.73)
- Vulnerable niche segments:
  - Cluster 4 (8.16%): very high support (0.80) but high risk (2.00), low satisfaction (0.29)
  - Cluster 3 (4.12%): no support (0.00), lower risk (0.97), low-moderate satisfaction (0.33)

Interpretation for HR leverage:
- Prioritize Cluster 5-like profiles for immediate intervention (low support + high risk + low satisfaction).
- Protect and replicate practices associated with Cluster 1-like profiles.
- Investigate mismatch cases (for example Cluster 4) where formal support exists but risk remains high.

## Output Locations
- Processed data: `data/processed/`
- Data processing stats and visuals: `Data Processing/results/`, `Data Processing/visualizations/`
- Exploration visuals: `Data Exploration/plots/`
- Dimensionality reduction artifacts: `Dimensionality Reduction/results/`
- Clustering insights, plots, metrics: `Clustering/insights/`, `Clustering/plots/`, `Clustering/metrics/`

## Visualizations:

Dimensionality Reduction: 

<img width="3010" height="2113" alt="pca_explained_variance" src="https://github.com/user-attachments/assets/559db8a0-c06b-4c24-969d-e963d6b04f4b" />

Clustering Insights:

<img width="4455" height="1902" alt="k_selection_analysis" src="https://github.com/user-attachments/assets/cb71e0ef-94ec-4b52-832e-f7f2e06a76f4" />

Clustering Analysis:

<img width="5965" height="6482" alt="kmeans_cluster_analysis" src="https://github.com/user-attachments/assets/095ac193-a811-4e5b-a8f9-a8018215779e" />

<img width="5965" height="6482" alt="hierarchical_cluster_analysis" src="https://github.com/user-attachments/assets/67c8722b-7240-4f96-9a57-cfe07075e3aa" />

<img width="5965" height="6482" alt="optimal_cluster_analysis" src="https://github.com/user-attachments/assets/e709bc0c-ddfe-449f-ab53-f483663bd6d8" />

## Recommended Next Improvements
- Recompute correlation matrix using numeric-only columns to remove plotting warning.
- Revisit `treatment_engagement_score` formulation (currently no variance).
- Add formal cluster stability checks across random seeds and bootstrap samples.
- Add a concise stakeholder dashboard consolidating cluster-specific HR actions.
 
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
