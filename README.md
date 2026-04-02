# A Configurable Pipeline for Mass Univariate Aggregation (MUA) Methods
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. A Configurable Pipeline for Mass Univariate Aggregation Methods

We present a unified, flexible, and accessible configurable pipeline that can be used for implementing CPM (Shen et al., 2017), PNRS (Byington et al., 2023), PCS (Libedinsky et al., 2024, 2025), and many new MUA configurations facilitated by user-specified parameters. The pipeline's core is built around two classes—**FeatureVectorizer** and **MUA**—both of which are engineered to be fully compatible with the scikit-learn ecosystem.

## 2. Installation

```bash
pip install git+https://github.com/neuroprismlab/_MUA_Pipeline.git
```

## 3. Project Structure

```
_MUA_Pipeline/
├── mua_pipeline/                        # Main package directory
│   ├── __init__.py                      # Package initialization
│   ├── core.py                          # Core classes: FeatureVectorizer and MUA
│   ├── RobustRegression.py              # Scikit-learn-compatible RobustRegression wrapper
│   ├── preprocessing.py                 # Data format auto-detection and missing data removal
│   └── visualization.py                 # Plotting and visualization utilities
├── examples/
│   ├── CPM_PNRS_PCS_CSS_tutorial.ipynb  # Notebook walkthrough tutorial of CPM, PNRS, PCS, and CSS derivation with MUA
│   └── example_usage.py                 # Example usage of MUA with a python script
│   ├── example_css_extraction           # Example usage of CSS extraction
├── validation/                          # Validation scripts and reference comparisons
│   ├── CPM_Validation.m                 # MATLAB CPM validation
│   ├── PNRS_Validation.m                # MATLAB PNRS validation
│   ├── PCS_CSS_Validation.py            # PCS computation validation, CSS derivation (Cohen's d) validation
├── setup.py                             # Package installation configuration
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 4. The Implementation Details

### 4.1 FeatureVectorizer Class
The FeatureVectorizer class handles both 2D and 3D input data. When handling 3D connectivity input (n subjects × m regions × m regions), it vectorizes the data by extracting the off-diagonal upper triangular elements. If the input is already in 2D feature format, the data is left unmodified. The class also supports inverse transformation, enabling reconstruction of full symmetric matrices from feature vectors. In this manner, our configurable pipeline can be used to apply MUA methods on neuroimaging data (connectivity matrices) and any other feature-outcome data.

### 4.2 MUA Class
The MUA class is our main Python class that extends scikit-learn's BaseEstimator and TransformerMixin classes to ensure compatibility with standard machine learning pipelines; the MUA class operates through three steps:

1. **Feature Selection:** Determines which features are included based on their association with behavior (e.g., p-value thresholding, top-k selection, or all features)
2. **Feature Weighting:** Specifies how selected features are weighted (e.g., binary, correlation-based, regression-derived weights, or externally provided weights)
3. **Feature Aggregation:** Defines how weighted features are combined into summary scores (e.g., sum or mean aggregation, with optional sign-based partitioning into positive and negative networks)

### 4.2.1 MUA Class Parameters
The MUA class employs a set of configurable parameters organized across the three computational steps described above that enable flexible implementation of established methods while facilitating exploration of novel approaches.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **filter_by_sign** | `True` | Separate positive/negative features (CPM-style) |
| | `False` | Keep all features together (single weighted score) |
| **direction** | `'difference'` | Positive network score minus negative network score as a single predictor |
| | `'positive'` | Positive network score only |
| | `'negative'` | Negative network score only |
| | *Ignored* | When `filter_by_sign=False` |
| **selection_method** | `'all'` | Use all features |
| | `'pvalue'` | Select features with p < α |
| | `'top_k'` | Select the top k features by absolute correlation |
| **selection_threshold** | `float (0, 1)` | 'pvalue' method: p-value threshold |
| | `positive integer` | 'top_k' method: number of features |
| | *Ignored* | For 'all' method |
| **weighting_method** | `'binary'` | ±1 based on correlation sign |
| | `'correlation'` | The strength of the correlation |
| | `'squared_correlation'` | r² preserving sign |
| | `'regression'` | Feature-specific regression coefficients |
| | `'external'` | Use externally provided edge weights (e.g., CSS) |
| **external_weights** | `array-like (n_features,)` | External edge-wise weights applied when `weighting_method='external'` |
| **correlation_type** | `'pearson'` | Linear correlation |
| | `'spearman'` | Rank-based correlation |
| **feature_aggregation** | `'sum'` | The sum of the weighted features |
| | `'mean'` | The mean of the weighted features |
| **standardize_scores** | `True` | Z-score normalize final scores |
| | `False` | Keep raw scores |

### 4.3 Prediction Strategy
MUA methods can serve different research goals depending on their configuration:

- **Score-based association** (e.g., PNRS, PCS): The aggregated score is used without a final regression step to quantify brain-behavior associations or connectivity pattern alignment.
- **Behavioral prediction** (e.g., CPM): A final regression step is applied to produce predicted values of the outcome variable.

Users can integrate any scikit-learn-compatible regression method directly into the pipeline via the `'regressor'` step. For robust regression matching the Yale MATLAB CPM implementation, a custom `RobustRegression()` wrapper is provided in the repository.

```python
from mua_pipeline import FeatureVectorizer, MUA
from mua_pipeline.RobustRegression import RobustRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# With final regression (prediction)
pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(...)),
    ('regressor', LinearRegression())  # or RobustRegression()
])

# Without final regression (score-based)
pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(...))
])
```

## 5. Tutorial: Getting Started

See [examples](examples/) directory for complete notebook tutorials.

### 5.1 Importing the Pipeline Components

```python
from mua_pipeline import FeatureVectorizer, MUA, preprocess, plot_results
from mua_pipeline.robust_regression import RobustRegression
```

### 5.2 Preprocessing Your Data
Before building a pipeline, you can use the **Preprocessing** module to handle data formatting and missing data removal. The `preprocess` function automatically detects the orientation of your data and converts it to the standard format expected by the pipeline:

* **3D connectivity matrices** are converted to `(n_subjects, n_regions, n_regions)` regardless of whether the input is `(n_regions, n_regions, n_subjects)`, `(n_regions, n_subjects, n_regions)`, or already in the standard format.
* **2D feature matrices** are converted to `(n_subjects, n_features)` even if provided as `(n_features, n_subjects)`.
* **Behavioral data** is similarly standardized.

```python
# Raw data — may have missing values and non-standard orientation
connectivity_matrices = ...   # e.g., shape (n_regions, n_regions, n_subjects)
behavioral_scores = ...       # e.g., shape (n_subjects,)

# Clean the data: auto-detects format, removes missing subjects, returns standard format
clean_connectivity, clean_behavioral, removed_indices = preprocess(
    connectivity_matrices,
    behavioral_scores,
    missing_strategy='any',   # Remove subjects with zeros, NaNs, or Infs
    verbose=True
)
```

The `missing_strategy` parameter controls what counts as missing data:

| Strategy | Removes subjects with |
| :--- | :--- |
| `'zero'` | Behavioral values equal to 0 |
| `'nan'` | NaN values |
| `'inf'` | Inf or -Inf values |
| `'any'` | Any of the above (default) |


### 5.3 Implementing CPM (Shen et al., 2017)
CPM identifies connectivity patterns that predict individual differences in behavior through sign-based aggregation. It uses binary weights, p-value-based feature selection, and splits features into positive and negative networks. A final linear regression maps a single summary score to the behavioral outcome.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Build the CPM pipeline (using robust regression to match Yale MATLAB implementation)
cpm_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=True,           # Separate positive/negative networks
        direction='difference',        # mean(pos) - mean(neg), matches original CPM
        selection_method='pvalue',     # p-value thresholding
        selection_threshold=0.05,      # p < 0.05
        weighting_method='binary',     # Binary weights (+1/−1)
        correlation_type='pearson',    # Pearson correlation
        feature_aggregation='mean',    # Mean of selected features
    )),
    ('regressor', RobustRegression())  # Robust regression (matches Yale MATLAB)
    # Or use LinearRegression() for ordinary least squares (matches Finn tutorial)
])

# Cross-validation
cpm_scores = cross_val_score(cpm_pipeline, brain_data, behavior, cv=10)
cpm_predictions = cross_val_predict(cpm_pipeline, brain_data, behavior, cv=10)
print(f"CPM R² (10-fold CV): {cpm_scores.mean():.3f} ± {cpm_scores.std():.3f}")

# Evaluation
cpm_r, cpm_p = pearsonr(behavior, cpm_predictions)
mae = mean_absolute_error(behavior, cpm_predictions)
rmse = np.sqrt(mean_squared_error(behavior, cpm_predictions))
r2 = r2_score(behavior, cpm_predictions)
```

With `filter_by_sign=True`,`direction='difference'`, and `feature_aggregation='mean'` the MUA transformer outputs a single column representing `mean(pos_edges) - mean(neg_edges)`, matching the original MATLAB CPM implementation. 

### 5.4 Implementing PNRS (Byington et al., 2023)
PNRS assesses brain-behavior associations through weighted aggregation, where each feature contributes proportionally to its univariate association strength. PNRS uses regression-derived weights, includes all features, and produces a single combined score.

```python
# Build the PNRS pipeline (score-based, no final regression)
pnrs_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,              # Single combined score
        selection_method='all',            # Use all features
        weighting_method='regression',     # Regression-derived β weights
        feature_aggregation='sum',         # Sum of weighted features
    ))
    # No regressor — PNRS produces an association score, not a prediction
])

pnrs_scores = pnrs_pipeline.fit_transform(brain_data, behavior)

# Evaluation: correlate scores with behavior
pnrs_r, pnrs_p = pearsonr(behavior, pnrs_scores.flatten())
```

#### Using PNRS as a Predictor with a Final Regression Step
PNRS scores can also be used as input to a final regression step for prediction (Mooney et al., 2024):

```python
# PNRS with final regression
pnrs_pred_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='regression',
        feature_aggregation='sum',
    )),
    ('regressor', LinearRegression())  # Final regression step
])

# Cross-validation
pnrs_scores = cross_val_score(pnrs_pred_pipeline, brain_data, behavior, cv=10)
pnrs_predictions = cross_val_predict(pnrs_pred_pipeline, brain_data, behavior, cv=10)
print(f"PNRS R² (10-fold CV): {pnrs_scores.mean():.3f} ± {pnrs_scores.std():.3f}")

# Evaluation
pnrs_r, pnrs_p = pearsonr(behavior, pnrs_predictions)
```


### 5.5 Implementing PCS (Libedinsky et al., 2024, 2025)
PCS measures how similar an individual's brain functional connectivity pattern is to the connectivity pattern associated with a specific disorder or behavior. It uses pre-computed connectome summary statistics (CSS) as external weights.

#### Applying Pre-computed CSS

```python
import numpy as np

# Load pre-computed CSS matrix from a discovery study
CSS_matrix = ...  # regions × regions matrix
# Extract upper triangle to match vectorized FC edges
CSS = CSS_matrix[np.triu_indices(n_nodes, k=1)]

# Build the PCS pipeline
pcs_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='external',       # Use externally provided weights
        external_weights=CSS,              # Pre-computed CSS vector
        feature_aggregation='mean',        # Weighted mean (PCS formula)
    ))
])

pcs_scores = pcs_pipeline.fit_transform(brain_data, behavior)
pcs_results = pcs_scores.flatten()
```

#### Deriving CSS from a Discovery Sample

**Scale variables** (regression-derived β-weights):

```python
# Step 1: Derive CSS (β-weights) from discovery sample
css_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='regression',
        feature_aggregation='sum',
    ))
])

css_pipeline.fit(brain_data_discovery, behavior_discovery)
CSS = css_pipeline.named_steps['mua'].edge_weights_

# Save CSS as a symmetric matrix CSV
vectorizer = css_pipeline.named_steps['vectorize']
CSS_matrix = vectorizer.inverse_transform(CSS.reshape(1, -1))[0]
pd.DataFrame(CSS_matrix, index=region_labels, columns=region_labels).to_csv('css_beta_weights.csv')

# Step 2: Apply CSS to compute PCS in validation sample
pcs_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='external',
        external_weights=CSS,
        feature_aggregation='mean',
    ))
])
pcs_scores = pcs_pipeline.fit_transform(brain_data_validation, behavior_validation)
```

**Group contrasts** (correlation-to-Cohen's d conversion):

```python
# Step 1: Derive correlations from discovery sample
css_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='correlation',
        feature_aggregation='sum',
    ))
])
css_pipeline.fit(brain_data_discovery, group_labels_discovery)
r = css_pipeline.named_steps['mua'].correlations_

# Convert correlation to Cohen's d
n1 = np.sum(group_labels_discovery == 0)  # e.g., controls
n2 = np.sum(group_labels_discovery == 1)  # e.g., patients
CSS_cohen_d = (r / np.sqrt(1 - r**2)) * np.sqrt(
    ((n1 + n2 - 2) / n1) + ((n1 + n2 - 2) / n2)
)

# Save CSS as a symmetric matrix CSV
vectorizer = css_pipeline.named_steps['vectorize']
CSS_cohen_d_matrix = vectorizer.inverse_transform(CSS_cohen_d.reshape(1, -1))[0]
pd.DataFrame(CSS_cohen_d_matrix, index=region_labels, columns=region_labels).to_csv('css_cohen_d.csv')

# Step 2: Apply CSS to compute PCS in validation sample
pcs_pipeline = Pipeline([
    ('vectorize', FeatureVectorizer()),
    ('mua', MUA(
        filter_by_sign=False,
        selection_method='all',
        weighting_method='external',
        external_weights=CSS_cohen_d,
        feature_aggregation='mean',
    ))
])
pcs_scores = pcs_pipeline.fit_transform(brain_data_validation, group_labels_validation)
```

> **Note:** The built-in correlation-to-Cohen's d conversion computes unadjusted effect sizes for two-group contrasts. For datasets requiring confound correction, multi-group contrasts, or other custom CSS, users can derive weights externally and supply them via the `external_weights` parameter. 


### 5.6 Visualizing the Results
The **Visualization** module provides the `plot_results` function for visualizing predicted vs. observed behavioral scores:

```python
plot_results(cpm_predictions, behavioral_scores, title="CPM")
plot_results(pnrs_scores.flatten(), behavioral_scores, title="PNRS")
```

## 6. Reference and Documentation

For comprehensive details, theoretical background, and extensive explanations regarding the use and validation of this pipeline, please refer to our paper:

**Mass Univariate Aggregation Methods for Machine Learning in Neuroscience**
Fatemeh Doshvargar, Fabricio Cravo, Hallee Shearer, Stephanie Noble
**DOI:** [https://doi.org/10.5281/zenodo.18436701]

### References
* Shen X, Finn ES, Scheinost D, et al. Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nat Protoc. 2017;12(3):506-518. doi:10.1038/nprot.2016.178
* Finn ES, Shen X, Scheinost D, et al. Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nat Neurosci. 2015;18(11):1664-1671. doi:10.1038/nn.4135
* Byington N, Grimsrud G, Mooney MA, et al. Polyneuro risk scores capture widely distributed connectivity patterns of cognition. Dev Cogn Neurosci. 2023;60:101231. doi:10.1016/j.dcn.2023.101231
* Mooney MA, Hermosillo RJM, Feczko E, et al. Cumulative effects of resting-state connectivity across all brain networks significantly correlate with ADHD symptoms. J Neurosci. 2024;44(10):e1202232023. doi:10.1523/JNEUROSCI.1202-23.2023
* Libedinsky I, Helwegen K, Guerrero Simón L, et al. Quantifying brain connectivity signatures by means of polyconnectomic scoring. bioRxiv. 2023. doi:10.1101/2023.09.26.559327
* Libedinsky I, Helwegen K, Boonstra J, et al. Polyconnectomic scoring of functional connectivity patterns across eight neuropsychiatric and three neurodegenerative disorders. Biol Psychiatry. 2025;97(11):1045-1058. doi:10.1016/j.biopsych.2024.10.007
* YaleMRRC. Connectome-based Predictive Modeling (CPM) [Code repository]. https://github.com/YaleMRRC/CPM
* DCAN-Labs. BWAS: Polyneuro Risk Score [Code repository]. https://github.com/DCAN-Labs/BWAS
* dutchconnectomelab. PCS Toolbox [Code repository]. https://github.com/dutchconnectomelab/pcs-toolbox
