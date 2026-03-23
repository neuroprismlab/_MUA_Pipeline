# examples/example_usage.py
"""
Example usage of the MUA Pipeline for CPM, PNRS, and PCS
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Import the custom modules
from mua_pipeline import FeatureVectorizer, MUA, plot_results


if __name__ == "__main__":

    # DATA LOADING SECTION
    # Replace these strings with your actual data
    functional_connectivity_matrices = 'your_connectivity_data'
    behavioral_measures = 'your_behavioral_data'
    cv = 10

    # CPM
    print("Original CPM")

    cpm_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=True,
            direction='difference',
            selection_method='pvalue',
            selection_threshold=0.05,
            weighting_method='binary',
            feature_aggregation='mean',
        )),
        ('regressor', LinearRegression())
    ])

    # Cross-validation
    cpm_scores = cross_val_score(
        cpm_pipeline, functional_connectivity_matrices,
        behavioral_measures, cv=cv)
    cpm_predictions = cross_val_predict(
        cpm_pipeline, functional_connectivity_matrices,
        behavioral_measures, cv=cv)

    print(f"CPM R² ({cv}-fold CV): "
          f"{cpm_scores.mean():.3f} ± {cpm_scores.std():.3f}")

    # Evaluation
    cpm_r, cpm_p = pearsonr(behavioral_measures, cpm_predictions)
    mae = mean_absolute_error(behavioral_measures, cpm_predictions)
    rmse = np.sqrt(mean_squared_error(behavioral_measures, cpm_predictions))
    r2 = r2_score(behavioral_measures, cpm_predictions)

    print(f"Correlation: r={cpm_r:.3f}, p={cpm_p:.2e}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # PNRS
    print("PNRS")

    pnrs_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='regression',
            feature_aggregation='sum',
        ))
    ])

    pnrs_scores = pnrs_pipeline.fit_transform(
        functional_connectivity_matrices, behavioral_measures)

    # Use scores directly as predictions
    pnrs = pnrs_scores.flatten()

    # Evaluation
    pnrs_r, pnrs_p = pearsonr(behavioral_measures, pnrs)
    print(f"Correlation: r={pnrs_r:.3f}, p={pnrs_p:.2e}")

    # PCS 
    print("PCS")

    # Load CSS (Connectome Summary Statistics) Matrix 
    css_path = 'your_CSS.csv'
    css_matrix = pd.read_csv(css_path, index_col=0).values

    # Extract upper triangle 
    upper_tri_indices = np.triu_indices(css_matrix.shape[0], k=1)
    css_weights = css_matrix[upper_tri_indices]
    
    pcs_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=css_weights,
            feature_aggregation='mean',
        ))
    ])

    pcs_scores = pcs_pipeline.fit_transform(
        functional_connectivity_matrices, behavioral_measures)

    # Use scores directly as predictions
    pcs = pcs_scores.flatten()

    # Evaluation
    pcs_r, pcs_p = pearsonr(behavioral_measures, pcs)
    print(f"Correlation: r={pcs_r:.3f}, p={pcs_p:.2e}")

    # Plot results
    plot_results(cpm_predictions, behavioral_measures, title="CPM")
    plot_results(pnrs, behavioral_measures, title="PNRS")
    plot_results(pcs, behavioral_measures, title="PCS")
