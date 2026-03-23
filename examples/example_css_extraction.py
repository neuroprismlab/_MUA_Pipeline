# examples/example_css_extraction.py
"""
Example usage of the MUA Pipeline for extracting CSS
(both for scale variables and group contrasts)
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Import the custom modules
from mua_pipeline import FeatureVectorizer, MUA

if __name__ == "__main__":

    # CSS Extraction for Scale Variables (β-weights)
  
    # Use this when your variable is continuous
    # (e.g., fluid intelligence, age, symptom severity)
    print("CSS Extraction: Scale Variable (β-weights)")

    # Replace these with your actual data
    scale_connectivity = 'your_connectivity_data'
    scale_behavioral = 'your_behavioral_data'
    scale_region_labels = 'your_region_labels'

    css_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='regression',
            feature_aggregation='sum',
        ))
    ])

    css_pipeline.fit(scale_connectivity, scale_behavioral)
    css_beta = css_pipeline.named_steps['mua'].edge_weights_

    # Save CSS as a symmetric matrix CSV
    vectorizer = css_pipeline.named_steps['vectorize']
    css_beta_matrix = vectorizer.inverse_transform(
        css_beta.reshape(1, -1))[0]
    pd.DataFrame(
        css_beta_matrix,
        index=scale_region_labels, columns=scale_region_labels
    ).to_csv('css_beta_weights.csv')

    print(f"Saved CSS (β-weights) to css_beta_weights.csv")

    # CSS Extraction for Group Contrasts (Cohen's d)
  
    # Use this when comparing two groups
    # (e.g., patients vs controls)
    # group_labels: 0 = controls, 1 = patients
    print("CSS Extraction: Group Contrast (Cohen's d)")

    # Replace these with your actual data
    group_connectivity = 'your_connectivity_data'
    group_labels = 'your_group_labels'
    group_region_labels = 'your_region_labels'

    css_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='correlation',
            feature_aggregation='sum',
        ))
    ])

    css_pipeline.fit(group_connectivity, group_labels)
    r = css_pipeline.named_steps['mua'].correlations_

    # Convert point-biserial r to Cohen's d
    n1 = np.sum(group_labels == 0)
    n2 = np.sum(group_labels == 1)
    css_cohen_d = (r / np.sqrt(1 - r ** 2)) * np.sqrt(
        (n1 + n2 - 2) / n1 + (n1 + n2 - 2) / n2
    )

    # Save CSS as a symmetric matrix CSV
    vectorizer = css_pipeline.named_steps['vectorize']
    css_cohen_d_matrix = vectorizer.inverse_transform(
        css_cohen_d.reshape(1, -1))[0]
    pd.DataFrame(
        css_cohen_d_matrix,
        index=group_region_labels, columns=group_region_labels
    ).to_csv('css_cohen_d.csv')

    print(f"Saved CSS (Cohen's d) to css_cohen_d.csv")
