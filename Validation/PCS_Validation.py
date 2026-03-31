from sklearn.pipeline import Pipeline
import HCP
from mua_pipeline import FeatureVectorizer, MUA
from Preprocessing import preprocess
from visualization import plot_results
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Load data
    mat_file_path = "s_hcp_fc_noble_corr.mat"
    connectome_data = HCP.reconstruct_fc_matrix(mat_file_path)
    behavioral_data = HCP.extract_behavioral_data_1d(mat_file_path, 'test1')

    # Preprocessing
    connectome_clean, behavioral_clean, removed_indices = preprocess(
        connectome_data, behavioral_data
    )

    if __name__ == "__main__":
        # Load data
        mat_file_path = "s_hcp_fc_noble_corr.mat"
        connectome_data = HCP.reconstruct_fc_matrix(mat_file_path)
        behavioral_data = HCP.extract_behavioral_data_1d(mat_file_path, 'test1')

        # Preprocessing
        connectome_clean, behavioral_clean, removed_indices = preprocess(
            connectome_data, behavioral_data
        )

        n_subjects = connectome_clean.shape[0]
        n_nodes = connectome_clean.shape[1]
        triu_idx = np.triu_indices(n_nodes, k=1)
        n_edges = len(triu_idx[0])

        cv = 10

        X_discovery = pd.read_csv('X_discovery.csv', header=None).to_numpy()
        group_labels = pd.read_csv('group_labels.csv', header=None).squeeze().to_numpy()
        effectsize_R_results = pd.read_csv('New_r_to_d_results.csv', header=None).squeeze().to_numpy()
        pingouin_results = pd.read_csv('New_cohens_d_pingouin_results.csv', header=None).squeeze().to_numpy()

        # Generate random CSS (the values don't matter — we're testing computation)
        np.random.seed(42)
        random_CSS = np.random.randn(n_edges)
        External_CSS = random_CSS

        pcs_pipeline_1 = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(
                filter_by_sign=False,
                selection_method='all',
                weighting_method='external',
                external_weights=External_CSS,
                feature_aggregation='mean',
            ))
        ])

        pcs_scores_1 = pcs_pipeline_1.fit_transform(connectome_clean, behavioral_clean).flatten()
        # Manual computation
        X = np.array([connectome_clean[i][triu_idx] for i in range(n_subjects)])
        pcs_scores_2 = np.mean(X * random_CSS, axis=1)

        css_pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(
                filter_by_sign=False,
                selection_method='all',
                weighting_method='correlation',
                feature_aggregation='sum',
            ))
        ])

        css_pipeline.fit(X_discovery, group_labels)
        r = css_pipeline.named_steps['mua'].correlations_
        # Convert correlation to Cohen's d (general formula)
        n1 = np.sum(group_labels == 0)  # e.g., controls
        n2 = np.sum(group_labels == 1)  # e.g., patients

        # Correlation-to-Cohen’s d conversion
        h = (n1 + n2 - 2) / n1 + (n1 + n2 - 2) / n2
        CSS_cohen_d = ((r / np.sqrt(1 - r ** 2)) * np.sqrt(h))
        # CSS_cohen_d.to_csv('CSS_cohens_d_check_results.csv', index=False)


        plot_results(pcs_scores_1, pcs_scores_2,
                        title="Validation of the Calculated PCS using the configurable pipeline against the manual computation")
        plot_results(CSS_cohen_d, effectsize_R_results,
                        title="Validation of the Calculated CSS (Cohen's d) using the configurable pipeline against the effectsize R package ")
        plot_results(CSS_cohen_d, pingouin_results,
                        title="Validation of the Calculated CSS (Cohen's d) using the configurable pipeline against the Pingouin Python library ")
