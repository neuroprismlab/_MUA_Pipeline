import numpy as np
from sklearn.pipeline import Pipeline
import HCP
from mua_pipeline import FeatureVectorizer, MUA
from Preprocessing import preprocess


def plot_validation(pipeline_values, manual_values, title=None):
    """
    Plot validation results comparing pipeline output against manual computation.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, norm

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Pipeline vs Manual
    ax1.scatter(manual_values, pipeline_values, alpha=0.7, s=30,
                edgecolors='k', linewidth=0.5, color='#4472C4', zorder=3)

    r_val, p_val = pearsonr(manual_values, pipeline_values)

    # Identity line (perfect match)
    val_min = min(manual_values.min(), pipeline_values.min())
    val_max = max(manual_values.max(), pipeline_values.max())
    margin = (val_max - val_min) * 0.05
    ax1.plot([val_min - margin, val_max + margin],
             [val_min - margin, val_max + margin],
             'r-', lw=2, alpha=0.8, zorder=2, label='Identity line')

    ax1.set_xlabel('Manual Computation', fontsize=11)
    ax1.set_ylabel('Pipeline Output', fontsize=11)

    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"

    ax1.text(0.05, 0.95, f'r = {r_val:.3f}\n{p_text}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Panel B: Difference distribution
    diff = pipeline_values - manual_values
    n, bins, patches = ax2.hist(diff, bins=20, edgecolor='k', linewidth=0.5,
                                alpha=0.8, color='#70AD47', density=True)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.8)

    mu, std = norm.fit(diff)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, 'k-', linewidth=1.5, alpha=0.8)

    max_abs_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    ax2.set_xlabel('Difference (Pipeline − Manual)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)

    ax2.text(0.95, 0.95,
             f'Max |diff| = {max_abs_diff:.2e}\nMean |diff| = {mean_diff:.2e}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax2.set_axisbelow(True)

    # Formatting
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=4, width=0.8)

    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    if title:
        fig.suptitle(title, y=0.98, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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

    # Verify FeatureVectorizer produces same ordering as triu_indices
    vectorizer = FeatureVectorizer()
    X_vectorized = vectorizer.fit_transform(connectome_clean)
    X_manual = np.array([connectome_clean[i][triu_idx] for i in range(n_subjects)])
    assert np.allclose(X_vectorized, X_manual), "FeatureVectorizer ordering mismatch!"
    print(f"FeatureVectorizer ordering verified: {X_vectorized.shape[1]} edges")

    # VALIDATION 1: PCS with External Weights (HCP Data + Random CSS)
    print("VALIDATION 1: PCS with External Weights Using HCP Data")
    print("_" * 50)

    # Generate random CSS (the values don't matter — we're testing computation)
    np.random.seed(42)
    random_CSS = np.random.randn(n_edges)

    pcs_pipeline = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=random_CSS,
            feature_aggregation='mean',
        ))
    ])

    pcs_scores = pcs_pipeline.fit_transform(connectome_clean, behavioral_clean).flatten()

    # Manual computation
    X = np.array([connectome_clean[i][triu_idx] for i in range(n_subjects)])
    manual_pcs = np.mean(X * random_CSS, axis=1)

    # Compare
    pcs_match = np.allclose(pcs_scores, manual_pcs, atol=1e-10)
    print(f"PCS scores match: {pcs_match}")

    # VALIDATION 2: CSS for Scale Variables (Simulated Data)
    print("VALIDATION 2: CSS for Scale Variables (Simulated Data)")
    print("_" * 50)

    np.random.seed(42)
    n_subjects = 200
    n_edges = 50

    # Simulated continuous behavioral variable (e.g., fluid intelligence)
    X_discovery = np.random.randn(n_subjects, n_edges)
    y_continuous = np.random.randn(n_subjects) * 10 + 100  # mean=100, sd=10

    X_validation = np.random.randn(100, n_edges)
    y_validation = np.random.randn(100) * 10 + 100

    # Step 1: Extract CSS (β-weights) using pipeline
    css_pipeline = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='regression',
            feature_aggregation='sum',
        ))
    ])

    css_pipeline.fit(X_discovery, y_continuous)
    CSS = css_pipeline.named_steps['mua'].edge_weights_

    # Step 1 manual: β = (Xᵀy) / (XᵀX) per edge
    manual_beta = np.zeros(n_edges)
    for e in range(n_edges):
        XtX = np.sum(X_discovery[:, e] ** 2)
        if XtX > 0:
            manual_beta[e] = np.dot(X_discovery[:, e], y_continuous) / XtX

    # Step 2: Apply CSS as external weights
    pcs_pipeline = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=CSS,
            feature_aggregation='mean',
        ))
    ])

    pcs_scores = pcs_pipeline.fit_transform(X_validation, y_validation).flatten()

    # Step 2 manual: PCSᵢ = (1/n) Σₑ (CSSₑ × FCᵢ,ₑ)
    manual_pcs = np.mean(X_validation * manual_beta, axis=1)

    # Compare
    beta_match = np.allclose(CSS, manual_beta, atol=1e-10)
    pcs_scale_match = np.allclose(pcs_scores, manual_pcs, atol=1e-10)
    print(f"CSS (β-weights) match:  {beta_match}")
    print(f"PCS scores match:       {pcs_scale_match}")

    # VALIDATION 3: CSS for Group Contrasts (Simulated Data)
    print("VALIDATION 3: CSS for Group Contrasts (Simulated Data)")
    print("_" * 50)

    np.random.seed(123)
    n1 = 120  # controls
    n2 = 80  # patients (unequal groups)
    n_edges = 50

    # Known group differences
    known_shift = np.random.randn(n_edges) * 0.5

    controls = np.random.randn(n1, n_edges)
    patients = np.random.randn(n2, n_edges) + known_shift

    X_discovery = np.vstack([controls, patients])
    group_labels = np.concatenate([np.zeros(n1), np.ones(n2)])

    X_validation = np.random.randn(100, n_edges)
    group_labels_val = np.concatenate([np.zeros(50), np.ones(50)])

    # Step 1: Extract correlations using pipeline
    css_pipeline = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='correlation',
            feature_aggregation='sum',
        ))
    ])

    css_pipeline.fit(X_discovery, group_labels)
    r = css_pipeline.named_steps['mua'].correlations_

    # Convert to Cohen's d (general formula)
    CSS_cohen_d = (r / np.sqrt(1 - r ** 2)) * np.sqrt(
        (n1 + n2 - 2) / n1 + (n1 + n2 - 2) / n2
    )

    # Step 1 manual: Cohen's d from group means and pooled SD
    mean_controls = np.mean(controls, axis=0)
    mean_patients = np.mean(patients, axis=0)
    var_controls = np.var(controls, axis=0, ddof=1)
    var_patients = np.var(patients, axis=0, ddof=1)
    pooled_sd = np.sqrt(
        ((n1 - 1) * var_controls + (n2 - 1) * var_patients) / (n1 + n2 - 2)
    )
    manual_d = (mean_patients - mean_controls) / pooled_sd

    # Step 2: Apply CSS as external weights
    pcs_pipeline = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=CSS_cohen_d,
            feature_aggregation='mean',
        ))
    ])

    pcs_scores = pcs_pipeline.fit_transform(X_validation, group_labels_val).flatten()

    # Step 2 manual: PCSᵢ = (1/n) Σₑ (dₑ × FCᵢ,ₑ)
    manual_pcs = np.mean(X_validation * manual_d, axis=1)

    # Compare
    d_match = np.allclose(CSS_cohen_d, manual_d, atol=1e-6)
    pcs_group_match = np.allclose(pcs_scores, manual_pcs, atol=1e-6)
    print(f"Cohen's d match:        {d_match}")
    print(f"PCS scores match:       {pcs_group_match}")

    # SUMMARY
    print("\n" + "_" * 50)
    print("SUMMARY")
    print("_" * 50)
    print(f"External weights (HCP) — PCS correct:        {pcs_match}")
    print(f"Scale variables (simulated) — CSS correct:    {beta_match}")
    print(f"Scale variables (simulated) — PCS correct:    {pcs_scale_match}")
    print(f"Group contrasts (simulated) — Cohen's d correct: {d_match}")
    print(f"Group contrasts (simulated) — PCS correct:    {pcs_group_match}")

    # PLOTS
    # Validation 1: External Weights (HCP)
    # Reload pipeline PCS and manual PCS from Validation 1
    vectorizer = FeatureVectorizer()
    X_hcp = vectorizer.fit_transform(connectome_clean)
    manual_pcs_hcp = np.mean(X_hcp * random_CSS, axis=1)

    pcs_pipeline_hcp = Pipeline([
        ('vectorize', FeatureVectorizer()),
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=random_CSS,
            feature_aggregation='mean',
        ))
    ])
    pipeline_pcs_hcp = pcs_pipeline_hcp.fit_transform(connectome_clean, behavioral_clean).flatten()

    plot_validation(pipeline_pcs_hcp, manual_pcs_hcp,
                    title="External Weights — Pipeline vs Manual PCS Using HCP")

    # Validation 2: Scale Variables (Simulated)
    # CSS match
    np.random.seed(42)
    n_sub_sim = 200
    n_e_sim = 50
    X_disc_sim = np.random.randn(n_sub_sim, n_e_sim)
    y_cont_sim = np.random.randn(n_sub_sim) * 10 + 100

    css_pipe_sim = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='regression',
            feature_aggregation='sum',
        ))
    ])
    css_pipe_sim.fit(X_disc_sim, y_cont_sim)
    pipe_beta = css_pipe_sim.named_steps['mua'].edge_weights_

    man_beta = np.zeros(n_e_sim)
    for e in range(n_e_sim):
        XtX = np.sum(X_disc_sim[:, e] ** 2)
        if XtX > 0:
            man_beta[e] = np.dot(X_disc_sim[:, e], y_cont_sim) / XtX

    plot_validation(pipe_beta, man_beta,
                    title="Compute CSS for a Scale Variable — Pipeline vs Manual CSS (β-weights)")

    # PCS match
    X_val_sim = np.random.randn(100, n_e_sim)
    y_val_sim = np.random.randn(100) * 10 + 100

    pcs_pipe_sim = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=pipe_beta,
            feature_aggregation='mean',
        ))
    ])
    pipe_pcs_sim = pcs_pipe_sim.fit_transform(X_val_sim, y_val_sim).flatten()
    man_pcs_sim = np.mean(X_val_sim * man_beta, axis=1)

    plot_validation(pipe_pcs_sim, man_pcs_sim,
                    title="Using Computed CSS for a Scale Variable — Pipeline vs Manual PCS")

    # Validation 3: Group Contrasts (Simulated)
    # Cohen's d match
    np.random.seed(123)
    n1_sim = 120
    n2_sim = 80
    n_e_sim = 50
    known_shift_sim = np.random.randn(n_e_sim) * 0.5

    ctrl_sim = np.random.randn(n1_sim, n_e_sim)
    pat_sim = np.random.randn(n2_sim, n_e_sim) + known_shift_sim

    X_disc_grp = np.vstack([ctrl_sim, pat_sim])
    grp_labels = np.concatenate([np.zeros(n1_sim), np.ones(n2_sim)])

    css_pipe_grp = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='correlation',
            feature_aggregation='sum',
        ))
    ])
    css_pipe_grp.fit(X_disc_grp, grp_labels)
    r_sim = css_pipe_grp.named_steps['mua'].correlations_

    pipe_d = (r_sim / np.sqrt(1 - r_sim ** 2)) * np.sqrt(
        (n1_sim + n2_sim - 2) / n1_sim + (n1_sim + n2_sim - 2) / n2_sim
    )

    mean_ctrl = np.mean(ctrl_sim, axis=0)
    mean_pat = np.mean(pat_sim, axis=0)
    var_ctrl = np.var(ctrl_sim, axis=0, ddof=1)
    var_pat = np.var(pat_sim, axis=0, ddof=1)
    pooled = np.sqrt(((n1_sim - 1) * var_ctrl + (n2_sim - 1) * var_pat) / (n1_sim + n2_sim - 2))
    man_d = (mean_pat - mean_ctrl) / pooled

    plot_validation(pipe_d, man_d,
                    title="Compute CSS for a Group Contrast — Pipeline vs Manual CSS (Cohen's d)")

    # PCS match
    X_val_grp = np.random.randn(100, n_e_sim)
    grp_labels_val = np.concatenate([np.zeros(50), np.ones(50)])

    pcs_pipe_grp = Pipeline([
        ('mua', MUA(
            filter_by_sign=False,
            selection_method='all',
            weighting_method='external',
            external_weights=pipe_d,
            feature_aggregation='mean',
        ))
    ])
    pipe_pcs_grp = pcs_pipe_grp.fit_transform(X_val_grp, grp_labels_val).flatten()
    man_pcs_grp = np.mean(X_val_grp * man_d, axis=1)

    plot_validation(pipe_pcs_grp, man_pcs_grp,
                    title="Using Computed CSS for a Group Contrast — Pipeline vs Manual PCS")