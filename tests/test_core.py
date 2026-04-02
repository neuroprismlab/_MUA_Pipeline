# tests/test_core.py
"""
Tests for FeatureVectorizer and MUA classes
"""

import warnings
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from mua_pipeline import FeatureVectorizer, MUA


class TestFeatureVectorizer:
    """Tests for FeatureVectorizer class"""

    @pytest.fixture
    def sample_3d_data(self):
        """Create sample 3D connectivity matrices"""
        np.random.seed(42)
        n_subjects, n_regions = 10, 20
        data = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            data[i] = (data[i] + data[i].T) / 2
            np.fill_diagonal(data[i], 1.0)
        return data

    @pytest.fixture
    def sample_2d_data(self):
        """Create sample 2D feature matrix"""
        np.random.seed(42)
        return np.random.randn(10, 190)

    def test_fit_3d_data(self, sample_3d_data):
        """Test fitting with 3D data"""
        vectorizer = FeatureVectorizer()
        vectorizer.fit(sample_3d_data)

        assert vectorizer.input_type_ == '3D'
        assert vectorizer.n_regions_ == 20
        assert vectorizer.n_features_ == 190

    def test_fit_2d_data(self, sample_2d_data):
        """Test fitting with 2D data"""
        vectorizer = FeatureVectorizer()
        vectorizer.fit(sample_2d_data)

        assert vectorizer.input_type_ == '2D'
        assert vectorizer.n_features_ == 190

    def test_transform_3d_data(self, sample_3d_data):
        """Test transforming 3D data to 2D"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_3d_data)

        assert transformed.shape == (10, 190)
        assert transformed.ndim == 2

    def test_transform_2d_data(self, sample_2d_data):
        """Test 2D data passes through unchanged"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_2d_data)

        np.testing.assert_array_equal(transformed, sample_2d_data)

    def test_inverse_transform_3d(self, sample_3d_data):
        """Test inverse transform reconstructs matrices"""
        vectorizer = FeatureVectorizer()
        transformed = vectorizer.fit_transform(sample_3d_data)
        reconstructed = vectorizer.inverse_transform(transformed)

        assert reconstructed.shape == sample_3d_data.shape
        for i in range(reconstructed.shape[0]):
            np.testing.assert_array_almost_equal(
                reconstructed[i], reconstructed[i].T
            )

    def test_error_on_non_square_matrices(self):
        """Test error raised for non-square matrices"""
        vectorizer = FeatureVectorizer()
        bad_data = np.random.randn(10, 20, 15)

        with pytest.raises(ValueError, match="must be square"):
            vectorizer.fit(bad_data)

    def test_error_on_unfitted_transform(self, sample_3d_data):
        """Test error when transforming before fitting"""
        vectorizer = FeatureVectorizer()

        with pytest.raises(ValueError, match="not fitted"):
            vectorizer.transform(sample_3d_data)


class TestMUA:
    """Tests for MUA class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for MUA"""
        np.random.seed(42)
        n_subjects = 50
        n_features = 100

        X = np.random.randn(n_subjects, n_features)
        y = np.sum(X[:, :10], axis=1) + np.random.randn(n_subjects) * 0.5

        return X, y

    def test_fit_filter_by_sign(self, sample_data):
        """Test MUA with filter_by_sign=True"""
        X, y = sample_data
        mua = MUA(filter_by_sign=True, selection_method='pvalue',
                  selection_threshold=0.05)
        mua.fit(X, y)

        assert hasattr(mua, 'correlations_')
        assert hasattr(mua, 'p_values_')
        assert hasattr(mua, 'selected_edges_')
        assert hasattr(mua, 'n_positive_')
        assert hasattr(mua, 'n_negative_')

    def test_fit_combined(self, sample_data):
        """Test MUA with filter_by_sign=False"""
        X, y = sample_data
        mua = MUA(filter_by_sign=False, selection_method='all')
        mua.fit(X, y)

        assert hasattr(mua, 'edge_weights_')
        assert not hasattr(mua, 'n_positive_')

    def test_transform_difference_scores(self, sample_data):
        """Test transform produces correct shape for difference scores"""
        X, y = sample_data
        mua = MUA(filter_by_sign=True, direction='difference')
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 1)

    def test_transform_positive_scores(self, sample_data):
        """Test transform produces correct shape for positive scores"""
        X, y = sample_data
        mua = MUA(filter_by_sign=True, direction='positive')
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 1)

    def test_transform_negative_scores(self, sample_data):
        """Test transform produces correct shape for negative scores"""
        X, y = sample_data
        mua = MUA(filter_by_sign=True, direction='negative')
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 1)

    def test_transform_combined_scores(self, sample_data):
        """Test transform produces correct shape for combined scores"""
        X, y = sample_data
        mua = MUA(filter_by_sign=False)
        scores = mua.fit_transform(X, y)

        assert scores.shape == (50, 1)

    def test_selection_methods(self, sample_data):
        """Test different selection methods"""
        X, y = sample_data

        mua_pval = MUA(selection_method='pvalue', selection_threshold=0.05)
        mua_pval.fit(X, y)
        n_selected_pval = np.sum(mua_pval.selected_edges_)

        mua_topk = MUA(selection_method='top_k', selection_threshold=20)
        mua_topk.fit(X, y)
        n_selected_topk = np.sum(mua_topk.selected_edges_)

        mua_all = MUA(selection_method='all')
        mua_all.fit(X, y)
        n_selected_all = np.sum(mua_all.selected_edges_)

        assert n_selected_topk == 20
        assert n_selected_all == 100
        assert 0 < n_selected_pval < 100

    def test_weighting_methods(self, sample_data):
        """Test different weighting methods"""
        X, y = sample_data

        methods = ['binary', 'correlation',
                   'squared_correlation', 'regression']

        for method in methods:
            mua = MUA(weighting_method=method, selection_method='all')
            mua.fit(X, y)
            scores = mua.transform(X)

            assert scores.shape[0] == 50
            assert not np.all(scores == 0)

    def test_sklearn_pipeline_integration(self, sample_data):
        """Test MUA works in sklearn pipeline"""
        X, y = sample_data

        pipeline = Pipeline([
            ('mua', MUA(filter_by_sign=True, selection_method='pvalue')),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (50,)

    def test_error_on_unfitted_transform(self, sample_data):
        """Test error when transforming before fitting"""
        X, y = sample_data
        mua = MUA()

        with pytest.raises(ValueError, match="not fitted"):
            mua.transform(X)

    def test_invalid_selection_method(self, sample_data):
        """Test error for invalid selection_method"""
        X, y = sample_data
        mua = MUA(selection_method='invalid')

        with pytest.raises(ValueError, match="selection_method"):
            mua.fit(X, y)

    def test_invalid_weighting_method(self, sample_data):
        """Test error for invalid weighting_method"""
        X, y = sample_data
        mua = MUA(weighting_method='invalid')

        with pytest.raises(ValueError, match="weighting_method"):
            mua.fit(X, y)

    def test_invalid_correlation_type(self, sample_data):
        """Test error for invalid correlation_type"""
        X, y = sample_data
        mua = MUA(correlation_type='invalid')

        with pytest.raises(ValueError, match="correlation_type"):
            mua.fit(X, y)

    def test_invalid_feature_aggregation(self, sample_data):
        """Test error for invalid feature_aggregation"""
        X, y = sample_data
        mua = MUA(feature_aggregation='invalid')

        with pytest.raises(ValueError, match="feature_aggregation"):
            mua.fit(X, y)

    def test_invalid_direction(self, sample_data):
        """Test error for invalid direction when filter_by_sign=True"""
        X, y = sample_data
        mua = MUA(filter_by_sign=True, direction='invalid')

        with pytest.raises(ValueError, match="direction"):
            mua.fit(X, y)

    def test_n_edges_stored(self, sample_data):
        """Test that n_edges_ is stored during fit"""
        X, y = sample_data
        mua = MUA()
        mua.fit(X, y)

        assert mua.n_edges_ == X.shape[1]

    def test_transform_wrong_n_edges(self, sample_data):
        """Test error when transform input has wrong number of edges"""
        X, y = sample_data
        mua = MUA()
        mua.fit(X, y)

        X_bad = np.random.randn(10, X.shape[1] + 5)
        with pytest.raises(ValueError, match="edges"):
            mua.transform(X_bad)

    def test_external_weights(self, sample_data):
        """Test MUA with external weights"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights)
        mua.fit(X, y)
        scores = mua.transform(X)

        np.testing.assert_array_almost_equal(mua.edge_weights_, ext_weights)
        assert scores.shape == (50, 1)

    def test_external_weights_selects_all_edges(self, sample_data):
        """Test external weights forces all edges selected"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights)
        mua.fit(X, y)

        assert np.all(mua.selected_edges_)

    def test_external_weights_skips_correlation(self, sample_data):
        """Test external weights skips correlation computation"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights)
        mua.fit(X, y)

        np.testing.assert_array_equal(
            mua.correlations_, np.zeros(X.shape[1]))
        np.testing.assert_array_equal(
            mua.p_values_, np.ones(X.shape[1]))

    def test_external_weights_none_raises(self, sample_data):
        """Test error when external weights not provided"""
        X, y = sample_data
        mua = MUA(weighting_method='external', external_weights=None)

        with pytest.raises(ValueError, match="external_weights must be provided"):
            mua.fit(X, y)

    def test_external_weights_wrong_length_raises(self, sample_data):
        """Test error when external weights has wrong length"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1] + 5)

        mua = MUA(weighting_method='external', external_weights=ext_weights)
        with pytest.raises(ValueError, match="must match number of edges"):
            mua.fit(X, y)

    def test_external_weights_warns_selection_method(self, sample_data):
        """Test warning when selection_method is not 'all' with external weights"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights,
                  selection_method='pvalue')

        with pytest.warns(UserWarning, match="selection_method and selection_threshold are ignored"):
            mua.fit(X, y)

    def test_external_weights_no_warn_selection_all(self, sample_data):
        """Test no warning when selection_method='all' with external weights"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights,
                  selection_method='all')

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mua.fit(X, y)

    def test_external_weights_with_filter_by_sign(self, sample_data):
        """Test external weights works with filter_by_sign"""
        X, y = sample_data
        ext_weights = np.random.randn(X.shape[1])

        mua = MUA(weighting_method='external', external_weights=ext_weights,
                  filter_by_sign=True, direction='difference')
        mua.fit(X, y)
        scores = mua.transform(X)

        assert hasattr(mua, 'pos_mask_')
        assert hasattr(mua, 'neg_mask_')
        assert scores.shape == (50, 1)

    def test_spearman_correlation(self, sample_data):
        """Test Spearman correlation produces different results than Pearson"""
        X, y = sample_data

        mua_p = MUA(correlation_type='pearson', selection_method='all')
        mua_s = MUA(correlation_type='spearman', selection_method='all')
        mua_p.fit(X, y)
        mua_s.fit(X, y)

        assert not np.allclose(mua_p.correlations_, mua_s.correlations_)

    def test_standardize_scores(self, sample_data):
        """Test standardized scores have mean~0 and std~1"""
        X, y = sample_data
        mua = MUA(standardize_scores=True, selection_method='all')
        mua.fit(X, y)
        scores = mua.transform(X)

        np.testing.assert_almost_equal(np.mean(scores), 0.0, decimal=5)
        np.testing.assert_almost_equal(np.std(scores), 1.0, decimal=1)


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_cpm_pipeline(self):
        """Test complete CPM pipeline"""
        np.random.seed(42)
        n_subjects, n_regions = 30, 20

        connectivity = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            connectivity[i] = (connectivity[i] + connectivity[i].T) / 2
            np.fill_diagonal(connectivity[i], 1.0)

        behavior = np.random.randn(n_subjects)

        pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(filter_by_sign=True, selection_method='pvalue')),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(connectivity, behavior)
        predictions = pipeline.predict(connectivity)

        assert predictions.shape == (n_subjects,)
        assert not np.all(np.isnan(predictions))

    def test_full_pnrs_pipeline(self):
        """Test complete PNRS pipeline"""
        np.random.seed(42)
        n_subjects, n_regions = 30, 20

        connectivity = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            connectivity[i] = (connectivity[i] + connectivity[i].T) / 2
            np.fill_diagonal(connectivity[i], 1.0)

        behavior = np.random.randn(n_subjects)

        pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(filter_by_sign=False, selection_method='all',
                        weighting_method='regression'))
        ])

        scores = pipeline.fit_transform(connectivity, behavior)

        assert scores.shape == (n_subjects, 1)
        assert not np.all(np.isnan(scores))

    def test_full_pipeline_with_external_weights(self):
        """Test complete pipeline with external weights"""
        np.random.seed(42)
        n_subjects, n_regions = 30, 20

        connectivity = np.random.randn(n_subjects, n_regions, n_regions)
        for i in range(n_subjects):
            connectivity[i] = (connectivity[i] + connectivity[i].T) / 2
            np.fill_diagonal(connectivity[i], 1.0)

        behavior = np.random.randn(n_subjects)

        n_features = n_regions * (n_regions - 1) // 2
        ext_weights = np.random.randn(n_features)

        pipeline = Pipeline([
            ('vectorize', FeatureVectorizer()),
            ('mua', MUA(weighting_method='external',
                        external_weights=ext_weights))
        ])

        scores = pipeline.fit_transform(connectivity, behavior)

        assert scores.shape == (n_subjects, 1)
        assert not np.all(np.isnan(scores))
