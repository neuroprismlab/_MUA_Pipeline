import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from mua_pipeline import RobustRegression


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.5
    return X, y


@pytest.fixture
def data_with_outliers():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.5
    # Inject outliers
    y[0] = 100
    y[1] = -100
    y[2] = 50
    return X, y


class TestRobustRegression:

    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        model = RobustRegression()
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, sample_data):
        X, y = sample_data
        model = RobustRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0]

    def test_predict_reasonable_values(self, sample_data):
        X, y = sample_data
        model = RobustRegression().fit(X, y)
        preds = model.predict(X)
        r = np.corrcoef(y, preds)[0, 1]
        assert r > 0.9, f"Expected r > 0.9, got {r:.3f}"

    def test_outlier_robustness(self, data_with_outliers):
        X, y = data_with_outliers
        model = RobustRegression().fit(X, y)
        # True coefficients are [2, -1.5, 0.5]
        # Intercept is at index 0, slopes at 1:
        coefs = model.model_.params[1:]
        assert abs(coefs[0] - 2.0) < 0.5, f"Expected ~2.0, got {coefs[0]:.3f}"
        assert abs(coefs[1] + 1.5) < 0.5, f"Expected ~-1.5, got {coefs[1]:.3f}"
        assert abs(coefs[2] - 0.5) < 0.5, f"Expected ~0.5, got {coefs[2]:.3f}"

    def test_single_feature(self):
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 3 * X[:, 0] + 1 + np.random.randn(50) * 0.3
        model = RobustRegression().fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == 50

    def test_sklearn_pipeline_compatible(self, sample_data):
        X, y = sample_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('robust', RobustRegression())
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape[0] == X.shape[0]

    def test_cross_val_score_runs(self, sample_data):
        X, y = sample_data
        model = RobustRegression()
        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(np.isfinite(scores)), f"Non-finite scores: {scores}"

    def test_model_attributes_after_fit(self, sample_data):
        X, y = sample_data
        model = RobustRegression().fit(X, y)
        assert hasattr(model, 'model_')
        assert model.model_.params is not None
        # 3 features + 1 intercept = 4 params
        assert len(model.model_.params) == 4

    def test_predict_before_fit_raises(self):
        model = RobustRegression()
        with pytest.raises(AttributeError):
            model.predict(np.array([[1, 2, 3]]))

    def test_tukey_biweight_used(self, sample_data):
        X, y = sample_data
        model = RobustRegression().fit(X, y)
        assert isinstance(model.model_.M, sm.robust.norms.TukeyBiweight)
