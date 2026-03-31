import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class RobustRegression(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X_const = sm.add_constant(np.array(X))
        self.model_ = sm.RLM(np.array(y), X_const, M=sm.robust.norms.TukeyBiweight()).fit()
        return self

    def predict(self, X):
        X_const = sm.add_constant(np.array(X))
        return self.model_.predict(X_const)
