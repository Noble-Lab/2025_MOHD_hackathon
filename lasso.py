from typing import Dict, List
import pandas as pd
from base_imputer import BaseImputer
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor

class LassoImputer(BaseImputer):
    name = "lasso"

    def __init__(self, alpha: float = 0.01, max_iter: int = 500, seed: int = 42):
        super().__init__(seed=seed)
        self.model = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=max_iter))
        self.mods = None

    def fit(self, train: Dict[str, pd.DataFrame], input_modalities: List[str], target_modalities: List[str]):
        super().fit(train, input_modalities, target_modalities)
        assert len(target_modalities) == 1
        t = target_modalities[0]
        self.mods = sorted(input_modalities)
        y = train[t]
        X = pd.concat([train[m].loc[y.index] for m in self.mods], axis=1)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, inputs: Dict[str, pd.DataFrame], target_modalities: List[str]):
        assert self.fitted_
        assert len(target_modalities) == 1
        t = target_modalities[0]
        idx = inputs[t].index
        mods = sorted([m for m in inputs if m != t])
        X = pd.concat([inputs[m].loc[idx] for m in mods], axis=1)
        Y_hat = self.model.predict(X.values)
        return {t: pd.DataFrame(Y_hat, index=idx, columns=inputs[t].columns)}
