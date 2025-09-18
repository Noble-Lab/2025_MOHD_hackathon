from typing import Dict, List
import numpy as np, pandas as pd
from base_imputer import BaseImputer

class GlobalMeanImputer(BaseImputer):
    """Predict each feature by its training mean for the target modality."""
    name = "global_mean"

    def fit(self, train: Dict[str, pd.DataFrame], input_modalities: List[str], target_modalities: List[str]):
        super().fit(train, input_modalities, target_modalities)
        self.per_feature_mean_ = {m: train[m].mean(axis=0) for m in target_modalities}
        return self

    def predict(self, inputs: Dict[str, pd.DataFrame], target_modalities: List[str]):
        assert self.fitted_
        preds = {}
        for m in target_modalities:
            idx = inputs[m].index
            cols = inputs[m].columns
            mu = self.per_feature_mean_[m].reindex(cols)
            preds[m] = pd.DataFrame(np.tile(mu.values, (len(idx), 1)), index=idx, columns=cols)
        return preds
