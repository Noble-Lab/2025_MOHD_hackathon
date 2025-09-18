from typing import Dict, List
import numpy as np, pandas as pd
from base_imputer import BaseImputer

class RandomTrainingSampleImputer(BaseImputer):
    """Copy a random training row (per target modality)."""
    name = "random_copy"

    def fit(self, train: Dict[str, pd.DataFrame], input_modalities: List[str], target_modalities: List[str]):
        super().fit(train, input_modalities, target_modalities)
        self.train_ = train
        self.rng_ = np.random.RandomState(self.seed)
        self.train_idx_ = {m: list(train[m].index) for m in target_modalities}
        return self

    def predict(self, inputs: Dict[str, pd.DataFrame], target_modalities: List[str]):
        assert self.fitted_
        preds = {}
        for m in target_modalities:
            idx = list(inputs[m].index)
            cols = list(inputs[m].columns)
            chosen = self.rng_.choice(self.train_idx_[m], size=len(idx), replace=True)
            rows = self.train_[m].loc[chosen, cols].to_numpy()
            preds[m] = pd.DataFrame(rows, index=idx, columns=cols)
        return preds
