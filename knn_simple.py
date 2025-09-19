from typing import Dict, List
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from base_imputer import BaseImputer

def _concat_inputs(split: Dict[str, pd.DataFrame], input_modalities: List[str]) -> np.ndarray:
    """
    Concatenate input modality matrices column-wise.
    Assumes inputs are already standardized. No scaling here.
    """
    if not input_modalities:
        raise ValueError("SimpleKNNImputer requires at least one input modality.")
    mats = [split[m].to_numpy() for m in input_modalities]
    return np.hstack(mats) if len(mats) > 1 else mats[0]

class SimpleKNNImputer(BaseImputer):
    """
    KNN imputer using concatenated INPUT modalities as the neighbor feature space.
    Predicts TARGET modality rows by **uniformly averaging** neighbors' TARGET rows.
    - Rows = samples, Columns = features for all DataFrames.
    - Inputs are assumed pre-standardized.
    """
    name = "knn_simple"

    def __init__(self, n_neighbors: int = 5, seed: int = 42):
        super().__init__(seed=seed)
        self.n_neighbors = n_neighbors
        self._input_modalities: List[str] = []
        self._nn: NearestNeighbors | None = None
        self._target_train: Dict[str, np.ndarray] = {}

    def fit(self, train: Dict[str, pd.DataFrame], input_modalities: List[str], target_modalities: List[str]):
        super().fit(train, input_modalities, target_modalities)
        if not input_modalities:
            raise ValueError("SimpleKNNImputer needs at least one input modality.")
        self._input_modalities = list(input_modalities)

        # Build neighbor space from inputs
        X_train = _concat_inputs(train, self._input_modalities)

        # Cache target matrices (numpy) for quick gather
        self._target_train = {m: train[m].to_numpy() for m in target_modalities}

        # Fit neighbor index (euclidean, uniform averaging later)
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self._nn.fit(X_train)
        return self

    def predict(self, inputs: Dict[str, pd.DataFrame], target_modalities: List[str]):
        assert self.fitted_, "Call fit() before predict()."
        # Build eval neighbor space
        X_eval = _concat_inputs(inputs, self._input_modalities)

        # Find neighbors in train
        dists, nbrs = self._nn.kneighbors(X_eval, n_neighbors=self.n_neighbors, return_distance=True)

        preds: Dict[str, pd.DataFrame] = {}
        for m in target_modalities:
            # Uniform average of neighbor rows from the training target matrix
            target_train = self._target_train[m]        # (n_train, n_feat)
            neighbor_rows = target_train[nbrs]          # (n_eval, k, n_feat)
            pred_mat = neighbor_rows.mean(axis=1)       # uniform averaging

            idx = inputs[m].index
            cols = inputs[m].columns
            preds[m] = pd.DataFrame(pred_mat, index=idx, columns=cols)
        return preds
