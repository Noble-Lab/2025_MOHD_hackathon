from typing import Dict, List
import pandas as pd

class BaseImputer:
    """Base API for cross-modal imputers."""
    name = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.fitted_ = False

    def fit(self, train: Dict[str, pd.DataFrame], input_modalities: List[str], target_modalities: List[str]):
        self.fitted_ = True
        return self

    def predict(self, inputs: Dict[str, pd.DataFrame], target_modalities: List[str]):
        raise NotImplementedError