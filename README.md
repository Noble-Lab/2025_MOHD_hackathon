# 2025 MOHD Imputation Hackathon

### Welcome to the 2025 MOHD Imputation Hackathon!

The immediate goal of this hackathon is to prototype and compare methods for **multi-omic data imputation** using CCLE data. In the long term, we hope to build a community interested in writing a **Nature Methods Registered Report**. Preparing such a Registered Report would benefit from dividing tasks across a group, and what we do here (creating baselines and designing a benchmarking pipeline) is a toy version of what that larger project could look like.

---

## Getting Started

### 1. Clone the repo

### 2. Set up your environment

Dependencies right now are quite minimal: `numpy`, `pandas`, `scipy`, `scikit-learn`.

---

## Data
The data splits are provided as pickled dictionaries.  
Each split file is a dictionary of `{modality_name: pandas.DataFrame}` where:

- **Rows = samples**  
- **Columns = features**

Available splits:

- `Data/ccle_split_train.pkl`
- `Data/ccle_split_val.pkl`
- `Data/ccle_split_test.pkl`

👉 Data for the hackathon can be found at this [link](https://drive.google.com/drive/folders/1w8rro2Vhynnf2uiqLOgXwOFBfgRk2SQm?usp=sharing)

---

## Code Structure

- `base_imputer.py` — defines the `BaseImputer` API (all models must inherit or follow this interface).
- `global_mean.py` — baseline model that predicts feature-wise global mean.
- `random_copy.py` — baseline model that copies feature values from a random training sample.
- `metrics.py` — evaluation metrics (MAE, RMSE, R², Spearman, Pearson).
- `Testing_model.ipynb` — notebook to test your model. (Note: please use this notebook locally to check your code, but *do not push changes* to it in the repo.)

---

## Writing Your Own Model

To add a new model:

1. Create a new file, e.g. `my_model.py`.
2. Define a class that implements the same API as the baselines:

```python
from base_imputer import BaseImputer

class MyModel(BaseImputer):
    name = "my_model"

    def fit(self, train, input_modalities, target_modalities):
        # train is dict[modality -> DataFrame]
        # store anything you need here
        return self

    def predict(self, inputs, target_modalities):
        # return dict[target_modality -> DataFrame]
        return {target_modality: preds_df}
```

3. Import it in the notebook and run with the shared `evaluate_model` function.

---

## Evaluation

We provide a common evaluation function (`evaluate_model`) to compare models consistently.  
It will:

- Fit your model on the training split.
- Predict the target modality on the validation/test split.
- Compute metrics: **MAE, RMSE, R², Spearman, Pearson**.
- Return both the results dictionary and the predictions DataFrame.

---

## Example

```python
from random_copy import RandomTrainingSampleImputer
from metrics import METRICS
from Testing_model import evaluate_model, load_split_dict

train = load_split_dict("Data/ccle_split_train.pkl")
test  = load_split_dict("Data/ccle_split_test.pkl")

target = list(train.keys())[0]
inputs = [m for m in train if m != target]

model = RandomTrainingSampleImputer(seed=42)
res, preds = evaluate_model(model, train, test, inputs, target)
print(res)
```

---

## Hackathon Schedule

### Day 1

**12:20–12:35 — Kickoff**

- Explain structure (3 groups).
- Explain CCLE data that is already processed.
- Show repo + input/output schema for metrics.

**12:35–1:30 — Work**

- Group A: start coding metrics and plots for given output structure.
- Group B: implement KNN imputation.
- Group C: implement LASSO imputation.

**1:30–1:40 — Sync**

- Group A shows sample metrics working on CCLE data + toy model.
- Group B and C share preliminary results from imputation methods.

**1:40–2:00 — Consortium update**

- Present early progress: evaluation pipeline in progress + baseline methods.

---

### Day 2

**10:45–10:50 — Quick regroup**

**10:50–11:50 — Work**

- Group A: finalize metrics and make demo plots on existing predictions.
- Group B and C: perform some hyperparameter optimization  
  (Nothing too complicated — just a small grid search).

**11:50–12:15 — Final sync & consortium prep**

- Group A runs metrics on top predictors from Group B and C.
- Combine outputs into single deck:
  - Slide 1: Hackathon goals
  - Slide 2: Evaluation metrics & demo results
  - Slide 3: Baseline/imputer comparisons
  - Slide 4: Next steps

---

