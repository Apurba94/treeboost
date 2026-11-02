# TreeBoost

Decision trees, random forests and histogram gradient boosting, written from scratch in NumPy. On tabular data these methods usually beat neural networks, and this code shows how they work.

```python
from treeboost import GradientBoostingClassifier, RandomForestClassifier

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, early_stopping=20)
model.fit(x_train, y_train, eval_set=(x_val, y_val))
print(model.predict(x_test), model.importance_)

forest = RandomForestClassifier(100).fit(x_train, y_train)
print(forest.oob_score_)
```

Or train on a CSV whose last column is the label:

```
python -m treeboost data.csv
```

## What is inside

| Model | Details |
|---|---|
| `DecisionTreeClassifier`, `DecisionTreeRegressor` | Exact CART with Gini and squared error, using one sorted sweep per feature |
| `RandomForestClassifier`, `RandomForestRegressor` | Bootstrap samples, random features at each split, and out-of-bag scores |
| `GradientBoostingClassifier`, `GradientBoostingRegressor` | Histogram binning and second-order gradients as in LightGBM and XGBoost, with log-loss or softmax, L2, gamma, row and column sampling, and early stopping |
| `permutation_importance`, `cross_validate` | Work with any fitted model |

## Why histograms?

A classic tree sorts every feature at every node. Binning each feature into at most 255 buckets replaces sorting with adding up gradients per bin, which is far faster and barely changes accuracy. Here is the core of a split, from `boost.py`:

```python
gain = 0.5 * (gl**2 / (hl + l2) + gr**2 / (hr + l2) - parent)
```

## How it is tested

When scikit-learn is installed, the trees are compared with its CART on the same data, and the forest and boosting models must come within 3 to 5 points of scikit-learn's on the breast cancer, wine and digits datasets. Other tests check non-linear problems, early stopping, regularisation and feature importance, and that noise features score low.

## Limits

It handles numeric features only, so encode categories first, and it has no handling of missing values. It runs on one CPU core and is much slower than LightGBM or XGBoost, so use those for large or production work.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
