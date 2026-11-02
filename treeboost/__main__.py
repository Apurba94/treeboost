"""Train on a CSV whose last column is the label: python -m treeboost data.csv"""
import csv
import sys

import numpy as np

from . import GradientBoostingClassifier, cross_validate


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    names, rows = rows[0], rows[1:]
    x = np.array([[float(v) for v in r[:-1]] for r in rows])
    y = np.array([r[-1] for r in rows])
    scores = cross_validate(lambda: GradientBoostingClassifier(n_estimators=150), x, y)
    print(f"5-fold accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    model = GradientBoostingClassifier(n_estimators=150).fit(x, y)
    for i in np.argsort(-model.importance_)[:10]:
        print(f"  {names[i]:<24}{model.importance_[i]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
