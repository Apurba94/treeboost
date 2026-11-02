"""Decision trees and random forests, in NumPy."""
import numpy as np


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value):
        self.feature = -1
        self.threshold = 0.0
        self.left = self.right = None
        self.value = value


def best_split_classes(x, y, weights_by_class, n_classes, min_leaf, gini_parent):
    """Best threshold on one feature for Gini impurity, by a single sorted sweep."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(ys)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), ys] = 1
    left = np.cumsum(one_hot, axis=0)[:-1]
    total = left[-1] + one_hot[-1]
    right = total - left
    nl = np.arange(1, n, dtype=float)
    nr = n - nl
    gini_l = 1 - (left ** 2).sum(axis=1) / nl ** 2
    gini_r = 1 - (right ** 2).sum(axis=1) / nr ** 2
    score = gini_parent - (nl * gini_l + nr * gini_r) / n
    valid = (xs[1:] > xs[:-1]) & (nl >= min_leaf) & (nr >= min_leaf)
    if not valid.any():
        return -np.inf, 0.0
    score = np.where(valid, score, -np.inf)
    i = int(np.argmax(score))
    return float(score[i]), float((xs[i] + xs[i + 1]) / 2)


def best_split_mse(x, y, min_leaf, parent_sse):
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(ys)
    s = np.cumsum(ys)[:-1]
    total = s[-1] + ys[-1]
    nl = np.arange(1, n, dtype=float)
    nr = n - nl
    gain = s ** 2 / nl + (total - s) ** 2 / nr - total ** 2 / n
    valid = (xs[1:] > xs[:-1]) & (nl >= min_leaf) & (nr >= min_leaf)
    if not valid.any():
        return -np.inf, 0.0
    gain = np.where(valid, gain, -np.inf)
    i = int(np.argmax(gain))
    return float(gain[i]), float((xs[i] + xs[i + 1]) / 2)


class _Tree:
    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=None, seed=0):
        self.max_depth = max_depth if max_depth is not None else 10 ** 6
        self.min_leaf, self.min_split, self.max_features = min_samples_leaf, min_samples_split, max_features
        self.rng = np.random.default_rng(seed)

    def _features(self, d):
        m = self.max_features
        if m is None:
            return np.arange(d)
        if m == "sqrt":
            m = max(1, int(np.sqrt(d)))
        return self.rng.choice(d, min(m, d), replace=False)

    def predict_values(self, x):
        x = np.asarray(x, float)
        out = np.zeros((len(x),) + np.shape(self.root.value))
        idx = [(self.root, np.arange(len(x)))]
        while idx:
            node, rows = idx.pop()
            if rows.size == 0:
                continue
            if node.feature < 0:
                out[rows] = node.value
                continue
            go = x[rows, node.feature] <= node.threshold
            idx.append((node.left, rows[go]))
            idx.append((node.right, rows[~go]))
        return out

    def depth(self, node=None):
        node = node or self.root
        return 0 if node.feature < 0 else 1 + max(self.depth(node.left), self.depth(node.right))

    def leaves(self, node=None):
        node = node or self.root
        return 1 if node.feature < 0 else self.leaves(node.left) + self.leaves(node.right)


class DecisionTreeClassifier(_Tree):
    def fit(self, x, y):
        x = np.asarray(x, float)
        self.classes_, ys = np.unique(y, return_inverse=True)
        self.k = len(self.classes_)
        self.importance_ = np.zeros(x.shape[1])
        self.n = len(ys)
        self.root = self._grow(x, ys, 0)
        total = self.importance_.sum()
        if total > 0:
            self.importance_ /= total
        return self

    def _grow(self, x, y, depth):
        counts = np.bincount(y, minlength=self.k).astype(float)
        node = Node(counts / counts.sum())
        if depth >= self.max_depth or len(y) < self.min_split or (counts > 0).sum() == 1:
            return node
        gini = 1 - ((counts / len(y)) ** 2).sum()
        best, feature, threshold = -1e-9, -1, 0.0       # a split with no gain now can still enable one later, as in XOR
        for f in self._features(x.shape[1]):
            score, t = best_split_classes(x[:, f], y, None, self.k, self.min_leaf, gini)
            if score > best:
                best, feature, threshold = score, int(f), t
        if feature < 0:
            return node
        self.importance_[feature] += max(best, 0.0) * len(y) / self.n
        go = x[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(x[go], y[go], depth + 1)
        node.right = self._grow(x[~go], y[~go], depth + 1)
        return node

    def predict_proba(self, x):
        return self.predict_values(x)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class DecisionTreeRegressor(_Tree):
    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.importance_ = np.zeros(x.shape[1])
        self.n = len(y)
        self.root = self._grow(x, y, 0)
        total = self.importance_.sum()
        if total > 0:
            self.importance_ /= total
        return self

    def _grow(self, x, y, depth):
        node = Node(float(y.mean()))
        if depth >= self.max_depth or len(y) < self.min_split or np.ptp(y) == 0:
            return node
        best, feature, threshold = -1e-9, -1, 0.0
        for f in self._features(x.shape[1]):
            gain, t = best_split_mse(x[:, f], y, self.min_leaf, None)
            if gain > best:
                best, feature, threshold = gain, int(f), t
        if feature < 0:
            return node
        self.importance_[feature] += max(best, 0.0) / self.n
        go = x[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(x[go], y[go], depth + 1)
        node.right = self._grow(x[~go], y[~go], depth + 1)
        return node

    def predict(self, x):
        return self.predict_values(x)


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", seed=0):
        self.n, self.max_depth, self.min_leaf, self.max_features, self.seed = n_estimators, max_depth, min_samples_leaf, max_features, seed

    def fit(self, x, y):
        x = np.asarray(x, float)
        self.classes_, ys = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.seed)
        n = len(ys)
        self.trees_, votes = [], np.zeros((n, len(self.classes_)))
        seen = np.zeros(n)
        for i in range(self.n):
            rows = rng.integers(0, n, n)
            t = DecisionTreeClassifier(self.max_depth, self.min_leaf, 2, self.max_features, int(rng.integers(1 << 31)))
            t.fit(x[rows], self.classes_[ys[rows]])
            self.trees_.append(t)
            out = np.setdiff1d(np.arange(n), rows)
            p = np.zeros((len(out), len(self.classes_)))
            p[:, np.searchsorted(self.classes_, t.classes_)] = t.predict_proba(x[out])
            votes[out] += p
            seen[out] += 1
        used = seen > 0
        self.oob_score_ = float((votes[used].argmax(axis=1) == ys[used]).mean())
        imp = np.mean([t.importance_ for t in self.trees_], axis=0)
        self.importance_ = imp / max(imp.sum(), 1e-12)
        return self

    def predict_proba(self, x):
        out = np.zeros((len(x), len(self.classes_)))
        for t in self.trees_:
            out[:, np.searchsorted(self.classes_, t.classes_)] += t.predict_proba(x)
        return out / len(self.trees_)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features=None, seed=0):
        self.n, self.max_depth, self.min_leaf, self.max_features, self.seed = n_estimators, max_depth, min_samples_leaf, max_features, seed

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n):
            rows = rng.integers(0, len(y), len(y))
            self.trees_.append(DecisionTreeRegressor(self.max_depth, self.min_leaf, 2, self.max_features, int(rng.integers(1 << 31))).fit(x[rows], y[rows]))
        return self

    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees_], axis=0)
