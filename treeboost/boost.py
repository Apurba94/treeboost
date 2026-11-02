"""Histogram gradient boosting, the idea behind LightGBM and XGBoost's hist mode.

Features are bucketed into at most 255 bins. A split search then only sums gradients per
bin, which makes training much faster than sorting every feature at every node. The
leaf values and gains use second-order gradients, as in XGBoost.
"""
import numpy as np


class Binner:
    def __init__(self, max_bins=255):
        self.max_bins = max_bins

    def fit(self, x):
        self.edges = []
        for j in range(x.shape[1]):
            uniq = np.unique(x[:, j])
            if len(uniq) <= self.max_bins:
                self.edges.append((uniq[:-1] + uniq[1:]) / 2)
            else:
                qs = np.quantile(x[:, j], np.linspace(0, 1, self.max_bins + 1)[1:-1])
                self.edges.append(np.unique(qs))
        return self

    def transform(self, x):
        out = np.zeros(x.shape, np.uint8)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, x[:, j], side="left")
        return out


class Leaf:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class Split:
    __slots__ = ("feature", "bin", "threshold", "left", "right", "gain", "cover")

    def __init__(self, feature, bin_, threshold, left, right, gain, cover):
        self.feature, self.bin, self.threshold, self.left, self.right, self.gain, self.cover = feature, bin_, threshold, left, right, gain, cover


class HistTree:
    def __init__(self, depth, min_child, l2, gamma, binner, feature_rows=None):
        self.depth, self.min_child, self.l2, self.gamma, self.binner, self.features = depth, min_child, l2, gamma, binner, feature_rows

    def fit(self, b, g, h):
        self.gains = {}
        self.root = self._grow(b, g, h, np.arange(len(g)), 0)
        return self

    def _leaf(self, gs, hs):
        return Leaf(-gs / (hs + self.l2))

    def _grow(self, b, g, h, rows, depth):
        gs, hs = g[rows].sum(), h[rows].sum()
        if depth >= self.depth or len(rows) < 2:
            return self._leaf(gs, hs)
        parent = gs * gs / (hs + self.l2)
        best = (self.gamma + 1e-12, -1, -1)
        cols = self.features if self.features is not None else range(b.shape[1])
        for j in cols:
            n_bins = len(self.binner.edges[j]) + 1
            if n_bins < 2:
                continue
            bins = b[rows, j]
            gh = np.bincount(bins, g[rows], n_bins)
            hh = np.bincount(bins, h[rows], n_bins)
            gl, hl = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1]
            gr, hr = gs - gl, hs - hl
            gain = 0.5 * (gl * gl / (hl + self.l2) + gr * gr / (hr + self.l2) - parent)
            gain[(hl < self.min_child) | (hr < self.min_child)] = -np.inf
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), int(j), k)
        if best[1] < 0:
            return self._leaf(gs, hs)
        gain, j, k = best
        go = b[rows, j] <= k
        self.gains[j] = self.gains.get(j, 0.0) + gain
        left = self._grow(b, g, h, rows[go], depth + 1)
        right = self._grow(b, g, h, rows[~go], depth + 1)
        return Split(j, k, float(self.binner.edges[j][k]), left, right, gain, float(hs))

    def predict_binned(self, b):
        out = np.zeros(len(b))
        todo = [(self.root, np.arange(len(b)))]
        while todo:
            node, rows = todo.pop()
            if rows.size == 0:
                continue
            if isinstance(node, Leaf):
                out[rows] = node.value
                continue
            go = b[rows, node.feature] <= node.bin
            todo.append((node.left, rows[go]))
            todo.append((node.right, rows[~go]))
        return out


class _Boost:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4, min_child_weight=1.0, l2=1.0, gamma=0.0, subsample=1.0, colsample=1.0, max_bins=255, early_stopping=None, seed=0):
        self.n, self.lr, self.depth, self.min_child, self.l2, self.gamma = n_estimators, learning_rate, max_depth, min_child_weight, l2, gamma
        self.subsample, self.colsample, self.max_bins, self.early, self.seed = subsample, colsample, max_bins, early_stopping, seed

    def _tree(self, b, g, h, rng):
        n, d = b.shape
        rows = None
        cols = None
        if self.colsample < 1:
            cols = rng.choice(d, max(1, int(d * self.colsample)), replace=False)
        t = HistTree(self.depth, self.min_child, self.l2, self.gamma, self.binner, cols)
        if self.subsample < 1:
            rows = rng.choice(n, max(2, int(n * self.subsample)), replace=False)
            t.fit(b[rows], g[rows], h[rows])
        else:
            t.fit(b, g, h)
        for j, v in t.gains.items():
            self.gain_[j] += v
        return t

    def _binned(self, x):
        return self.binner.transform(np.asarray(x, float))

    @property
    def importance_(self):
        total = self.gain_.sum()
        return self.gain_ / total if total > 0 else self.gain_


class GradientBoostingRegressor(_Boost):
    def fit(self, x, y, eval_set=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.binner = Binner(self.max_bins).fit(x)
        b = self.binner.transform(x)
        self.gain_ = np.zeros(x.shape[1])
        self.base = float(y.mean())
        rng = np.random.default_rng(self.seed)
        f = np.full(len(y), self.base)
        h = np.ones(len(y))
        self.trees_ = []
        self.history_ = []
        best, wait, keep = np.inf, 0, 0
        if eval_set is not None:
            bv, yv = self._binned(eval_set[0]), np.asarray(eval_set[1], float)
            fv = np.full(len(yv), self.base)
        for i in range(self.n):
            t = self._tree(b, f - y, h, rng)
            f += self.lr * t.predict_binned(b)
            self.trees_.append(t)
            if eval_set is not None:
                fv += self.lr * t.predict_binned(bv)
                loss = float(((fv - yv) ** 2).mean())
                self.history_.append(loss)
                if loss < best - 1e-12:
                    best, wait, keep = loss, 0, i + 1
                else:
                    wait += 1
                    if self.early and wait >= self.early:
                        self.trees_ = self.trees_[:keep]
                        break
        return self

    def predict(self, x):
        b = self._binned(x)
        return self.base + self.lr * sum((t.predict_binned(b) for t in self.trees_), np.zeros(len(b)))


class GradientBoostingClassifier(_Boost):
    """Binary log-loss, or softmax with one tree per class for more than two classes."""

    def fit(self, x, y, eval_set=None):
        x = np.asarray(x, float)
        self.classes_, ys = np.unique(y, return_inverse=True)
        self.k = len(self.classes_)
        self.binner = Binner(self.max_bins).fit(x)
        b = self.binner.transform(x)
        self.gain_ = np.zeros(x.shape[1])
        rng = np.random.default_rng(self.seed)
        n = len(ys)
        outputs = 1 if self.k == 2 else self.k
        prior = np.bincount(ys, minlength=self.k) / n
        self.base = np.array([np.log(prior[1] / prior[0])]) if self.k == 2 else np.log(np.maximum(prior, 1e-12))
        f = np.tile(self.base, (n, 1))
        one = (ys == 1).astype(float)[:, None] if self.k == 2 else np.eye(self.k)[ys]
        self.trees_ = []
        self.history_ = []
        best, wait, keep = np.inf, 0, 0
        if eval_set is not None:
            bv = self._binned(eval_set[0])
            yv = np.searchsorted(self.classes_, eval_set[1])
            fv = np.tile(self.base, (len(yv), 1))
        for i in range(self.n):
            p = self._prob(f)
            g, h = p - one, np.maximum(p * (1 - p), 1e-6)
            round_ = []
            for c in range(outputs):
                t = self._tree(b, g[:, c], h[:, c], rng)
                f[:, c] += self.lr * t.predict_binned(b)
                round_.append(t)
            self.trees_.append(round_)
            if eval_set is not None:
                for c, t in enumerate(round_):
                    fv[:, c] += self.lr * t.predict_binned(bv)
                pv = self._full(self._prob(fv))
                loss = float(-np.log(np.maximum(pv[np.arange(len(yv)), yv], 1e-15)).mean())
                self.history_.append(loss)
                if loss < best - 1e-12:
                    best, wait, keep = loss, 0, i + 1
                else:
                    wait += 1
                    if self.early and wait >= self.early:
                        self.trees_ = self.trees_[:keep]
                        break
        return self

    def _prob(self, f):
        if self.k == 2:
            return 1 / (1 + np.exp(-f))
        z = f - f.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _full(self, p):
        return np.hstack([1 - p, p]) if self.k == 2 else p

    def decision_function(self, x):
        b = self._binned(x)
        f = np.tile(self.base, (len(b), 1))
        for round_ in self.trees_:
            for c, t in enumerate(round_):
                f[:, c] += self.lr * t.predict_binned(b)
        return f

    def predict_proba(self, x):
        return self._full(self._prob(self.decision_function(x)))

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def permutation_importance(model, x, y, metric=None, rounds=5, seed=0):
    """How much the score falls when one feature is shuffled. Works with any fitted model."""
    x = np.asarray(x, float)
    y = np.asarray(y)
    metric = metric or (lambda t, p: float((t == p).mean()))
    base = metric(y, model.predict(x))
    rng = np.random.default_rng(seed)
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        drops = []
        for _ in range(rounds):
            z = x.copy()
            z[:, j] = rng.permutation(z[:, j])
            drops.append(base - metric(y, model.predict(z)))
        out[j] = np.mean(drops)
    return out


def cross_validate(make, x, y, folds=5, seed=0, metric=None):
    x, y = np.asarray(x, float), np.asarray(y)
    metric = metric or (lambda t, p: float((t == p).mean()))
    order = np.random.default_rng(seed).permutation(len(y))
    out = []
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        out.append(metric(y[te], make().fit(x[tr], y[tr]).predict(x[te])))
    return out
