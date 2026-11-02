import unittest

import numpy as np

try:
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier as SkForest
    from sklearn.tree import DecisionTreeClassifier as SkTree, DecisionTreeRegressor as SkReg
except ImportError:
    SkTree = None

from treeboost import Binner, DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, cross_validate, permutation_importance


def rings(n=1200, seed=0, noise=0.08):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    r = rng.integers(0, 3, n)
    x = np.column_stack([(r + 1) * np.cos(a), (r + 1) * np.sin(a)]) + rng.normal(0, noise, (n, 2))
    return np.column_stack([x, rng.normal(size=(n, 3))]), r


def wave(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 4))
    return x, np.sin(x[:, 0]) * 2 + x[:, 1] ** 2 * 0.5 + rng.normal(0, 0.1, n)


def split(x, y, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(len(y) * 0.7)
    return x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]]


def acc(m, x, y):
    return float((m.predict(x) == y).mean())


class TestTree(unittest.TestCase):
    def test_learns_xor(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, float)
        y = np.array([0, 1, 1, 0] * 5)
        self.assertEqual(acc(DecisionTreeClassifier().fit(x, y), x, y), 1)

    def test_string_labels(self):
        m = DecisionTreeClassifier().fit([[0], [1]], ["no", "yes"])
        self.assertEqual(m.predict([[0.1], [0.9]]).tolist(), ["no", "yes"])

    def test_depth_and_leaf_limits(self):
        x, y = rings()
        self.assertLessEqual(DecisionTreeClassifier(max_depth=3).fit(x, y).depth(), 3)
        big = DecisionTreeClassifier(min_samples_leaf=50).fit(x, y)
        self.assertLess(big.leaves(), DecisionTreeClassifier().fit(x, y).leaves())

    def test_regressor_fits_a_step(self):
        x = np.arange(20.0)[:, None]
        y = (x[:, 0] > 9) * 5.0
        self.assertEqual(DecisionTreeRegressor(max_depth=1).fit(x, y).root.threshold, 9.5)

    def test_constant_target_and_duplicates(self):
        self.assertEqual(DecisionTreeRegressor().fit(np.ones((5, 2)), np.arange(5.0)).leaves(), 1)
        self.assertEqual(DecisionTreeClassifier().fit(np.ones((5, 2)), [0, 1, 0, 1, 0]).leaves(), 1)

    def test_importance_finds_the_real_features(self):
        x, y = rings()
        m = DecisionTreeClassifier(max_depth=8).fit(x, y)
        self.assertGreater(m.importance_[:2].sum(), 0.8)
        self.assertAlmostEqual(m.importance_.sum(), 1)

    @unittest.skipIf(SkTree is None, "scikit-learn is not installed")
    def test_same_tree_as_scikit_learn(self):
        """With every feature searched, an exact CART has one best answer up to ties."""
        for seed in range(6):
            x, y = rings(300, seed)
            x = x[:, :3]
            a = DecisionTreeClassifier(max_depth=4).fit(x, y).predict(x)
            b = SkTree(max_depth=4, random_state=0).fit(x, y).predict(x)
            self.assertGreater((a == b).mean(), 0.97)
        x, y = wave(400)
        a = DecisionTreeRegressor(max_depth=5).fit(x, y).predict(x)
        b = SkReg(max_depth=5, random_state=0).fit(x, y).predict(x)
        self.assertLess(np.abs(a - b).mean(), 0.02)


class TestForest(unittest.TestCase):
    def test_beats_a_single_tree(self):
        x, y = rings(noise=0.35)
        xt, yt, xv, yv = split(x, y)
        tree = acc(DecisionTreeClassifier().fit(xt, yt), xv, yv)
        forest = acc(RandomForestClassifier(60).fit(xt, yt), xv, yv)
        self.assertGreater(forest, tree)

    def test_out_of_bag_estimates_test_score(self):
        x, y = rings(noise=0.35)
        xt, yt, xv, yv = split(x, y)
        m = RandomForestClassifier(200).fit(xt, yt)       # each out-of-bag vote sees about a third of the trees
        self.assertAlmostEqual(m.oob_score_, acc(m, xv, yv), delta=0.04)

    def test_probabilities_sum_to_one(self):
        x, y = rings(300)
        p = RandomForestClassifier(10).fit(x, y).predict_proba(x)
        self.assertTrue(np.allclose(p.sum(axis=1), 1))

    def test_a_rare_class_missing_from_a_bootstrap_is_handled(self):
        x = np.arange(30.0)[:, None]
        y = np.array([0] * 28 + [1, 2])
        self.assertEqual(RandomForestClassifier(30).fit(x, y).predict_proba(x).shape, (30, 3))

    def test_regressor(self):
        x, y = wave()
        xt, yt, xv, yv = split(x, y)
        p = RandomForestRegressor(30).fit(xt, yt).predict(xv)
        self.assertLess(((p - yv) ** 2).mean(), 0.1)

    def test_seed_repeats(self):
        x, y = rings(300)
        a = RandomForestClassifier(5, seed=1).fit(x, y).predict_proba(x)
        b = RandomForestClassifier(5, seed=1).fit(x, y).predict_proba(x)
        self.assertTrue(np.array_equal(a, b))


class TestBoosting(unittest.TestCase):
    def test_binner(self):
        x = np.array([[1.0], [2.0], [3.0]])
        b = Binner().fit(x)
        self.assertEqual(b.transform(x).ravel().tolist(), [0, 1, 2])
        big = np.random.default_rng(0).normal(size=(5000, 1))
        bins = Binner(32).fit(big).transform(big)
        self.assertLessEqual(bins.max(), 31)
        self.assertLess(np.bincount(bins.ravel()).max(), 5000 / 32 * 1.5)

    def test_regression(self):
        x, y = wave()
        xt, yt, xv, yv = split(x, y)
        p = GradientBoostingRegressor(200).fit(xt, yt).predict(xv)
        self.assertLess(((p - yv) ** 2).mean(), 0.05)

    def test_more_rounds_lower_training_loss(self):
        x, y = wave(500)
        a = ((GradientBoostingRegressor(5).fit(x, y).predict(x) - y) ** 2).mean()
        b = ((GradientBoostingRegressor(100).fit(x, y).predict(x) - y) ** 2).mean()
        self.assertLess(b, a / 5)

    def test_binary_and_multiclass(self):
        x, y = rings()
        xt, yt, xv, yv = split(x, y)
        self.assertGreater(acc(GradientBoostingClassifier(60).fit(xt, yt), xv, yv), 0.97)
        self.assertGreater(acc(GradientBoostingClassifier(60).fit(xt, yt == 1), xv, yv == 1), 0.95)

    def test_probabilities_are_valid(self):
        x, y = rings(400)
        p = GradientBoostingClassifier(20).fit(x, y).predict_proba(x)
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_early_stopping(self):
        x, y = wave(600)
        xt, yt, xv, yv = split(x, y)
        m = GradientBoostingRegressor(2000, 0.3, 6, early_stopping=15).fit(xt, yt, (xv, yv))
        self.assertLess(len(m.trees_), 500)
        self.assertEqual(len(m.trees_), int(np.argmin(m.history_)) + 1)

    def test_shrinkage_controls(self):
        x, y = rings(400)
        m = GradientBoostingClassifier(30, subsample=0.7, colsample=0.6, gamma=0.1, seed=2).fit(x, y)
        self.assertGreater(acc(m, x, y), 0.9)

    def test_regularisation_shrinks_leaves(self):
        x, y = wave(300)
        a = GradientBoostingRegressor(1, 1.0, 2, l2=0.0).fit(x, y).predict(x)
        b = GradientBoostingRegressor(1, 1.0, 2, l2=500.0).fit(x, y).predict(x)
        self.assertLess(b.std(), a.std())

    def test_importances(self):
        x, y = rings()
        m = GradientBoostingClassifier(40).fit(x, y)
        self.assertGreater(m.importance_[:2].sum(), 0.8)
        p = permutation_importance(m, x, y, rounds=2)
        self.assertGreater(p[:2].min(), 5 * p[2:].max())

    def test_cross_validation(self):
        x, y = rings(300)
        s = cross_validate(lambda: GradientBoostingClassifier(20), x, y, 3)
        self.assertEqual(len(s), 3)
        self.assertGreater(np.mean(s), 0.9)

    @unittest.skipIf(SkTree is None, "scikit-learn is not installed")
    def test_real_datasets_against_scikit_learn(self):
        for load, tolerance in ((load_breast_cancer, 0.03), (load_wine, 0.05), (load_digits, 0.03)):
            d = load()
            xt, yt, xv, yv = split(d.data, d.target, 1)
            ours = acc(GradientBoostingClassifier(120, 0.1, 3).fit(xt, yt), xv, yv)
            theirs = acc(HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_depth=3, random_state=0).fit(xt, yt), xv, yv)
            self.assertGreater(ours, theirs - tolerance, load.__name__)
            forest = acc(RandomForestClassifier(60, seed=1).fit(xt, yt), xv, yv)
            sk = acc(SkForest(60, random_state=1).fit(xt, yt), xv, yv)
            self.assertGreater(forest, sk - tolerance, load.__name__)


if __name__ == "__main__":
    unittest.main()
