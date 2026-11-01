"""Unit tests required by T1 PREREG section 17 (run before any outer run).

    python -I code/tests/test_t1.py            (all tests; the trainer test takes ~1-2 min on CPU)
    python -I code/tests/test_t1.py --fast     (skips the trainer-copy reproduction test)
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import t1_common as C  # noqa: E402
import numpy as np  # noqa: E402

import seedvig_eog as S  # noqa: E402
import summarize as SM  # noqa: E402
import tc_models as TM  # noqa: E402

FAST = "--fast" in sys.argv


class TestFolds(unittest.TestCase):
    def test_loso_persons(self):
        b = C.base()
        self.assertEqual(len(b["persons"]), 21)
        for p, tr, te in S.loso_splits(b["groups"]):
            S.assert_no_person_overlap(b["groups"][tr], b["groups"][te])
            if p in (4, 5):
                self.assertEqual(len(np.unique(b["rec"][te])), 2)
                self.assertEqual(len(te), 2 * S.N_EPOCH)
        for tr, va in S.inner_group_folds(b["groups"][b["groups"] != 3], k=5, seed=0):
            g = b["groups"][b["groups"] != 3]
            S.assert_no_person_overlap(g[tr], g[va])

    def test_labels(self):
        b = C.base()
        self.assertTrue(np.array_equal(C.lab3(b["perclos"], 0.35, 0.70), b["yi"] - 1))
        l2 = C.lab2(b["perclos"], 0.35, 0.70)
        self.assertTrue(np.array_equal(l2 >= 0, b["yi"] != 2))
        self.assertEqual(int(C.valid_binary_cells().sum()), 47)
        bad = [C.CELLS[j] for j in np.flatnonzero(~C.valid_binary_cells())]
        self.assertEqual(sorted(bad), [(0.2, 0.8), (0.2, 0.85)])
        self.assertEqual(C.canonical_binary_persons(), [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 21])


class TestCellMetric(unittest.TestCase):
    def test_canonical_equals_summarize(self):
        b = C.base()
        for task, f in (("three", "three_litc-trad+kf_lda.npz"), ("binary", "binary_litc-trad+kf_lda.npz")):
            z = np.load(C.COMMON / "results" / "feature_check" / f)
            ref = SM.per_person(z, task)
            col = np.full(len(b["perclos"]), -1, np.int8)
            keep = np.ones(len(col), bool) if task == "three" else b["yi"] != 2
            col[keep] = z["pred"]
            ctask = "three" if task == "three" else "binarypair"
            mine = C.cell_person_metrics(b["perclos"], b["groups"], col, ctask, C.CANON, want_f1=True)
            self.assertEqual(sorted(ref), sorted(mine))
            for p in ref:
                self.assertAlmostEqual(ref[p]["bacc"], mine[p]["bacc"], places=12)
                if task == "three":
                    self.assertAlmostEqual(ref[p]["f1_macro"], mine[p]["f1_macro"], places=12)


class TestDistributions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.mu = rng.normal(-0.5, 1.5, 200)
        self.sig = rng.uniform(0.2, 2.0, 200)
        self.y = np.clip(rng.uniform(0, 1, 200), 0.022, 1.0)

    def test_probs(self):
        F = C.ln_Fcut(self.mu, self.sig)
        self.assertTrue(np.all(np.diff(F, axis=1) >= 0))
        for c in C.CELLS:
            P = C.three_from_F(F, c)
            self.assertTrue(np.allclose(P.sum(1), 1) and (P >= 0).all())
            Q = C.binpair_from_F(F, c)
            self.assertTrue(np.allclose(Q.sum(1), 1) and (Q >= 0).all())
        ex = np.stack([C.single_from_F(F, th)[:, 1] for th in C.SINGLE], 1)
        self.assertTrue(np.all(np.diff(ex, axis=1) <= 1e-15))

    def test_crps_grid(self):
        from scipy.special import logit
        from scipy.special import ndtr
        c512 = C.ln_crps(self.mu, self.sig, self.y)
        g = (np.arange(20000) + 0.5) / 20000
        Fr = ndtr((logit(g)[None, :] - self.mu[:, None]) / self.sig[:, None])
        ref = np.mean((Fr - (self.y[:, None] <= g[None, :])) ** 2, axis=1)
        self.assertLess(np.max(np.abs(c512 - ref)), 1e-3)
        G = C.ln_cdf_grid(self.mu, self.sig)
        self.assertTrue(np.allclose(C.crps_from_grid(G, self.y), c512))

    def test_decision_rules(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        self.assertEqual(C.bayes_balanced(P, [1 / 3, 1 / 3, 1 / 3]).tolist(), [0, 2, 0])
        self.assertEqual(C.bayes_balanced(P, [0.5, 0.3, 0.2]).tolist(), [0, 2, 1])
        self.assertEqual(C.bayes_balanced(np.array([[0.5, 0.5]]), [0.5, 0.5]).tolist(), [0])

    def test_qhgb_cdf_monotone(self):
        rng = np.random.default_rng(1)
        Q = rng.uniform(0, 1.05, (50, 19))                # unsorted, some outside [0.005, 0.995]
        K = TM.qhgb_knots(Q)
        G = TM.qhgb_cdf(K, C.GRID512)
        self.assertTrue(np.all(np.diff(G, axis=1) >= -1e-12))
        self.assertTrue(np.all((G >= 0) & (G <= 1)))
        Fc = TM.qhgb_cdf(K, np.array(C.CUTS))
        self.assertTrue(np.all(np.diff(Fc, axis=1) >= -1e-12))
        # F at a knot equals its quantile level
        self.assertAlmostEqual(TM.qhgb_cdf(K[:1], K[0, 9:10])[0, 0], 0.5, places=6)

    def test_ordlogit_gradient(self):
        from scipy.optimize import check_grad
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 5))
        y = np.digitize(X @ rng.normal(size=5) + rng.normal(size=300), [-0.5, 0.7])
        for K, yy in ((2, y), (1, (y == 2).astype(int))):
            th = np.concatenate([rng.normal(size=5) * 0.1, [0.1], [0.2] * (K - 1)])
            err = check_grad(lambda t: TM._ord_nll_grad(t, X, yy, 0.01, K)[0],
                             lambda t: TM._ord_nll_grad(t, X, yy, 0.01, K)[1], th)
            self.assertLess(err, 1e-5)
        th, _ = TM.ord_fit(X, y, 1e-3, 2)
        P = TM.ord_proba(th, X, 2)
        self.assertTrue(np.allclose(P.sum(1), 1, atol=1e-9))

    def test_tc_grid_shapes(self):
        b = C.base()
        n = len(b["perclos"])
        mu = C.zlogit(b["perclos"])           # oracle-like synthetic mu (no model output)
        F = C.ln_Fcut(mu, np.full(n, 0.5))
        g = C.tc_grid(F, "bayes")
        self.assertEqual(g["three"][0].shape, (n, 49))
        self.assertTrue(np.all(g["binarypair"][0][C.lab2(b["perclos"], 0.35, 0.70) < 0, C.CANON_IDX] == -1))


class TestPrep(unittest.TestCase):
    def test_median_imputer_equals_sklearn(self):
        from sklearn.impute import SimpleImputer
        for fs in ("litc-trad+kf", "litc-fore+kf"):
            X = C.load_features(fs)["X"]
            g = C.base()["groups"]
            tr = g != 19
            ref = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
            mine = TM.MedianImputer().fit(X[tr])
            self.assertTrue(np.array_equal(ref.transform(X[~tr]), mine.transform(X[~tr])))
            self.assertTrue(np.array_equal(ref.transform(X[tr]), mine.transform(X[tr])))


class TestFastLDA(unittest.TestCase):
    def test_equals_sklearn(self):
        import fast_lda as FL
        d = C.load_features("litc-trad+kf")
        b = C.base()
        X, g, y = d["X"], b["groups"], b["perclos"]
        tr, te = np.flatnonzero(g != 2), np.flatnonzero(g == 2)
        prep = TM.make_prep().fit(X[tr])
        Xt, Xe = prep.transform(X[tr]), prep.transform(X[te])
        M = FL.BinMoments(Xt, y[tr])
        for task, c in (("three", (0.25, 0.6)), ("three", C.CANON), ("binarypair", C.CANON), ("single", 0.45)):
            lab = C.cell_labels(y[tr], task, c)
            m = lab >= 0
            k = len(np.unique(lab[m]))
            ref = TM.make_lda(k).fit(Xt[m], lab[m])
            fa = FL.GridLDA(M, FL.cell_classes(task, c))
            self.assertTrue(np.array_equal(ref.predict(Xe), fa.predict(Xe)))
            self.assertLess(np.abs(ref.predict_proba(Xe) - fa.predict_proba(Xe)).max(), 1e-6)


@unittest.skipIf(FAST, "fast mode")
class TestTrainerCopy(unittest.TestCase):
    def test_reproduces_shared_trainer(self):
        import deep_train as DT
        import deep_train_t1 as DT1
        with tempfile.TemporaryDirectory() as td:
            kw = dict(montage="trad6", norm="global", K=8, seed=0, folds=[3], save_weights=False, verbose=False,
                      max_epochs=1, device="cpu")
            r0 = DT.run_loso("three", "feat_gru", out_dir=pathlib.Path(td) / "shared", **kw)
            r1 = DT1.run_loso("three", "feat_gru", out_dir=pathlib.Path(td) / "t1", **kw)
            with np.load(pathlib.Path(r0["run_dir"]) / "fold_3.npz") as a, \
                    np.load(pathlib.Path(r1["run_dir"]) / "fold_3.npz") as c:
                self.assertTrue(np.array_equal(a["rows"], c["rows"]))
                self.assertTrue(np.array_equal(a["score"], c["score"]))
                self.assertTrue(np.array_equal(a["all_score"], c["all_score"]))
            # outputs are deleted unread (temporary directory)


if __name__ == "__main__":
    argv = [a for a in sys.argv if a != "--fast"]
    unittest.main(argv=argv, verbosity=2)
