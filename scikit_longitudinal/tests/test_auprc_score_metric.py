import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scikit_longitudinal import auprc_score


class TestAUPRCScore:
    def setup_method(self):
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=2,
            n_redundant=10,
            n_classes=2,
            random_state=1,
        )
        self.y = [int(value) for value in self.y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=1
        )
        self.dummy_clf = DummyClassifier(strategy="stratified")
        self.dummy_clf.fit(self.X_train, self.y_train)
        self.dummy_scores = self.dummy_clf.predict_proba(self.X_test)[:, 1]

    def test_auprc_score_output_type(self):
        assert isinstance(auprc_score(self.y_test, self.dummy_scores), float)

    def test_auprc_score_value_range(self):
        auprc = auprc_score(self.y_test, self.dummy_scores)
        assert 0.0 <= auprc <= 1.0

    def test_auprc_score_matches_trapezoidal_pr_auc_for_binary_targets(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        assert auprc_score(y_true, y_score) == pytest.approx(auc(recall, precision))
        assert auprc_score(y_true, y_score) != pytest.approx(
            average_precision_score(y_true, y_score)
        )

    def test_auprc_score_accepts_binary_two_column_scores(self):
        two_column_scores = np.column_stack(
            [1.0 - self.dummy_scores, self.dummy_scores]
        )
        assert auprc_score(self.y_test, two_column_scores) == pytest.approx(
            auprc_score(self.y_test, self.dummy_scores)
        )

    def test_auprc_score_input_length_mismatch(self):
        with pytest.raises(ValueError):
            auprc_score([1, 0], [0.1, 0.9, 0.8])

    def test_auprc_score_non_numeric_values_in_y_score(self):
        with pytest.raises(ValueError):
            auprc_score(self.y_test, ["0.1", "0.9", "0.8", "0.2", "0.65"])

    def test_auprc_score_non_integer_values_in_y_true(self):
        scores = np.array([0.1, 0.9, 0.8, 0.2, 0.65])
        result = auprc_score([1.0, 0.0, 1.0, 0.0, 1.0], scores)
        assert isinstance(result, float)

    def test_auprc_score_supports_multiclass_averaging(self):
        X, y = make_classification(
            n_samples=150,
            n_features=8,
            n_informative=8,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=3,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=3
        )
        clf = RandomForestClassifier(random_state=3)
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)
        y_test_binarized = label_binarize(y_test, classes=np.array([0, 1, 2]))

        per_class = []
        for class_index in range(scores.shape[1]):
            precision, recall, _ = precision_recall_curve(
                y_test_binarized[:, class_index], scores[:, class_index]
            )
            per_class.append(auc(recall, precision))
        per_class = np.asarray(per_class)
        support = y_test_binarized.sum(axis=0)

        assert np.allclose(auprc_score(y_test, scores, average=None), per_class)
        assert auprc_score(y_test, scores, average="macro") == pytest.approx(
            float(np.mean(per_class))
        )
        assert auprc_score(y_test, scores, average="weighted") == pytest.approx(
            float(np.average(per_class, weights=support))
        )
        assert auprc_score(y_test, scores, average="micro") == pytest.approx(
            auc(
                *precision_recall_curve(y_test_binarized.ravel(), scores.ravel())[:2][
                    ::-1
                ]
            )
        )

    def test_auprc_score_respects_explicit_label_order_for_multiclass_scores(self):
        X, y = make_classification(
            n_samples=150,
            n_features=8,
            n_informative=8,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=11,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=11
        )
        clf = RandomForestClassifier(random_state=11)
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)

        reordered_labels = np.array([2, 0, 1])
        reordered_scores = scores[:, [2, 0, 1]]
        expected = auprc_score(y_test, scores, average=None)

        assert np.allclose(
            auprc_score(
                y_test, reordered_scores, average=None, labels=reordered_labels
            ),
            expected[[2, 0, 1]],
        )

    def test_auprc_score_requires_two_dimensional_scores_for_multiclass_targets(self):
        with pytest.raises(ValueError):
            auprc_score([0, 1, 2], np.array([0.1, 0.2, 0.3]))
