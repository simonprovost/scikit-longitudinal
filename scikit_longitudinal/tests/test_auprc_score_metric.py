import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from scikit_longitudinal import auprc_score


class TestAUPRCScore:
    def setup_method(self):
        self.X, self.y = make_classification(
            n_samples=100, n_features=20, n_informative=2, n_redundant=10, n_classes=2, random_state=1
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

    def test_auprc_score_input_length_mismatch(self):
        with pytest.raises(ValueError):
            auprc_score([1, 0], [0.1, 0.9, 0.8])

    def test_auprc_score_non_numeric_values_in_y_score(self):
        with pytest.raises(ValueError):
            auprc_score(self.y_test, ["0.1", "0.9", "0.8", "0.2", "0.65"])

    def test_auprc_score_non_integer_values_in_y_true(self):
        with pytest.raises(ValueError):
            auprc_score([1.0, 0.0, 1.0, 0.0, 1.0], self.dummy_scores)

    def test_auprc_score_non_binary_values_in_y_true(self):
        with pytest.raises(ValueError):
            auprc_score([1, 2, 1, 0, 1], self.dummy_scores)
