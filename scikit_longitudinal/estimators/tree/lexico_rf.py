# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
#
#
# # Custom Decision Tree Classifier
# class CustomDecisionTreeClassifier(DecisionTreeClassifier):
#     def __init__(self, custom_hyperparameter=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.custom_hyperparameter = custom_hyperparameter
#
#     # ... (rest of the CustomDecisionTreeClassifier code remains unchanged)
#
#
# # Custom Random Forest Classifier
# class CustomRandomForestClassifier(RandomForestClassifier):
#     def __init__(self, custom_hyperparameter=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.custom_hyperparameter = custom_hyperparameter
#
#     def _make_estimator(self, append=True, random_state=None):
#         estimator = CustomDecisionTreeClassifier(
#             custom_hyperparameter=self.custom_hyperparameter,
#             criterion=self.criterion,
#             splitter=self.splitter,
#             max_depth=self.max_depth,
#             min_samples_split=self.min_samples_split,
#             min_samples_leaf=self.min_samples_leaf,
#             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
#             max_features=self.max_features,
#             max_leaf_nodes=self.max_leaf_nodes,
#             min_impurity_decrease=self.min_impurity_decrease,
#             min_impurity_split=self.min_impurity_split,
#             random_state=random_state,
#             ccp_alpha=self.ccp_alpha,
#         )
#
#         if append:
#             self.estimators_.append(estimator)
#
#         return estimator
#
#
# # Load dataset and split into train and test sets
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Train custom random forest classifier
# custom_rf = CustomRandomForestClassifier(custom_hyperparameter="your_value", n_estimators=100, random_state=42)
# custom_rf.fit(X_train, y_train)
#
# # Test custom random forest classifier
# y_pred = custom_rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
