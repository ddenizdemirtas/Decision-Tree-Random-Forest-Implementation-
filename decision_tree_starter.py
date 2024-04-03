"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter
import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import combinations
import pandas as pd
from scipy.stats import mode
import io
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
from sklearn.model_selection import KFold
from sklearn.utils import resample
from itertools import product

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree(BaseEstimator, ClassifierMixin):

    def __init__(self, max_depth=5, feature_labels=None, max_features=None, min_samples_split=2):
        self.max_depth = max_depth
        self.features = feature_labels
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO
        pass

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO
        return np.random.rand()

    def is_feature_categorical(self, index):

        return index in [0, 1, 7, 8]

    @staticmethod
    def gini_impurity(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini_impurity = 1 - np.sum(probabilities ** 2)
        return gini_impurity

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO
        pass

    def find_best_grouping(self, values, y):
        unique_values = np.unique(values)
        n_categories = len(unique_values)

        if n_categories == 1:
            return None, float('inf')

        best_score = float('inf')
        best_grouping = None

        # Iterate through each unique value to perform one vs all splits
        for category in unique_values:

            current_grouping = (category,)

            current_score = self.gini_impurity_grouping(
                values, y, current_grouping)

            if current_score < best_score:
                best_score = current_score
                best_grouping = current_grouping

        return best_grouping, best_score

    def gini_impurity_grouping(self, values, y, current_grouping):

        # Split data into two groups based on the current_grouping
        is_in_group = np.isin(values, current_grouping)
        y_group = y[is_in_group]
        y_not_group = y[~is_in_group]

        gini_group = DecisionTree.gini_impurity(y_group)
        gini_not_group = DecisionTree.gini_impurity(y_not_group)

        # Calculate weighted average of the Gini impurities
        n = len(y)
        n_group = len(y_group)
        n_not_group = len(y_not_group)

        weighted_gini = (n_group / n) * gini_group + \
            (n_not_group / n) * gini_not_group

        return weighted_gini

    def split_dataset(self, X, feature_idx, threshold):
        # Handle categorical splits differently
        if isinstance(threshold, (set, list, tuple)):
            # Categorical split: Check if each sample's feature value is in the threshold grouping
            mask = np.array([x in threshold for x in X[:, feature_idx]])
        else:

            mask = X[:, feature_idx] <= threshold

        left_indices = np.where(mask)[0]
        right_indices = np.where(~mask)[0]

        return left_indices, right_indices

    def calculate_prediction(self, y):

        if len(y) == 0:
            return None

        prediction = mode(y).mode

        return prediction

    def fit(self, X, y):
        total_features = X.shape[1]

        if self.max_features is None:
            self.feature_indices_ = np.arange(total_features)
        else:
            n_features = min(self.max_features, total_features)
            self.feature_indices_ = np.random.choice(
                total_features, n_features, replace=False)

        self._fit(X, y, self.feature_indices_, depth=self.max_depth)

    def _fit(self, X, y, feature_indices, depth):
        if len(set(y)) == 1 or depth == 0 or len(y) < self.min_samples_split:
            self.pred = self.calculate_prediction(y)
            self.max_depth = 0
            return

        best_impurity = float('inf')
        best_split = None
        best_indices = None
        initial_impurity = DecisionTree.gini_impurity(y)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            threshold, impurity = self.find_best_threshold(feature_values, y)

            if impurity < best_impurity:
                left_indices, right_indices = self.split_dataset(
                    X, feature_idx, threshold)
                if len(left_indices) >= self.min_samples_split and len(right_indices) >= self.min_samples_split:
                    best_impurity = impurity
                    best_split = (feature_idx, threshold)
                    best_indices = (left_indices, right_indices)

        impurity_reduction = initial_impurity - best_impurity

        if best_split is not None and impurity_reduction >= 0.005:
            self.split_idx, self.thresh = best_split

            self.left = DecisionTree(max_depth=depth-1, feature_labels=self.features,
                                     max_features=self.max_features, min_samples_split=self.min_samples_split)
            self.right = DecisionTree(max_depth=depth-1, feature_labels=self.features,
                                      max_features=self.max_features, min_samples_split=self.min_samples_split)

            X_left, y_left = X[best_indices[0]], y[best_indices[0]]
            X_right, y_right = X[best_indices[1]], y[best_indices[1]]

            self.left._fit(X_left, y_left, feature_indices, depth-1)
            self.right._fit(X_right, y_right, feature_indices, depth-1)
        else:
            self.pred = self.calculate_prediction(y)
            self.max_depth = 0

    def is_categorical(self, index):

        if self.dataset == "titanic":
            return index in [0, 1, 7, 8]
        else:
            return False

    def predict(self, X):

        if X.ndim == 1:
            return self._predict_single_sample(X)
        else:
            return np.array([self._predict_single_sample(sample) for sample in X])

    def _predict_single_sample(self, sample):
        if self.left is None and self.right is None:
            return self.pred

        if isinstance(self.thresh, (set, list, tuple)):
            # Categorical split
            decision = sample[self.split_idx] in self.thresh
        else:
            # Numerical split
            decision = sample[self.split_idx] <= self.thresh

        if decision:
            return self.left._predict_single_sample(sample)
        else:
            return self.right._predict_single_sample(sample)

    @staticmethod
    def accuracy_score(y_true, y_pred):

        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def find_best_threshold(self, values, y):

        sorted_indices = np.argsort(values)
        values_sorted = values[sorted_indices]
        y_sorted = y[sorted_indices]

        best_score = float('inf')
        best_threshold = None

        for i in range(1, len(values_sorted)):
            if values_sorted[i] == values_sorted[i-1]:
                continue

            potential_threshold = (values_sorted[i] + values_sorted[i-1]) / 2

            current_score = self.gini_impurity_split(y_sorted, i)

            if current_score < best_score:
                best_score = current_score
                best_threshold = potential_threshold

        return best_threshold, best_score

    def gini_impurity_split(self, y, split_index):

        y_left = y[:split_index]
        y_right = y[split_index:]

        gini_left = DecisionTree.gini_impurity(y_left)
        gini_right = DecisionTree.gini_impurity(y_right)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        return weighted_gini

    def score(self, X, y):

        predictions = self.predict(X)

        accuracy = DecisionTree.accuracy_score(y, predictions)
        return accuracy

    def find_best_split(self, X, y, dataset):

        best_score = float('inf')
        best_feature_idx = None
        best_threshold = None
        n_features = X.shape[1]

        best_scores = []

        for feature_idx in range(n_features):
            values = X[:, feature_idx]

            if self.is_categorical(feature_idx):

                best_grouping, current_score = self.find_best_grouping(
                    values, y)
                current_best = best_grouping
            else:

                best_threshold, current_score = self.find_best_threshold(
                    values, y)
                current_best = best_threshold

            best_scores.append((current_score, feature_idx, current_best))

        best_score, best_feature_idx, best_threshold = min(
            best_scores, key=lambda x: x[0])

        return best_feature_idx, best_threshold, best_score

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, len(self.features))
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(**self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        for tree in self.decision_trees:
            X_sample, y_sample = resample(X, y, replace=True)
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.decision_trees]
        predictions = np.array(predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)
        return final_predictions


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=None):
        if params is None:
            params = {}
        if m is not None:
            params['max_features'] = m
        else:
            pass
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass


def preprocess(data, categorical_cols):

    category_mappings = {}
    ticket_number_col_index = 5

    # Clean the ticket number column
    for i, ticket_str in enumerate(data[:, ticket_number_col_index]):
        data[i, ticket_number_col_index] = clean_ticket_number(ticket_str)

    cabin_col_index = 7
    for i, cabin_str in enumerate(data[:, cabin_col_index]):
        data[i, cabin_col_index] = extract_deck(cabin_str)
    data = np.where(data == '', np.nan, data)

    for col in categorical_cols:
        unique_categories = np.unique(data[:, col])
        category_to_index = {category: index for index,
                             category in enumerate(unique_categories)}
        data[:, col] = np.array([category_to_index[category]
                                for category in data[:, col]])
        category_mappings[col] = category_to_index

    data = data.astype(float)
    imputer = KNNImputer(n_neighbors=5, add_indicator=False)
    data_imputed = imputer.fit_transform(data)
    data_imputed = np.round(data_imputed)

    return data_imputed.astype(int), category_mappings


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def clean_ticket_number(ticket_str):

    numeric_parts = re.findall(r'\d+', ticket_str)
    return numeric_parts[0] if numeric_parts else ''


def extract_deck(cabin_str):
    if cabin_str == "":
        return np.nan
    match = re.match(r"([a-zA-Z]+)", cabin_str)
    if match:
        return match.group(1)
    return np.nan


def cross_validate(model, X, y, dataset, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = DecisionTree.accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, accuracies


def calculate_accuracies(X, y, max_depth_range, model, test_size=0.2, n=None, m=None, params=None):
    depths = []
    training_accuracies = []
    validation_accuracies = []

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)

    if params is None:
        params = {}

    for depth in max_depth_range:
        current_params = params.copy()
        current_params['max_depth'] = depth

        if model == RandomForest:

            current_params['n'] = n if n is not None else 100
            current_params['m'] = m  # max features
            classifier = model(params=current_params, n=n, m=m)
        else:

            classifier = model(**current_params)

        classifier.fit(X_train, y_train)

        # Predict on training and validation sets
        y_train_pred = classifier.predict(X_train)
        y_val_pred = classifier.predict(X_val)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Store results
        depths.append(depth)
        training_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)

    # Plotting
    # plt.figure(figsize=(10, 5))
    # plt.plot(depths, training_accuracies, label='Training Accuracy')
    # plt.plot(depths, validation_accuracies, label='Validation Accuracy')
    # plt.xlabel('Max Depth')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Max Depth')
    # plt.legend()
    # # plt.savefig('./accuracy_vs_max_depth.png', dpi=300)
    # plt.show()

    return depths, training_accuracies, validation_accuracies


def find_best_hyperparameters(X, y, hyperparameter_grid, n_estimators_range, max_features_range, n_splits=5):
    def all_combinations(grid):
        keys, values = zip(*grid.items())
        for value_combination in product(*values):
            yield dict(zip(keys, value_combination))

    def cross_validate(model, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores.append(accuracy_score(y_test, predictions))
        return np.mean(scores)

    best_score = 0
    best_params = {}
    best_n_estimators = 0
    best_m = None

    for n_estimators in n_estimators_range:
        print("DEBUG: N ", n_estimators)
        for m in max_features_range:
            print("DEBUG: M ", m)

            for params in all_combinations(hyperparameter_grid):
                print("DEBUG: P ", params)

                # Assuming RandomForest is your RandomForest implementation
                mean_accuracy = cross_validate(RandomForest(
                    params=params, n=n_estimators, m=m), X, y, n_splits=n_splits)

                if mean_accuracy > best_score:
                    best_score = mean_accuracy
                    best_params = params
                    best_n_estimators = n_estimators
                    best_m = m

    print(f"Best Score: {best_score}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best n_estimators: {best_n_estimators}")
    print(f"Best max_features (m): {best_m}")
    return best_score, best_params, best_n_estimators, best_m


def results_to_csv(y_test, dataset, model):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1
    df.to_csv(f'./{model}-{dataset}-submission.csv', index_label='Id')


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"

    N = 100

    if dataset == "titanic":
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',',
                          dtype=None, encoding='utf-8')
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',',
                               dtype=None, encoding='utf-8')

        y = data[1:, 0]
        labeled_idx = np.where(y != '')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)

        # Assuming categorical columns indices are known
        # Adjust this based on actual categorical columns
        categorical_cols_indices = [0, 1, 7, 8]

        # Preprocess training data
        X, train_mappings = preprocess(
            data[1:, 1:][labeled_idx, :], categorical_cols_indices)

        X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
        # Preprocess test data similarly
        Z, test_mappings = preprocess(
            test_data[1:, :], categorical_cols_indices)

        # Verify that the feature dimensions match
        assert X.shape[1] == Z.shape[1], "Feature dimensions of X_train and X_test do not match."
        class_names = ['Dead', 'Alive']
        features = list(data[0, 1:])

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    params = {
        "max_depth": 3,
        # "random_state": 6,
        # "min_samples_leaf": 10,
        "feature_labels": features,

    }
    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    # print("\n\nsklearn's decision tree")
    # clf = DecisionTreeClassifier(random_state=0, **params)
    # clf.fit(X, y)
    # evaluate(clf)

    # out = io.StringIO()
    # export_graphviz(
    #     clf, out_file=out, feature_names=features, class_names=class_names)
    # # For OSX, may need the following for dot: brew install gprof2dot
    # graph = graph_from_dot_data(out.getvalue())
    # graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    ######### FIND BEST DEPTH ##############

    # Best depth candidates for Decision Tree using spam =[8,15]
    # Best depth candidates for Decision Tree using titanic = [3,4]

    calculate_accuracies(X, y, range(1, 40), DecisionTree,
                         params={'feature_labels': features})

    ##########################################
    ######## RANDOM FOREST HYPERPARAMETER FINE TUNING #####################
    # Titanic best hyperparameters =
    # Spam best hyperparameters =
    # hyperparameter_grid = {
    #     'max_depth': [3, 4],
    #     'min_samples_split': [2, 5, 10],
    # }

    # n_estimators_range = [100, 200, 300]
    # max_features_range = [4, 6, 8]

    # best_score, best_params, best_n_estimators, best_m = find_best_hyperparameters(
    #     X, y, hyperparameter_grid, n_estimators_range, max_features_range)
    ##########################################

    ## TEST IMPLEMENTATION DECISION TREE ##

    classifier = DecisionTree(**params)
    classifier.fit(X, y)
    mean_accuracy, acc_list = cross_validate(
        classifier, X, y, dataset, n_splits=5)
    print("MEAN ACCURACY: ", mean_accuracy)
    predictions = classifier.predict(X)
    # training_accuracy = accuracy_score(y, predictions)
    # print("TRAINING ACC: ", training_accuracy)
    print(classifier)

    # RANDOM FOREST
    # rf_best_hyperparameters = {
    #     'max_depth': 25,
    #     'min_samples_split': 5,
    #     'max_features': 16  # Note: 'max_features' is handled separately as 'm'
    # }
    # best_m = rf_best_hyperparameters.pop('max_features')
    # best_n_estimators = 100

    # # ### TEST IMPLEMENTATION RANDOM FOREST ###
    # classifier = RandomForest(
    #     params=rf_best_hyperparameters, n=best_n_estimators, m=best_m)

    # classifier.fit(X, y)
    # mean_accuracy, acc_list = cross_validate(
    #     classifier, X, y, dataset, n_splits=5)
    # print("MEAN ACCURACY: ", mean_accuracy)
    # predictions = classifier.predict(X)
    # training_accuracy = accuracy_score(y, predictions)
    # print("TRAINING ACC: ", training_accuracy)

    # ### PLOTTING ####
    # max_depth_range = range(1, 40)  # Adjust max depth range as needed
    # depths, training_accuracies, validation_accuracies = calculate_accuracies(
    #     X, y, max_depth_range, n=best_n_estimators, m=best_m, params=rf_best_hyperparameters)

    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(depths, training_accuracies, label='Training Accuracy')
    # plt.plot(depths, validation_accuracies, label='Validation Accuracy')
    # plt.xlabel('Depth of Tree')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Depth of Tree')
    # plt.legend()
    # plt.show()

    ### GENERATE TEST PREDICTION ###
    # y_test_pred = classifier.predict(Z)
    # results_to_csv(y_test_pred, dataset, RandomForest)

    ### RANDOM FOREST HYPER PARAMETER TUNING ###

    ### RANDOM FOREST GENERATE TEST PREDICTIONS ###
