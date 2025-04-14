import numpy as np
from typing import Any, Callable, Iterable

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu, wilcoxon


def train_distance_test(
    test_features, generated_features, train_features, metric: str = "l2"
):
    """Extracts distances to training nearest neighbor
    L(P_n), L(Q_m), and runs Z-scored Mann Whitney U-test.
    For the global test, this is used on the samples within each cell.

    Inputs:
        test_features: (n X d) np array representing test sample of
            length n (with dimension d)

        generated_features: (m X d) np array representing generated sample of
            length n (with dimension d)

        train_features: (l X d) np array representing training sample of
            length l (with dimension d)

    Ouptuts:
        Zu: Z-scored U value. A large value >>0 indicates
            underfitting by generated_features. A small value <<0 indicates.
    """
    m = generated_features.shape[0]
    n = test_features.shape[0]

    # fit NN model to training sample to get distances to test and generated samples
    knn = NearestNeighbors(n_neighbors=1, metric=metric).fit(train_features)
    dist_generated_features, _ = knn.kneighbors(
        X=generated_features, n_neighbors=1
    )
    dist_test_features, _ = knn.kneighbors(X=test_features, n_neighbors=1)

    # Get Mann-Whitney U score and manually Z-score it using the conditions of null hypothesis H_0
    u, p_value = mannwhitneyu(
        dist_generated_features, dist_test_features, alternative="less"
    )
    mean = (n * m / 2) - 0.5  # 0.5 is continuity correction
    std = np.sqrt(n * m * (n + m + 1) / 12)
    z_u = (u - mean) / std
    return z_u, p_value.item()


class TrainDistanceCopyingMetric:
    def __init__(
        self,
        training_set: Iterable[nx.Graph],
        test_set: Iterable[nx.Graph],
        descriptor: Callable[[Iterable[nx.Graph]], Any],
        distance: str = "l1",
    ):
        self.training_set = training_set
        self.test_set = test_set
        self.descriptor = descriptor
        self.distance = distance

        self.training_descriptors = self.descriptor(self.training_set)
        self.test_descriptors = self.descriptor(self.test_set)

    def compute(self, generated_graphs: Iterable[nx.Graph]):
        generated_descriptors = self.descriptor(generated_graphs)
        _, p_value = train_distance_test(
            self.test_descriptors,
            generated_descriptors,
            self.training_descriptors,
            self.distance,
        )
        return p_value


def train_test_distance_test(
    test_features, generated_features, train_features, metric: str = "l2"
):
    assert len(test_features) == len(train_features)
    knn_train = NearestNeighbors(n_neighbors=1, metric=metric).fit(
        train_features
    )
    knn_test = NearestNeighbors(n_neighbors=1, metric=metric).fit(test_features)
    dist_train, _ = knn_train.kneighbors(X=generated_features, n_neighbors=1)
    dist_test, _ = knn_test.kneighbors(X=generated_features, n_neighbors=1)

    # Use Wilcoxon signed-rank test for paired samples
    # Flatten arrays to ensure they're 1D
    dist_train_flat = dist_train[:, 0]
    dist_test_flat = dist_test[:, 0]
    assert dist_train_flat.ndim == 1 and dist_test_flat.ndim == 1
    assert len(dist_train_flat) == len(dist_test_flat) and len(
        dist_train_flat
    ) == len(generated_features)

    # Perform Wilcoxon test
    statistic, p_value = wilcoxon(
        dist_train_flat, dist_test_flat, alternative="less"
    )
    return statistic, p_value


class TrainTestDistanceCopyingMetric:
    def __init__(
        self,
        training_set: Iterable[nx.Graph],
        test_set: Iterable[nx.Graph],
        descriptor: Callable[[Iterable[nx.Graph]], Any],
        distance: str = "l1",
    ):
        assert len(training_set) == len(test_set)
        self.training_set = training_set
        self.test_set = test_set
        self.descriptor = descriptor
        self.distance = distance

        self.training_descriptors = self.descriptor(self.training_set)
        self.test_descriptors = self.descriptor(self.test_set)

    def compute(self, generated_graphs: Iterable[nx.Graph]):
        generated_descriptors = self.descriptor(generated_graphs)
        _, p_value = train_test_distance_test(
            test_features=self.test_descriptors,
            generated_features=generated_descriptors,
            train_features=self.training_descriptors,
            metric=self.distance,
        )
        return p_value
