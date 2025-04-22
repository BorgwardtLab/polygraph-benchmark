from typing import Any, Callable, Iterable, Optional

import networkx as nx
import numpy as np
from loguru import logger
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def train_distance_test(
    test_features, generated_features, train_features, metric: str = "l2"
):
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


def train_distance_test_cells(
    test_features,
    test_features_cells,
    generated_features,
    generated_features_cells,
    train_features,
    train_features_cells,
):
    k = len(np.unique(test_features_cells))
    # k = len(test_features_cells)
    z_u_cells = np.zeros(k)
    p_values_cells = np.zeros(k)
    for i in range(k):
        test_features_cell_i = test_features[test_features_cells == i]
        generated_features_cell_i = generated_features[
            generated_features_cells == i
        ]
        train_features_cells_i = train_features[train_features_cells == i]
        if len(test_features_cell_i) * len(train_features_cells_i) == 0:
            raise ValueError(
                f"Cell {i} lacks test samples and/or training samples. Consider reducing the number of cells in partition."
            )

        if len(generated_features_cell_i) > 0:
            z_u_cells[i], p_values_cells[i] = train_distance_test(
                test_features_cell_i,
                generated_features_cell_i,
                train_features_cells_i,
            )
        else:
            z_u_cells[i] = 0
            p_values_cells[i] = 1
    return z_u_cells, p_values_cells


def celled_distance_train_test(
    test_features,
    test_features_cells,
    generated_features,
    generated_features_cells,
    train_features,
    train_features_cells,
    tau,
):
    m = generated_features.shape[0]
    n = test_features.shape[0]
    k = len(np.unique(train_features_cells))

    labels, cts = np.unique(generated_features_cells, return_counts=True)
    generated_features_counts = np.zeros(k)
    generated_features_counts[labels.astype(int)] = cts
    generated_feature_probability = generated_features_counts / m
    pi_tau = generated_feature_probability > tau

    labels, cts = np.unique(test_features_cells, return_counts=True)
    test_features_count = np.zeros(k)
    test_features_count[labels.astype(int)] = cts
    test_features_probability = test_features_count / n

    z_u, p_values = train_distance_test_cells(
        test_features,
        test_features_cells,
        generated_features,
        generated_features_cells,
        train_features,
        train_features_cells,
    )
    p_value = test_features_probability[pi_tau].dot(p_values[pi_tau]) / np.sum(
        test_features_probability[pi_tau]
    )
    return z_u, p_value


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
    assert (
        len(dist_train_flat) == len(dist_test_flat)
        and len(dist_train_flat) == generated_features.shape[0]
    )

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


class CelledTrainDistanceCopyingMetric:
    def __init__(
        self,
        training_set: Iterable[nx.Graph],
        test_set: Iterable[nx.Graph],
        descriptor: Callable[[Iterable[nx.Graph]], Any],
        distance: str = "l1",
        k: int = 10,
    ):
        self.training_set = training_set
        self.test_set = test_set
        self.descriptor = descriptor
        self.distance = distance

        self.training_descriptors = self.descriptor(self.training_set)
        self.test_descriptors = self.descriptor(self.test_set)
        self.kmeans = KMeans(n_clusters=k)
        self.kmeans.fit(self.training_descriptors)
        self.training_descriptors_cells = self.kmeans.predict(
            self.training_descriptors
        )
        self.test_descriptors_cells = self.kmeans.predict(self.test_descriptors)
        logger.trace(
            f"Training descriptors cells: {np.unique(self.training_descriptors_cells, return_counts=True)}"
        )
        logger.trace(
            f"Test descriptors cells: {np.unique(self.test_descriptors_cells, return_counts=True)}"
        )

    def compute(
        self,
        generated_graphs: Iterable[nx.Graph],
        tau: Optional[float] = None,
    ) -> float:
        if tau is None:
            tau = 20 / len(generated_graphs)
            logger.trace(
                f"No tau provided, using default value of {tau} (20 / "
                f"n_generated) as suggested by Meehan et al. (2021)",
            )

        generated_descriptors = self.descriptor(generated_graphs)
        generated_descriptors_cells = self.kmeans.predict(generated_descriptors)
        logger.trace(
            f"Generated descriptors cells: {len(generated_descriptors_cells)}"
        )
        logger.trace(
            f"Generated descriptors cells: {np.unique(generated_descriptors_cells, return_counts=True)}"
        )
        _, p_value = celled_distance_train_test(
            self.test_descriptors,
            self.test_descriptors_cells,
            generated_descriptors,
            generated_descriptors_cells,
            self.training_descriptors,
            self.training_descriptors_cells,
            tau,
        )
        return p_value
