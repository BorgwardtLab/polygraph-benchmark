"""PolyGraphScores to compare graph distributions.

PolyGraphScores compare generated graphs to reference graphs by fitting a binary classifier to discriminate between the two.
Performance metrics of this classifier lower-bound intrinsic probability metrics.
Multiple graph descriptors may be combined within PolyGraphScores to yield a theoretically grounded summary metric.
"""

from typing import (
    Collection,
    Literal,
    Optional,
    Tuple,
    Dict,
    Union,
    TypedDict,
)
from collections import Counter
import warnings
from sklearn.metrics import roc_curve
from scipy.sparse import csr_array

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from tabpfn import TabPFNClassifier
from polygraph.metrics.base.metric_interval import MetricInterval
from polygraph.utils.graph_descriptors import GraphDescriptor
from polygraph.metrics.base.interfaces import GenerationMetric, GenerationMetricInterval


__all__ = [
    "ClassifierMetric",
    "PolyGraphScore",
    "PolyGraphScoreInterval",
]


class PolyGraphScoreResult(TypedDict):
    """Return type for PolyGraphScore.compute method."""
    polygraphscore: float
    polygraphscore_descriptor: str
    subscores: Dict[str, float]


class PolyGraphScoreIntervalResult(TypedDict):
    """Return type for PolyGraphScoreInterval.compute method."""
    polyscore: MetricInterval
    subscores: Dict[str, MetricInterval]
    polyscore_descriptor: Dict[str, float]


def _scores_to_jsd(ref_scores, gen_scores, eps: float = 1e-10):
    """Estimate Jensen-Shannon distance based on classifier probabilities."""
    divergence = 0.5 * (
        np.log2(ref_scores + eps).mean()
        + np.log2(1 - gen_scores + eps).mean()
        + 2
    )
    return np.sqrt(np.clip(divergence, 0, 1))


def _scores_to_informedness_and_threshold(ref_scores: np.ndarray, gen_scores: np.ndarray) -> Tuple[float, float]:
    ground_truth = np.concatenate(
        [np.ones(len(ref_scores)), np.zeros(len(gen_scores))]
    )
    if ref_scores.ndim != 1:
        raise RuntimeError("ref_scores must be 1-dimensional, got shape {ref_scores.shape}. This should not happen, please file a bug report.")

    assert ref_scores.ndim == 1 and gen_scores.ndim == 1
    fpr, tpr, thresholds = roc_curve(
        ground_truth, np.concatenate([ref_scores, gen_scores])
    )
    j_statistic = tpr - fpr
    optimal_idx = np.argmax(j_statistic)
    optimal_threshold = thresholds[optimal_idx]
    j_statistic = j_statistic[optimal_idx]
    return j_statistic, optimal_threshold


def _scores_and_threshold_to_informedness(ref_scores: np.ndarray, gen_scores: np.ndarray, threshold: float) -> float:
    assert ref_scores.ndim == 1 and gen_scores.ndim == 1
    ref_pred = (ref_scores >= threshold).astype(int)
    gen_pred = (gen_scores >= threshold).astype(int)
    tpr = np.mean(ref_pred, axis=0)
    fpr = np.mean(gen_pred, axis=0)
    return tpr - fpr



def _classifier_cross_validation(
    classifier: Union[LogisticRegression, TabPFNClassifier],
    train_ref_descriptions: np.ndarray,
    train_gen_descriptions: np.ndarray,
    variant: Literal["informedness", "jsd"],
    n_folds: int = 4,
):
    """
    Perform stratified k-fold cross-validation with proper threshold selection for informedness.

    Args:
        classifier: The classifier model to use
        train_ref_descriptions: Feature vectors for reference graphs
        train_gen_descriptions: Feature vectors for generated graphs
        variant: Either "informedness" or "jsd"
        n_folds: Number of cross-validation folds

    Returns:
        List of scores for each fold
    """
    # Combine data and create labels
    X = np.concatenate([train_ref_descriptions, train_gen_descriptions], axis=0)
    y = np.concatenate(
        [
            np.ones(len(train_ref_descriptions)),
            np.zeros(len(train_gen_descriptions)),
        ]
    )

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        if np.all(X_train == X_train[0]):
            warnings.warn(
                "Input to classifier is constant, setting all scores to 0.5"
            )
            predict_proba = lambda x: np.ones((x.shape[0], 2)) * 0.5
        else:
            classifier.fit(X_train, y_train)
            predict_proba = classifier.predict_proba

        # Get validation predictions
        val_pred = predict_proba(X_val)[:, 1]

        # Get reference and generated indices in validation set
        val_ref_idx = np.where(y_val == 1)[0]
        val_gen_idx = np.where(y_val == 0)[0]

        # Get predictions for reference and generated samples
        val_ref_pred = val_pred[val_ref_idx]
        val_gen_pred = val_pred[val_gen_idx]

        if variant == "informedness":
            train_pred = predict_proba(X_train)[:, 1]
            train_ref_idx = np.where(y_train == 1)[0]
            train_gen_idx = np.where(y_train == 0)[0]
            train_ref_pred = train_pred[train_ref_idx]
            train_gen_pred = train_pred[train_gen_idx]

            # Get threshold from training data
            _, threshold = _scores_to_informedness_and_threshold(
                train_ref_pred, train_gen_pred
            )

            # Apply threshold to validation data
            score = _scores_and_threshold_to_informedness(
                val_ref_pred, val_gen_pred, threshold
            )
        elif variant == "jsd":
            score = _scores_to_jsd(val_ref_pred, val_gen_pred)
        else:
            raise ValueError(f"Invalid variant: {variant}")

        scores.append(score)

    return scores


def _descriptions_to_classifier_metric(
    ref_descriptions: Union[np.ndarray, csr_array],
    gen_descriptions: Union[np.ndarray, csr_array],
    variant: Literal[
        "informedness", "jsd"
    ] = "jsd",
    classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    rng: Optional[np.random.Generator] = None,
):
    rng = np.random.default_rng(0) if rng is None else rng

    if isinstance(ref_descriptions, csr_array):
        assert isinstance(gen_descriptions, csr_array)
        # Convert to dense array
        num_features = (
            max(
                ref_descriptions.indices.max(),
                gen_descriptions.indices.max(),
            )
            + 1
        )
        gen_descriptions = csr_array(
            (
                gen_descriptions.data,
                gen_descriptions.indices,
                gen_descriptions.indptr,
            ),
            shape=(gen_descriptions.shape[0], num_features),
        ).toarray()
        ref_descriptions = csr_array(
            (
                ref_descriptions.data,
                ref_descriptions.indices,
                ref_descriptions.indptr,
            ),
            shape=(ref_descriptions.shape[0], num_features),
        ).toarray()

    ref_train_idx = rng.choice(
        len(ref_descriptions),
        size=len(ref_descriptions) // 2,
        replace=False,
    )
    ref_test_idx = np.setdiff1d(np.arange(len(ref_descriptions)), ref_train_idx)
    gen_train_idx = rng.choice(
        len(gen_descriptions), size=len(gen_descriptions) // 2, replace=False
    )
    gen_test_idx = np.setdiff1d(np.arange(len(gen_descriptions)), gen_train_idx)

    scaler = StandardScaler()
    train_ref_descriptions = ref_descriptions[ref_train_idx]
    train_gen_descriptions = gen_descriptions[gen_train_idx]
    test_ref_descriptions = ref_descriptions[ref_test_idx]
    test_gen_descriptions = gen_descriptions[gen_test_idx]
    scaler.fit(
        np.concatenate([train_ref_descriptions, train_gen_descriptions], axis=0)
    )
    test_ref_descriptions = scaler.transform(test_ref_descriptions)
    test_gen_descriptions = scaler.transform(test_gen_descriptions)
    train_ref_descriptions = scaler.transform(train_ref_descriptions)
    train_gen_descriptions = scaler.transform(train_gen_descriptions)

    if classifier == "logistic":
        clf = LogisticRegression(penalty="l2", max_iter=1000)
    elif classifier == "tabpfn":
        clf = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Invalid classifier: {classifier}")

    # Use custom cross-validation function
    scores = _classifier_cross_validation(
        clf,
        train_ref_descriptions,
        train_gen_descriptions,
        variant,
        n_folds=4,
    )

    train_metric = np.mean(scores)

    # Refit on all training data
    train_all_descriptions = np.concatenate(
        [train_ref_descriptions, train_gen_descriptions], axis=0
    )
    train_labels = np.concatenate(
        [
            np.ones(len(train_ref_descriptions)),
            np.zeros(len(train_gen_descriptions)),
        ]
    )

    # Check if train_all_descriptions is constant across rows
    if np.all(train_all_descriptions == train_all_descriptions[0]):
        warnings.warn(
            "Input to classifier is constant, setting all scores to 0.5"
        )
        predict_proba = lambda x: np.ones((x.shape[0], 2)) * 0.5
    else:
        clf.fit(train_all_descriptions, train_labels)
        predict_proba = clf.predict_proba

    ref_test_pred = predict_proba(test_ref_descriptions)[:, 1]
    gen_test_pred = predict_proba(test_gen_descriptions)[:, 1]

    if variant == "informedness":
        ref_train_pred = predict_proba(train_ref_descriptions)[:, 1]
        gen_train_pred = predict_proba(train_gen_descriptions)[:, 1]
        _, threshold = _scores_to_informedness_and_threshold(
            ref_train_pred, gen_train_pred
        )
        test_metric = _scores_and_threshold_to_informedness(
            ref_test_pred, gen_test_pred, threshold
        )
    elif variant == "jsd":
        test_metric = _scores_to_jsd(ref_test_pred, gen_test_pred)
    else:
        raise ValueError(f"Invalid variant: {variant}")

    return train_metric, test_metric


class ClassifierMetric(GenerationMetric):
    """Classifier-based metric using a single graph descriptor.

    Args:
        reference_graphs: Reference graphs
        descriptor: Descriptor function
        variant: Classifier metric to compute. To estimate the Jensen-Shannon distance, use "jsd". To estimate total variation distance, use "informedness".
        classifier: Binary classifier to fit
    """
    _variant: Literal["informedness", "jsd"]
    _classifier: Literal["logistic", "tabpfn"]

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor: GraphDescriptor,
        variant: Literal[
            "informedness", "jsd"
        ] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant
        self._classifier = classifier

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, float]:
        """Compute the classifier metric.

        Args:
            generated_graphs: Generated graphs

        Returns:
            Tuple of train and test metric
        """
        descriptions = self._descriptor(generated_graphs)
        return _descriptions_to_classifier_metric(
            self._reference_descriptions,
            descriptions,
            variant=self._variant,
            classifier=self._classifier,
        )


class _ClassifierMetricSamples:
    _variant: Literal["informedness", "jsd"]
    _classifier: Literal["logistic", "tabpfn"]

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor: GraphDescriptor,
        variant: Literal[
            "informedness", "jsd"
        ] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant
        self._classifier = classifier

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 100,
    ) -> np.ndarray:
        descriptions = self._descriptor(generated_graphs)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(num_samples):
            ref_idx = rng.choice(
                self._reference_descriptions.shape[0],
                size=subsample_size,
                replace=False,
            )
            gen_idx = rng.choice(
                descriptions.shape[0], size=subsample_size, replace=False
            )
            samples.append(
                _descriptions_to_classifier_metric(
                    self._reference_descriptions[ref_idx],
                    descriptions[gen_idx],
                    variant=self._variant,
                    rng=rng,
                    classifier=self._classifier,
                )
            )
        samples = np.array(samples)
        return samples


class PolyGraphScore(GenerationMetric):
    """PolyGraphScore to compare graph distributions.

    Args:
        reference_graphs: Reference graphs
        descriptors: Dictionary of descriptor names and descriptor functions
        variant: Classifier metric to compute. To estimate the Jensen-Shannon distance, use "jsd". To estimate total variation distance, use "informedness".
        classifier: Binary classifier to fit
    """
    _variant: Literal["informedness", "jsd"]
    _classifier: Literal["logistic", "tabpfn"]

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptors: Dict[str, GraphDescriptor],
        variant: Literal[
            "informedness", "jsd"
        ] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._sub_metrics = {
            name: ClassifierMetric(
                reference_graphs, descriptors[name], variant, classifier
            )
            for name in descriptors
        }

    def compute(self, generated_graphs: Collection[nx.Graph]) -> PolyGraphScoreResult:
        """Compute the PolyGraphScore.

        Args:
            generated_graphs: Generated graphs

        Returns:
            Dictionary of scores. 
                The key `"polygraphscore"` specifies the PolyGraphScore, giving the estimated tightest lower-bound on the probability metric. 
                The key `"polygraphscore_descriptor"` specifies the descriptor that achieves this bound. 
                All descritor-wise scores are returned in the key `"subscores"`.
        """
        all_metrics = {
            name: metric.compute(generated_graphs)
            for name, metric in self._sub_metrics.items()
        }  # Select the descriptor with the optimal train metric
        optimal_descriptor = max(
            all_metrics.keys(), key=lambda x: all_metrics[x][0]
        )
        aggregate_metric = all_metrics[optimal_descriptor][1]
        result = {
            "polygraphscore": aggregate_metric,
            "polygraphscore_descriptor": optimal_descriptor,
            "subscores": {
                name: metric[1] for name, metric in all_metrics.items()
            },
        }
        return PolyGraphScoreResult(**result)


class PolyGraphScoreInterval(GenerationMetricInterval):
    _variant: Literal["informedness", "jsd"]
    _classifier: Literal["logistic", "tabpfn"]

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptors: Dict[str, GraphDescriptor],
        variant: Literal[
            "informedness", "jsd"
        ] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._sub_metrics = {
            name: _ClassifierMetricSamples(
                reference_graphs, descriptors[name], variant, classifier
            )
            for name in descriptors
        }

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 100,
    ) -> PolyGraphScoreIntervalResult:
        all_sub_samples = {
            name: metric.compute(generated_graphs, subsample_size, num_samples)
            for name, metric in self._sub_metrics.items()
        }
        all_sub_intervals = {
            name: MetricInterval.from_samples(all_sub_samples[name][:, 1])
            for name in self._sub_metrics.keys()
        }

        optimal_descriptors = [
            max(
                self._sub_metrics.keys(), key=lambda x: all_sub_samples[x][i, 0]
            )
            for i in range(num_samples)
        ]
        aggregate_samples = np.array(
            [
                all_sub_samples[descriptor][i, 1]
                for i, descriptor in enumerate(optimal_descriptors)
            ]
        )
        aggregate_interval = MetricInterval.from_samples(aggregate_samples)

        descriptor_counts = Counter(optimal_descriptors)

        result = {
            "polyscore": aggregate_interval,
            "subscores": all_sub_intervals,
            "polyscore_descriptor": {
                key: descriptor_counts[key] / num_samples
                for key in self._sub_metrics.keys()
            },
        }
        return PolyGraphScoreIntervalResult(**result)
