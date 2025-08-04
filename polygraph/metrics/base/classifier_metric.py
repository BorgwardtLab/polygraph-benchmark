"""PolyGraphScores to compare graph distributions.

PolyGraphScores compare generated graphs to reference graphs by fitting a binary classifier to discriminate between the two.
Performance metrics of this classifier lower-bound intrinsic probability metrics.
Multiple graph descriptors may be combined within PolyGraphScores to yield a theoretically grounded summary metric.
"""

from typing import (
    Collection,
    Literal,
    Optional,
    Callable,
    List,
    Tuple,
    Dict,
    Union,
)
from collections import Counter
import warnings

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.sparse import csr_array

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from tabpfn import TabPFNClassifier
from polygraph.utils.kernels import DescriptorKernel
from polygraph.metrics.base.metric_interval import MetricInterval

__all__ = [
    "KernelClassifierMetric",
    "ClassifierMetric",
    "MultiKernelClassifierMetric",
]


def _scores_to_auroc(ref_scores, gen_scores):
    ground_truth = np.concatenate(
        [np.ones(len(ref_scores)), np.zeros(len(gen_scores))]
    )
    if ref_scores.ndim == 2:
        assert (
            gen_scores.ndim == 2 and ref_scores.shape[1] == gen_scores.shape[1]
        )
        auroc = [
            roc_auc_score(
                ground_truth,
                np.concatenate([ref_scores[:, i], gen_scores[:, i]]),
            )
            for i in range(gen_scores.shape[1])
        ]
        auroc = np.array(auroc)
    else:
        assert ref_scores.ndim == 1 and gen_scores.ndim == 1
        auroc = roc_auc_score(
            ground_truth, np.concatenate([ref_scores, gen_scores])
        )
    return auroc


def _scores_to_jsd(ref_scores, gen_scores, eps: float = 1e-10):
    """Estimate Jensen-Shannon distance based on classifier probabilities."""
    divergence = 0.5 * (
        np.log2(ref_scores + eps).mean()
        + np.log2(1 - gen_scores + eps).mean()
        + 2
    )
    return np.sqrt(np.clip(divergence, 0, 1))


def _scores_to_informedness_and_threshold(ref_scores, gen_scores):
    ground_truth = np.concatenate(
        [np.ones(len(ref_scores)), np.zeros(len(gen_scores))]
    )
    if ref_scores.ndim == 2:
        assert (
            gen_scores.ndim == 2 and ref_scores.shape[1] == gen_scores.shape[1]
        )
        all_rocs = [
            roc_curve(
                ground_truth,
                np.concatenate([ref_scores[:, i], gen_scores[:, i]]),
            )
            for i in range(gen_scores.shape[1])
        ]
        all_j_statistics = [tpr - fpr for fpr, tpr, _ in all_rocs]
        all_thresholds = [thresholds for _, _, thresholds in all_rocs]
        optimal_idxs = [
            np.argmax(j_statistic) for j_statistic in all_j_statistics
        ]
        assert len(all_thresholds) == len(optimal_idxs)
        optimal_threshold = [
            thresholds[idx]
            for thresholds, idx in zip(all_thresholds, optimal_idxs)
        ]
        assert len(all_j_statistics) == len(optimal_threshold)
        j_statistic = np.array(
            [
                j_statistic[idx]
                for j_statistic, idx in zip(all_j_statistics, optimal_idxs)
            ]
        )
    else:
        assert ref_scores.ndim == 1 and gen_scores.ndim == 1
        fpr, tpr, thresholds = roc_curve(
            ground_truth, np.concatenate([ref_scores, gen_scores])
        )
        j_statistic = tpr - fpr
        optimal_idx = np.argmax(j_statistic)
        optimal_threshold = thresholds[optimal_idx]
        j_statistic = j_statistic[optimal_idx]
    return j_statistic, optimal_threshold


def _scores_and_threshold_to_informedness(ref_scores, gen_scores, threshold):
    ref_pred = (ref_scores >= threshold).astype(int)
    gen_pred = (gen_scores >= threshold).astype(int)
    tpr = np.mean(ref_pred, axis=0)
    fpr = np.mean(gen_pred, axis=0)
    return tpr - fpr


def _train_test_eval(
    ref_vs_ref,
    ref_vs_gen,
    gen_vs_gen,
    ref_train_idx,
    ref_eval_idx,
    gen_train_idx,
    gen_eval_idx,
    variant: Literal[
        "auroc", "informedness-adaptive"
    ] = "informedness-adaptive",
    threshold: Optional[float] = None,
):
    ref_scores = np.mean(
        ref_vs_ref[ref_train_idx][:, ref_eval_idx], axis=0
    ) - np.mean(ref_vs_gen[ref_eval_idx][:, gen_train_idx], axis=1)
    gen_scores = np.mean(
        ref_vs_gen[ref_train_idx][:, gen_eval_idx], axis=0
    ) - np.mean(gen_vs_gen[gen_train_idx][:, gen_eval_idx], axis=0)
    if variant == "auroc":
        return _scores_to_auroc(ref_scores, gen_scores)
    elif variant == "informedness-adaptive":
        return _scores_and_threshold_to_informedness(
            ref_scores, gen_scores, threshold
        )
    else:
        raise ValueError(f"Invalid variant: {variant}")


def _leave_one_out_eval(
    ref_vs_ref,
    ref_vs_gen,
    gen_vs_gen,
    variant: Literal[
        "auroc", "informedness-adaptive"
    ] = "informedness-adaptive",
):
    num_ref = len(ref_vs_ref)
    num_gen = len(gen_vs_gen)
    assert ref_vs_ref.ndim == 2 or ref_vs_ref.ndim == 3
    ref_scores = (
        np.sum(ref_vs_ref, axis=0) - np.diagonal(ref_vs_ref, axis1=0, axis2=1).T
    ) / (num_ref - 1) - np.sum(ref_vs_gen, axis=1) / num_gen
    gen_scores = np.sum(ref_vs_gen, axis=0) / num_ref - (
        np.sum(gen_vs_gen, axis=0) - np.diagonal(gen_vs_gen, axis1=0, axis2=1).T
    ) / (num_gen - 1)
    if variant == "auroc":
        return _scores_to_auroc(ref_scores, gen_scores), None
    elif variant == "informedness-adaptive":
        return _scores_to_informedness_and_threshold(ref_scores, gen_scores)
    else:
        raise ValueError(f"Invalid variant: {variant}")


class KernelClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal[
            "auroc", "informedness-adaptive"
        ] = "informedness-adaptive",
    ):
        self._kernel = kernel
        self._reference_descriptions = self._kernel.featurize(reference_graphs)
        self._variant = variant

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, float]:
        descriptions = self._kernel.featurize(
            generated_graphs,
        )
        ref_vs_ref, ref_vs_gen, gen_vs_gen = self._kernel(
            self._reference_descriptions, descriptions
        )
        num_ref = len(ref_vs_ref)
        num_gen = len(gen_vs_gen)

        rng = np.random.default_rng(0)
        ref_train_idx = rng.choice(num_ref, size=num_ref // 2, replace=False)
        ref_test_idx = np.setdiff1d(np.arange(num_ref), ref_train_idx)
        gen_train_idx = rng.choice(num_gen, size=num_gen // 2, replace=False)
        gen_test_idx = np.setdiff1d(np.arange(num_gen), gen_train_idx)
        train_metric, threshold = _leave_one_out_eval(
            ref_vs_ref[ref_train_idx][:, ref_train_idx],
            ref_vs_gen[ref_train_idx][:, gen_train_idx],
            gen_vs_gen[gen_train_idx][:, gen_train_idx],
            variant=self._variant,
        )
        test_metric = _train_test_eval(
            ref_vs_ref,
            ref_vs_gen,
            gen_vs_gen,
            ref_train_idx,
            ref_test_idx,
            gen_train_idx,
            gen_test_idx,
            variant=self._variant,
            threshold=threshold,
        )
        return train_metric, test_metric


class MultiKernelClassifierMetric(KernelClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal[
            "auroc", "informedness-adaptive"
        ] = "informedness-adaptive",
    ):
        self._kernel = kernel
        self._reference_descriptions = self._kernel.featurize(reference_graphs)
        if not self._kernel.num_kernels > 1:
            raise ValueError(
                "Must provide several kernels, i.e. a kernel with multiple parameters"
            )
        super().__init__(reference_graphs, kernel, variant)

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, float]:
        train_metric, test_metric = super().compute(generated_graphs)
        best_kernel_idx = np.argmax(train_metric)
        return train_metric[best_kernel_idx], test_metric[best_kernel_idx]


def _classifier_cross_validation(
    classifier,
    train_ref_descriptions,
    train_gen_descriptions,
    variant,
    n_folds=4,
):
    """
    Perform stratified k-fold cross-validation with proper threshold selection for informedness.

    Args:
        classifier: The classifier model to use
        train_ref_descriptions: Feature vectors for reference graphs
        train_gen_descriptions: Feature vectors for generated graphs
        variant: Either "auroc" or "informedness"
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

        if variant == "auroc":
            # Compute AUROC
            score = _scores_to_auroc(val_ref_pred, val_gen_pred)
        elif variant == "informedness":
            threshold = 0.5
            score = _scores_and_threshold_to_informedness(
                val_ref_pred, val_gen_pred, threshold
            )
        elif variant == "informedness-adaptive":
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
        "auroc", "informedness", "informedness-adaptive", "jsd"
    ] = "informedness-adaptive",
    classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    rng: Optional[np.random.Generator] = None,
):
    rng = np.random.default_rng(0) if rng is None else rng

    if isinstance(ref_descriptions, csr_array):
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
        classifier = LogisticRegression(penalty="l2", max_iter=1000)
    elif classifier == "tabpfn":
        classifier = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Invalid classifier: {classifier}")

    # Use custom cross-validation function
    scores = _classifier_cross_validation(
        classifier,
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
        classifier.fit(train_all_descriptions, train_labels)
        predict_proba = classifier.predict_proba

    ref_test_pred = predict_proba(test_ref_descriptions)[:, 1]
    gen_test_pred = predict_proba(test_gen_descriptions)[:, 1]

    if variant == "auroc":
        test_metric = _scores_to_auroc(ref_test_pred, gen_test_pred)
    elif variant == "informedness":
        threshold = 0.5
        test_metric = _scores_and_threshold_to_informedness(
            ref_test_pred, gen_test_pred, threshold
        )
    elif variant == "informedness-adaptive":
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


class ClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor: Callable[[List[nx.Graph]], np.ndarray],
        variant: Literal[
            "auroc", "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant
        self._classifier = classifier

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, float]:
        descriptions = self._descriptor(generated_graphs)
        return _descriptions_to_classifier_metric(
            self._reference_descriptions,
            descriptions,
            variant=self._variant,
            classifier=self._classifier,
        )


class _ClassifierMetricSamples:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor: Callable[[List[nx.Graph]], np.ndarray],
        variant: Literal[
            "auroc", "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
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


class AggregateClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptors: Dict[str, Callable[[List[nx.Graph]], np.ndarray]],
        variant: Literal[
            "auroc", "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        self._sub_metrics = {
            name: ClassifierMetric(
                reference_graphs, descriptors[name], variant, classifier
            )
            for name in descriptors
        }

    def compute(self, generated_graphs: Collection[nx.Graph]) -> Dict:
        all_metrics = {
            name: metric.compute(generated_graphs)
            for name, metric in self._sub_metrics.items()
        }  # Select the descriptor with the optimal train metric
        optimal_descriptor = max(
            all_metrics.keys(), key=lambda x: all_metrics[x][0]
        )
        aggregate_metric = all_metrics[optimal_descriptor][1]
        result = {
            "polyscore": aggregate_metric,
            "polyscore_descriptor": optimal_descriptor,
            "subscores": {
                name: metric[1] for name, metric in all_metrics.items()
            },
        }
        return result


class AggregateClassifierMetricInterval:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptors: Dict[str, Callable[[List[nx.Graph]], np.ndarray]],
        variant: Literal[
            "auroc", "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
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
    ) -> Dict:
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
        return result
