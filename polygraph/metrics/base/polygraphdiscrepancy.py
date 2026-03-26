"""
PolyGraph discrepancy metrics compare generated graphs to reference graphs by fitting a binary classifier to discriminate between the two.
Performance metrics of this classifier lower-bound intrinsic probability metrics.
Multiple graph descriptors may be combined within the metric to yield a theoretically grounded summary metric.

Any binary classifier implementing the standard scikit-learn interface (as defined in [`ClassifierProtocol`][polygraph.metrics.base.polygraphdiscrepancy.ClassifierProtocol])
may be used for classification. By default, we use [TabPFN](https://github.com/PriorLabs/TabPFN).
The classifiers may be then be evaluated by either:

- Data log-likelihood (default) - Provides a lower bound on the Jensen-Shannon distance, or
- Informedness - Provides a lower bound on the total variation distance.

The [`PolyGraphDiscrepancy`][polygraph.metrics.base.polygraphdiscrepancy.PolyGraphDiscrepancy] class combines metrics across multiple graph descriptors,
providing the tightest lower-bound on the probability metrics.
The [`ClassifierMetric`][polygraph.metrics.base.polygraphdiscrepancy.ClassifierMetric] class, on the other hand, computes a lower bound
for a single graph descriptor.

The [`PolyGraphDiscrepancyInterval`][polygraph.metrics.base.polygraphdiscrepancy.PolyGraphDiscrepancyInterval] class implements a variant of the PolyGraphDiscrepancy
with uncertainty quantification.

Example:
    ```python
    from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
    from polygraph.metrics.base import PolyGraphDiscrepancy
    from polygraph.utils.descriptors import OrbitCounts, SparseDegreeHistogram

    reference = PlanarGraphDataset("val").to_nx()
    generated = SBMGraphDataset("val").to_nx()

    benchmark = PolyGraphDiscrepancy(
        reference,
        descriptors={
            "orbit": OrbitCounts(),
            "degree": SparseDegreeHistogram(),
        },
    )
    print(benchmark.compute(generated))         # {'pgd': 0.9975117559449073, 'pgd_descriptor': 'degree', 'subscores': {'orbit': 0.9962500491652303, 'degree': 0.9975117559449073}}
    ```

"""

import warnings
from collections import Counter
from importlib.metadata import version
from typing import (
    Collection,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from scipy.sparse import csr_array, issparse
from scipy.sparse import vstack as sparse_vstack
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from packaging.version import Version
from tabpfn import TabPFNClassifier
from tabpfn.classifier import ModelVersion

from polygraph import GraphType
from polygraph.metrics.base.interface import GenerationMetric
from polygraph.metrics.base.metric_interval import MetricInterval
from polygraph.utils.descriptors import GraphDescriptor

__all__ = [
    "ClassifierMetric",
    "PolyGraphDiscrepancy",
    "PolyGraphDiscrepancyInterval",
    "default_classifier",
]


class ClassifierProtocol(Protocol):
    """Protocol for binary classifiers used in the PolyGraph discrepancy metric."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierProtocol":
        """Fit the classifier to the data.

        Args:
            X: Feature matrix of shape `(n_samples, n_features)`
            y: Labels of shape `(n_samples,)`

        Returns:
            self: The fitted classifier
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability of the positive class.

        Args:
            X: Feature matrix of shape `(n_samples, n_features)`

        Returns:
            Probabilities of shape `(n_samples, 2)`
        """
        ...


def default_classifier() -> TabPFNClassifier:
    """Create the default TabPFN classifier used by PGD.

    Returns:
        A TabPFNClassifier with v2.5 weights (auto device, 4
        estimators). Requires ``tabpfn >= 2.0.9``.
    """
    tabpfn_ver = Version(version("tabpfn"))
    if tabpfn_ver < Version("2.0.9"):
        raise RuntimeError(
            "TabPFN >= 2.0.9 is required. "
            "Install with `pip install 'tabpfn>=2.0.9'`."
        )
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        device="auto",
        n_estimators=4,
    )


class PolyGraphDiscrepancyResult(TypedDict):
    """Return type for PolyGraphDiscrepancy.compute method."""

    pgd: float
    pgd_descriptor: str
    subscores: Dict[str, float]


class PolyGraphDiscrepancyIntervalResult(TypedDict):
    """Return type for `PolyGraphDiscrepancyInterval.compute` method."""

    pgd: MetricInterval
    subscores: Dict[str, MetricInterval]
    pgd_descriptor: Dict[str, float]


def _vstack(arrays):
    """Stack arrays vertically, handling both dense and sparse."""
    if any(issparse(a) for a in arrays):
        return sparse_vstack(arrays, format="csr")
    return np.concatenate(arrays, axis=0)


def _is_constant(X) -> bool:
    """Check if all rows of X are identical."""
    if issparse(X):
        if X.shape[0] <= 1:
            return True
        col_min = X.min(axis=0).toarray().ravel()
        col_max = X.max(axis=0).toarray().ravel()
        return bool(np.array_equal(col_min, col_max))
    return bool(np.all(X == X[0]))


def _scores_to_jsd(ref_scores, gen_scores, eps: float = 1e-10) -> float:
    """Estimate Jensen-Shannon distance based on classifier probabilities."""
    divergence = 0.5 * (
        np.log2(ref_scores + eps).mean()
        + np.log2(1 - gen_scores + eps).mean()
        + 2
    )
    return np.sqrt(np.clip(divergence, 0, 1)).item()


def _scores_to_informedness_and_threshold(
    ref_scores: np.ndarray, gen_scores: np.ndarray
) -> Tuple[float, float]:
    ground_truth = np.concatenate(
        [np.ones(len(ref_scores)), np.zeros(len(gen_scores))]
    )
    if ref_scores.ndim != 1:
        raise RuntimeError(
            f"ref_scores must be 1-dimensional, got shape {ref_scores.shape}. This should not happen, please file a bug report."
        )

    assert ref_scores.ndim == 1 and gen_scores.ndim == 1
    fpr, tpr, thresholds = roc_curve(
        ground_truth, np.concatenate([ref_scores, gen_scores])
    )
    j_statistic = tpr - fpr
    optimal_idx = np.argmax(j_statistic)
    optimal_threshold = thresholds[optimal_idx]
    j_statistic = j_statistic[optimal_idx]
    return j_statistic, optimal_threshold


def _scores_and_threshold_to_informedness(
    ref_scores: np.ndarray, gen_scores: np.ndarray, threshold: float
) -> float:
    assert ref_scores.ndim == 1 and gen_scores.ndim == 1
    ref_pred = (ref_scores >= threshold).astype(int)
    gen_pred = (gen_scores >= threshold).astype(int)
    tpr = np.mean(ref_pred, axis=0).item()
    fpr = np.mean(gen_pred, axis=0).item()
    return tpr - fpr


def _classifier_cross_validation(
    classifier: ClassifierProtocol,
    train_ref_descriptions: Union[np.ndarray, csr_array],
    train_gen_descriptions: Union[np.ndarray, csr_array],
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
    X = _vstack([train_ref_descriptions, train_gen_descriptions])
    n_ref = train_ref_descriptions.shape[0]  # pyright: ignore[reportOptionalSubscript]
    n_gen = train_gen_descriptions.shape[0]  # pyright: ignore[reportOptionalSubscript]
    y = np.concatenate(
        [
            np.ones(n_ref),
            np.zeros(n_gen),
        ]
    )

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    scores = []

    for train_idx, val_idx in skf.split(
        X if not issparse(X) else np.zeros((X.shape[0], 1)),  # pyright: ignore[reportOptionalSubscript]
        y,
    ):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]  # pyright: ignore[reportIndexIssue]
        y_train, y_val = y[train_idx], y[val_idx]

        try:
            # Train model
            if _is_constant(X_train):
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
        except (ValueError, RuntimeError) as e:
            # TabPFN v2.0.9 can raise ValueError with NaN in encoder when
            # features are near-constant (see github.com/PriorLabs/TabPFN/issues/108).
            # Near-constant features mean distributions are indistinguishable,
            # so the correct score is 0.
            warnings.warn(
                f"Classifier failed in CV fold ({e}), treating as "
                f"indistinguishable (score=0)."
            )
            score = 0.0

        scores.append(score)

    return scores


def _descriptions_to_classifier_metric(
    ref_descriptions: Union[np.ndarray, csr_array],
    gen_descriptions: Union[np.ndarray, csr_array],
    variant: Literal["informedness", "jsd"] = "jsd",
    *,
    classifier: ClassifierProtocol,
    rng: Optional[np.random.Generator] = None,
    scale: bool = True,
) -> Tuple[float, Union[int, float]]:
    # scale=False is needed for kernel-based classifiers (e.g. GKLR)
    # that operate on precomputed kernel matrices, not raw features.
    rng = np.random.default_rng(0) if rng is None else rng

    if isinstance(ref_descriptions, csr_array):
        assert isinstance(gen_descriptions, csr_array)
        # Align sparse dimensions
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
            shape=(gen_descriptions.shape[0], num_features),  # pyright: ignore
        )
        ref_descriptions = csr_array(
            (
                ref_descriptions.data,
                ref_descriptions.indices,
                ref_descriptions.indptr,
            ),
            shape=(ref_descriptions.shape[0], num_features),  # pyright: ignore
        )
        if scale or isinstance(classifier, TabPFNClassifier):
            gen_descriptions = gen_descriptions.toarray()
            ref_descriptions = ref_descriptions.toarray()

    n_ref = ref_descriptions.shape[0]  # pyright: ignore[reportOptionalSubscript]
    n_gen = gen_descriptions.shape[0]  # pyright: ignore[reportOptionalSubscript]
    ref_train_idx = rng.choice(
        n_ref,
        size=n_ref // 2,
        replace=False,
    )
    ref_test_idx = np.setdiff1d(np.arange(n_ref), ref_train_idx)
    gen_train_idx = rng.choice(n_gen, size=n_gen // 2, replace=False)
    gen_test_idx = np.setdiff1d(np.arange(n_gen), gen_train_idx)

    train_ref_descriptions = ref_descriptions[ref_train_idx]
    train_gen_descriptions = gen_descriptions[gen_train_idx]
    test_ref_descriptions = ref_descriptions[ref_test_idx]
    test_gen_descriptions = gen_descriptions[gen_test_idx]
    if scale:
        scaler = StandardScaler()
        scaler.fit(
            np.concatenate(
                [train_ref_descriptions, train_gen_descriptions], axis=0
            )
        )
        test_ref_descriptions = scaler.transform(test_ref_descriptions)
        test_gen_descriptions = scaler.transform(test_gen_descriptions)
        train_ref_descriptions = scaler.transform(train_ref_descriptions)
        train_gen_descriptions = scaler.transform(train_gen_descriptions)
    assert isinstance(train_ref_descriptions, (np.ndarray, csr_array))
    assert isinstance(train_gen_descriptions, (np.ndarray, csr_array))
    assert isinstance(test_ref_descriptions, (np.ndarray, csr_array))
    assert isinstance(test_gen_descriptions, (np.ndarray, csr_array))

    scores = _classifier_cross_validation(
        classifier,
        train_ref_descriptions,
        train_gen_descriptions,
        variant,
        n_folds=4,
    )

    train_metric = np.mean(scores).item()

    # Refit on all training data
    train_all_descriptions = _vstack(
        [train_ref_descriptions, train_gen_descriptions]
    )
    train_labels = np.concatenate(
        [
            np.ones(train_ref_descriptions.shape[0]),  # pyright: ignore[reportOptionalSubscript]
            np.zeros(train_gen_descriptions.shape[0]),  # pyright: ignore[reportOptionalSubscript]
        ]
    )

    # Check if train_all_descriptions is constant across rows
    try:
        if _is_constant(train_all_descriptions):
            warnings.warn(
                "Input to classifier is constant, setting all scores to 0.5"
            )
            predict_proba = lambda x: np.ones((x.shape[0], 2)) * 0.5
        else:
            classifier.fit(train_all_descriptions, train_labels)  # pyright: ignore[reportArgumentType]
            predict_proba = classifier.predict_proba

        ref_test_pred = predict_proba(test_ref_descriptions)[:, 1]  # pyright: ignore[reportArgumentType]
        gen_test_pred = predict_proba(test_gen_descriptions)[:, 1]  # pyright: ignore[reportArgumentType]

        if variant == "informedness":
            ref_train_pred = predict_proba(train_ref_descriptions)[:, 1]  # pyright: ignore[reportArgumentType]
            gen_train_pred = predict_proba(train_gen_descriptions)[:, 1]  # pyright: ignore[reportArgumentType]
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
    except (ValueError, RuntimeError) as e:
        # TabPFN v2.0.9 can raise ValueError with NaN in encoder when
        # features are near-constant (see github.com/PriorLabs/TabPFN/issues/108).
        # Near-constant features mean distributions are indistinguishable,
        # so the correct metric is 0.
        warnings.warn(
            f"Classifier failed during refit ({e}), treating as "
            f"indistinguishable (metric=0)."
        )
        test_metric = 0.0

    assert isinstance(train_metric, float)
    assert isinstance(test_metric, float)
    return train_metric, test_metric


class ClassifierMetric(GenerationMetric[GraphType], Generic[GraphType]):
    """Classifier-based metric using a single graph descriptor.

    Args:
        reference_graphs: Reference graphs
        descriptor: Descriptor function
        variant: Classifier metric to compute. To estimate the Jensen-Shannon distance, use "jsd". To estimate total variation distance, use "informedness".
        classifier: Binary classifier to fit. Defaults to TabPFN
            via ``default_classifier()``.
    """

    _variant: Literal["informedness", "jsd"]
    _classifier: ClassifierProtocol

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        descriptor: GraphDescriptor[GraphType],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant
        self._classifier = (
            classifier if classifier is not None else default_classifier()
        )

    def compute(
        self, generated_graphs: Collection[GraphType]
    ) -> Tuple[float, float]:
        """Compute the classifier metric.

        Args:
            generated_graphs: Generated graphs

        Returns:
            Tuple of train and test metric
        """
        descriptions = self._descriptor(generated_graphs)
        assert (
            self._reference_descriptions.shape is not None
            and descriptions.shape is not None
        )

        if descriptions.shape[0] != self._reference_descriptions.shape[0]:
            raise ValueError(
                f"Number of generated graphs must be equal to the number of reference graphs. Got {descriptions.shape[0]} and {self._reference_descriptions.shape[0]}."
            )

        return _descriptions_to_classifier_metric(
            self._reference_descriptions,
            descriptions,
            variant=self._variant,
            classifier=self._classifier,
        )


class _ClassifierMetricSamples(Generic[GraphType]):
    _variant: Literal["informedness", "jsd"]
    _classifier: ClassifierProtocol

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        descriptor: GraphDescriptor[GraphType],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
        scale: bool = True,
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant
        self._classifier = (
            classifier if classifier is not None else default_classifier()
        )
        self._scale = scale

    def compute(
        self,
        generated_graphs: Collection[GraphType],
        subsample_size: int,
        num_samples: int = 100,
    ) -> np.ndarray:
        descriptions = self._descriptor(generated_graphs)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(num_samples):
            ref_idx = rng.choice(
                self._reference_descriptions.shape[0],  # pyright: ignore
                size=subsample_size,
                replace=False,
            )
            gen_idx = rng.choice(
                descriptions.shape[0],  # pyright: ignore
                size=subsample_size,
                replace=False,
            )
            samples.append(
                _descriptions_to_classifier_metric(
                    self._reference_descriptions[ref_idx],
                    descriptions[gen_idx],
                    variant=self._variant,
                    rng=rng,
                    classifier=self._classifier,
                    scale=self._scale,
                )
            )
        samples = np.array(samples)
        return samples


class PolyGraphDiscrepancy(GenerationMetric[GraphType], Generic[GraphType]):
    """PolyGraphDiscrepancy to compare graph distributions, combining multiple graph descriptors.

    Args:
        reference_graphs: Reference graphs
        descriptors: Dictionary of descriptor names and descriptor functions
        variant: Classifier metric to compute. To estimate the Jensen-Shannon distance, use "jsd". To estimate total variation distance, use "informedness".
        classifier: Binary classifier to fit. Defaults to TabPFN
            via ``default_classifier()``.
    """

    _variant: Literal["informedness", "jsd"]

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        descriptors: Dict[str, GraphDescriptor[GraphType]],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        resolved = (
            classifier if classifier is not None else default_classifier()
        )
        self._sub_metrics = {
            name: ClassifierMetric(
                reference_graphs, descriptors[name], variant, resolved
            )
            for name in descriptors
        }

    def compute(
        self, generated_graphs: Collection[GraphType]
    ) -> PolyGraphDiscrepancyResult:
        """Compute the PolyGraphDiscrepancy.

        Args:
            generated_graphs: Generated graphs

        Returns:
            Typed dictionary of scores.
                The key `"pgd"` specifies the PolyGraphDiscrepancy, giving the estimated tightest lower-bound on the probability metric.
                The key `"pgd_descriptor"` specifies the descriptor that achieves this bound.
                All descriptor-wise scores are returned in the key `"subscores"`.
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
            "pgd": aggregate_metric,
            "pgd_descriptor": optimal_descriptor,
            "subscores": {
                name: metric[1] for name, metric in all_metrics.items()
            },
        }
        return PolyGraphDiscrepancyResult(**result)


class PolyGraphDiscrepancyInterval(
    GenerationMetric[GraphType], Generic[GraphType]
):
    """Uncertainty quantification for [`PolyGraphDiscrepancy`][polygraph.metrics.base.polygraphdiscrepancy.PolyGraphDiscrepancy].

    Args:
        reference_graphs: Reference graphs. Must provide at least `2 * subsample_size` graphs.
        descriptors: Dictionary of descriptor names and descriptor functions
        subsample_size: Size of each subsample, should be consistent with the number
            of reference and generated graphs passed to [`PolyGraphDiscrepancy`][polygraph.metrics.base.polygraphdiscrepancy.PolyGraphDiscrepancy]
            for point estimates.
        num_samples: Number of samples to draw for uncertainty quantification.
    """

    _variant: Literal["informedness", "jsd"]

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        descriptors: Dict[str, GraphDescriptor[GraphType]],
        subsample_size: int,
        num_samples: int = 10,
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
        scale: bool = True,
    ):
        if len(reference_graphs) < 2 * subsample_size:
            raise ValueError(
                "Number of reference graphs must be at least 2 * subsample_size"
            )

        resolved = (
            classifier if classifier is not None else default_classifier()
        )
        self._sub_metrics = {
            name: _ClassifierMetricSamples(
                reference_graphs, descriptors[name], variant, resolved, scale
            )
            for name in descriptors
        }
        self._subsample_size = subsample_size
        self._num_samples = num_samples

    def compute(
        self,
        generated_graphs: Collection[GraphType],
    ) -> PolyGraphDiscrepancyIntervalResult:
        """Compute the PolyGraphDiscrepancyInterval.

        Args:
            generated_graphs: Generated graphs. Must provide at least `2 * subsample_size` graphs.

        Returns:
            Typed dictionary of scores.
                The key `"pgd"` specifies the PolyGraphDiscrepancy, giving mean and standard deviation as [`MetricInterval`][polygraph.metrics.base.metric_interval.MetricInterval] objects.
                The key `"pgd_descriptor"` describes which descriptors achieve this score. This is a dictionary mapping descriptor names to the ratio of samples in which the descriptor was chosen.
                All descriptor-wise scores are returned in the key `"subscores"`. These are [`MetricInterval`][polygraph.metrics.base.metric_interval.MetricInterval] objects.
        """
        if len(generated_graphs) < 2 * self._subsample_size:
            raise ValueError(
                "Number of generated graphs must be at least 2 * subsample_size"
            )
        all_sub_samples = {
            name: metric.compute(
                generated_graphs, self._subsample_size, self._num_samples
            )
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
            for i in range(self._num_samples)
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
            "pgd": aggregate_interval,
            "subscores": all_sub_intervals,
            "pgd_descriptor": {
                key: descriptor_counts[key] / self._num_samples
                for key in self._sub_metrics.keys()
            },
        }
        return PolyGraphDiscrepancyIntervalResult(**result)
