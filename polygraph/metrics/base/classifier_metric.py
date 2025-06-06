from typing import Collection, Literal, Optional, Callable, List, Tuple, Dict

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.sparse import csr_array

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from polygraph.utils.kernels import DescriptorKernel

__all__ = [
    "KernelClassifierMetric",
    "LogisticRegressionClassifierMetric",
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
    variant: Literal["auroc", "informedness"] = "auroc",
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
    elif variant == "informedness":
        return _scores_and_threshold_to_informedness(
            ref_scores, gen_scores, threshold
        )


def _leave_one_out_eval(
    ref_vs_ref,
    ref_vs_gen,
    gen_vs_gen,
    variant: Literal["auroc", "informedness"] = "auroc",
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
    elif variant == "informedness":
        return _scores_to_informedness_and_threshold(ref_scores, gen_scores)
    else:
        raise ValueError(f"Invalid variant: {variant}")


class KernelClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
        variant: Literal["auroc", "informedness"] = "informedness",
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
        variant: Literal["auroc", "informedness"] = "informedness",
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
    regressor,
    train_ref_descriptions,
    train_gen_descriptions,
    variant,
    n_folds=5,
):
    """
    Perform stratified k-fold cross-validation with proper threshold selection for informedness.

    Args:
        regressor: The classifier model to use
        train_ref_descriptions: Feature vectors for reference graphs
        train_gen_descriptions: Feature vectors for generated graphs
        variant: Either "auroc" or "informedness"
        n_folds: Number of cross-validation folds

    Returns:
        List of scores for each fold
    """
    from sklearn.model_selection import StratifiedKFold

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
        regressor.fit(X_train, y_train)

        # Get validation predictions
        val_pred = regressor.predict_proba(X_val)[:, 1]

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
            # For informedness, we need to find the threshold on training data
            train_pred = regressor.predict_proba(X_train)[:, 1]
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
        else:
            raise ValueError(f"Invalid variant: {variant}")

        scores.append(score)

    return scores


class LogisticRegressionClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor: Callable[[List[nx.Graph]], np.ndarray],
        variant: Literal["auroc", "informedness"] = "informedness",
    ):
        self._descriptor = descriptor
        self._reference_descriptions = self._descriptor(reference_graphs)
        self._variant = variant

    def compute(
        self, generated_graphs: Collection[nx.Graph]
    ) -> Tuple[float, float]:
        descriptions = self._descriptor(generated_graphs)
        rng = np.random.default_rng(0)
        if isinstance(self._reference_descriptions, csr_array):
            # Convert to dense array
            num_features = (
                max(
                    self._reference_descriptions.indices.max(),
                    descriptions.indices.max(),
                )
                + 1
            )
            descriptions = csr_array(
                (descriptions.data, descriptions.indices, descriptions.indptr),
                shape=(descriptions.shape[0], num_features),
            ).toarray()
            ref_descriptions = csr_array(
                (
                    self._reference_descriptions.data,
                    self._reference_descriptions.indices,
                    self._reference_descriptions.indptr,
                ),
                shape=(self._reference_descriptions.shape[0], num_features),
            ).toarray()
        else:
            ref_descriptions = self._reference_descriptions

        ref_train_idx = rng.choice(
            len(ref_descriptions),
            size=len(ref_descriptions) // 2,
            replace=False,
        )
        ref_test_idx = np.setdiff1d(
            np.arange(len(ref_descriptions)), ref_train_idx
        )
        gen_train_idx = rng.choice(
            len(descriptions), size=len(descriptions) // 2, replace=False
        )
        gen_test_idx = np.setdiff1d(np.arange(len(descriptions)), gen_train_idx)

        scaler = StandardScaler()
        train_ref_descriptions = ref_descriptions[ref_train_idx]
        train_gen_descriptions = descriptions[gen_train_idx]
        test_ref_descriptions = ref_descriptions[ref_test_idx]
        test_gen_descriptions = descriptions[gen_test_idx]
        scaler.fit(
            np.concatenate(
                [train_ref_descriptions, train_gen_descriptions], axis=0
            )
        )
        test_ref_descriptions = scaler.transform(test_ref_descriptions)
        test_gen_descriptions = scaler.transform(test_gen_descriptions)
        train_ref_descriptions = scaler.transform(train_ref_descriptions)
        train_gen_descriptions = scaler.transform(train_gen_descriptions)

        regressor = LogisticRegression(penalty="l2", max_iter=1000)

        # Use custom cross-validation function
        scores = _classifier_cross_validation(
            regressor,
            train_ref_descriptions,
            train_gen_descriptions,
            self._variant,
            n_folds=5,
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
        regressor.fit(train_all_descriptions, train_labels)

        ref_test_pred = regressor.decision_function(test_ref_descriptions)
        gen_test_pred = regressor.decision_function(test_gen_descriptions)

        if self._variant == "auroc":
            test_metric = _scores_to_auroc(ref_test_pred, gen_test_pred)
        elif self._variant == "informedness":
            ref_train_pred = regressor.decision_function(train_ref_descriptions)
            gen_train_pred = regressor.decision_function(train_gen_descriptions)
            _, threshold = _scores_to_informedness_and_threshold(
                ref_train_pred, gen_train_pred
            )
            test_metric = _scores_and_threshold_to_informedness(
                ref_test_pred, gen_test_pred, threshold
            )
        else:
            raise ValueError(f"Invalid variant: {self._variant}")
        return train_metric, test_metric


class AggregateLogisticRegressionClassifierMetric:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptors: Dict[str, Callable[[List[nx.Graph]], np.ndarray]],
        variant: Literal["auroc", "informedness"] = "informedness",
    ):
        self._sub_metrics = {
            name: LogisticRegressionClassifierMetric(
                reference_graphs, descriptors[name], variant
            )
            for name in descriptors
        }

    def compute(self, generated_graphs: Collection[nx.Graph]) -> Dict:
        all_metrics = {
            name: metric.compute(generated_graphs)
            for name, metric in self._sub_metrics.items()
        }
        # Select the descriptor with the optimal train metric
        optimal_descriptor = max(
            all_metrics.values(), key=lambda x: all_metrics[x][0]
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
