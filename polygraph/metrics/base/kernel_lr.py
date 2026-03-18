"""Kernel Logistic Regression classifier.

Implements kernel logistic regression for graph classification using
kernel functions computed on graph descriptor features.
"""

import warnings
from typing import List, Optional, Union

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KernelCenterer

from polygraph.utils.descriptors import WeisfeilerLehmanDescriptor
from polygraph.utils.kernels import (
    DescriptorKernel,
    LinearKernel,
)

__all__ = ["KernelLogisticRegression"]


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """Kernel Logistic Regression classifier.

    Uses the representer theorem where the decision function is expressed as
    a linear combination of kernel evaluations with training examples.
    """

    def __init__(
        self,
        kernel: Optional[DescriptorKernel] = None,
        wl_iterations: int = 3,
        use_node_labels: bool = False,
        node_label_key: Optional[str] = None,
        digest_size: int = 4,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_jobs: int = 1,
        verbose: bool = False,
        normalize_kernel: bool = True,
        center_kernel: bool = True,
    ):
        self.kernel = kernel
        self.wl_iterations = wl_iterations
        self.use_node_labels = use_node_labels
        self.node_label_key = node_label_key
        self.digest_size = digest_size
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.normalize_kernel = normalize_kernel
        self.center_kernel = center_kernel

        self.alpha_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.classes_ = None
        self.train_diag_ = None
        self.centerer_ = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _resolve_kernel(
        self,
        is_array: bool,
        show_progress: bool = False,
    ) -> DescriptorKernel:
        """Return the kernel to use for computation."""
        if self.kernel is not None:
            return self.kernel
        if is_array:
            return LinearKernel(lambda x: x)  # pyright: ignore[reportArgumentType,reportReturnType]
        wl_descriptor = WeisfeilerLehmanDescriptor(
            iterations=self.wl_iterations,
            use_node_labels=self.use_node_labels,
            node_label_key=self.node_label_key,
            digest_size=self.digest_size,
            n_jobs=self.n_jobs,
            show_progress=show_progress,
        )
        return LinearKernel(wl_descriptor)

    def _compute_kernel_matrix(
        self,
        X1: Union[List[nx.Graph], np.ndarray, csr_array],
        X2: Optional[Union[List[nx.Graph], np.ndarray, csr_array]] = None,
    ) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        is_array = isinstance(X1, (np.ndarray, csr_array))
        kernel = self._resolve_kernel(is_array, show_progress=self.verbose)

        if X2 is None:
            features1 = X1 if is_array else kernel.featurize(X1)
            return kernel.pre_gram_block(features1, features1)  # pyright: ignore[reportArgumentType]

        features1 = X1 if is_array else kernel.featurize(X1)
        features2 = X2 if is_array else kernel.featurize(X2)  # pyright: ignore[reportArgumentType]
        return kernel.pre_gram_block(features1, features2)  # pyright: ignore[reportArgumentType]

    def _compute_kernel_diag(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array]
    ) -> np.ndarray:
        """Compute diagonal of the kernel matrix for X."""
        is_array = isinstance(X, (np.ndarray, csr_array))
        kernel = self._resolve_kernel(is_array)
        features = X if is_array else kernel.featurize(X)
        return kernel.kernel_diag(features)  # pyright: ignore[reportArgumentType]

    def _objective_and_gradient(
        self,
        alpha: np.ndarray,
        K: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Objective and gradient for L-BFGS-B optimization.

        Returns both to avoid redundant K @ alpha computation.
        """
        Ka = K @ alpha
        yf = y * Ka
        log_likelihood = np.sum(np.logaddexp(0, -yf))
        regularization = (1.0 / (2.0 * self.C)) * (alpha.T @ Ka)
        objective = log_likelihood + regularization

        sigmoid_neg_yf = self._sigmoid(-yf)
        grad_log_likelihood = -K.T @ (y * sigmoid_neg_yf)
        grad_regularization = (1.0 / self.C) * Ka
        gradient = grad_log_likelihood + grad_regularization

        return objective, gradient

    def fit(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array], y: np.ndarray
    ) -> "KernelLogisticRegression":
        """Fit the kernel logistic regression model."""
        y = np.asarray(y).flatten()
        self.classes_ = np.unique(y)

        if set(y) == {0, 1}:
            y = 2 * y - 1

        if not set(y) <= {-1, 1}:
            raise ValueError("Labels must be binary (-1/1 or 0/1)")

        self.X_train_ = X
        self.y_train_ = y

        K = self._compute_kernel_matrix(X)
        if self.normalize_kernel:
            self.train_diag_ = np.diag(K).astype(np.float64)
            inv_sqrt_diag = np.zeros(len(self.train_diag_), dtype=np.float64)
            mask = self.train_diag_ > 0
            inv_sqrt_diag[mask] = 1.0 / np.sqrt(self.train_diag_[mask])

            K = K * np.outer(inv_sqrt_diag, inv_sqrt_diag)

        if self.center_kernel:
            self.centerer_ = KernelCenterer().fit(K)
            K = self.centerer_.transform(K)

        self.kernel_ = K
        alpha_init = np.zeros(K.shape[0])

        result = minimize(
            fun=self._objective_and_gradient,
            x0=alpha_init,
            args=(K, y),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if not result.success:
            warnings.warn(
                f"Optimization did not converge. Message: {result.message}"
            )

        self.alpha_ = result.x
        return self

    def decision_function(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array]
    ) -> np.ndarray:
        """Compute decision function values for test data."""
        if self.alpha_ is None:
            raise ValueError("Model must be fitted before prediction")

        K_test = self._compute_kernel_matrix(X, self.X_train_)

        if self.normalize_kernel:
            assert self.train_diag_ is not None
            test_diag = self._compute_kernel_diag(X)

            inv_sqrt_train = np.zeros(len(self.train_diag_), dtype=np.float64)
            mask_train = self.train_diag_ > 0
            inv_sqrt_train[mask_train] = 1.0 / np.sqrt(
                self.train_diag_[mask_train]
            )

            test_diag = np.asarray(test_diag, dtype=np.float64)
            inv_sqrt_test = np.zeros(len(test_diag), dtype=np.float64)
            mask_test = test_diag > 0
            inv_sqrt_test[mask_test] = 1.0 / np.sqrt(test_diag[mask_test])

            K_test = K_test * inv_sqrt_test[:, None] * inv_sqrt_train[None, :]

        if self.center_kernel:
            assert self.centerer_ is not None
            K_test = self.centerer_.transform(K_test)

        return K_test @ self.alpha_

    def predict_proba(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array]
    ) -> np.ndarray:
        """Predict class probabilities."""
        f = self.decision_function(X)
        prob_positive = self._sigmoid(f)
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    def predict(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array]
    ) -> np.ndarray:
        """Predict class labels."""
        f = self.decision_function(X)
        return np.sign(f).astype(int)
