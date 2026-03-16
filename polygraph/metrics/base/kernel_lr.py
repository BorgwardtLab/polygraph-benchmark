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
    AdaptiveRBFKernel,
    DescriptorKernel,
    GaussianTV,
    LaplaceKernel,
    LinearKernel,
    RBFKernel,
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
        project_dim: Optional[int] = None,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_jobs: int = 1,
        verbose: bool = False,
        random_state: Optional[int] = None,
        normalize_kernel: bool = True,
        center_kernel: bool = True,
    ):
        self.kernel = kernel
        self.wl_iterations = wl_iterations
        self.use_node_labels = use_node_labels
        self.node_label_key = node_label_key
        self.digest_size = digest_size
        self.project_dim = project_dim
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
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

    def _compute_kernel_matrix(
        self,
        X1: Union[List[nx.Graph], np.ndarray, csr_array],
        X2: Optional[Union[List[nx.Graph], np.ndarray, csr_array]] = None,
    ) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if X2 is None:
            X2 = X1

        if isinstance(X1, (np.ndarray, csr_array)):
            features1 = X1
            features2 = X2

            if self.kernel is not None:
                return self.kernel.pre_gram_block(features1, features2)

            kernel = LinearKernel(lambda x: x)  # pyright: ignore[reportArgumentType]
            return kernel.pre_gram_block(features1, features2)  # pyright: ignore[reportArgumentType]

        if self.kernel is not None:
            features1 = self.kernel.featurize(X1)
            features2 = self.kernel.featurize(X2)  # pyright: ignore[reportArgumentType]
            return self.kernel.pre_gram_block(features1, features2)

        wl_descriptor = WeisfeilerLehmanDescriptor(
            iterations=self.wl_iterations,
            use_node_labels=self.use_node_labels,
            node_label_key=self.node_label_key,
            digest_size=self.digest_size,
            n_jobs=self.n_jobs,
            show_progress=self.verbose,
        )
        kernel = LinearKernel(wl_descriptor)

        features1 = kernel.featurize(X1)
        features2 = kernel.featurize(X2)  # pyright: ignore[reportArgumentType]

        K = kernel.pre_gram_block(features1, features2)
        return K

    def _compute_kernel_diag(
        self, X: Union[List[nx.Graph], np.ndarray, csr_array]
    ) -> np.ndarray:
        """Compute diagonal of the kernel matrix for X."""
        if isinstance(X, (np.ndarray, csr_array)):
            features = X
            if self.kernel is not None:
                kernel = self.kernel
            else:
                kernel = LinearKernel(lambda x: x)  # pyright: ignore[reportArgumentType]
        else:
            if self.kernel is not None:
                kernel = self.kernel
            else:
                wl_descriptor = WeisfeilerLehmanDescriptor(
                    iterations=self.wl_iterations,
                    use_node_labels=self.use_node_labels,
                    node_label_key=self.node_label_key,
                    digest_size=self.digest_size,
                    n_jobs=self.n_jobs,
                    show_progress=False,
                )
                kernel = LinearKernel(wl_descriptor)

            if isinstance(
                kernel,
                (RBFKernel, LaplaceKernel, AdaptiveRBFKernel, GaussianTV),
            ):
                return np.ones(len(X))

            features = kernel.featurize(X)

        if isinstance(
            kernel, (RBFKernel, LaplaceKernel, AdaptiveRBFKernel, GaussianTV)
        ):
            return np.ones(features.shape[0])  # pyright: ignore[reportOptionalSubscript]

        if hasattr(features, "multiply"):
            sq_norms = features.multiply(features).sum(axis=1)  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(sq_norms, "toarray"):
                sq_norms = sq_norms.toarray()
            return np.asarray(sq_norms, dtype=np.float64).flatten()
        else:
            return np.einsum("ij,ij->i", features, features)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def _objective(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> float:
        """Objective function minimizing log-likelihood plus regularization."""
        f = K @ alpha
        yf = y * f
        log_likelihood = np.sum(np.log(1 + np.exp(-yf)))
        regularization = (1.0 / (2.0 * self.C)) * (alpha.T @ K @ alpha)
        return log_likelihood + regularization

    def _gradient(
        self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Gradient of the objective function."""
        f = K @ alpha
        yf = y * f
        sigmoid_neg_yf = self._sigmoid(-yf)
        grad_log_likelihood = -K.T @ (y * sigmoid_neg_yf)
        grad_regularization = (1.0 / self.C) * (K @ alpha)
        return grad_log_likelihood + grad_regularization

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
        alpha_init = np.zeros(len(X))

        result = minimize(
            fun=self._objective,
            x0=alpha_init,
            args=(K, y),
            method="L-BFGS-B",
            jac=self._gradient,
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
