import numpy as np
import pytest
from scipy.sparse import csr_array

from polygraph.utils.sparse_dist import (
    sparse_dot_product,
    sparse_euclidean,
    sparse_manhattan,
)


def test_small_arrays():
    # Create small dense arrays
    X_dense = np.array([[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 5, 0]])
    Y_dense = np.array([[1, 0, 1, 0], [0, 2, 0, 0], [3, 0, 2, 1]])

    # Convert to sparse
    X_sparse = csr_array(X_dense)
    Y_sparse = csr_array(Y_dense)

    # Test dot product
    sparse_dot = sparse_dot_product(X_sparse, Y_sparse)
    dense_dot = X_dense @ Y_dense.T
    np.testing.assert_allclose(sparse_dot, dense_dot)

    # Test euclidean distance
    sparse_euc = sparse_euclidean(X_sparse, Y_sparse)
    X_exp = X_dense[:, np.newaxis, :]
    Y_exp = Y_dense[np.newaxis, :, :]
    dense_euc = np.sqrt(np.sum((X_exp - Y_exp) ** 2, axis=2))
    np.testing.assert_allclose(sparse_euc, dense_euc)

    # Test manhattan distance
    sparse_man = sparse_manhattan(X_sparse, Y_sparse)
    dense_man = np.sum(np.abs(X_exp - Y_exp), axis=2)
    np.testing.assert_allclose(sparse_man, dense_man)


def test_edge_cases():
    # Test empty matrices
    X = csr_array((1, 10))
    Y = csr_array((2, 10))

    for dist_func in [sparse_dot_product, sparse_euclidean, sparse_manhattan]:
        result = dist_func(X, Y)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result, np.zeros((1, 2)))

    # Test identical vectors
    X_dense = np.array([[1, 0, 2, 0], [0, 3, 0, 0]])
    X_sparse = csr_array(X_dense)

    dot_result = sparse_dot_product(X_sparse, X_sparse)
    np.testing.assert_allclose(np.diag(dot_result), np.array([5, 9]))

    for dist_func in [sparse_euclidean, sparse_manhattan]:
        result = dist_func(X_sparse, X_sparse)
        np.testing.assert_allclose(np.diag(result), np.zeros(2))

    # Test zero vectors
    X = csr_array((2, 5))  # All zeros
    Y = csr_array(([1.0, 2.0], ([0, 1], [2, 3])), shape=(2, 5))

    man_result = sparse_manhattan(X, Y)
    expected = np.array([[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(man_result, expected)


def test_dimension_mismatch():
    X = csr_array((2, 10))
    Y = csr_array((2, 11))

    for dist_func in [sparse_dot_product, sparse_euclidean, sparse_manhattan]:
        with pytest.raises(ValueError):
            dist_func(X, Y)


@pytest.mark.slow
def test_large_sparse_arrays():
    # Create large sparse arrays (100 x 10^6)
    n_rows = 100
    n_cols = 10**6
    density = 0.0001  # 0.01% density

    rng = np.random.RandomState(42)  # For reproducibility
    nnz = int(n_rows * n_cols * density)

    def create_sparse_matrix():
        elements_per_row = min(
            nnz // n_rows, n_cols // 10
        )  # Ensure we don't exceed columns
        actual_nnz = elements_per_row * n_rows
        indptr = np.arange(0, actual_nnz + 1, elements_per_row, dtype=np.int32)
        data = rng.randn(actual_nnz)
        indices = np.zeros(actual_nnz, dtype=np.int32)

        for i in range(n_rows):
            start, end = indptr[i], indptr[i + 1]
            row_indices = np.sort(
                rng.choice(n_cols, size=elements_per_row, replace=False)
            )
            indices[start:end] = row_indices

        return csr_array((data, indices, indptr), shape=(n_rows, n_cols))

    X = create_sparse_matrix()
    Y = create_sparse_matrix()

    # Test all distance functions
    for dist_func in [sparse_dot_product, sparse_euclidean, sparse_manhattan]:
        try:
            result = dist_func(X, Y)
            assert result.shape == (n_rows, n_rows)
        except Exception as e:
            pytest.fail(
                f"Failed to compute {dist_func.__name__} for large arrays: {str(e)}"
            )


def test_direct_csr_construction():
    # Create CSR arrays directly from components
    X_data = np.array([1.0, 2.0, 3.0])
    X_indices = np.array([0, 2, 1])
    X_indptr = np.array([0, 2, 3])  # 2 rows
    X = csr_array((X_data, X_indices, X_indptr), shape=(2, 1000))

    Y_data = np.array([2.0, 1.0, 3.0])
    Y_indices = np.array([0, 2, 1])
    Y_indptr = np.array([0, 2, 3])  # 2 rows
    Y = csr_array((Y_data, Y_indices, Y_indptr), shape=(2, 1000))

    # Convert to dense for computing expected results
    X_dense = X.toarray()
    Y_dense = Y.toarray()

    # Test dot product
    dot_result = sparse_dot_product(X, Y)
    expected_dot = X_dense @ Y_dense.T
    np.testing.assert_allclose(dot_result, expected_dot)

    # Test euclidean distance
    euc_result = sparse_euclidean(X, Y)
    X_exp = X_dense[:, np.newaxis, :]
    Y_exp = Y_dense[np.newaxis, :, :]
    expected_euc = np.sqrt(np.sum((X_exp - Y_exp) ** 2, axis=2))
    np.testing.assert_allclose(euc_result, expected_euc)

    # Test manhattan distance
    man_result = sparse_manhattan(X, Y)
    expected_man = np.sum(np.abs(X_exp - Y_exp), axis=2)
    np.testing.assert_allclose(man_result, expected_man)
