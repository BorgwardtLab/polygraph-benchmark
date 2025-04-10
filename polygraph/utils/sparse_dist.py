import numpy as np
import numba as nb


@nb.njit
def _is_sorted(indices, indptr):
    """
    Check if indices are sorted within each row of a CSR matrix.

    Parameters
    ----------
    indices : numpy.ndarray
        Array of column indices
    indptr : numpy.ndarray
        Array of row pointers

    Returns
    -------
    bool
        True if indices are sorted within each row, False otherwise
    """
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        for j in range(start, end - 1):
            if indices[j] >= indices[j + 1]:
                return False
    return True


# Numba-optimized core function that works with raw arrays
@nb.njit(parallel=True)
def _sparse_dot_product_core(
    X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
):
    D = np.zeros((m, n), dtype=np.float64)

    for px in nb.prange(m):
        X_indptr_end = X_indptr[px + 1]
        for py in range(n):
            Y_indptr_end = Y_indptr[py + 1]
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0

            while i < X_indptr_end and j < Y_indptr_end:
                ix = X_indices[i]
                iy = Y_indices[j]
                if ix == iy:
                    d = d + X_data[i] * Y_data[j]
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    i = i + 1
                else:
                    j = j + 1

            D[px, py] = d

    return D


# Wrapper function that handles scipy sparse matrices
def sparse_dot_product(X, Y):
    """
    Compute pairwise dot products between rows of two CSR matrices.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        First input matrix.
    Y : scipy.sparse.csr_matrix
        Second input matrix.

    Returns
    -------
    D : numpy.ndarray
        Pairwise dot product matrix where D[i,j] is the dot product of X[i] and Y[j].
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input matrices must have the same number of columns.")

    # Extract CSR components
    X_data = X.data.astype(np.float64)
    X_indices = X.indices
    X_indptr = X.indptr

    Y_data = Y.data.astype(np.float64)
    Y_indices = Y.indices
    Y_indptr = Y.indptr

    # Check if indices are sorted
    if not _is_sorted(X_indices, X_indptr) or not _is_sorted(
        Y_indices, Y_indptr
    ):
        raise ValueError("Matrix indices must be sorted within each row")

    m, n = X.shape[0], Y.shape[0]

    # Call the numba-optimized core function
    return _sparse_dot_product_core(
        X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
    )


# Numba-optimized core function for Euclidean distance
@nb.njit(parallel=True)
def _sparse_euclidean_core(
    X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
):
    D = np.zeros((m, n), dtype=np.float64)

    for px in nb.prange(m):
        X_indptr_end = X_indptr[px + 1]
        for py in range(n):
            Y_indptr_end = Y_indptr[py + 1]
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0

            while i < X_indptr_end and j < Y_indptr_end:
                ix = X_indices[i]
                iy = Y_indices[j]
                if ix == iy:
                    d = d + (X_data[i] - Y_data[j]) * (X_data[i] - Y_data[j])
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + X_data[i] * X_data[i]
                    i = i + 1
                else:
                    d = d + Y_data[j] * Y_data[j]
                    j = j + 1

            # Handle remaining elements in X
            while i < X_indptr_end:
                d = d + X_data[i] * X_data[i]
                i = i + 1

            # Handle remaining elements in Y
            while j < Y_indptr_end:
                d = d + Y_data[j] * Y_data[j]
                j = j + 1

            D[px, py] = np.sqrt(d)

    return D


# Wrapper function for Euclidean distance
def sparse_euclidean(X, Y):
    """
    Compute pairwise Euclidean distances between rows of two CSR matrices.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        First input matrix.
    Y : scipy.sparse.csr_matrix
        Second input matrix.

    Returns
    -------
    D : numpy.ndarray
        Pairwise distance matrix where D[i,j] is the Euclidean distance between X[i] and Y[j].
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input matrices must have the same number of columns.")

    # Extract CSR components
    X_data = X.data.astype(np.float64)
    X_indices = X.indices
    X_indptr = X.indptr

    Y_data = Y.data.astype(np.float64)
    Y_indices = Y.indices
    Y_indptr = Y.indptr

    # Check if indices are sorted
    if not _is_sorted(X_indices, X_indptr) or not _is_sorted(
        Y_indices, Y_indptr
    ):
        raise ValueError("Matrix indices must be sorted within each row")

    m, n = X.shape[0], Y.shape[0]

    # Call the numba-optimized core function
    return _sparse_euclidean_core(
        X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
    )


@nb.njit(parallel=True)
def _sparse_manhattan_core(
    X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
):
    D = np.zeros((m, n), dtype=np.float64)

    for px in nb.prange(m):
        X_indptr_end = X_indptr[px + 1]
        for py in range(n):
            Y_indptr_end = Y_indptr[py + 1]
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0

            while i < X_indptr_end and j < Y_indptr_end:
                ix = X_indices[i]
                iy = Y_indices[j]
                if ix == iy:
                    d = d + np.abs(X_data[i] - Y_data[j])
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + np.abs(X_data[i])
                    i = i + 1
                else:
                    d = d + np.abs(Y_data[j])
                    j = j + 1

            # Handle remaining elements in X
            while i < X_indptr_end:
                d = d + np.abs(X_data[i])
                i = i + 1

            # Handle remaining elements in Y
            while j < Y_indptr_end:
                d = d + np.abs(Y_data[j])
                j = j + 1

            D[px, py] = d

    return D


# Wrapper function for Manhattan distance
def sparse_manhattan(X, Y):
    """
    Compute pairwise Manhattan (L1) distances between rows of two CSR matrices.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        First input matrix.
    Y : scipy.sparse.csr_matrix
        Second input matrix.

    Returns
    -------
    D : numpy.ndarray
        Pairwise distance matrix where D[i,j] is the Manhattan distance between X[i] and Y[j].
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input matrices must have the same number of columns.")

    # Extract CSR components
    X_data = X.data.astype(np.float64)
    X_indices = X.indices
    X_indptr = X.indptr

    Y_data = Y.data.astype(np.float64)
    Y_indices = Y.indices
    Y_indptr = Y.indptr

    # Check if indices are sorted
    if not _is_sorted(X_indices, X_indptr) or not _is_sorted(
        Y_indices, Y_indptr
    ):
        raise ValueError("Matrix indices must be sorted within each row")

    m, n = X.shape[0], Y.shape[0]

    # Call the numba-optimized core function
    return _sparse_manhattan_core(
        X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, m, n
    )
