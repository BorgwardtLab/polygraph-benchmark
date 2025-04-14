import pytest
import numpy as np
from polygraph.utils.mmd_utils import mmd_from_full_gram, mmd_from_gram


@pytest.mark.parametrize("variant", ["biased", "umve", "ustat"])
@pytest.mark.parametrize("n_samples", [(8, 8), (4, 16), (32, 16), (1024, 1024)])
@pytest.mark.parametrize("n_kernels", [1, 4])
def test_full_gram_vs_gram(variant, n_samples, n_kernels):
    """Test that mmd_from_full_gram gives the same results as mmd_from_gram."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Unpack sample sizes
    n, m = n_samples
    total_samples = n + m

    # Create a random gram matrix
    if n_kernels == 1:
        gram = np.random.randn(total_samples, total_samples)
        # Make it symmetric
        gram = (gram + gram.T) / 2
    else:
        gram = np.random.randn(total_samples, total_samples, n_kernels)
        # Make each kernel symmetric
        for k in range(n_kernels):
            gram[:, :, k] = (gram[:, :, k] + gram[:, :, k].T) / 2

    # Create random permutation for indices
    perm = np.random.permutation(total_samples)
    x_idx = perm[:n]
    y_idx = perm[n : n + m]

    # Skip ustat test if sample sizes are different
    if variant == "ustat" and n != m:
        pytest.skip("Unequal sample sizes not supported for ustat variant")

    # Extract submatrices using np.ix_
    kxx = gram[np.ix_(x_idx, x_idx)]
    kyy = gram[np.ix_(y_idx, y_idx)]
    kxy = gram[np.ix_(x_idx, y_idx)]

    # Compute MMD using both methods
    mmd_original = mmd_from_gram(kxx, kyy, kxy, variant)
    mmd_full = mmd_from_full_gram(gram, x_idx, y_idx, variant)

    assert np.isclose(mmd_original, mmd_full, rtol=1e-10, atol=1e-10).all()
