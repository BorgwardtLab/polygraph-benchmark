import numpy as np
import pytest
from scipy.sparse import csr_array

from polygraph.utils.descriptors import WeisfeilerLehmanDescriptor
from polygraph.utils.kernels import (
    AdaptiveRBFKernel,
    DescriptorKernel,
    GaussianTV,
    GramBlocks,
    LaplaceKernel,
    LinearKernel,
    RBFKernel,
)


class MockDescriptorKernel(DescriptorKernel):
    def __init__(self, descriptor_fn):
        super().__init__(descriptor_fn)

    def pre_gram_block(self, x, y):
        return np.ones((x.shape[0], y.shape[0]))

    def get_subkernel(self, idx):
        return self

    def kernel_diag(self, features):
        assert features.shape is not None
        return np.ones(features.shape[0])

    @property
    def num_kernels(self):
        return 1


@pytest.fixture
def mock_descriptor_fn():
    def _descriptor_fn(graphs):
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    return _descriptor_fn


def test_adaptive():
    kernel = AdaptiveRBFKernel(mock_descriptor_fn, 1.0)
    assert kernel.is_adaptive
    kernel = LaplaceKernel(mock_descriptor_fn, 1.0)
    assert not kernel.is_adaptive
    kernel = LinearKernel(mock_descriptor_fn)
    assert not kernel.is_adaptive


def test_descriptor_kernel_base(mock_descriptor_fn, sample_graphs):
    kernel = MockDescriptorKernel(mock_descriptor_fn)

    features = kernel.featurize(sample_graphs)
    assert features.shape == (3, 2)

    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    gen = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    blocks = kernel.pre_gram(ref, gen)
    assert isinstance(blocks, GramBlocks)
    assert blocks.ref_vs_ref.shape == (2, 2)
    assert blocks.ref_vs_gen.shape == (2, 3)
    assert blocks.gen_vs_gen.shape == (3, 3)


@pytest.mark.parametrize("lbd", [0.1, np.array([0.1, 0.5, 1.0])])
def test_laplace_kernel(mock_descriptor_fn, sample_features, lbd):
    ref, gen = sample_features
    kernel = LaplaceKernel(mock_descriptor_fn, lbd)

    if isinstance(lbd, np.ndarray):
        assert kernel.num_kernels == lbd.size
    else:
        assert kernel.num_kernels == 1

    result = kernel.pre_gram_block(ref, gen)

    if isinstance(lbd, np.ndarray):
        assert result.shape == (ref.shape[0], gen.shape[0], lbd.size)
    else:
        assert result.shape == (ref.shape[0], gen.shape[0])

    if isinstance(lbd, np.ndarray):
        subkernel = kernel.get_subkernel(0)
        assert isinstance(subkernel, LaplaceKernel)
        assert subkernel.lbd == lbd[0]


@pytest.mark.parametrize("bw", [0.1, np.array([0.1, 0.5, 1.0])])
def test_gaussian_tv_kernel(mock_descriptor_fn, sample_features, bw):
    ref, gen = sample_features
    kernel = GaussianTV(mock_descriptor_fn, bw)

    if isinstance(bw, np.ndarray):
        assert kernel.num_kernels == bw.size
    else:
        assert kernel.num_kernels == 1

    result = kernel.pre_gram_block(ref, gen)

    if isinstance(bw, np.ndarray):
        assert result.shape == (ref.shape[0], gen.shape[0], bw.size)
    else:
        assert result.shape == (ref.shape[0], gen.shape[0])

    if isinstance(bw, np.ndarray):
        subkernel = kernel.get_subkernel(0)
        assert isinstance(subkernel, GaussianTV)
        assert subkernel.bw == bw[0]


@pytest.mark.parametrize("bw", [0.1, np.array([0.1, 0.5, 1.0])])
def test_rbf_kernel(mock_descriptor_fn, sample_features, bw):
    ref, gen = sample_features
    kernel = RBFKernel(mock_descriptor_fn, bw)

    if isinstance(bw, np.ndarray):
        assert kernel.num_kernels == bw.size
    else:
        assert kernel.num_kernels == 1

    result = kernel.pre_gram_block(ref, gen)

    if isinstance(bw, np.ndarray):
        assert result.shape == (ref.shape[0], gen.shape[0], bw.size)
    else:
        assert result.shape == (ref.shape[0], gen.shape[0])

    if isinstance(bw, np.ndarray):
        subkernel = kernel.get_subkernel(0)
        assert isinstance(subkernel, RBFKernel)
        assert subkernel.bw == bw[0]


@pytest.mark.parametrize("bw", [0.1, np.array([0.1, 0.5, 1.0])])
@pytest.mark.parametrize("variant", ["mean", "median"])
def test_adaptive_rbf_kernel(mock_descriptor_fn, sample_features, bw, variant):
    ref, gen = sample_features
    kernel = AdaptiveRBFKernel(mock_descriptor_fn, bw, variant=variant)

    if isinstance(bw, np.ndarray):
        assert kernel.num_kernels == bw.size
    else:
        assert kernel.num_kernels == 1

    pre_gram_result = kernel.pre_gram_block(ref, gen)
    assert pre_gram_result.shape == (ref.shape[0], gen.shape[0])

    ref_ref = np.ones((ref.shape[0], ref.shape[0]))
    ref_gen = np.ones((ref.shape[0], gen.shape[0]))
    gen_gen = np.ones((gen.shape[0], gen.shape[0]))
    blocks = GramBlocks(ref_ref, ref_gen, gen_gen)

    adapted_blocks = kernel.adapt(blocks)
    assert isinstance(adapted_blocks, GramBlocks)

    if isinstance(bw, np.ndarray):
        subkernel = kernel.get_subkernel(0)
        assert isinstance(subkernel, AdaptiveRBFKernel)
        assert subkernel.bw == bw[0]
        assert subkernel._variant == variant


def test_linear_kernel(mock_descriptor_fn, sample_features):
    ref, gen = sample_features
    kernel = LinearKernel(mock_descriptor_fn)

    assert kernel.num_kernels == 1

    result = kernel.pre_gram_block(ref, gen)
    assert result.shape == (ref.shape[0], gen.shape[0])

    expected = ref @ gen.T
    assert np.allclose(result, expected)

    subkernel = kernel.get_subkernel(0)
    assert isinstance(subkernel, LinearKernel)


def test_linear_kernel_with_sparse_input(mock_descriptor_fn):
    ref_data = np.array([1.0, 2.0, 3.0])
    ref_indices = np.array([0, 1, 0])
    ref_indptr = np.array([0, 2, 3])
    ref = csr_array((ref_data, ref_indices, ref_indptr), shape=(2, 2))

    gen_data = np.array([4.0, 5.0, 6.0])
    gen_indices = np.array([1, 0, 1])
    gen_indptr = np.array([0, 1, 3])
    gen = csr_array((gen_data, gen_indices, gen_indptr), shape=(2, 2))

    kernel = LinearKernel(mock_descriptor_fn)
    result = kernel.pre_gram_block(ref, gen)

    assert isinstance(result, np.ndarray)

    expected = (ref @ gen.T).toarray()
    assert np.allclose(result, expected)


def test_kernel_call_method(mock_descriptor_fn, sample_features):
    ref, gen = sample_features
    kernel = LinearKernel(mock_descriptor_fn)

    kernel.featurize = lambda x: ref if x == "ref" else gen  # type: ignore[method-assign]

    blocks = kernel(ref, gen)
    assert isinstance(blocks, GramBlocks)
    assert blocks.ref_vs_ref.shape == (ref.shape[0], ref.shape[0])
    assert blocks.ref_vs_gen.shape == (ref.shape[0], gen.shape[0])
    assert blocks.gen_vs_gen.shape == (gen.shape[0], gen.shape[0])


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_weisfeiler_lehman(sample_graphs, iterations):
    wl_descriptor = WeisfeilerLehmanDescriptor(iterations=iterations)
    kernel = LinearKernel(wl_descriptor)

    ref_graphs = sample_graphs[:2]
    gen_graphs = sample_graphs[2:]

    ref_features = kernel.featurize(ref_graphs)
    gen_features = kernel.featurize(gen_graphs)

    assert ref_features.shape[0] == len(ref_graphs)
    assert gen_features.shape[0] == len(gen_graphs)

    blocks = kernel(ref_features, gen_features)

    assert isinstance(blocks, GramBlocks)
    assert blocks.ref_vs_ref.shape == (len(ref_graphs), len(ref_graphs))
    assert blocks.ref_vs_gen.shape == (len(ref_graphs), len(gen_graphs))
    assert blocks.gen_vs_gen.shape == (len(gen_graphs), len(gen_graphs))

    expected_ref_gen = ref_features @ gen_features.T

    if isinstance(expected_ref_gen, np.ndarray):
        assert np.allclose(blocks.ref_vs_gen, expected_ref_gen)
    else:
        assert np.allclose(blocks.ref_vs_gen, expected_ref_gen.toarray())


@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize("use_node_labels", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_weisfeiler_lehman_with_molecules(
    sample_molecules, iterations, use_node_labels, n_jobs
):
    """Test the Weisfeiler-Lehman kernel with QM9 molecules."""
    # Use the WeisfeilerLehmanDescriptor with QM9 molecules
    # QM9 molecules have atom types as node attributes
    wl_descriptor = WeisfeilerLehmanDescriptor(
        iterations=iterations,
        use_node_labels=use_node_labels,
        node_label_key="atom_labels" if use_node_labels else None,
        n_jobs=n_jobs,
        digest_size=2,
    )
    kernel = LinearKernel(wl_descriptor)

    ref_molecules = sample_molecules[:2]
    gen_molecules = sample_molecules[2:]

    ref_features = kernel.featurize(ref_molecules)
    gen_features = kernel.featurize(gen_molecules)

    assert ref_features.shape[0] == len(ref_molecules)
    assert gen_features.shape[0] == len(gen_molecules)

    blocks = kernel(ref_features, gen_features)

    assert isinstance(blocks, GramBlocks)
    assert blocks.ref_vs_ref.shape == (len(ref_molecules), len(ref_molecules))
    assert blocks.ref_vs_gen.shape == (len(ref_molecules), len(gen_molecules))
    assert blocks.gen_vs_gen.shape == (len(gen_molecules), len(gen_molecules))

    expected_ref_gen = ref_features @ gen_features.T

    assert np.allclose(blocks.ref_vs_gen, expected_ref_gen.toarray())


# Reference gram matrices precomputed with grakel 0.1.10 (numpy<2).
#
# grakel is incompatible with numpy>=2 so we cannot use it as a runtime
# dependency. The values below were produced once in an isolated environment
# and are checked in as frozen references.
#
# To regenerate, create a temporary environment and run:
#
#     pip install "numpy<2" grakel networkx
#     python -c "
#     import grakel, networkx as nx
#
#     # ---- ER graphs (same seeds as the sample_graphs fixture) ----
#     graphs = [nx.erdos_renyi_graph(n, p, seed=s)
#               for n, p, s in [(10,.3,42),(15,.2,43),(12,.25,44),(10,.3,45)]]
#     for g in graphs:
#         for node in g.nodes():
#             g.nodes[node]['degree'] = g.degree(node)
#     ref, gen = graphs[:2], graphs[2:]
#     for it in (2, 3):
#         k = grakel.WeisfeilerLehman(n_iter=it)
#         k.fit(grakel.graph_from_networkx(ref, node_labels_tag='degree'))
#         r = k.transform(grakel.graph_from_networkx(gen, node_labels_tag='degree'))
#         print(f'ER  it={it}  ref_vs_gen={r.T.tolist()}')
#
#     # ---- QM9 molecules (first 5 from the test split) ----
#     # Molecules must be exported from the pixi env as JSON because
#     # QM9 requires rdkit. See tests/conftest.py::sample_molecules.
#     import json
#     from networkx.readwrite import json_graph
#     with open('/tmp/sample_molecules_fixed.json') as f:
#         mols = [json_graph.node_link_graph(d) for d in json.load(f)]
#     ref, gen = mols[:2], mols[2:]
#     for it in (2, 3):
#         k = grakel.WeisfeilerLehman(n_iter=it)
#         k.fit(grakel.graph_from_networkx(ref, node_labels_tag='atom_labels'))
#         r = k.transform(grakel.graph_from_networkx(gen, node_labels_tag='atom_labels'))
#         print(f'MOL it={it}  ref_vs_gen={r.T.tolist()}')
#     "
GRAKEL_ER_REF_VS_GEN = {
    2: np.array([[32.0, 21.0], [42.0, 35.0]]),
    3: np.array([[32.0, 21.0], [42.0, 35.0]]),
}

GRAKEL_MOL_REF_VS_GEN = {
    2: np.array([[10.0, 12.0, 9.0], [56.0, 50.0, 86.0]]),
    3: np.array([[10.0, 12.0, 9.0], [56.0, 50.0, 86.0]]),
}


@pytest.mark.parametrize("iterations", [2, 3])
def test_weisfeiler_lehman_vs_grakel_er_graphs(sample_graphs, iterations):
    """Test our WL kernel against grakel 0.1.10 reference values.

    See GRAKEL_ER_REF_VS_GEN above for how to regenerate.
    """
    wl_descriptor = WeisfeilerLehmanDescriptor(
        iterations=iterations, use_node_labels=False, n_jobs=1
    )
    our_kernel = LinearKernel(wl_descriptor)

    ref_graphs = sample_graphs[:2]
    gen_graphs = sample_graphs[2:]

    ref_features = our_kernel.featurize(ref_graphs)
    gen_features = our_kernel.featurize(gen_graphs)
    our_blocks = our_kernel(ref_features, gen_features)

    assert np.allclose(
        our_blocks.ref_vs_gen, GRAKEL_ER_REF_VS_GEN[iterations]
    ), (ref_features, gen_features)


@pytest.mark.parametrize("iterations", [2, 3])
def test_weisfeiler_lehman_vs_grakel_molecules(sample_molecules, iterations):
    """Test our WL kernel against grakel 0.1.10 reference values.

    See GRAKEL_MOL_REF_VS_GEN above for how to regenerate.
    """
    wl_descriptor_mol = WeisfeilerLehmanDescriptor(
        iterations=iterations,
        use_node_labels=True,
        node_label_key="atom_labels",
        n_jobs=1,
    )
    our_kernel_mol = LinearKernel(wl_descriptor_mol)

    ref_molecules = sample_molecules[:2]
    gen_molecules = sample_molecules[2:]

    ref_features_mol = our_kernel_mol.featurize(ref_molecules)
    gen_features_mol = our_kernel_mol.featurize(gen_molecules)
    our_blocks_mol = our_kernel_mol(ref_features_mol, gen_features_mol)

    assert np.allclose(
        our_blocks_mol.ref_vs_gen, GRAKEL_MOL_REF_VS_GEN[iterations]
    )
