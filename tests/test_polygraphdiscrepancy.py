import pytest

from sklearn.linear_model import LogisticRegression
import networkx as nx
from polygraph.utils.descriptors import (
    SparseDegreeHistogram,
    DegreeHistogram,
    ClusteringHistogram,
)
from polygraph.metrics.base import (
    PolyGraphDiscrepancy,
    PolyGraphDiscrepancyInterval,
    ClassifierMetric,
)
from polygraph.metrics.base.metric_interval import MetricInterval
from polygraph.metrics.standard_pgd import (
    StandardPGD,
    StandardPGDInterval,
    ClassifierOrbit4Metric,
    ClassifierOrbit5Metric,
    ClassifierClusteringMetric,
    ClassifierDegreeMetric,
    ClassifierSpectralMetric,
    GraphNeuralNetworkClassifierMetric,
)


@pytest.fixture
def dense_graphs():
    return [nx.erdos_renyi_graph(10, 0.8) for _ in range(128)]


@pytest.fixture
def sparse_graphs():
    return [nx.erdos_renyi_graph(10, 0.1) for _ in range(128)]


@pytest.mark.parametrize(
    "descriptor", [SparseDegreeHistogram(), DegreeHistogram(100)]
)
@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_classifier_metric(
    descriptor, classifier, variant, dense_graphs, sparse_graphs
):
    if classifier == "tabpfn":
        classifier = None
    else:
        classifier = LogisticRegression()

    clf_metric = ClassifierMetric(dense_graphs, descriptor, variant, classifier)
    train, test = clf_metric.compute(sparse_graphs)

    assert isinstance(train, float) and isinstance(test, float)
    assert train >= 0.7, f"Train score {train} is less than 0.7"
    assert test >= 0.8, f"Test score {test} is less than 0.8"

    train, test = clf_metric.compute(dense_graphs)
    assert train <= 0.2, f"Train score {train} is greater than 0.2"
    assert test <= 0.2, f"Test score {test} is greater than 0.2"


@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_polygraphdiscrepancy(classifier, variant, dense_graphs, sparse_graphs):
    descriptors = {
        "degree": SparseDegreeHistogram(),
        "clustering": ClusteringHistogram(100),
    }

    if classifier == "tabpfn":
        classifier = None
    else:
        classifier = LogisticRegression()

    pgd = PolyGraphDiscrepancy(dense_graphs, descriptors, variant, classifier)
    result = pgd.compute(sparse_graphs)

    assert isinstance(result, dict)
    assert "pgd" in result
    assert "pgd_descriptor" in result
    assert "subscores" in result
    assert len(result["subscores"]) == len(descriptors)
    assert result["pgd"] == result["subscores"][result["pgd_descriptor"]]

    assert result["pgd"] >= 0.8, (
        f"PolyGraphDiscrepancy {result['pgd']} is less than 0.8"
    )

    result = pgd.compute(dense_graphs)
    assert result["pgd"] <= 0.2, (
        f"PolyGraphDiscrepancy {result['pgd']} is greater than 0.2"
    )


@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_polygraphdiscrepancy_interval(
    classifier, variant, dense_graphs, sparse_graphs
):
    descriptors = {
        "degree": SparseDegreeHistogram(),
        "clustering": ClusteringHistogram(100),
    }
    if classifier == "tabpfn":
        classifier = None
    else:
        classifier = LogisticRegression()

    pgd = PolyGraphDiscrepancyInterval(
        dense_graphs,
        descriptors,
        subsample_size=10,
        num_samples=4,
        variant=variant,
        classifier=classifier,
    )
    result = pgd.compute(sparse_graphs)
    assert isinstance(result, dict)
    assert "pgd" in result
    assert "pgd_descriptor" in result
    assert "subscores" in result
    assert len(result["subscores"]) == len(descriptors)
    assert isinstance(result["pgd"], MetricInterval)
    assert isinstance(result["pgd_descriptor"], dict)


def test_standard_pgd(dense_graphs, sparse_graphs):
    metric = StandardPGD(dense_graphs)
    result = metric.compute(sparse_graphs)

    individual_metrics = {
        "orbit4": ClassifierOrbit4Metric(
            dense_graphs, variant="jsd", classifier=None
        ),
        "orbit5": ClassifierOrbit5Metric(
            dense_graphs, variant="jsd", classifier=None
        ),
        "clustering": ClassifierClusteringMetric(
            dense_graphs, variant="jsd", classifier=None
        ),
        "degree": ClassifierDegreeMetric(
            dense_graphs, variant="jsd", classifier=None
        ),
        "spectral": ClassifierSpectralMetric(
            dense_graphs, variant="jsd", classifier=None
        ),
        "gin": GraphNeuralNetworkClassifierMetric(
            dense_graphs, variant="jsd", classifier=None
        ),
    }
    individual_results = {
        name: metric.compute(sparse_graphs)
        for name, metric in individual_metrics.items()
    }

    for name, (_, individual_result) in individual_results.items():
        assert isinstance(individual_result, float)
        assert individual_result == result["subscores"][name], (
            f"Individual result {individual_result} for descriptor {name} does not match the overall result {result['subscores'][name]}"
        )

    metric = StandardPGDInterval(dense_graphs, subsample_size=10, num_samples=4)
    result = metric.compute(sparse_graphs)
    assert isinstance(result, dict)
    assert "pgd" in result
    assert "pgd_descriptor" in result
    assert "subscores" in result
    assert len(result["subscores"]) == len(individual_metrics)
    assert isinstance(result["pgd"], MetricInterval)
    assert isinstance(result["pgd_descriptor"], dict)
