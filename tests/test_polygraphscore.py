import pytest

import networkx as nx
from polygraph.utils.graph_descriptors import SparseDegreeHistogram, DegreeHistogram, ClusteringHistogram
from polygraph.metrics.base import PolyGraphScore, PolyGraphScoreInterval, ClassifierMetric
from polygraph.metrics.base.metric_interval import MetricInterval


@pytest.fixture
def dense_graphs():
    return [nx.erdos_renyi_graph(10, 0.8) for _ in range(128)]

@pytest.fixture
def sparse_graphs():
    return [nx.erdos_renyi_graph(10, 0.1) for _ in range(128)]


@pytest.mark.parametrize("descriptor", [SparseDegreeHistogram(), DegreeHistogram(100)])
@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_classifier_metric(descriptor, classifier, variant, dense_graphs, sparse_graphs):
    clf_metric = ClassifierMetric(dense_graphs, descriptor, variant, classifier)
    train, test = clf_metric.compute(sparse_graphs)

    assert isinstance(train, float) and isinstance(test, float)
    assert train >= 0.7, f"Train score {train} is less than 0.9"
    assert test >= 0.9, f"Test score {test} is less than 0.9"

    train, test = clf_metric.compute(dense_graphs)
    assert train <= 0.2, f"Train score {train} is greater than 0.2"
    assert test <= 0.2, f"Test score {test} is greater than 0.2"



@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_polygraphscore(classifier, variant, dense_graphs, sparse_graphs):
    descriptors = {
        "degree": SparseDegreeHistogram(),
        "clustering": ClusteringHistogram(100),
    }
    polygraphscore = PolyGraphScore(dense_graphs, descriptors, variant, classifier)
    result = polygraphscore.compute(sparse_graphs)

    assert isinstance(result, dict)
    assert "polygraphscore" in result
    assert "polygraphscore_descriptor" in result
    assert "subscores" in result
    assert len(result["subscores"]) == len(descriptors)
    assert result["polygraphscore"] == result["subscores"][result["polygraphscore_descriptor"]]

    assert result["polygraphscore"] >= 0.9, f"PolyGraphScore {result['polygraphscore']} is less than 0.9"

    result = polygraphscore.compute(dense_graphs)
    assert result["polygraphscore"] <= 0.2, f"PolyGraphScore {result['polygraphscore']} is greater than 0.2"


@pytest.mark.parametrize("classifier", ["logistic", "tabpfn"])
@pytest.mark.parametrize("variant", ["jsd", "informedness"])
def test_polygraphscore_interval(classifier, variant, dense_graphs, sparse_graphs):
    descriptors = {
        "degree": SparseDegreeHistogram(),
        "clustering": ClusteringHistogram(100),
    }
    polygraphscore = PolyGraphScoreInterval(dense_graphs, descriptors, subsample_size=10, num_samples=4, variant=variant, classifier=classifier)
    result = polygraphscore.compute(sparse_graphs)
    assert isinstance(result, dict)
    assert "polygraphscore" in result
    assert "polygraphscore_descriptor" in result
    assert "subscores" in result
    assert len(result["subscores"]) == len(descriptors)
    assert isinstance(result["polygraphscore"], MetricInterval)
    assert isinstance(result["polygraphscore_descriptor"], dict)