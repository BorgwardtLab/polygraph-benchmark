from typing import Optional
import numpy as np


class MetricInterval:
    """Class for representing uncertainty quantifications of a metric.

    Attributes:
        mean: Mean of the metric samples.
        std: Standard deviation of the metric samples.
        low: Lower bound of the metric interval, if `coverage` is provided.
        high: Upper bound of the metric interval, if `coverage` is provided.
        coverage: Coverage of the metric interval. If `None`, no bounds are computed, only the mean and standard deviation are set.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        low: Optional[float] = None,
        high: Optional[float] = None,
        coverage: Optional[float] = None,
    ):
        self.mean = mean
        self.std = std

        if coverage is not None and (low is None or high is None):
            raise ValueError("Coverage is provided but low or high is not")

        if coverage is None and (low is not None or high is not None):
            raise ValueError(
                "Coverage is not provided but low or high is provided"
            )

        self.low = low
        self.high = high
        self.coverage = coverage

    @classmethod
    def from_samples(
        cls, samples: np.ndarray, coverage: Optional[float] = None
    ):
        """Create a `MetricInterval` from a collection of samples.

        Args:
            samples: 1D array of samples.
            coverage: Coverage of the interval. If `None`, no bounds are computed, only the mean and standard deviation are set.
        """
        if samples.ndim != 1:
            raise ValueError("Samples must be a 1D array")

        if coverage is not None:
            low, high = (
                np.quantile(samples, (1 - coverage) / 2, axis=0),
                np.quantile(samples, coverage + (1 - coverage) / 2, axis=0),
            )
        else:
            low, high = None, None

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        return cls(mean=mean, std=std, low=low, high=high, coverage=coverage)

    def __getitem__(self, key: str) -> Optional[float]:
        if key == "mean":
            return self.mean
        elif key == "std":
            return self.std
        elif key == "low":
            return self.low
        elif key == "high":
            return self.high
        else:
            raise ValueError(f"Invalid key: {key}")

    def __repr__(self):
        if self.coverage is not None:
            return f"MetricInterval(mean={self.mean}, std={self.std}, low={self.low}, high={self.high}, coverage={self.coverage})"
        else:
            return f"MetricInterval(mean={self.mean}, std={self.std})"
