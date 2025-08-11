from typing import Optional

from polygraph.datasets.base import SplitGraphDataset


class PointCloudGraphDataset(SplitGraphDataset):
    """Dataset of KNN-graphs of point clouds, proposed by Neumann et al. [1].

    {{ plot_first_k_graphs("PointCloudGraphDataset", "train", 3, node_size=20) }}


    Dataset statistics:

    {{ summary_md_table("PointCloudGraphDataset", ["train", "val", "test"]) }}


    Graph attributes:
        - `coords`: node-level feature describing the 3D coordinates of the point cloud.
        - `object_class`: graph-level attribute describing the object represented by the point cloud.

    References:
        [1] Neumann, M., Moreno, P., Antanas, L., Garnett, R., & Kersting, K. (2013).
            [Graph kernels for object category prediction in task-dependent robot grasping](http://snap.stanford.edu/mlg2013/submissions/mlg2013_submission_11.pdf).
            In International Workshop on Mining and Learning with Graphs at KDD.
    """

    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/ccnBfchstXblFCl/download",
        "val": "https://datashare.biochem.mpg.de/s/qYpuHH3HhhYAimi/download",
        "test": "https://datashare.biochem.mpg.de/s/w1CKclswdcxLbpK/download",
    }

    _HASH_FOR_SPLIT = {
        "train": None,
        "val": None,
        "test": None,
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]
