from typing import Optional

from polygraph.datasets.base import SplitGraphDataset


class ModelNet10GraphDataset(SplitGraphDataset):
    """Dataset of kNN-graphs sampled from objects in ModelNet10 by Wu et al. [1].

    The graphs are constructed by sampling a random number of points on the object's surface and computing a 4-NN graph.

    {{ plot_first_k_graphs("ModelNet10GraphDataset", "train", 3, node_size=7) }}

    Dataset statistics:

    {{ summary_md_table("ModelNet10GraphDataset", ["train", "val", "test"]) }}


    Graph attributes:
        - `coords`: node-level feature describing the 3D coordinates of the point cloud.
        - `object_class`: graph-level attribute describing the object represented by the point cloud.

    References:
        [1] Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., & Xiao, J. (2015).
            [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf).
            In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1912-1920).
    """

    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/JEpFoe1lJKO33vR/download",
        "val": "https://datashare.biochem.mpg.de/s/BA02AmdCzAhcXKc/download",
        "test": "https://datashare.biochem.mpg.de/s/jxDLhYuzK2ijSx1/download",
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
