# -*- coding: utf-8 -*-
"""dataset.py
Implementation of datasets.
"""

from typing import List, Union

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.datasets.graph import Graph


class GraphDataset(AbstractDataset):
    def __init__(self, data_store: Graph):
        super().__init__()
        self._data_store = data_store

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Graph, List[Graph]]:
        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)
