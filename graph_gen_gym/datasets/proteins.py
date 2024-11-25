from graph_gen_gym.datasets.dataset import OnlineGraphDataset


class DobsonDoigGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/IUzyKrF6T1wjqqG/download",
        "val": "https://datashare.biochem.mpg.de/s/NhaictDUDb7UTpr/download",
        "test": "https://datashare.biochem.mpg.de/s/ecJCDZVTNOpbvy4/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]
