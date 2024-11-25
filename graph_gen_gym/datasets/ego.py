from graph_gen_gym.datasets.dataset import OnlineGraphDataset


class EgoGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/F0MGpYS7sMGjMIS/download",
        "val": "https://datashare.biochem.mpg.de/s/o5wq4MRMTsA9uu3/download",
        "test": "https://datashare.biochem.mpg.de/s/bASBL8VCUVm2jai/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]


class SmallEgoGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/RtsHhHBTFkZMIap/download",
        "val": "https://datashare.biochem.mpg.de/s/dWUWhuRj1ipGOVw/download",
        "test": "https://datashare.biochem.mpg.de/s/ey00DsRG1Zm7SQt/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]
