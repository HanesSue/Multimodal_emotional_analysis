# -*- coding : utf-8 -*-

from torch.utils.data import Dataset
import pandas as pd
import os


class SIMSData(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.root = kwargs["root"]
        self.mode = kwargs["mode"]
        self._raw_meta = None
        self._video_split = None

    @property
    def meta(self):
        if self._raw_meta is None:
            meta_path = os.path.join(self.root, "meta.csv")
            self._raw_meta = pd.read_csv(meta_path)

        if self.mode not in ["train", "test", "val"]:
            raise ValueError("mode must be one of train, test, val")

        return self._raw_meta[self._raw_meta["mode"] == self.mode]

    @property
    def video_split(self):
        if self._video_split is None:
            self._video_split = []
            raw_path = os.path.join(self.root, "Raw")
            for root, _, files in os.walk(raw_path):
                video_id = root.split("/")[-1]
                for file in files:
                    split_id = file.split(".")[0]
                    split_path = os.path.join(root, file)
                    self._video_split.append((video_id, split_id, split_path))
        return self._video_split

    def _load_data(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return index


if __name__ == "__main__":
    test = SIMSData(root="./data/ch-sims2s/ch-simsv2s", mode="test")
    print(test.meta.head())
    print(test.video_split[:10])
