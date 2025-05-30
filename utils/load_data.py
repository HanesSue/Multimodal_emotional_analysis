# -*- coding : utf-8 -*-
import pandas as pd
import os
from torch.utils.data import Dataset


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
            splits = []
            raw_path = os.path.join(self.root, "Raw")
            for root, _, files in os.walk(raw_path):
                video_id = os.path.basename(root)
                for file in files:
                    clip_id = file.split(".")[0]
                    split_path = os.path.join(root, file)
                    splits.append((video_id, clip_id, split_path))
            self._video_split = pd.DataFrame(splits, columns=["video_id", "clip_id", "split_path"])
        return self._video_split

    def _get_path(self, video_id, clip_id):
        return self.video_split.loc[(self.video_split["video_id"] == video_id) & (self.video_split["clip_id"] == clip_id), "split_path"].values[0]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        video_id, clip_id, text = self.meta.iloc[idx][["video_id", "clip_id", "text"]]
        split_path = self._get_path(video_id, clip_id)
        return split_path, text

if __name__ == "__main__":
    test = SIMSData(root="./data/ch-sims2s/ch-simsv2s", mode="test")
    print(test.video_split.head())
    print(test.meta.head())
    print(test[0])
