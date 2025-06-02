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

    @property
    def meta(self):
        if self._raw_meta is None:
            meta_path = os.path.join(self.root, "meta.csv")
            self._raw_meta = pd.read_csv(meta_path)

        if self.mode not in ["train", "test", "val"]:
            raise ValueError("mode must be one of train, test, val")

        return self._raw_meta[self._raw_meta["mode"] == self.mode]

    def _get_path(self, video_id, clip_id):
        return os.path.join(self.root, "Raw", f"{video_id}/{clip_id}.mp4")

    def _get_labels(self, video_id, clip_id):
        return (
            self.meta.loc[
                (self.meta["video_id"] == video_id) & (self.meta["clip_id"] == clip_id),
                ["label", "label_T", "label_A", "label_V"],
            ]
            .values[0]
            .tolist()
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        video_id, clip_id, text = self.meta.iloc[idx][["video_id", "clip_id", "text"]]
        split_path = self._get_path(video_id, clip_id)
        labels = self._get_labels(video_id, clip_id)
        return split_path, text, labels


if __name__ == "__main__":
    test = SIMSData(root="./data/ch-sims2s/ch-simsv2s", mode="test")
    print(test.meta.head())
    print(test[0])
