# -*- coding : utf-8 -*-
import pandas as pd
import os
import torchaudio
import tempfile
from torch.utils.data import Dataset, DataLoader
from .Processor import MakeProcessor
from functools import partial
import torch
import cv2
from typing import List


class SIMSData(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.root = kwargs["root"]
        self.mode = kwargs["mode"]
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self._raw_meta = None

    @property
    def meta(self):
        if self._raw_meta is None:
            meta_path = os.path.join(self.root, "meta.csv")
            self._raw_meta = pd.read_csv(meta_path)

        if self.mode not in ["train", "test", "valid"]:
            raise ValueError("mode must be one of train, test, valid")

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

    @staticmethod
    def _separate_audio(video_path, sample_rate=16000):
        from moviepy.editor import AudioFileClip

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=True
        ) as temp_audio_file:
            tmp_path = temp_audio_file.name
        with AudioFileClip(video_path) as clip:
            clip.write_audiofile(tmp_path, fps=sample_rate, verbose=False, logger=None)
        waveform, sample_rate = torchaudio.load(tmp_path)
        if waveform.ndim == 2:
            # If stereo, convert to mono by averaging channels
            waveform = waveform.mean(dim=0, keepdim=True)
        os.remove(tmp_path)
        return waveform, sample_rate

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        video_id, clip_id, text = self.meta.iloc[idx][["video_id", "clip_id", "text"]]
        split_path = self._get_path(video_id, clip_id)
        audio_waveform, sample_rate = self._separate_audio(split_path, self.sample_rate)
        labels = self._get_labels(video_id, clip_id)
        frames_tensors = None
        # return frames_tensors, target_score
        return frames_tensors, audio_waveform, text, labels


class SIMSLoader:
    def __init__(
        self, root="./data/ch-sims2s/ch-simsv2s", mode="audio", batch_size=32, num_workers=0, **kwargs
    ):
        self.trainset = SIMSData(root=root, mode="train")
        self.testset = SIMSData(root=root, mode="test")
        self.valset = SIMSData(root=root, mode="valid")
        self.mode = mode
        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers
        if mode not in ["audio", "text", "video"]:
            raise ValueError("mode must be one of audio, text or video")

    @property
    def trainloader(self):
        collate_fn = self.make_collate_fn()
        return DataLoader(
            self.trainset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
        )

    @property
    def testloader(self):
        collate_fn = self.make_collate_fn()
        return DataLoader(
            self.testset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
        )

    @property
    def valloader(self):
        collate_fn = self.make_collate_fn()
        return DataLoader(
            self.valset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
        )

    def make_collate_fn(self):
        if self.mode == "audio":
            processor = MakeProcessor(
                mode="audio", model_name="facebook/hubert-base-ls960"
            )
            return partial(self.collate_fn, processor=processor, mode=self.mode)
        elif self.mode == "text":
            ...
        elif self.mode == "video":
            ...
        else:
            raise ValueError("mode must be one of audio, text or video")
        return None

    @staticmethod
    def collate_fn(batch, processor: MakeProcessor, mode="audio"):
        videos, audio_waveforms, texts, labels = zip(*batch)
        if mode == "audio":
            waveforms = [waveform.squeeze(0).numpy() for waveform in audio_waveforms]
            audios_feats = processor.process(waveforms, sampling_rate=16000)
            label_As = [torch.tensor(label[2]) for label in labels]
            return audios_feats, torch.stack(label_As)
        elif mode == "text":
            ...
        elif mode == "video":
            ...
        else:
            raise ValueError("mode must be one of audio, text or video")


if __name__ == "__main__":
    test = SIMSData(root="./data/ch-sims2s/ch-simsv2s", mode="test")
    test_loader = SIMSLoader(
        root="./data/ch-sims2s/ch-simsv2s", mode="audio", batch_size=32
    ).testloader

    for batch in test_loader:
        audio_feats, labels = batch
        print(audio_feats["input_values"].shape)  # Print shape of audio features
        print(labels.shape)
        break  # Just to demonstrate one batch
    print(test.meta.head())
