import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.opus')


def scan_audio_files(roots):
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(AUDIO_EXTENSIONS):
                    files.append(os.path.join(dirpath, name))
    return sorted(files)


class AudioCropDataset(Dataset):
    """Yields random fixed-duration mono crops, energy-normalized to `average_energy`.

    The normalization mirrors the inference pipeline in silentcipher.server.Model,
    which rescales every input to the average energy of the VCTK corpus before the STFT.
    """

    def __init__(self, roots, sr, crop_seconds, average_energy, epoch_len=10000, min_power=1e-7):
        self.files = scan_audio_files(roots)
        if not self.files:
            raise FileNotFoundError(f'No audio files found under {roots}')
        self.sr = sr
        self.crop_seconds = crop_seconds
        self.crop_len = int(sr * crop_seconds)
        self.average_energy = average_energy
        self.epoch_len = epoch_len
        self.min_power = min_power
        self._durations = {}

    def __len__(self):
        return self.epoch_len

    def _duration(self, path):
        if path not in self._durations:
            try:
                self._durations[path] = librosa.get_duration(path=path)
            except Exception:
                self._durations[path] = 0.0
        return self._durations[path]

    def _load_crop(self, path):
        total = self._duration(path)
        if total <= 0:
            return None
        offset = random.uniform(0, max(0.0, total - self.crop_seconds))
        y, _ = librosa.load(path, sr=self.sr, mono=True, offset=offset, duration=self.crop_seconds)
        if y.shape[0] < self.crop_len:
            y = np.pad(y, (0, self.crop_len - y.shape[0]))
        return y[:self.crop_len]

    def __getitem__(self, idx):
        for _ in range(20):
            path = random.choice(self.files)
            try:
                y = self._load_crop(path)
            except Exception:
                continue
            if y is None:
                continue
            power = float(np.mean(y ** 2))
            if power < self.min_power:  # skip silence, normalization would blow up
                continue
            y = y * np.sqrt(self.average_energy / power)
            return torch.from_numpy(y.astype(np.float32))
        raise RuntimeError('Could not sample a non-silent audio crop after 20 attempts')
