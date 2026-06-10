"""End-to-end smoke test for the training pipeline.

1. Generates a few synthetic audio files (noise + chirps).
2. Runs a handful of training iterations with a config derived from
   train/config_44k.yaml (shortened duration, CPU-friendly overrides).
3. Loads the resulting checkpoint through the real inference API
   (silentcipher.get_model) and runs an encode/decode round trip.

This verifies shapes, devices, the checkpoint layout, and inference
compatibility — NOT model quality (a few iterations decode garbage).

Usage:
    python train/smoke_test.py
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Trainer  # noqa: E402


def generate_data(data_dir, sr, n_files=6, seconds=4.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    for i in range(n_files):
        f0, f1 = rng.uniform(100, 2000), rng.uniform(2000, 8000)
        chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / seconds / 2) * t)
        noise = rng.normal(0, 0.3, t.shape)
        y = (0.5 * chirp + 0.5 * noise).astype(np.float32)
        y /= np.abs(y).max() * 1.25
        sf.write(os.path.join(data_dir, f'smoke_{i}.wav'), y, sr)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'config_44k.yaml')) as f:
        config = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, 'data')
        out_dir = os.path.join(tmp, 'ckpt')
        os.makedirs(data_dir)
        generate_data(data_dir, config['SR'])

        config.update({
            'data_dirs': [data_dir],
            'val_dirs': None,
            'output_dir': out_dir,
            'duration_seconds': 3,       # keep CPU cost down; training default is 12
            'num_iterations': 3,
            'batch_size': 1,
            'num_workers': 0,
            'device': 'cpu',
            'distortions': ['none', 'gaussian_noise', 'time_jitter', 'equalization'],
            'log_every': 1,
            'save_every': 3,
        })

        print('--- training 3 iterations on synthetic data ---')
        Trainer(config).train()

        ckpt_dir = os.path.join(out_dir, '3_iteration')
        assert os.path.isdir(ckpt_dir), f'checkpoint not written: {ckpt_dir}'
        print(f'--- loading {ckpt_dir} via silentcipher.get_model ---')

        import silentcipher
        model = silentcipher.get_model(
            model_type='44.1k',
            ckpt_path=ckpt_dir,
            config_path=os.path.join(ckpt_dir, 'hparams.yaml'),
            device='cpu',
        )

        sr = config['SR']
        y, _ = sf.read(os.path.join(data_dir, 'smoke_0.wav'), dtype='float32')
        message = [123, 234, 111, 222, 11]
        encoded, sdr = model.encode_wav(y, sr, message)
        print(f'encode_wav ok | sdr {sdr:.2f} dB (target lower bound '
              f'{config["message_sdr"]} dB)')

        result = model.decode_wav(encoded, sr, phase_shift_decoding=False)
        print(f'decode_wav ok | status={result["status"]} messages={result["messages"]} '
              f'(decoded bytes are expected to be wrong after 3 iterations)')

    print('SMOKE TEST PASSED: train -> checkpoint -> get_model -> encode/decode all ran')


if __name__ == '__main__':
    main()
