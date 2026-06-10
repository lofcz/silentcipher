"""Table-1 attack benchmark (arXiv:2406.03822, Sec. 5) for a SilentCipher checkpoint.

Replicates the paper's objective evaluation attack suite against any checkpoint
loadable by silentcipher.get_model (released or retrained):

  No attack | GN (Gaussian noise 40 dB) | 50C (random 50% crop) |
  EQ (15 dB band-limited at 35/200/1000/4000 Hz) | MX (mix speech at -15 dB) |
  Q (16-bit float quantization) | TJ (time jitter) |
  RS (random resampling) | MP3/OGG/AAC at 64/128/256 kbps

Results (message match, char accuracy, confidence) are printed and written to CSV.

Usage:
    python train/benchmark_attacks.py --release_dir Models/44_1_khz/73999_iteration \
        --audio examples/colab/test.wav [--mix_audio path/to/speech.wav] \
        [--phase_shift] [--out train/benchmark_results.csv]
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import silentcipher  # noqa: E402
from distortions import _codec_roundtrip_np, equalization, time_jitter  # noqa: E402

MESSAGE = [123, 234, 111, 222, 11]


def gn(y, sr, snr_db=40):
    noise_power = np.mean(y ** 2) / 10 ** (snr_db / 10)
    return y + np.random.default_rng(0).normal(0, np.sqrt(noise_power), y.shape).astype(np.float32)


def crop50(y, sr):
    keep = len(y) // 2
    start = random.randint(0, len(y) - keep)
    return y[start:start + keep]


def eq(y, sr, n_fft, hop):
    out = equalization(torch.from_numpy(y).unsqueeze(0), sr, n_fft, hop,
                       max_gain_db=15, center_freqs=[35, 200, 1000, 4000])
    return out[0].numpy()


def mix(y, sr, speech):
    speech = np.resize(speech, y.shape)
    target_power = np.mean(y ** 2) / 10 ** (15 / 10)  # speech at -15 dB
    speech = speech * np.sqrt(target_power / max(np.mean(speech ** 2), 1e-12))
    return y + speech.astype(np.float32)


def quant16(y, sr):
    return y.astype(np.float16).astype(np.float32)


def tj(y, sr):
    return time_jitter(torch.from_numpy(y).unsqueeze(0), 50)[0].numpy()


def rs(y, sr):
    low, high = int(0.4 * sr), sr
    new_sr = random.randint(low, high)
    t = torch.from_numpy(y).unsqueeze(0)
    down = torchaudio.functional.resample(t, sr, new_sr)
    up = torchaudio.functional.resample(down, new_sr, sr)[0].numpy()
    return up[:len(y)]


def codec(name, kbps):
    def attack(y, sr):
        return _codec_roundtrip_np(y, sr, name, kbps)
    return attack


def char_accuracy(decoded, expected):
    if not decoded:
        return 0.0
    bits_dec = ''.join(f'{b:08b}' for b in decoded[:len(expected)])
    bits_exp = ''.join(f'{b:08b}' for b in expected)
    n = min(len(bits_dec), len(bits_exp))
    if n == 0:
        return 0.0
    return sum(a == b for a, b in zip(bits_dec[:n], bits_exp[:n])) / len(bits_exp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release_dir', default='Models/44_1_khz/73999_iteration')
    parser.add_argument('--model_type', default='44.1k', choices=['44.1k', '16k'])
    parser.add_argument('--audio', default='examples/colab/test.wav')
    parser.add_argument('--mix_audio', default=None,
                        help='speech file for the MX attack; skipped if not given')
    parser.add_argument('--phase_shift', action='store_true',
                        help='enable phase-shift decoding (slow, helps 50C/TJ)')
    parser.add_argument('--out', default='train/benchmark_results.csv')
    args = parser.parse_args()

    random.seed(0)
    model = silentcipher.get_model(
        model_type=args.model_type, ckpt_path=args.release_dir,
        config_path=os.path.join(args.release_dir, 'hparams.yaml'), device='cpu')
    sr = model.config.SR
    n_fft, hop = model.config.N_FFT, model.config.HOP_LENGTH

    y, file_sr = sf.read(args.audio, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]

    encoded, sdr = model.encode_wav(y, file_sr, MESSAGE)
    encoded = encoded.numpy() if isinstance(encoded, torch.Tensor) else encoded
    print(f'encoded {args.audio} ({len(y)/file_sr:.1f}s) | SDR {sdr:.2f} dB | '
          f'phase_shift_decoding={args.phase_shift}')

    attacks = [
        ('no_attack', lambda x, s: x),
        ('GN_40dB', gn),
        ('50C', crop50),
        ('EQ_15dB', lambda x, s: eq(x, s, n_fft, hop)),
        ('Q_float16', quant16),
        ('TJ', tj),
        ('RS', rs),
    ]
    if args.mix_audio:
        speech, _ = sf.read(args.mix_audio, dtype='float32')
        if speech.ndim > 1:
            speech = speech[:, 0]
        attacks.append(('MX_-15dB', lambda x, s: mix(x, s, speech)))
    else:
        print('MX attack skipped (pass --mix_audio to enable)')
    for codec_name in ['mp3', 'ogg', 'aac']:
        for kbps in [64, 128, 256]:
            attacks.append((f'{codec_name}_{kbps}k', codec(codec_name, kbps)))

    rows = []
    for name, attack in attacks:
        try:
            attacked = attack(encoded, file_sr)
            res = model.decode_wav(attacked, file_sr, phase_shift_decoding=args.phase_shift)
            ok = res['status'] and res['messages'][0] == MESSAGE
            acc = char_accuracy(res['messages'][0], MESSAGE) if res['status'] else 0.0
            conf = res['confidences'][0] if res['status'] else 0.0
            rows.append([name, ok, f'{acc:.3f}', f'{conf:.3f}'])
            print(f'{name:12s} | match={str(ok):5s} | bit-acc {acc:.3f} | confidence {conf:.3f}')
        except Exception as e:
            rows.append([name, False, 'error', str(e)])
            print(f'{name:12s} | ERROR: {e}')

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['attack', 'message_match', 'bit_accuracy', 'confidence'])
        writer.writerows(rows)
    print(f'results written to {args.out}')


if __name__ == '__main__':
    main()
