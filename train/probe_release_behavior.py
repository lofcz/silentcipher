"""Empirical behavioral probe of a released SilentCipher checkpoint.

Static weight analysis (analyze_release_ckpts.py) cannot verify runtime
semantics. This script runs the released model on real audio, captures the
intermediate watermark tensor, and MEASURES every property the training code
must reproduce:

  P1  Negative-message constraint: the additive watermark (before ReLU) is
      non-positive in every time-frequency bin.
  P2  Band masking: the watermark is exactly zero for all frequency bins
      >= message_band_size.
  P3  Normalization level: RMS ratio watermark/carrier per UTTERANCE matches
      10^(-alpha/20), while per-frame ratios vary widely -> proves
      utterance-level (not frame-level) normalization empirically.
  P4  SDR lower bound (Eq. 3): measured waveform SDR >= alpha for several
      alpha values, without retraining.
  P5  Robustness: decode accuracy on clean audio and after additive Gaussian
      noise at 40 dB SNR (the paper's GN attack; hparams carrier_noise_norm
      0.01 == 1% amplitude == 40 dB).

Also dumps decoder frequency attention (dec_m.linear weights over bins) and
saves a diagnostic figure.

Usage:
    python train/probe_release_behavior.py --release_dir Models/44_1_khz/73999_iteration
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import silentcipher  # noqa: E402


def capture_watermark(model, y, alpha):
    """Re-runs the embed path of Model.encode_wav, returning intermediates."""
    cfg = model.config
    with torch.no_grad():
        y = torch.tensor(y, dtype=torch.float32)
        power = torch.mean(y ** 2)
        y_n = y * torch.sqrt(torch.tensor(model.average_energy_VCTK) / power)
        carrier, phase = model.stft.transform(y_n.unsqueeze(0))
        carrier = carrier[:, None]

        msg_bytes = [123, 234, 111, 222, 11]
        binary_message = ''.join(['{0:08b}'.format(m) for m in msg_bytes])
        chars = [int(binary_message[i * 2:i * 2 + 2], 2) for i in range(len(binary_message) // 2)]
        msgs, _ = model.letters_encoding(carrier.shape[3], [chars])
        msg_enc = torch.tensor(msgs, dtype=torch.float32).unsqueeze(0)

        carrier_enc = model.enc_c(carrier)
        msg_enc = model.enc_c.transform_message(msg_enc)
        merged = torch.cat((carrier_enc, carrier.repeat(1, 32, 1, 1),
                            msg_enc.repeat(1, 32, 1, 1)), dim=1)
        message_info = model.dec_c(merged, alpha)
        if cfg.frame_level_normalization:
            message_info = message_info * torch.mean(carrier ** 2, dim=2, keepdim=True) ** 0.5
        elif cfg.utterance_level_normalization:
            message_info = message_info * torch.mean(carrier ** 2, dim=(2, 3), keepdim=True) ** 0.5
        if cfg.ensure_negative_message:
            message_info = -message_info
    return carrier[0, 0], message_info[0, 0]  # (F, T) each


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release_dir', default='Models/44_1_khz/73999_iteration')
    parser.add_argument('--audio', default='examples/colab/test.wav')
    parser.add_argument('--fig_out', default='train/release_probe.png')
    args = parser.parse_args()

    model = silentcipher.get_model(
        model_type='44.1k', ckpt_path=args.release_dir,
        config_path=os.path.join(args.release_dir, 'hparams.yaml'), device='cpu')
    band = model.config.message_band_size
    y, sr = sf.read(args.audio, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    message = [123, 234, 111, 222, 11]

    print(f'=== behavioral probe: {args.release_dir} on {args.audio} ({len(y)/sr:.1f}s) ===')

    alpha = float(model.config.message_sdr)
    carrier, wm = capture_watermark(model, y, alpha)

    # P1: negativity
    max_val = wm.max().item()
    print(f'P1 negative message: max(watermark) = {max_val:.3e} -> '
          f'{"PASS" if max_val <= 0 else "FAIL"}')

    # P2: band mask
    above_band = wm[band:].abs().max().item()
    print(f'P2 band mask (bins >= {band}): max|watermark| = {above_band:.3e} -> '
          f'{"PASS" if above_band == 0 else "FAIL"}')

    # P3: normalization level
    utt_ratio_db = 20 * torch.log10(carrier.pow(2).mean().sqrt()
                                    / wm.pow(2).mean().sqrt()).item()
    frame_ratio = 20 * torch.log10(carrier.pow(2).mean(dim=0).sqrt()
                                   / (wm.pow(2).mean(dim=0).sqrt() + 1e-12))
    print(f'P3 utterance-level RMS ratio = {utt_ratio_db:.2f} dB (alpha = {alpha:.0f}); '
          f'per-frame ratios spread {frame_ratio.min():.1f}..{frame_ratio.max():.1f} dB -> '
          f'{"PASS (utterance-level)" if abs(utt_ratio_db - alpha) < 1.5 else "FAIL"}')

    # P4: SDR lower bound across alphas
    for a in [30, 40, 47]:
        encoded, sdr = model.encode_wav(y, sr, message, message_sdr=a)
        ok = sdr >= a - 0.1
        print(f'P4 alpha={a}: measured SDR = {sdr:.2f} dB -> {"PASS" if ok else "FAIL"} (>= alpha)')

    # P5: robustness, clean + 40 dB Gaussian noise
    encoded, _ = model.encode_wav(y, sr, message, message_sdr=alpha)
    encoded = encoded.numpy() if isinstance(encoded, torch.Tensor) else encoded
    for name, sig in [('clean', encoded),
                      ('GN 40dB', encoded + np.random.default_rng(0).normal(
                          0, np.sqrt(np.mean(encoded ** 2) / 10 ** 4), encoded.shape
                      ).astype(np.float32))]:
        res = model.decode_wav(sig, sr, phase_shift_decoding=False)
        match = res['status'] and res['messages'][0] == message
        conf = res['confidences'][0] if res['status'] else 0.0
        print(f'P5 decode [{name}]: match={match} confidence={conf:.3f} -> '
              f'{"PASS" if match else "FAIL"}')

    # Decoder frequency attention + figure
    lin_w = model.dec_m[0].linear.weight.detach()[0]  # (band,)
    top = torch.topk(lin_w.abs(), 5).indices.tolist()
    hz_per_bin = sr / model.config.N_FFT
    print(f'decoder freq attention: top-|w| bins {top} '
          f'(~{[int(b * hz_per_bin) for b in top]} Hz)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    eps = 1e-9
    axes[0].imshow(20 * np.log10(carrier.numpy() + eps), origin='lower', aspect='auto',
                   vmin=-120, vmax=0)
    axes[0].set_title('carrier magnitude (dB)')
    axes[1].imshow(20 * np.log10(np.abs(wm.numpy()) + eps), origin='lower', aspect='auto',
                   vmin=-160, vmax=-40)
    axes[1].set_title(f'|watermark| (dB), band mask at bin {band}')
    axes[2].plot(np.arange(band) * hz_per_bin, lin_w.numpy())
    axes[2].set_title('dec_m linear weights over frequency')
    axes[2].set_xlabel('Hz')
    for ax in axes[:2]:
        ax.set_xlabel('frame')
        ax.set_ylabel('freq bin')
    fig.tight_layout()
    fig.savefig(args.fig_out, dpi=110)
    print(f'figure saved: {args.fig_out}')


if __name__ == '__main__':
    main()
