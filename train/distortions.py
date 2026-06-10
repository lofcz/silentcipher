"""Training-time distortions for SilentCipher (Sec. 3.3 of the paper).

One distortion is sampled uniformly per iteration from:
  none, gaussian_noise, time_jitter, equalization, compression

Compression (MP3/OGG/AAC) is non-differentiable, so it is applied as a
pseudo-differentiable layer: the codec round-trip runs in the forward pass
while gradients bypass it (straight-through estimator).

NOTE: the official training code is unreleased, so the implementations below
are reconstructions. The distortion set, the noise level (40 dB), the EQ gain
(15 dB) and the codec bitrates come from the paper / released hparams; the
jitter mechanics, EQ curve shape, and codec delay alignment are inferred.
"""

import io
import random
import warnings

import numpy as np
import torch
from scipy import signal as sps

_FFMPEG_WARNED = False


def gaussian_noise(y, snr_db):
    power = torch.mean(y ** 2, dim=-1, keepdim=True)
    noise_power = power / (10 ** (snr_db / 10))
    return y + torch.randn_like(y) * torch.sqrt(noise_power)


def time_jitter(y, max_jitter_samples):
    """Deletes k random samples and duplicates k others, preserving length."""
    batch, length = y.shape
    out = []
    for b in range(batch):
        k = random.randint(1, max_jitter_samples)
        keep = torch.ones(length, dtype=torch.bool, device=y.device)
        keep[torch.randperm(length, device=y.device)[:k]] = False
        kept = y[b][keep]
        counts = torch.ones(kept.shape[0], dtype=torch.long, device=y.device)
        dup_idx = torch.randint(0, kept.shape[0], (k,), device=y.device)
        counts.scatter_add_(0, dup_idx, torch.ones(k, dtype=torch.long, device=y.device))
        out.append(torch.repeat_interleave(kept, counts)[:length])
    return torch.stack(out)


def equalization(y, sr, n_fft, hop_length, max_gain_db, center_freqs):
    """Applies a random band-limited gain of up to +-max_gain_db (STFT domain, differentiable)."""
    length = y.shape[-1]
    window = torch.hann_window(n_fft, device=y.device)
    spec = torch.stft(y, n_fft, hop_length, n_fft, window=window, return_complex=True)

    f0 = random.choice(center_freqs)
    gain_db = random.uniform(-max_gain_db, max_gain_db)
    freqs = torch.linspace(0, sr / 2, spec.shape[1], device=y.device)
    # Gaussian bump, one octave wide, centered at f0 (log-frequency domain)
    log_f = torch.log2(torch.clamp(freqs, min=1.0))
    bump = torch.exp(-0.5 * ((log_f - np.log2(f0)) / 0.5) ** 2)
    gain = 10 ** (gain_db * bump / 20)

    spec = spec * gain[None, :, None]
    return torch.istft(spec, n_fft, hop_length, n_fft, window=window, length=length)


def _codec_roundtrip_np(x, sr, codec, bitrate_kbps):
    from pydub import AudioSegment

    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    segment = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    # libvorbis rejects high managed bitrates for mono input ("encoder setup
    # failed" at 256k); encode OGG as dual-mono stereo and downmix afterwards.
    if codec == 'ogg':
        segment = AudioSegment.from_mono_audiosegments(segment, segment)

    fmt = {'mp3': 'mp3', 'ogg': 'ogg', 'aac': 'adts'}[codec]
    buf = io.BytesIO()
    segment.export(buf, format=fmt, bitrate=f'{bitrate_kbps}k')
    buf.seek(0)
    decoded = AudioSegment.from_file(buf).set_frame_rate(sr).set_channels(1)
    out = np.array(decoded.get_array_of_samples(), dtype=np.float32) / 32768.0

    # Codecs introduce a leading delay; align via cross-correlation of the head
    head = x[:min(sr, len(x))]
    search = out[:len(head) + 8192]
    if len(search) > len(head):
        corr = sps.correlate(search, head, mode='valid', method='fft')
        lag = int(np.argmax(corr))
        out = out[lag:]
    if len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)))
    return out[:len(x)]


def pseudo_differentiable_compression(y, sr, codecs, bitrates_kbps):
    """Codec round-trip in the forward pass; identity in the backward pass."""
    global _FFMPEG_WARNED
    codec = random.choice(codecs)
    bitrate = random.choice(bitrates_kbps)
    compressed = []
    for b in range(y.shape[0]):
        x = y[b].detach().cpu().numpy()
        try:
            compressed.append(_codec_roundtrip_np(x, sr, codec, bitrate))
        except Exception as e:
            if not _FFMPEG_WARNED:
                warnings.warn(f'Compression distortion failed ({e}). Is ffmpeg installed? '
                              'Falling back to identity for this batch.')
                _FFMPEG_WARNED = True
            compressed.append(x)
    compressed = torch.from_numpy(np.stack(compressed)).to(y.device, y.dtype)
    return y + (compressed - y).detach()


def apply_distortion(name, y, sr, config):
    if name == 'none':
        return y
    if name == 'gaussian_noise':
        return gaussian_noise(y, config['gaussian_snr_db'])
    if name == 'time_jitter':
        return time_jitter(y, config['max_jitter_samples'])
    if name == 'equalization':
        return equalization(y, sr, config['N_FFT'], config['HOP_LENGTH'],
                            config['eq_gain_db'], config['eq_center_freqs'])
    if name == 'compression':
        return pseudo_differentiable_compression(y, sr, config['compression_codecs'],
                                                 config['compression_bitrates'])
    raise ValueError(f'Unknown distortion: {name}')
