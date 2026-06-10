"""SilentCipher training script — a RECONSTRUCTION, not the official training code.

Sony has not released the training code for SilentCipher (arXiv:2406.03822); the
repository README says it is still pending. This script rebuilds the training
procedure around the released inference code. Be aware of what is verified and
what is inferred:

Verified against released artifacts:
  - Model architecture and dimensions (src/silentcipher/model.py, server.py);
    strict state_dict loads of both released checkpoints pass
    (train/verify_against_release.py).
  - The differentiable embed path in `Trainer.embed` mirrors
    `Model.encode_wav` line by line; behavioral probes of the released 44k
    model confirm negativity, band masking, utterance-level normalization at
    exactly 10^(-alpha/20), and the SDR lower bound
    (train/probe_release_behavior.py).
  - Checkpoint layout ({iter}_iteration/{enc_c,dec_c,dec_m_i}.ckpt + hparams.yaml)
    consumed by `silentcipher.get_model`.
  - Config values generated directly from the official hparams.yaml files
    (train/make_config_from_release.py).
  - Optimizer forensics from opt.ckpt (train/analyze_release_ckpts.py):
    ONE joint Adam(lr=1e-3, betas=(0.9, 0.999), wd=0) over all modules,
    registered enc_c -> dec_c -> dec_m_0 — matching this Trainer. A scheduler
    object existed but lr was still 1e-3 at the final step, so constant LR is
    behaviorally identical over the released horizon.
  - BatchNorm num_batches_tracked == optimizer steps in every module: each
    module ran exactly ONE training forward per iteration (single distortion,
    single decode — the structure of `Trainer.step`). The released 16k model
    is a 16001-step fine-tune of an 81560-step run (81561 + 16001 = 97562).
  - Gaussian noise level: hparams carrier_noise_norm 0.01 == 1% amplitude
    == 40 dB SNR, matching gaussian_snr_db: 40.

Taken from the paper (Sec. 3-4):
  - Cross-entropy-only loss; the SDR lower bound is enforced architecturally
    (Eq. 1-3), the watermark is negative w.r.t. the carrier with ReLU(C + W).
  - 80k iterations (released 44k ran ~74k), 12-second crops; one distortion
    sampled uniformly per iteration; pseudo-differentiable (straight-through)
    compression.

Inferred / undocumented (filled with reasonable guesses — review before trusting):
  - Jitter magnitude, EQ curve shape and width, codec round-trip alignment —
    see distortions.py.
  - Energy re-normalization before the decoder (mirrors decode_wav, but the
    original training-time behavior is unknown).
  - Message sampling details, data pipeline, and crop/silence handling.

A model trained with this script is NOT guaranteed to match the released
checkpoints' behavior.

Usage:
    python train/train.py --config train/config_44k.yaml
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from silentcipher.model import CarrierDecoder, Encoder, MsgDecoder  # noqa: E402
from silentcipher.stft import STFT  # noqa: E402

from dataset import AudioCropDataset  # noqa: E402
from distortions import apply_distortion  # noqa: E402

INFERENCE_HPARAM_KEYS = [
    'model_type', 'SR', 'N_FFT', 'HOP_LENGTH', 'message_band_size', 'message_dim',
    'message_len', 'n_messages', 'enc_n_layers', 'dec_c_n_layers', 'message_sdr',
    'ensure_negative_message', 'ensure_constrained_message', 'no_normalization',
    'frame_level_normalization', 'utterance_level_normalization',
]


class _CheckpointedLayer(torch.nn.Module):
    """Recomputes the wrapped layer during backward to save activation memory.

    CAVEAT: recomputation runs BatchNorm twice per training step, so running
    stats and num_batches_tracked advance twice as fast as the released
    checkpoints' one-forward-per-iteration pattern. Prefer amp +
    grad_accum_steps first; use this only if memory still does not fit.
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        if self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self.mod, x, use_reentrant=False)
        return self.mod(x)


class Trainer:

    def __init__(self, config):
        self.config = config
        self.cfg_ns = argparse.Namespace(**config)
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Model dimensions follow silentcipher.server.Model so that the trained
        # checkpoints are drop-in compatible with the inference code.
        self.encoder_out_dim = 32
        self.dec_c_conv_dim = 32 * 3

        self.enc_c = Encoder(
            n_layers=config['enc_n_layers'],
            message_dim=config['message_dim'],
            out_dim=self.encoder_out_dim,
            message_band_size=config['message_band_size'],
            n_fft=config['N_FFT'],
        ).to(self.device)

        self.dec_c = CarrierDecoder(
            config=self.cfg_ns,
            conv_dim=self.dec_c_conv_dim,
            n_layers=config['dec_c_n_layers'],
            message_band_size=config['message_band_size'],
        ).to(self.device)

        self.dec_m = torch.nn.ModuleList([
            MsgDecoder(message_dim=config['message_dim'],
                       message_band_size=config['message_band_size'])
            for _ in range(config['n_messages'])
        ]).to(self.device)

        self.stft = STFT(config['N_FFT'], config['HOP_LENGTH']).to(self.device)

        # Memory/throughput options (off by default; semantics-preserving except
        # for the BatchNorm caveat documented on _CheckpointedLayer)
        self.amp = bool(config.get('amp')) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.accum = int(config.get('grad_accum_steps') or 1)
        if config.get('activation_checkpointing'):
            for module in [self.enc_c, self.dec_c, *self.dec_m]:
                module.main = torch.nn.Sequential(
                    *[_CheckpointedLayer(m) for m in module.main])

        params = (list(self.enc_c.parameters()) + list(self.dec_c.parameters())
                  + list(self.dec_m.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

        dataset = AudioCropDataset(
            roots=config['data_dirs'],
            sr=config['SR'],
            crop_seconds=config['duration_seconds'],
            average_energy=config['average_energy'],
            epoch_len=config['batch_size'] * 1000,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True,
            persistent_workers=config['num_workers'] > 0,
        )

        self.val_loader = None
        if config.get('val_dirs'):
            val_dataset = AudioCropDataset(
                roots=config['val_dirs'],
                sr=config['SR'],
                crop_seconds=config['duration_seconds'],
                average_energy=config['average_energy'],
                epoch_len=config['batch_size'] * config.get('eval_batches', 8),
            )
            self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                         num_workers=0, drop_last=True)

        os.makedirs(config['output_dir'], exist_ok=True)
        self.train_log_path = os.path.join(config['output_dir'], 'train_log.csv')
        self.val_log_path = os.path.join(config['output_dir'], 'val_log.csv')
        self._init_csv(self.train_log_path, ['iteration', 'loss', 'frame_acc', 'sdr_db', 'sec_per_iter'])
        self._init_csv(self.val_log_path,
                       ['iteration', 'distortion', 'frame_acc', 'message_acc', 'sdr_db'])

    @staticmethod
    def _init_csv(path, header):
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    @staticmethod
    def _append_csv(path, row):
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def set_train(self, mode):
        self.enc_c.train(mode)
        self.dec_c.train(mode)
        self.dec_m.train(mode)

    # ------------------------------------------------------------------ #

    def sample_messages(self, batch_size, n_frames):
        """Random messages tiled along the time axis (one char per spectrogram frame).

        With an end token (SC-44: message_dim 5, message_len 21) the message is
        (message_len - 1) chars in [1, D-1] followed by the end token 0. Without
        one (SC-16: message_dim 4, message_len 16) all chars are uniform in [0, D-1].
        """
        d, m_len = self.config['message_dim'], self.config['message_len']
        if self.config.get('end_token', True):
            chars = torch.randint(1, d, (batch_size, m_len - 1), device=self.device)
            end = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            msg = torch.cat([chars, end], dim=1)
        else:
            msg = torch.randint(0, d, (batch_size, m_len), device=self.device)
        reps = math.ceil(n_frames / m_len)
        target = msg.repeat(1, reps)[:, :n_frames]                       # (B, T)
        one_hot = F.one_hot(target, d).float().transpose(1, 2)[:, None]  # (B, 1, D, T)
        return target, one_hot

    def sample_message_sdr(self, batch_size):
        rng = self.config.get('message_sdr_range')
        if rng:
            sdr = torch.empty(batch_size, 1, 1, 1, device=self.device).uniform_(rng[0], rng[1])
        else:
            sdr = torch.full((batch_size, 1, 1, 1), float(self.config['message_sdr']),
                             device=self.device)
        return sdr

    def embed(self, y, one_hot, message_sdr):
        """Mirrors silentcipher.server.Model.encode_wav (the differentiable part).

        Under AMP only the conv stacks run in reduced precision; the STFT/iSTFT
        and the carrier arithmetic stay fp32.
        """
        cfg = self.config
        carrier, carrier_phase = self.stft.transform(y)   # (B, F, T)
        carrier = carrier[:, None]

        with torch.autocast(self.device.type, enabled=self.amp):
            carrier_enc = self.enc_c(carrier)
            msg_enc = self.enc_c.transform_message(one_hot)
            merged = torch.cat(
                (carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1)
            message_info = self.dec_c(merged, message_sdr)
        message_info = message_info.float()

        if cfg['frame_level_normalization']:
            message_info = message_info * torch.mean(carrier ** 2, dim=2, keepdim=True) ** 0.5
        elif cfg['utterance_level_normalization']:
            message_info = message_info * torch.mean(carrier ** 2, dim=(2, 3), keepdim=True) ** 0.5

        if cfg['ensure_negative_message']:
            message_info = -message_info
            carrier_reconst = F.relu(message_info + carrier)
        elif cfg['ensure_constrained_message']:
            message_info = torch.clamp(message_info, min=-carrier, max=carrier)
            carrier_reconst = message_info + carrier
        else:
            carrier_reconst = torch.abs(message_info + carrier)

        self.stft.num_samples = y.shape[-1]
        y_wm = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase)[:, 0]
        return y_wm, carrier.shape[3]

    def decode_logits(self, y):
        # Re-normalize the energy, matching Model.decode_wav at inference time
        power = torch.mean(y ** 2, dim=-1, keepdim=True)
        y = y * torch.sqrt(self.config['average_energy'] / power)
        carrier, _ = self.stft.transform(y)
        with torch.autocast(self.device.type, enabled=self.amp):
            logits_list = [dec(carrier[:, None]).squeeze(1) for dec in self.dec_m]
        return [logits.float() for logits in logits_list]  # each (B, D, T)

    def forward_loss(self, y, distortion_name):
        """One micro-batch forward; returns the loss tensor plus metrics."""
        batch_size = y.shape[0]
        with torch.no_grad():
            # Frame count of the padded STFT, needed to tile the message
            n_frames = self.stft.transform(y[:1])[0].shape[-1]

        target, one_hot = self.sample_messages(batch_size, n_frames)
        message_sdr = self.sample_message_sdr(batch_size)

        y_wm, _ = self.embed(y, one_hot, message_sdr)
        y_dist = apply_distortion(distortion_name, y_wm, self.config['SR'], self.config)

        logits_list = self.decode_logits(y_dist)
        loss = sum(F.cross_entropy(logits, target) for logits in logits_list) / len(logits_list)

        with torch.no_grad():
            acc = (logits_list[0].argmax(dim=1) == target).float().mean().item()
            sdr = self.sdr(y, y_wm)
        return loss, acc, sdr

    def step(self, batches, distortion_name):
        """One optimizer step over grad_accum_steps micro-batches.

        The same distortion is used for every micro-batch of a step, preserving
        the released models' one-distortion-per-iteration pattern; the effective
        batch size is batch_size * grad_accum_steps.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = acc_sum = sdr_sum = 0.0
        for _ in range(self.accum):
            y = next(batches).to(self.device)
            loss, acc, sdr = self.forward_loss(y, distortion_name)
            self.scaler.scale(loss / self.accum).backward()
            loss_sum += loss.item()
            acc_sum += acc
            sdr_sum += sdr

        if self.config.get('grad_clip'):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']],
                self.config['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_sum / self.accum, acc_sum / self.accum, sdr_sum / self.accum

    @staticmethod
    def sdr(orig, recon):
        rms1 = torch.mean(orig ** 2, dim=-1) ** 0.5
        rms2 = torch.mean((orig - recon) ** 2, dim=-1) ** 0.5
        return (20 * torch.log10(rms1 / (rms2 + 1e-12))).mean().item()

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self, iteration):
        """Per-distortion decode accuracy on held-out audio.

        `message_acc` decodes the way inference does (mode over the message
        repetitions along time, without the phase-shift search), so it is the
        closer proxy for the accuracy numbers reported in the paper.
        """
        if self.val_loader is None:
            return
        self.set_train(False)
        m_len = self.config['message_len']

        for distortion in self.config['distortions']:
            frame_accs, msg_accs, sdrs = [], [], []
            for y in self.val_loader:
                y = y.to(self.device)
                n_frames = self.stft.transform(y[:1])[0].shape[-1]
                target, one_hot = self.sample_messages(y.shape[0], n_frames)
                message_sdr = torch.full((y.shape[0], 1, 1, 1),
                                         float(self.config['message_sdr']), device=self.device)

                y_wm, _ = self.embed(y, one_hot, message_sdr)
                y_dist = apply_distortion(distortion, y_wm, self.config['SR'], self.config)
                logits = self.decode_logits(y_dist)[0]          # (B, D, T)
                preds = logits.argmax(dim=1)                    # (B, T)

                frame_accs.append((preds == target).float().mean().item())

                n_reps = preds.shape[1] // m_len
                pred_mode = torch.mode(preds[:, :n_reps * m_len].reshape(-1, n_reps, m_len),
                                       dim=1).values            # (B, m_len)
                msg_accs.append((pred_mode == target[:, :m_len]).float().mean().item())
                sdrs.append(self.sdr(y, y_wm))

            row = [iteration, distortion,
                   float(np.mean(frame_accs)), float(np.mean(msg_accs)), float(np.mean(sdrs))]
            self._append_csv(self.val_log_path, row)
            print(f'[val] iter {iteration} | {distortion:15s} | '
                  f'frame-acc {row[2]:.4f} | msg-acc {row[3]:.4f} | sdr {row[4]:5.2f} dB')
        self.set_train(True)

    @staticmethod
    def _portable_state_dict(module):
        # Strip the '.mod.' segment introduced by _CheckpointedLayer so saved
        # checkpoints stay loadable by silentcipher.get_model
        return {k.replace('.mod.', '.'): v for k, v in module.state_dict().items()}

    def save_checkpoint(self, iteration):
        out_dir = os.path.join(self.config['output_dir'], f'{iteration}_iteration')
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self._portable_state_dict(self.enc_c), os.path.join(out_dir, 'enc_c.ckpt'))
        torch.save(self._portable_state_dict(self.dec_c), os.path.join(out_dir, 'dec_c.ckpt'))
        for i, dec in enumerate(self.dec_m):
            torch.save(self._portable_state_dict(dec), os.path.join(out_dir, f'dec_m_{i}.ckpt'))
        hparams = {k: self.config[k] for k in INFERENCE_HPARAM_KEYS}
        with open(os.path.join(out_dir, 'hparams.yaml'), 'w') as f:
            yaml.safe_dump(hparams, f)
        print(f'[ckpt] saved {out_dir}')

    def _batches(self):
        while True:
            for y in self.loader:
                yield y

    def train(self):
        cfg = self.config
        distortion_set = cfg['distortions']
        log_every, save_every = cfg['log_every'], cfg['save_every']
        eval_every = cfg.get('eval_every', save_every)
        running = {'loss': 0.0, 'acc': 0.0, 'sdr': 0.0, 'n': 0}
        iteration, t0 = 0, time.time()
        batches = self._batches()

        self.set_train(True)

        while iteration < cfg['num_iterations']:
            distortion = random.choice(distortion_set)
            loss, acc, sdr = self.step(batches, distortion)

            running['loss'] += loss
            running['acc'] += acc
            running['sdr'] += sdr
            running['n'] += 1
            iteration += 1

            if iteration % log_every == 0:
                n = running['n']
                sec_per_iter = (time.time() - t0) / n
                print(f'iter {iteration:6d}/{cfg["num_iterations"]} | '
                      f'loss {running["loss"]/n:.4f} | frame-acc {running["acc"]/n:.4f} | '
                      f'sdr {running["sdr"]/n:5.2f} dB | '
                      f'{sec_per_iter:.2f} s/it')
                self._append_csv(self.train_log_path,
                                 [iteration, running['loss'] / n, running['acc'] / n,
                                  running['sdr'] / n, sec_per_iter])
                running = {'loss': 0.0, 'acc': 0.0, 'sdr': 0.0, 'n': 0}
                t0 = time.time()

            if iteration % eval_every == 0:
                self.evaluate(iteration)
                t0 = time.time()

            if iteration % save_every == 0:
                self.save_checkpoint(iteration)

        self.evaluate(iteration)
        if iteration % save_every != 0:
            self.save_checkpoint(iteration)


def main():
    parser = argparse.ArgumentParser(description='Train SilentCipher')
    parser.add_argument('--config', required=True, help='Path to the training config yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Trainer(config).train()


if __name__ == '__main__':
    main()
