"""Verifies the training-side models against a released SilentCipher checkpoint.

Checks performed:
  1. Builds Encoder / CarrierDecoder / MsgDecoder with the exact dimensions used
     by silentcipher.server.Model and the released hparams.yaml, then loads the
     released state dicts with strict=True. A strict load passing proves the
     training-side architecture is identical to the released one.
  2. Dumps the optimizer hyperparameters recorded in opt.ckpt (optimizer type,
     lr, betas, eps, weight_decay, step counts) — ground truth for the training
     configuration, not guesses.

Usage:
    python train/verify_against_release.py --release_dir Models/44_1_khz/73999_iteration
    python train/verify_against_release.py --release_dir Models/16_khz/97561_iteration
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from silentcipher.model import CarrierDecoder, Encoder, MsgDecoder  # noqa: E402


def strip_dataparallel(state):
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}


def load_state(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # opt.ckpt may contain non-tensor python objects
        return torch.load(path, map_location='cpu', weights_only=False)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release_dir', required=True,
                        help='Checkpoint folder containing hparams.yaml and *.ckpt files')
    args = parser.parse_args()

    with open(os.path.join(args.release_dir, 'hparams.yaml')) as f:
        hp = yaml.safe_load(f)
    cfg_ns = argparse.Namespace(**hp)

    print(f'=== {args.release_dir} ===')
    print(f"hparams: SR={hp['SR']} N_FFT={hp['N_FFT']} HOP={hp['HOP_LENGTH']} "
          f"band={hp['message_band_size']} message_dim={hp['message_dim']} "
          f"message_len={hp['message_len']} batch={hp.get('batch_size')} lr={hp.get('lr')} "
          f"seed={hp.get('seed')}")
    print(f"normalization: frame={hp['frame_level_normalization']} "
          f"utterance={hp['utterance_level_normalization']} no_norm={hp['no_normalization']}")
    print(f"augmentation: mp3={hp.get('mp3_aug')} ogg={hp.get('ogg_aug')} "
          f"aac={hp.get('aac_aug')} noise={hp.get('add_carrier_noise')} "
          f"carrier_noise_norm={hp.get('carrier_noise_norm')}")

    # --- 1. Architecture check: strict state_dict loads -------------------
    # Dimensions copied from silentcipher.server.Model.__init__
    enc_c = Encoder(n_layers=hp['enc_n_layers'], message_dim=hp['message_dim'],
                    out_dim=32, message_band_size=hp['message_band_size'],
                    n_fft=hp['N_FFT'])
    dec_c = CarrierDecoder(config=cfg_ns, conv_dim=32 * 3,
                           n_layers=hp['dec_c_n_layers'],
                           message_band_size=hp['message_band_size'])

    checks = [('enc_c.ckpt', enc_c), ('dec_c.ckpt', dec_c)]
    for i in range(hp['n_messages']):
        dec_m = MsgDecoder(message_dim=hp['message_dim'],
                           message_band_size=hp['message_band_size'])
        checks.append((f'dec_m_{i}.ckpt', dec_m))

    all_ok = True
    for fname, module in checks:
        state = strip_dataparallel(load_state(os.path.join(args.release_dir, fname)))
        try:
            module.load_state_dict(state, strict=True)
            print(f'[OK]   {fname}: strict load passed '
                  f'({len(state)} tensors, {n_params(module):,} params)')
        except RuntimeError as e:
            all_ok = False
            print(f'[FAIL] {fname}: {e}')

    # --- 2. Optimizer state: recorded training hyperparameters ------------
    opt_path = os.path.join(args.release_dir, 'opt.ckpt')
    if os.path.exists(opt_path):
        opt = load_state(opt_path)
        if isinstance(opt, dict) and 'param_groups' in opt:
            for gi, group in enumerate(opt['param_groups']):
                known = {k: v for k, v in group.items() if k != 'params'}
                print(f'[opt]  param_group {gi}: {known}')
            steps = [s['step'] for s in opt.get('state', {}).values() if 'step' in s]
            if steps:
                steps = [int(s.item()) if torch.is_tensor(s) else int(s) for s in steps]
                print(f'[opt]  recorded step counts: min={min(steps)} max={max(steps)}')
        else:
            print(f'[opt]  unexpected opt.ckpt structure: {type(opt)}')
    else:
        print('[opt]  opt.ckpt not found')

    print('RESULT:', 'architecture matches release' if all_ok else 'MISMATCH — see failures above')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
