"""Forensic analysis of the released SilentCipher checkpoints.

Extracts training evidence recorded inside the tensors themselves, beyond what
hparams.yaml states:

  1. BatchNorm `num_batches_tracked` per module — the exact number of training
     forward passes each module performed (survives checkpoint resumes, so it
     reveals cumulative training across fine-tuning stages).
  2. Adam `opt.ckpt` state — step counts per parameter, and a shape-based
     mapping of optimizer entries onto module parameters, which recovers the
     parameter ordering / grouping the original trainer used (i.e. whether
     enc_c, dec_c and dec_m were optimized jointly by one optimizer and in
     which order they were registered).
  3. Basic weight statistics per module (sanity signal for training health).

Usage:
    python train/analyze_release_ckpts.py --release_dir Models/44_1_khz/73999_iteration
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
        return torch.load(path, map_location='cpu', weights_only=False)


def bn_counters(name, state):
    counts = sorted({int(v) for k, v in state.items() if k.endswith('num_batches_tracked')})
    print(f'[bn]   {name}: num_batches_tracked = {counts}')
    return counts


def weight_stats(name, state):
    weights = [v.float() for k, v in state.items() if k.endswith('.weight') and v.dim() > 1]
    flat = torch.cat([w.flatten() for w in weights])
    print(f'[w]    {name}: {len(weights)} weight tensors | '
          f'mean {flat.mean():+.5f} | std {flat.std():.5f} | absmax {flat.abs().max():.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release_dir', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.release_dir, 'hparams.yaml')) as f:
        hp = yaml.safe_load(f)
    cfg_ns = argparse.Namespace(**hp)

    print(f'=== {args.release_dir} ===')
    print(f"[hp]   load_ckpt (resume source): {hp.get('load_ckpt')}")

    states = {}
    for fname in ['enc_c.ckpt', 'dec_c.ckpt'] + [f'dec_m_{i}.ckpt' for i in range(hp['n_messages'])]:
        states[fname] = strip_dataparallel(load_state(os.path.join(args.release_dir, fname)))

    # --- 1. BatchNorm forward-pass counters --------------------------------
    for fname, state in states.items():
        bn_counters(fname, state)

    # --- 2. Optimizer state forensics ---------------------------------------
    opt = load_state(os.path.join(args.release_dir, 'opt.ckpt'))
    groups = opt['param_groups']
    opt_state = opt['state']
    print(f'[opt]  {len(groups)} param_group(s); '
          f'{sum(len(g["params"]) for g in groups)} params registered; '
          f'{len(opt_state)} params with Adam state')

    steps = sorted({int(s['step'].item() if torch.is_tensor(s['step']) else s['step'])
                    for s in opt_state.values()})
    print(f'[opt]  distinct step counts: {steps}')

    # Map optimizer entries to module parameters by shape sequence.
    # The original trainer registered parameters in some module order; if the
    # concatenated shape sequence of [enc_c, dec_c, dec_m_0] (in that order)
    # equals the optimizer's shape sequence, the modules were jointly optimized
    # by ONE optimizer in that registration order.
    def trainable_shapes(module):
        return [tuple(p.shape) for p in module.parameters()]

    enc_c = Encoder(n_layers=hp['enc_n_layers'], message_dim=hp['message_dim'], out_dim=32,
                    message_band_size=hp['message_band_size'], n_fft=hp['N_FFT'])
    dec_c = CarrierDecoder(config=cfg_ns, conv_dim=32 * 3, n_layers=hp['dec_c_n_layers'],
                           message_band_size=hp['message_band_size'])
    dec_m = [MsgDecoder(message_dim=hp['message_dim'], message_band_size=hp['message_band_size'])
             for _ in range(hp['n_messages'])]

    modules = {'enc_c': enc_c, 'dec_c': dec_c}
    for i, m in enumerate(dec_m):
        modules[f'dec_m_{i}'] = m

    opt_shapes = [tuple(opt_state[i]['exp_avg'].shape) for i in sorted(opt_state.keys())]

    import itertools
    match = None
    for order in itertools.permutations(modules.keys()):
        concat = [s for name in order for s in trainable_shapes(modules[name])]
        if concat == opt_shapes:
            match = order
            break
    if match:
        print(f'[opt]  shape sequence matches JOINT optimization, '
              f'module registration order: {" -> ".join(match)}')
    else:
        total = sum(len(trainable_shapes(m)) for m in modules.values())
        print(f'[opt]  no permutation of module orders matches '
              f'({len(opt_shapes)} opt entries vs {total} module params) — '
              f'parameter grouping differed from a plain jointly-registered optimizer')

    # Gradient magnitude footprint at save time (per module, from exp_avg_sq)
    if match:
        idx = 0
        for name in match:
            n = len(trainable_shapes(modules[name]))
            keys = sorted(opt_state.keys())[idx:idx + n]
            rms = torch.cat([opt_state[k]['exp_avg_sq'].flatten() for k in keys]).mean().sqrt()
            print(f'[grad] {name}: sqrt(mean exp_avg_sq) = {rms:.3e}')
            idx += n

    # --- 3. Weight statistics ------------------------------------------------
    for fname, state in states.items():
        weight_stats(fname, state)


if __name__ == '__main__':
    main()
