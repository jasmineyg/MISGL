# coding=utf-8

import argparse
import json

import torch

from MISGL.bin import train_eval
from MISGL.models import encoder
from MISGL.utils import get_loss, hparam, hparams_lib, load_data, reproducibility
from MISGL.utils.global_variables import g_key


def _run_one(hparams, data_name, training_loader, loss_type):
    hparams.loss_type = loss_type
    loss_config = train_eval._configure_fold_loss(hparams, training_loader)
    model = encoder.MISGLEncoder(hparams, data_name=data_name).to(torch.device(hparams.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    batch = train_eval._move_batch_to_device(next(iter(training_loader)), torch.device(hparams.device))
    optimizer.zero_grad()
    output = model(batch)
    loss = get_loss.fused_loss(output, batch[g_key.y], 0, hparams)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke-test loss: {}'.format(float(loss.detach().cpu())))
    loss.backward()
    optimizer.step()
    result = {
        'dataset': data_name,
        'loss_config': loss_config,
        'loss': float(loss.detach().cpu()),
        'batch_size': int(batch[g_key.y].numel()),
    }
    del model, optimizer, batch, output, loss
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparam_path', required=True)
    parser.add_argument('--data_name', required=True)
    parser.add_argument('--all_losses', action='store_true')
    args = parser.parse_args()

    hparams = hparam.HParams()
    hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(hparams)
    hparams.data_name = args.data_name
    reproducibility.set_seed(int(hparams.cv_seed), cuda_deterministic=True)

    wrapper = load_data.GraphDataLoaderWrapper(hparams, data_name=args.data_name)
    manifest = wrapper.load_cv_split_manifest(wrapper.get_cv_split_path(ensure_dir=False))
    training_loader, _, _, _ = wrapper.get_cv_loaders_from_manifest(manifest, 0)
    loss_types = ('bce', 'focal', 'weighted_bce') if args.all_losses else (hparams.loss_type,)
    for loss_type in loss_types:
        print(json.dumps(
            _run_one(hparams, args.data_name, training_loader, loss_type),
            ensure_ascii=False,
        ))


if __name__ == '__main__':
    main()
