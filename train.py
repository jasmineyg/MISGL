'''Thin command-line entry point for MISGL training.'''

import argparse
from typing import Optional, Sequence

from MISGL.config import apply_overrides, load_config
from MISGL.trainer import run


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MISGL')
    parser.add_argument(
        '--config',
        default='config/train.yml',
        help='Path to the single training YAML file.',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Dataset names. Replaces the datasets listed in the YAML file.',
    )
    parser.add_argument('--data-dir', help='Override the processed-data directory.')
    parser.add_argument('--output-dir', help='Override the result directory.')
    parser.add_argument(
        '--device',
        choices=('cpu', 'cuda'),
        help='Override the configured execution device.',
    )
    parser.add_argument(
        '--mil-head',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable or disable MIL-HEAD.',
    )
    parser.add_argument(
        '--pos-head',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable or disable POS-HEAD. POS-HEAD requires MIL-HEAD.',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(
        load_config(args.config),
        datasets=args.datasets,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        mil_head=args.mil_head,
        pos_head=args.pos_head,
    )
    run(config)


if __name__ == '__main__':
    main()
