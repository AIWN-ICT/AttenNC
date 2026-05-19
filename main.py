"""Command-line entry point for AttenNC training and evaluation."""

from __future__ import annotations

import argparse

import test
import train


def build_parser():
    """Create the top-level CLI parser with train and test subcommands."""
    parser = argparse.ArgumentParser(description="AttenNC command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from scratch")
    train_parser.set_defaults(_target="train")

    test_parser = subparsers.add_parser("test", help="Evaluate using saved best checkpoints")
    test_parser.set_defaults(_target="test")

    return parser



def main(argv=None):
    """Dispatch the requested subcommand to the training or testing module."""
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command == "train":
        train.main(remaining)
    elif args.command == "test":
        test.main(remaining)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
