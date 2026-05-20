"""Training entry point for AttenNC experiments."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import SEED
from config_parser import load_config
from evaluate import run_evaluate
from network_layers import set_tdma_seed
from utils.relay_dqn_agent import DQNAgent_R
from utils.source_dqn_agent import DQNAgent_S
from utils.gnn_model import GNNMARL


torch.backends.cudnn.enabled = False
np.set_printoptions(threshold=np.inf)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "checkpoints" / "best_by_avg_source_send"
DEFAULT_RESULT_DIR = BASE_DIR / "results"


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
set_tdma_seed(SEED)


def build_parser():
    """Create the command-line argument parser for training."""
    parser = argparse.ArgumentParser(description="AttenNC training")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory for saving best model files (default: models/checkpoints/best_by_avg_source_send)",
    )
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory for generated outputs")
    parser.add_argument("--best-state", default="best_epoch_GNN.pkl", help="Filename or path for best random-state snapshot")
    parser.add_argument("--skip-compile", action="store_true", help="Disable torch.compile for compatibility")
    parser.add_argument(
        "--mode",
        choices=["on", "off"],
        default=None,
        help="Relay coding-node selection mode: on=enable selection, off=disable selection",
    )
    return parser



def resolve_paths(args, enable_relay_coding_selection):
    """Resolve output paths and create directories when they do not exist."""
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = BASE_DIR / result_dir

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir

    mode_subdir = "selection_on" if enable_relay_coding_selection else "selection_off"
    model_dir = model_dir / mode_subdir
    result_dir = result_dir / mode_subdir

    best_state_path = Path(args.best_state)
    if not best_state_path.is_absolute():
        best_state_path = model_dir / best_state_path

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return result_dir, model_dir, best_state_path



def build_agents(config, skip_compile=False):
    """Construct the source/relay agents and optionally the GNN controller."""
    parallel_path = config["parallel_path"]
    max_nb = config["max_nb"]
    K = config["K"]
    M = config["M"]
    batch_size = config["batch_size"]
    device = config["device"]
    max_buffer_size = config["max_buffer_size"]
    enable_relay_coding_selection = config.get("enable_relay_coding_selection", True)

    S_state_size = (M * parallel_path + 2) * K
    R_state_size = (M * max_nb + 2) * K

    model = None
    if enable_relay_coding_selection:
        model = GNNMARL(
            arg_dict=config,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            buffer_size=max_buffer_size,
            batch_size=batch_size,
            target_update_freq=10,
        ).to(device)

    agent_s = DQNAgent_S(S_state_size, config, device)
    agent_r = DQNAgent_R(R_state_size, config, device)

    if not skip_compile and hasattr(torch, "compile"):
        agent_s = torch.compile(agent_s)
        agent_r = torch.compile(agent_r)

    return model, agent_s, agent_r



def save_best_models(model_dir, best_state_path, best_state, agent_s, agent_r, model, enable_relay_coding_selection=True):
    """Persist the current best-performing checkpoints and random state."""
    agent_r.save_network(str(model_dir / "dqn_agent_r_min.pt"))
    agent_s.save_network(str(model_dir / "dqn_agent_s_min.pt"))

    gnn_model_path = model_dir / "gnn_model_dqn.pt"
    if enable_relay_coding_selection:
        model.save_network(str(gnn_model_path))
    elif gnn_model_path.exists():
        os.remove(gnn_model_path)

    with open(best_state_path, "wb") as f:
        pickle.dump({"best_state": best_state}, f)



def main(argv=None):
    """Run the training loop and keep the best checkpoint by source transmissions."""
    args = build_parser().parse_args(argv)
    start_time = time.time()

    config = load_config()
    if args.mode is not None:
        config["enable_relay_coding_selection"] = args.mode.lower() == "on"
    enable_relay_coding_selection = config.get("enable_relay_coding_selection", True)
    episodes = config["EPISODES"]
    max_s_f = config["Max_s_f"]

    result_dir, model_dir, best_state_path = resolve_paths(args, enable_relay_coding_selection)
    model, agent_s, agent_r = build_agents(config, args.skip_compile)

    min_f = max_s_f

    for e in range(episodes):
        metrics = run_evaluate(
            e=e,
            config=config,
            agent_s=agent_s,
            agent_r=agent_r,
            model=model,
            result_dir=str(result_dir),
            min_f=min_f,
            load_best_state=False,
            best_state_path=best_state_path,
            save_csv=False,
            enable_training_updates=True,
        )

        if metrics["is_best"]:
            min_f = metrics["avg_s_f"]
            save_best_models(
                model_dir,
                best_state_path,
                metrics["best_state"],
                agent_s,
                agent_r,
                model,
                enable_relay_coding_selection=enable_relay_coding_selection,
            )

        elapsed_time = time.time() - start_time
        average_time = elapsed_time / (e + 1)
        remaining_time = average_time * (episodes - (e + 1))
        if e % 10 == 0:
            print("\r episode: %d, remaining: %d s" % (e, remaining_time), end="")


if __name__ == "__main__":
    main()
