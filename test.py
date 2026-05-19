"""Evaluation entry point for running AttenNC with saved checkpoints."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

import data_processor
from config_parser import load_config
from evaluate import run_evaluate
from network_layers import set_tdma_seed
from utils.relay_dqn_agent import DQNAgent_R
from utils.source_dqn_agent import DQNAgent_S
from utils.gnn_model import GNNMARL


torch.backends.cudnn.enabled = False
np.set_printoptions(threshold=np.inf)

seed = 555
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "examples" / "best_by_avg_source_send"
DEFAULT_RESULT_DIR = BASE_DIR / "results"
PACKET_LOG_DIR = BASE_DIR / "packet_log"
METRICS_DIR = BASE_DIR / "data_attenNC"
PACKET_COUNT_LOG = "source_send_counts.csv"
AGGREGATED_METRICS_BASENAME = "decode_probability_summary"


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_tdma_seed(seed)


def build_parser():
    """Create the command-line argument parser for evaluation."""
    parser = argparse.ArgumentParser(description="AttenNC testing")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing pretrained model files. Defaults to the example checkpoint folder under models/examples/.",
    )
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory for generated outputs")
    parser.add_argument(
        "--best-state",
        default="best_epoch_GNN.pkl",
        help="Filename or path for best random-state snapshot inside the selected model directory",
    )
    parser.add_argument("--skip-compile", action="store_true", help="Disable torch.compile for compatibility")
    parser.add_argument(
        "--mode",
        choices=["on", "off"],
        default=None,
        help="Relay coding-node selection mode: on=enable selection, off=disable selection",
    )
    return parser



def resolve_paths(args, enable_relay_coding_selection):
    """Resolve evaluation output paths relative to the project directory."""
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
    return result_dir, model_dir, best_state_path



def safe_load_network(agent, model_path):
    """Load a checkpoint and raise a clear error when the file is missing."""
    if os.path.exists(model_path):
        agent.load_network(model_path)
    else:
        raise FileNotFoundError(f"Required model file not found: {model_path}")



def build_agents(config, skip_compile=False):
    """Construct the source agent, relay agent, and GNN controller for testing."""
    parallel_path = config["parallel_path"]
    max_nb = config["max_nb"]
    K = config["K"]
    M = config["M"]
    batch_size = config["batch_size"]
    device = config["device"]
    max_buffer_size = config["max_buffer_size"]

    S_state_size = (M * parallel_path + 2) * K
    R_state_size = (M * max_nb + 2) * K

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



def main(argv=None):
    """Run evaluation with pretrained checkpoints and export summary metrics."""
    args = build_parser().parse_args(argv)
    config = load_config()
    if args.mode is not None:
        config["enable_relay_coding_selection"] = args.mode.lower() == "on"
    max_s_f = config["Max_s_f"]
    enable_relay_coding_selection = config.get("enable_relay_coding_selection", True)

    result_dir, model_dir, best_state_path = resolve_paths(args, enable_relay_coding_selection)
    model, agent_s, agent_r = build_agents(config, args.skip_compile)

    safe_load_network(agent_r, str(model_dir / "dqn_agent_r_min.pt"))
    safe_load_network(agent_s, str(model_dir / "dqn_agent_s_min.pt"))

    enable_relay_coding_selection = config.get("enable_relay_coding_selection", True)
    if enable_relay_coding_selection:
        safe_load_network(model, str(model_dir / "gnn_model_dqn.pt"))
    else:
        model = None

    metrics = run_evaluate(
        e=0,
        config=config,
        agent_s=agent_s,
        agent_r=agent_r,
        model=model,
        result_dir=str(result_dir),
        min_f=max_s_f,
        load_best_state=True,
        best_state_path=best_state_path,
        save_csv=True,
    )

    mode_subdir = "selection_on" if enable_relay_coding_selection else "selection_off"
    packet_log_dir = PACKET_LOG_DIR / mode_subdir
    metrics_dir = METRICS_DIR / mode_subdir
    os.makedirs(packet_log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    data_processor.write_counts_to_csv(metrics["source_send_count_list"], str(packet_log_dir), PACKET_COUNT_LOG)
    decode_probability = data_processor.calculate_decode_probability(metrics["source_send_count_list"])
    data_processor.write_results_to_csv(
        AGGREGATED_METRICS_BASENAME,
        decode_probability,
        metrics["avg_overhead"],
        data_processor.std_dev(metrics["source_send_count_list"]),
        metrics["avg_s_f"],
        str(metrics_dir),
    )

    print(f"Test done. avg_s_f={metrics['avg_s_f']:.4f}, avg_overhead={metrics['avg_overhead']:.4f}")


if __name__ == "__main__":
    main()
