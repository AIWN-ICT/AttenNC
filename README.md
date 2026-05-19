# AttenNC

AttenNC is an adaptive learning and simulation framework for generalizable network coding, integrating Source/Relay DQN with optional GNN-based node selection. The repository contains training and evaluation scripts, topology and protocol configuration files, reinforcement-learning agents, and utilities for recording experiment outputs.

## Project structure

### 1) Minimal must-read (quick start map)

```text
AttenNC/
├─ main.py                 # CLI entry: `train` / `test`
├─ train.py                # Training workflow
├─ test.py                 # Test/evaluation command wrapper
├─ evaluate.py             # Main simulation + evaluation loop
├─ config.py               # RL/MAC/PHY hyperparameters
├─ config_topology.py      # Topology and link configuration
├─ config_parser.py        # Runtime config merge/parse
├─ simulation_slots.py     # Episode/slot-level simulation orchestration
├─ utils/                  # Source/Relay DQN, GNN, replay buffer
└─ models/examples/        # Lightweight demo checkpoints
```

### 2) Complete file map (full list)

```text
AttenNC/
├─ Core entry files
│  ├─ main.py
│  ├─ train.py
│  ├─ test.py
│  ├─ evaluate.py
│  ├─ config.py
│  ├─ config_topology.py
│  └─ config_parser.py
│
├─ Simulation core modules
│  ├─ simulation_slots.py
│  ├─ slot_simulator.py
│  ├─ simulation_context.py
│  ├─ network_layers.py
│  ├─ layer_node.py
│  ├─ node.py
│  ├─ mac_layer.py
│  ├─ phy_layer.py
│  ├─ forwarding.py
│  ├─ coding_actions.py
│  ├─ coding_buffers.py
│  ├─ link_utils.py
│  ├─ link_events.py
│  ├─ topology_utils.py
│  ├─ rewards.py
│  ├─ efficiency_stats.py
│  ├─ interference_radius_check.py
│  ├─ pdu.py
│  ├─ results_writer.py
│  └─ data_processor.py
│
├─ Learning modules
│  └─ utils/
│     ├─ source_dqn_agent.py
│     ├─ relay_dqn_agent.py
│     ├─ gnn_model.py
│     ├─ replay_buffer.py
│     ├─ noisy_linear.py
│     ├─ source_agent_pytorch.py
│     ├─ relay_agent_pytorch.py
│     └─ __init__.py
│
├─ Models and artifacts
│  ├─ models/
│  │  ├─ examples/
│  │  ├─ checkpoints/
│  │  └─ README.md
│  ├─ results/
│  ├─ packet_log/
│  └─ data_incdeep_llm/
│
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Environment

- Python 3.10+ recommended
- PyTorch-compatible CPU or GPU environment

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependency installation notes

`torch-geometric` may require a version combination that matches your installed `torch` build, Python version, and CPU/CUDA environment. If `pip install -r requirements.txt` fails at `torch-geometric`, install PyTorch first, verify it works, and then install PyG separately according to the official instructions for your platform.

Recommended order:

```bash
pip install torch
pip install -r requirements.txt
```

If needed, replace the second step with a platform-specific `torch-geometric` install command from the PyTorch Geometric documentation:

- PyG install guide: `https://pytorch-geometric.readthedocs.io/`
- PyTorch install guide: `https://pytorch.org/get-started/locally/`

This is especially important when using:

- CUDA-enabled PyTorch
- Windows environments
- newer Python versions
- custom GPU driver/toolkit combinations

## Quick start

Run commands from the `AttenNC` directory.

### Train

```bash
python main.py train
```

Optional arguments:

- `--model-dir`: directory for saving/loading checkpoints
- `--result-dir`: directory for generated outputs
- `--best-state`: filename or path of the saved random-state snapshot
- `--skip-compile`: disable `torch.compile` for compatibility
- `--mode`: `on` / `off`

Example:

```bash
python main.py train --model-dir ./models/checkpoints/best_by_avg_source_send --result-dir ./results --skip-compile --mode on
```

### Test / Evaluate

```bash
python main.py test
```

Example:

```bash
python main.py test --model-dir ./models/examples/best_by_avg_source_send --result-dir ./results --skip-compile --mode off
```

## Mode behavior (`--mode on|off`)

`--mode` controls relay coding-node selection and automatically chooses mode-specific subfolders (`selection_on/` or `selection_off/`) under the provided `--model-dir` and output directories.

- `--mode on`
  - Enables relay coding-node selection (with GNN gating when applicable).
  - Train/load components: Source DQN + Relay DQN + GNN.
  - Typical checkpoint set includes `dqn_agent_s_min.pt`, `dqn_agent_r_min.pt`, `gnn_model_dqn.pt`, and `best_epoch_GNN.pkl`.

- `--mode off`
  - Disables relay coding-node selection (relay follows coding path without GNN gating decisions).
  - Train/load components: Source DQN + Relay DQN only (no GNN update).
  - Typical checkpoint set includes `dqn_agent_s_min.pt`, `dqn_agent_r_min.pt`, and `best_epoch_GNN.pkl`.

Default locations used in practice:

- Training checkpoints: `models/checkpoints/best_by_avg_source_send/<selection_on|selection_off>/`
- Example test checkpoints: `models/examples/best_by_avg_source_send/<selection_on|selection_off>/`

## Outputs

By default, generated artifacts are written to mode-separated subfolders (`selection_on/` / `selection_off/`) under:

- `results/`: per-run summaries and logs
- `packet_log/`: packet transmission count CSV files
- `data_incdeep_llm/`: aggregated decode-probability and related metrics
- `models/checkpoints/`: local training checkpoints (usually not committed)
- `models/examples/`: lightweight demo checkpoints for quick reproducibility

In practice, local outputs are usually not committed. Keep `models/examples/` small and runnable for demos.

See `models/README.md` for checkpoint organization details.

## Configuration

Main settings are defined in:

- `config.py`: episode count, RL hyperparameters, timing parameters, MAC/PHY packet settings
- `config_topology.py`: node layout, connectivity, and link structure

Relay coding-node selection can be set in code and overridden by CLI:

- `ENABLE_RELAY_CODING_SELECTION = True/False`
- `python main.py train --mode on|off`
- `python main.py test --mode on|off`

If you want reproducible experiments for publication or sharing, record the exact values you used in these files.

## Notes for GitHub publishing

- Keep source code and documentation in the repository.
- Ignore generated outputs and temporary files with `.gitignore`.
- Only commit small demo checkpoints under `models/examples/`; keep training outputs in `models/checkpoints/` local-only (or distribute large ones via Git LFS/GitHub Releases).
