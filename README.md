# AttenNC

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/AIWN-ICT/AttenNC/blob/main/LICENSE)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

AttenNC is an adaptive learning and simulation framework for generalizable network coding. It combines Source/Relay DQN with optional GNN-based relay-node selection, and includes training, evaluation, topology/protocol configuration, and experiment-output utilities.

---

## Table of Contents

- [Project structure](#project-structure)
- [Environment](#environment)
- [Quick start](#quick-start)
- [Mode behavior (`--mode on|off`)](#mode-behavior---mode-onoff)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Notes for GitHub publishing](#notes-for-github-publishing)
- [Simulation time model and event scheduling](#simulation-time-model-and-event-scheduling)
- [Packet loss model](#packet-loss-model)
- [FAQ](#faq)

---

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
│  └─ data_attenNC/
│
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Environment

- Python 3.10+
- PyTorch-compatible CPU or GPU environment

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependency installation notes

`torch-geometric` must match your installed `torch` build, Python version, and CPU/CUDA setup. If `pip install -r requirements.txt` fails at `torch-geometric`, install PyTorch first, verify it works, then install PyG with platform-specific instructions.

Recommended order:

```bash
pip install torch
pip install -r requirements.txt
```

References:

- PyG install guide: `https://pytorch-geometric.readthedocs.io/`
- PyTorch install guide: `https://pytorch.org/get-started/locally/`

This is especially important on:

- CUDA-enabled environments
- Windows
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
- `--best-state`: filename or path of the random-state snapshot
- `--skip-compile`: disable `torch.compile` for compatibility
- `--mode`: `on` / `off`
- `--eval-interval`: run evaluation every N episodes (`>= 1`), overriding `TRAIN_EVAL_INTERVAL` in `config.py` for that run

Example:

```bash
python main.py train --model-dir ./models/checkpoints/best_by_avg_source_send --result-dir ./results --skip-compile --mode on --eval-interval 5
```

### Test / evaluate

```bash
python main.py test
```

Example:

```bash
python main.py test --model-dir ./models/examples/best_by_avg_source_send --result-dir ./results --skip-compile --mode off
```

You can also run `test.py` directly with explicit checkpoint paths:

```bash
python test.py --model-dir models/best_by_avg_source_send --best-state models/best_by_avg_source_send/best_epoch.pkl
```

If `--model-dir` / `--best-state` are omitted, `test.py` uses built-in default paths.

`--best-state` supports:
- filename only (resolved under final `--model-dir`), e.g. `best_epoch_GNN.pkl`
- relative path (resolved from project directory)
- absolute path

## Mode behavior (`--mode on|off`)

`--mode` controls relay coding-node selection and automatically uses mode-specific subfolders (`selection_on/` or `selection_off/`) under the provided `--model-dir` and output directories.

- `--mode on`
  - Enables relay coding-node selection (with GNN gating when applicable).
  - Trains/loads Source DQN + Relay DQN + GNN.
  - Typical checkpoints: `dqn_agent_s_min.pt`, `dqn_agent_r_min.pt`, `gnn_model_dqn.pt`, `best_epoch_GNN.pkl`.

- `--mode off`
  - Disables relay coding-node selection (relay follows coding path without GNN gating decisions).
  - Trains/loads Source DQN + Relay DQN only.
  - Typical checkpoints: `dqn_agent_s_min.pt`, `dqn_agent_r_min.pt`, `best_epoch_GNN.pkl`.

Default locations used in practice:

- Training checkpoints: `models/checkpoints/best_by_avg_source_send/<selection_on|selection_off>/`
- Example test checkpoints: `models/examples/best_by_avg_source_send/<selection_on|selection_off>/`

## Outputs

By default, artifacts are written to mode-specific subfolders (`selection_on/` / `selection_off/`) under:

- `results/`: per-run summaries and logs
  - `reward_log`: episode reward metric
  - definition: `(sum of episode rewards) / (source packet transmissions at decode time)`
  - in training: single evaluated episode value
  - in testing: mean value over `Max_test` runs
- `packet_log/`: packet transmission count CSV files
- `data_attenNC/`: aggregated decode-probability and related metrics
- `models/checkpoints/`: local training checkpoints (typically untracked)
- `models/examples/`: lightweight demo checkpoints for reproducibility

See `models/README.md` for checkpoint layout details.

## Configuration

Main settings:

- `config.py`: episode count, RL hyperparameters, timing parameters, MAC/PHY packet settings
- `config_topology.py`: node layout, connectivity, and link structure

Relay coding-node selection can be set in code and overridden by CLI:

- `ENABLE_RELAY_CODING_SELECTION = True/False`
- `python main.py train --mode on|off`
- `python main.py test --mode on|off`

For reproducible experiments, record exact configuration values used in these files.

## Notes for GitHub publishing

- Keep source code and documentation in the repository.
- Ignore generated outputs and temporary files via `.gitignore`.
- Commit only small demo checkpoints under `models/examples/`.
- Keep training outputs in `models/checkpoints/` local-only, or distribute large files via Git LFS/GitHub Releases.

## Simulation time model and event scheduling

AttenNC uses a slot-driven simulation loop with delayed event commitment. In each frame, source/relay transmission decisions are made in TDMA slots, while packet arrivals, ACK/NACK feedback, and decode/reward acknowledgments are applied when modeled arrival times are reached.

Why this design is used now:

- Aligns control decisions with TDMA slots for RL state/action construction and reward accounting.
- Supports explicit propagation/transmission delay modeling on data and control paths.
- Keeps behavior stable and reproducible across topology-level experiments.

Scope note:

- This release is not a fully asynchronous event-only network simulator.
- Delay values are more suitable for relative algorithm comparison under fixed configurations than for absolute real-world latency matching.
- In high-concurrency cases, a pure event-driven engine with globally ordered same-timestamp events can provide better delay fidelity.

Future direction: evolve from hybrid slot+event scheduling toward event-level concurrent scheduling while preserving metric comparability with existing results.

## Packet loss model

AttenNC currently uses a link-quality packet-loss model: each directed link has a delivery probability, and each packet reception is sampled from that probability during simulation.

Per-link delivery probabilities are defined in `config_topology.py` and consumed at runtime.

Alternative: a SINR/interference-threshold model, where packet success depends on instantaneous signal, noise, and interference.

Why link-quality model is used in this release:

- easy to calibrate
- stable for RL training
- reproducible for topology-level evaluation

SINR/interference-threshold modeling remains a natural next step for stronger wireless-physics fidelity.

---

## FAQ

### Q1. `pip install -r requirements.txt` fails at `torch-geometric`.

**Symptoms**
- Installation stops at `torch-geometric` or related wheel/build steps.

**Checklist**
- Confirm your Python version.
- Confirm whether you use CPU or CUDA PyTorch.
- Confirm your PyTorch version is installed and importable.

**Recommended action**
1. Install PyTorch first.
2. Install remaining dependencies.
3. If needed, install `torch-geometric` using the official platform-specific command.

References:
- `https://pytorch.org/get-started/locally/`
- `https://pytorch-geometric.readthedocs.io/`

### Q2. `test` cannot find checkpoint files.

**Symptoms**
- Missing `*.pt` or `best_epoch_GNN.pkl` errors.

**Checklist**
- Run training first.
- Ensure `--model-dir` points to the correct mode subfolder (`selection_on` or `selection_off`).
- Ensure required files exist for that mode.

**Recommended action**
- Re-run with explicit `--model-dir` and `--best-state` paths.
- If model dimensions/config changed, retrain and test with newly generated checkpoints.

### Q3. Should I use `--mode on` or `--mode off`?

**Short answer**
- Use `on` when you want GNN-assisted relay coding-node selection.
- Use `off` when you want the simpler DQN-only relay path without GNN gating.

**Why**
- The two modes train/load different model sets and use different checkpoint subfolders.
- Mixing mode and checkpoint folder is a common source of loading errors.