# models/

This directory is split into two purposes:

- `models/examples/`: demo checkpoints tracked in GitHub, used for quick testing and README examples.
- `models/checkpoints/`: local training outputs (regenerable), intentionally ignored by `.gitignore`.

## Recommended layout

```text
models/
├─ examples/
│  └─ best_by_avg_source_send/
│     ├─ dqn_agent_s_min.pt
│     ├─ dqn_agent_r_min.pt
│     ├─ gnn_model_dqn.pt
│     └─ best_epoch_GNN.pkl
└─ checkpoints/
   └─ best_by_avg_source_send/
      ├─ dqn_agent_s_min.pt
      ├─ dqn_agent_r_min.pt
      ├─ gnn_model_dqn.pt
      └─ best_epoch_GNN.pkl
```

## Default evaluation path

`python main.py test` now uses the example checkpoint directory by default:

```bash
python main.py test --model-dir ./models/examples/best_by_avg_source_send
```

## Mode default and path behavior

Relay coding-node selection can be set in code and overridden by CLI:

- `ENABLE_RELAY_CODING_SELECTION = True/False` (default: `True`, equivalent to `--mode on`)
- `python main.py train --mode on|off`
- `python main.py test --mode on|off`

`--mode` controls relay coding-node selection and automatically chooses mode-specific subfolders (`selection_on/` or `selection_off/`) under the provided `--model-dir` and output directories.

## Notes

1. Keep only minimal demo artifacts in `models/examples/`.
2. Store large or frequently changing training outputs in `models/checkpoints/`.
3. If `.pt`/`.pkl` files are large, prefer Git LFS or GitHub Releases.
