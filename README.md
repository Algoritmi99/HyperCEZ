# HyperCEZ

Continual Model-Based Reinforcement Learning using **Hypernetworks** applied to **EfficientZero V2**. :contentReference[oaicite:3]{index=3}

HyperCEZ extends EfficientZero-V2 with a continual learning mechanism where a hypernetwork conditions parts of the model (and/or heads) on a task embedding, aiming to reduce catastrophic forgetting while maintaining strong planning-based sample efficiency.

---

## Repository layout

Top-level files and folders: :contentReference[oaicite:4]{index=4}

- `hypercez/` Core implementation
- `main.py` Main entrypoint
- `requirements.txt` Python dependencies
- `LICENSE` MIT license

---

## Installation
Requires Python 3.12

### Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick Start
Run the default entrypoint:
```bash
python main.py
```

## What HyperCEZ does

### HyperCEZ combines:

- Planning-based model learning (EfficientZero-V2 style)
- Continual learning across tasks
- Hypernetwork conditioning with task embeddings to modulate the agent per task without full retraining

### A typical continual learning workflow:

- Train Task 0
- Move to Task 1 (create/unlock Task 1 embedding, optionally warm-start from Task 0)
- Repeat for all tasks
- Evaluate:
per-task performance,
retention (forgetting),
forward transfer (speedup on new tasks)

## License
MIT License.

## Citation
If you use this repository in academic work, cite it as:

```bibtex
@misc{hypercez,
  title        = {HyperCEZ: Continual Model-Based Reinforcement Learning using Hypernetworks applied to EfficientZero V2},
  author       = {Algoritmi99},
  year         = {2026},
  howpublished = {GitHub repository},
}

@thesis{thesis,
author = {Torabi Goodarzi, Arash and Günther, Waxenegger-Wilfing and D'Eramo, Carlo},
year = {2026},
month = {01},
pages = {},
title = {Continual Model-based Reinforcement Learning for Fault-Adaptive Control},
doi = {10.13140/RG.2.2.35948.42885}
}
```

## Acknowledgements
- EfficientZero / MuZero family of model-based RL methods
- Continual learning literature on mitigating catastrophic forgetting