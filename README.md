# Oracle - DOTA2 Draft Recommendation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

AlphaZero-style reinforcement learning system for DOTA2 hero draft recommendation using Monte Carlo Tree Search.

## Overview

This project applies AlphaZero techniques to DOTA2 draft strategy. The system uses:
- **Monte Carlo Tree Search** for exploring draft possibilities
- **Neural networks** for evaluating draft outcomes
- **Self-play** for generating training data

The goal is to recommend optimal hero picks during the draft phase based on current team compositions.

## Architecture

```
Data Pipeline:
  Draft History (2M matches) → Feature Engineering → Neural Network

MCTS System:
  Current Draft → Tree Search → Policy Network → Action Recommendation
```

**Core Components:**
- `oracle/drafting/mcts.py` - MCTS implementations (Classic & AlphaZero)
- `oracle/drafting/game.py` - Draft game state management
- `oracle/models.py` - Neural network architectures
- `oracle/utils/data.py` - Data loading and preprocessing

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download data (~500MB)
wget "https://www.dropbox.com/s/vy4zei33725l8a4/dota.pickle?dl=0" -O data/dota.pickle

# Train supervised win rate model (baseline)
python scripts/train_model.py --epochs 50 --batch-size 512

# Train AlphaZero via self-play (advanced)
python scripts/train_alphazero.py --iterations 100 --games-per-iter 25 --simulations 400
```

## Training Results

**Supervised Win Rate Model (DraftNet):**
- **Test Accuracy**: ~68% on unseen drafts
- **Architecture**: Hero-aware attention network with team encoders
- **Parameters**: ~480K trainable parameters
- **Training time**: ~15 min on GPU

**AlphaZero Self-Play Model:**
- **Architecture**: Dual-head network (policy + value)
- **Training**: Self-play with 400 MCTS simulations per move
- **Features**: Dirichlet noise exploration, temperature scheduling, TensorBoard logging
- **Policy head**: 112-dim hero selection probabilities
- **Value head**: Position evaluation [-1, 1]
- **Parameters**: ~490K trainable parameters

## Data

The dataset contains:
- 2M+ DOTA2 matches
- 112 heroes (hero IDs 1-114, excluding 24 and 108)
- Draft picks: 5 heroes per team (10 total)
- Hero embeddings: roles, attributes, attack types

Hero encoding:
- `+1`: picked by Radiant (team 1)
- `-1`: picked by Dire (team 2)
- `0`: unpicked

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black oracle/ scripts/
ruff oracle/ scripts/
```

## Project Structure

```
oracle/
├── drafting/
│   ├── mcts.py              # ClassicMCTS, AlphaZeroMCTS
│   ├── game.py              # DraftState, game rules
│   └── alpha_zero/
│       ├── neural_net.py    # AlphaZeroNet, NeuralNetWrapper
│       └── training.py      # Self-play coordinator
├── evaluation/              # Evaluation framework
│   ├── metrics.py           # Policy accuracy, value calibration
│   └── benchmark.py         # Performance benchmarking
├── models.py                # DraftNet (supervised)
├── config.py                # ModelConfig, TrainingConfig, AlphaZeroConfig
└── utils/
    ├── data.py              # Data loading, preprocessing
    └── logger.py            # Logging utilities

scripts/
├── train_model.py           # Supervised win rate training
├── train_alphazero.py       # AlphaZero self-play training
└── demo_alphazero.py        # Interactive recommendations
```

## Implementation Details

### AlphaZero Architecture

**Network Design:**
- **Input**: Draft state (112 heroes, values in {-1, 0, +1})
- **Hero Encoder**: Learnable embeddings → attention-based team composition
- **Shared Backbone**: Team encoders with multi-head self-attention
- **Policy Head**: Outputs action probabilities over 112 heroes
- **Value Head**: Outputs position evaluation in [-1, 1]

**Training Loop:**
1. **Self-Play**: Generate games using MCTS with current network
2. **Train**: Update network on (state, policy, outcome) triplets
3. **Checkpoint**: Save model every N iterations
4. **Iterate**: Repeat with improved network

### MCTS Algorithm

**AlphaZero MCTS:**
1. **Selection**: Use PUCT (Predictor + UCT) to choose actions
   ```
   U(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
   ```
2. **Expansion**: When reaching leaf, query neural network for (policy, value)
3. **Backpropagation**: Update Q-values up the tree from leaf value
4. **Action**: Select action based on visit counts

**Key Parameters:**
- `cpuct`: Exploration constant (default: 1.0)
- `num_simulations`: MCTS simulations per move (default: 400)
- `temperature`: Controls action selection randomness during training

## Command Reference

### Supervised Training (Baseline)
```bash
python scripts/train_model.py \
  --epochs 50 \
  --batch-size 512 \
  --lr 1e-3 \
  --device auto
```

### AlphaZero Training
```bash
python scripts/train_alphazero.py \
  --iterations 100 \
  --games-per-iter 25 \
  --simulations 400 \
  --batch-size 256 \
  --lr 1e-3 \
  --cpuct 1.0 \
  --device cuda
```

**Parameters:**
- `--iterations`: Number of self-play → train cycles
- `--games-per-iter`: Self-play games per iteration
- `--simulations`: MCTS simulations per draft move
- `--batch-size`: Training mini-batch size
- `--cpuct`: MCTS exploration constant
- `--device`: `auto` (default), `cpu`, or `cuda`
- `--resume`: Resume from checkpoint (optional)
- `--no-tensorboard`: Disable TensorBoard logging

## Future Work

- [ ] Distributed self-play across multiple workers
- [ ] Model evaluation arena (Elo rating system)
- [ ] Real-time draft recommendations API
- [ ] Web interface for draft analysis

## License

MIT License
