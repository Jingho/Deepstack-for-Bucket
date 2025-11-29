# Quick Start Guide

This guide will help you get started with the Deepstack Bucket project.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
pip install -e .
```

## Basic Usage

### 1. Generate Data

Generate hand representation data for river round:

```bash
python scripts/generate_data.py --street river --data-dir data/
```

This creates `data/river_data.csv` with equity values for all possible river hands.

### 2. Cluster Data

Cluster the river data:

```bash
python scripts/cluster_data.py --street river --k 5 --data-dir data/
```

This creates `data/river_cluster_k5.csv` with cluster centroids.

### 3. Generate Turn Data

After clustering river, generate turn data:

```bash
python scripts/generate_data.py --street turn --data-dir data/
```

This creates `data/turn_data.csv` with distributions over river clusters.

### 4. Cluster Turn Data

```bash
python scripts/cluster_data.py --street turn --k 5 --data-dir data/
```

### 5. Generate Flop Data

After clustering turn, generate flop data:

```bash
python scripts/generate_data.py --street flop --data-dir data/
```

### 6. Cluster Flop Data

```bash
python scripts/cluster_data.py --street flop --k 80 --data-dir data/
```

## Programmatic Usage

### Using Configuration

```python
from src.config import Config, PokerRound
from src.core.data import DataGenerator
from src.core.clustering import Clusterer

# Load configuration
config = Config.from_env()

# Generate data
generator = DataGenerator(street=PokerRound.RIVER, config=config)
generator.generate()

# Cluster data
clusterer = Clusterer(
    street=PokerRound.RIVER,
    k=5,
    config=config
)
centroids = clusterer.cluster()
```

### Using Poker Modules

```python
from src.core.poker import CardConverter, HandEvaluator, EquityCalculator
from src.config import Config

config = Config.default()

# Card conversion
converter = CardConverter(config)
cards = converter.generate_all_cards()

# Hand evaluation
evaluator = HandEvaluator(converter)
result = evaluator.compare_hands("AcKd", "QhJs", "AhKhQh")

# Equity calculation
equity_calc = EquityCalculator(evaluator, converter, config)
equity = equity_calc.compute_equity(
    "AcKd",
    "AhKhQh",
    ["2c", "2d", "3c", "3d"],
    use_exact=True
)
```

## Environment Variables

Set these before running:

```bash
export RIVER_CLUSTER_COUNT=5
export TURN_CLUSTER_COUNT=5
export FLOP_CLUSTER_COUNT=80
export DATA_DIR=data/
export LOG_LEVEL=INFO
export NUM_WORKERS=4
export SHOW_PROGRESS=true
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_poker.py
```

## Logging

Logs are written to:
- Console: Standard output
- File: `logs/deepstack_bucket.log` (with rotation)

Log levels: DEBUG, INFO, WARNING, ERROR

## Progress Bars

The project includes progress bars for long-running operations. Progress bars show:
- Current progress percentage
- Processing speed (items/second)
- Estimated time remaining (ETA)

To disable progress bars (useful for logging or CI environments):

```bash
export SHOW_PROGRESS=false
```

Or in code:

```python
config = Config(show_progress=False)
```

## Troubleshooting

### File Not Found Errors
- Ensure you run clustering in order: river → turn → flop
- Check that data files exist in the specified data directory

### Import Errors
- Make sure you've installed the package: `pip install -e .`
- Or add `src/` to your Python path

### Configuration Errors
- Check environment variables are set correctly
- Verify configuration values are valid (positive integers, etc.)

### Progress Bar Issues
- If progress bars don't display correctly, try setting `SHOW_PROGRESS=false`
- In non-interactive environments (CI, scripts), disable progress bars to avoid formatting issues

