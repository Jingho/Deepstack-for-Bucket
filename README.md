# Deepstack Bucket - Industrial-Grade Refactoring

> Unofficial implementation - A community-maintained Python implementation of the algorithm

A Python implementation of the algorithm described in:

> Ganzfried S, Sandholm T. Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games. AAAI 2014.

## 🏗️ Project Structure

```
deepstack_bucket/
├── src/                      # Source code
│   ├── core/                # Core business logic
│   │   ├── poker/          # Poker-related functionality
│   │   │   ├── card.py     # Card representation and conversion
│   │   │   ├── hand_evaluator.py  # Hand evaluation and comparison
│   │   │   └── equity_calculator.py  # Equity calculation
│   │   ├── clustering/     # Clustering algorithms
│   │   │   ├── distance.py # Distance matrix computation
│   │   │   ├── kmeans.py   # K-means implementation
│   │   │   ├── kmeans_plus_plus.py  # K-means++ initialization
│   │   │   └── clusterer.py  # Main clustering class
│   │   └── data/           # Data generation and mapping
│   │       ├── generator.py  # Data generator
│   │       └── mapper.py     # Cluster mapping
│   ├── utils/              # Utility modules
│   │   ├── logging.py      # Logging configuration
│   │   ├── metrics.py      # Performance monitoring
│   │   ├── cache.py         # Caching utilities
│   │   ├── validation.py   # Data validation
│   │   └── parallel.py      # Parallel processing
│   ├── config/             # Configuration management
│   │   ├── settings.py     # Configuration class
│   │   └── constants.py     # Constants and enums
│   └── analysis/           # Analysis tools
│       ├── visualizer.py   # Data visualization
│       └── cluster_analyzer.py  # Cluster analysis
├── tests/                  # Test suite
├── scripts/                # Executable scripts
│   ├── generate_data.py    # Generate hand representation data
│   ├── cluster_data.py     # Cluster data
│   ├── map_clusters.py     # Map hands to clusters
│   └── analyze_clusters.py # Analyze cluster results
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # Architecture documentation
│   └── QUICKSTART.md       # Quick start guide
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── pyproject.toml          # Modern Python project config (PEP 621)
```

## ✨ Key Improvements

### 1. **Industrial-Grade Architecture**
- Modular design with clear separation of concerns
- Type hints throughout
- Comprehensive error handling
- Configuration management with environment variable support

### 2. **Logging System**
- Structured logging with rotation
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- File and console handlers

### 3. **Performance Monitoring**
- Performance metrics collection
- Function timing decorators
- Memory usage tracking (ready for implementation)

### 4. **Caching**
- Disk-based caching for expensive computations
- LRU memory cache
- Automatic cache key generation

### 5. **Parallel Processing**
- Support for multiprocessing and multithreading
- Batch processing utilities
- Configurable worker counts

### 6. **Configuration Management**
- Environment variable support
- Configuration validation
- Type-safe configuration classes

### 7. **Progress Tracking**
- Real-time progress bars for long-running operations
- Configurable progress display (can be disabled for non-interactive environments)
- Detailed progress information (percentage, speed, ETA)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
pip install -e .
```

### Usage

```python
from src.config.settings import Config
from src.core.poker.card import CardConverter
from src.core.poker.hand_evaluator import HandEvaluator
from src.core.poker.equity_calculator import EquityCalculator

# Load configuration
config = Config.from_env()

# Use poker modules
card_converter = CardConverter(config)
hand_evaluator = HandEvaluator()
equity_calc = EquityCalculator(hand_evaluator, card_converter, config)

# Calculate equity
equity = equity_calc.compute_equity(
    hand_string="AcKd",
    board_string="AhKhQh",
    available_cards=["2c", "2d", "3c", "3d"]
)
```

### Command Line

```bash
# Generate data
python scripts/generate_data.py --street river --data-dir data/

# Cluster data
python scripts/cluster_data.py --street river --k 5 --data-dir data/
```

## 📋 Implementation Status

### ✅ Completed
- [x] Directory structure
- [x] Configuration management (`src/config/`)
- [x] Logging system (`src/utils/logging.py`)
- [x] Poker module (`src/core/poker/`)
  - [x] Card conversion (`card.py`)
  - [x] Hand evaluation (`hand_evaluator.py`)
  - [x] Equity calculation (`equity_calculator.py`)
- [x] Clustering module (`src/core/clustering/`)
  - [x] Distance matrix calculator (`distance.py`)
  - [x] K-means implementation (`kmeans.py`)
  - [x] K-means++ initialization (`kmeans_plus_plus.py`)
  - [x] Main clusterer (`clusterer.py`)
- [x] Data generation module (`src/core/data/`)
  - [x] Data generator (`generator.py`)
  - [x] Cluster mapper (`mapper.py`)
- [x] Utility modules (`src/utils/`)
  - [x] Performance monitoring (`metrics.py`)
  - [x] Caching (`cache.py`)
  - [x] Validation (`validation.py`)
  - [x] Parallel processing (`parallel.py`)
- [x] Analysis module (`src/analysis/`)
  - [x] Visualizer (`visualizer.py`)
  - [x] Cluster analyzer (`cluster_analyzer.py`)
- [x] Scripts (`scripts/`)
  - [x] `generate_data.py`
  - [x] `cluster_data.py`
  - [x] `map_clusters.py`
  - [x] `analyze_clusters.py`
- [x] Project configuration files
  - [x] `requirements.txt`
  - [x] `requirements-dev.txt`
  - [x] `pyproject.toml` (PEP 621)
  - [x] `pytest.ini`
- [x] Basic test framework (`tests/`)
  - [x] Unit test structure
  - [x] Test fixtures
- [x] Documentation (`docs/`)
  - [x] Architecture documentation (`ARCHITECTURE.md`)
  - [x] Quick start guide (`QUICKSTART.md`)
- [x] Progress tracking
  - [x] Progress bars for data generation
  - [x] Progress bars for clustering
  - [x] Progress bars for mapping

### 🚧 Remaining Tasks
- [ ] Complete test coverage
  - [ ] More unit tests
  - [ ] Integration tests
- [ ] Additional documentation
  - [ ] API documentation (auto-generated from docstrings)
  - [ ] Advanced tutorials

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
```

### Environment Variables

Key environment variables (see `src/config/settings.py` for full list):

- `SUIT_COUNT`: Number of suits (default: 2)
- `RANK_COUNT`: Number of ranks (default: 6)
- `RIVER_CLUSTER_COUNT`: Number of river clusters (default: 5)
- `TURN_CLUSTER_COUNT`: Number of turn clusters (default: 5)
- `FLOP_CLUSTER_COUNT`: Number of flop clusters (default: 80)
- `DATA_DIR`: Data directory (default: "data")
- `LOG_DIR`: Log directory (default: "logs")
- `NUM_WORKERS`: Number of parallel workers (default: 4)
- `LOG_LEVEL`: Logging level (default: "INFO")
- `SHOW_PROGRESS`: Show progress bars (default: "true")

## 📝 Migration from Old Code

The old code structure has been refactored into the new structure:

- `Generate_data.py` → `src/core/data/generator.py`
- `Cluster_data.py` → `src/core/clustering/clusterer.py`
- `calc_matrix.py` → `src/core/clustering/distance.py`
- `judging.py` → `src/core/poker/hand_evaluator.py`
- `card_to_string_conversion.py` → `src/core/poker/card.py`
- `Cluster_result.py` → `src/core/data/mapper.py`
- `settings.py` → `src/config/settings.py` + `src/config/constants.py`

## 🤝 Contributing

1. Follow the code style (black, pylint, mypy)
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## 🙏 Acknowledgments

Based on the paper:
- Ganzfried S, Sandholm T. Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games. AAAI 2014.
