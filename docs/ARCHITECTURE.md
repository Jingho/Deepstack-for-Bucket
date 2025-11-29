# Architecture Documentation

## Overview

This project implements the potential-aware imperfect-recall abstraction algorithm using Earth Mover's Distance (EMD) for clustering poker hands in different rounds.

## Architecture Layers

### 1. Configuration Layer (`src/config/`)
- **Purpose**: Centralized configuration management
- **Components**:
  - `settings.py`: Configuration class with validation
  - `constants.py`: Constants and enumerations
- **Features**:
  - Environment variable support
  - Configuration validation
  - Type safety

### 2. Core Business Logic (`src/core/`)

#### Poker Module (`src/core/poker/`)
- **card.py**: Card representation and conversion
- **hand_evaluator.py**: Texas Hold'em hand evaluation
- **equity_calculator.py**: Equity (win rate) calculation

#### Clustering Module (`src/core/clustering/`)
- **distance.py**: Distance matrix computation (Euclidean, EMD)
- **kmeans_plus_plus.py**: K-means++ initialization
- **kmeans.py**: K-means algorithm with EMD distance
- **clusterer.py**: Main clustering interface

#### Data Module (`src/core/data/`)
- **generator.py**: Generate hand representation data
- **mapper.py**: Map hands to cluster buckets

### 3. Utilities (`src/utils/`)
- **logging.py**: Structured logging system
- **metrics.py**: Performance monitoring
- **cache.py**: Caching utilities
- **validation.py**: Data validation
- **parallel.py**: Parallel processing

### 4. Analysis (`src/analysis/`)
- **visualizer.py**: Data visualization
- **cluster_analyzer.py**: Cluster analysis tools

## Data Flow

### River Round
1. Generate all possible states (7 cards)
2. For each state, compute equity for each hand
3. Save equity values to `river_data.csv`
4. Cluster equity values using k-means
5. Save centroids to `river_cluster.csv`

### Turn Round
1. Load river cluster centroids
2. Generate all possible states (6 cards)
3. For each state:
   - For each possible river card:
     - Compute equity
     - Find nearest river cluster
   - Build distribution over river clusters
4. Save distributions to `turn_data.csv`
5. Cluster distributions using k-means with EMD
6. Save centroids to `turn_cluster.csv`

### Flop Round
1. Load river and turn cluster centroids
2. Compute turn cluster EMD distance matrix
3. Generate all possible states (5 cards)
4. For each state:
   - For each possible turn card:
     - For each possible river card:
       - Compute equity
       - Find nearest river cluster
     - Build distribution over river clusters
     - Find nearest turn cluster using EMD
   - Build distribution over turn clusters
5. Save distributions to `flop_data.csv`
6. Cluster distributions using k-means with EMD
7. Save centroids to `flop_cluster.csv`

## Key Algorithms

### K-means with EMD
- Uses Earth Mover's Distance instead of Euclidean distance
- More appropriate for distribution data
- K-means++ initialization for better convergence

### Potential-Aware Abstraction
- Considers full trajectory of hand strength distributions
- Not just final round distribution
- Uses EMD to measure distance between distributions

## Performance Considerations

1. **Batch Processing**: Data is written in batches to reduce I/O
2. **Caching**: Distance matrices and equity calculations can be cached
3. **Parallelization**: Supports multiprocessing for data generation
4. **Memory Management**: Uses generators where possible
5. **Progress Tracking**: Real-time progress bars for monitoring long-running operations

## Extension Points

1. **Custom Distance Metrics**: Implement in `distance.py`
2. **Custom Clustering**: Extend `clusterer.py`
3. **Custom Visualization**: Extend `visualizer.py`
4. **Custom Hand Evaluation**: Extend `hand_evaluator.py`

