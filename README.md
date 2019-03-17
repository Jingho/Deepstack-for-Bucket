# Deepstack-bucket


This repository contains the code for running the experiments for the following paper:

> Ganzfried S , Sandholm T . Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games[C]// Twenty-eighth Aaai Conference on Artificial Intelligence. AAAI Press, 2014.

This code is designed to:

- Implement an abstract method to significantly improve performance on no-limit Texas Hold'em.
- Apply and test the effect of this abstract method on the Deepstack algorithm.

## Dependencies

- Python 3
- Pyemd
- Numpy
- Scipy
- Scikit-learn
- Matplotlib

## How to Run


- generate data

  `python Generate_data.py [OPTIONS]`

  - Options

  | Name, shorthand | Default | Range                 | Description                        |
  | --------------- | ------- | --------------------- | ---------------------------------- |
  | --street        | river   | river \| turn \| flop | the round name                     |
  | --file_path     | data/   | -                     | the relative path for storing data |

- Clustering data

  ·python Cluster_data.py [OPTIONS]

  - Options

  | Name, shorthand | Default | Range                 | Description                              |
  | --------------- | ------- | --------------------- | ---------------------------------------- |
  | --street        | river   | river \| turn \| flop | the round name                           |
  | --file_path     | data/   | -                     | the relative path for storing data       |
  | --k             | 5       | Positive integer      | the number of clusters                   |
  | --initialMethod | kmean++ | kmeans++ \| random    | initialize cluster center point method   |
  | --ifsave        | True    | True \| False         | whether to save the cluster center point |

- Save the correspondence between the hand and the bucket

  `python Cluster_result.py`

- Visual clustering results

  `python Data_Visualization.py [OPTIONS]`

  - Options

  | Name, shorthand | Default | Range                   | Description                              |
  | --------------- | ------- | ----------------------- | ---------------------------------------- |
  | --street        | river   | river \| turn \| flop   | the round name                           |
  | --mode          | test    | data \| results \| test | plot data mode                           |
  | --ifsave        | True    | True \| False           | whether to save the cluster center point |

  - Details

    data mode : Plot raw data distribution

    results mode : Plot bucket data distribution

    test mode :Plot both raw and bucket data distribution

- Check the minimum cluster center point EMD distance

  `python Check_MinEMD.py`

## Algorithm


This is the first algorithm for computing potential-aware imperfect-recall abstractions, using EMD as the distance metric. Experiments on no-limit Texas Hold’em show that our algorithm leads to a statistically significant improvement in performance over the previously best abstraction algorithm.

If you want to know more details, please read this [paper](https://www.researchgate.net/publication/287088563_Potential-aware_imperfect-recall_abstraction_with_earth_mover's_distance_in_imperfect-information_games)



