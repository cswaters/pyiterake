# PyIterake

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for creating weights using iterative raking (rim weighting).

<p align="center">
  <img src="https://raw.githubusercontent.com/cswaters/pyiterake/main/docs/images/logo.png" alt="PyIterake Logo" width="200"/>
</p>

## Overview

PyIterake's main utility is creating row weights using a process called iterative raking. Iterative raking (also known as rim weighting), is one of several methods used to correct the deviation between the _marginal_ proportions in a sample and a known population for a given set of variables.

This is a Python port of the [iterake R package](https://github.com/ttrodrigz/iterake) by ttrodrigz.

PyIterake is designed with speed and simplicity in mind, using pandas for efficient data manipulation and seaborn for beautiful visualizations.

## Key Features

- Create and manage weighting categories with target proportions
- Generate weights using iterative raking algorithm
- Visualize the differences between unweighted, weighted, and target proportions
- Analyze the quality of weights with detailed statistics
- Modern and beautiful visualizations using Seaborn

## Installation

### From PyPI (Recommended)

```bash
pip install pyiterake
```

### From Source

```bash
pip install git+https://github.com/cswaters/pyiterake.git
```

Or clone the repository and install in development mode:

```bash
git clone https://github.com/cswaters/pyiterake.git
cd pyiterake
pip install -e .
```

## Dependencies

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn

## Workflow

The weighting process with PyIterake is straightforward:

1. Use the `universe()` function to build your population.  
   1. The universe is constructed with one or more categories where the marginal probabilities are known. These categories are built with the `category()` function.  
   2. If you want to use the natural marginal proportions from an existing dataset as your targets, you can use `inherit_category()`. 
2. Compare the marginal proportions in your sample with the population with `compare_margins()` function.
3. If needed, create weights for your data using `iterake()`.
4. Use `compare_margins()` again to verify that the weighted proportions in your sample now match the population.
5. Check the performance of the weighting model with `weight_stats()`.

## Basic Example

```python
import pandas as pd
import numpy as np
import pyiterake as pi

# Create sample data
N = 400
np.random.seed(101)

df = pd.DataFrame({
    'id': range(1, N+1),
    'Sex': np.random.choice(
        ['Male', 'Female'],
        size=N,
        replace=True,
        p=[0.42, 0.58]
    ),
    'Under50': np.random.choice(
        [True, False], 
        size=N,
        replace=True,
        p=[0.22, 0.78]
    )
})

# Step 1: Build the universe
uni = pi.universe(
    df,
    pi.category(
        name="Sex",
        buckets=["Male", "Female"],
        targets=[0.4, 0.6]
    ),
    pi.category(
        name="Under50",
        buckets=[True, False],
        targets=[0.2, 0.8]
    )
)

# Step 2: Compare margins prior to weighting
pi.compare_margins(uni)

# Step 3: Weight the data
df_wgt = pi.iterake(uni)

# Step 4: Compare margins after weighting
pi.compare_margins(
    uni,
    data=df_wgt,
    weight="weight",
    plot=True
)

# Step 5: Inspect weights
pi.weight_stats(df_wgt["weight"])
```

## Advanced Example

For a more complex example with visualizations, see the `visualization_example.py` file in the repository.

## API Reference

### Main Functions

- `category(name, buckets, targets)`: Create a category for iterative raking
- `inherit_category(data, name)`: Create a category using natural proportions from data
- `universe(data, *categories)`: Create a universe for iterative raking
- `compare_margins(universe, data, weight, plot)`: Compare margins between sample and target
- `iterake(universe, max_iterations, tolerance, cap_weights)`: Perform iterative raking
- `weight_stats(weights)`: Calculate statistics for weights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ttrodrigz](https://github.com/ttrodrigz) for the original R implementation of iterake
- The pandas, numpy, matplotlib, and seaborn communities for their excellent libraries
