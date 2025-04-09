import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class Category:
    """
    A class representing a category for iterative raking.
    
    Parameters
    ----------
    name : str
        The name of the category, must match a column in the data.
    buckets : list
        The possible values of the category.
    targets : list
        The target proportions for each bucket. Must sum to 1.
    """
    def __init__(self, name: str, buckets: List, targets: List[float]):
        if len(buckets) != len(targets):
            raise ValueError("Length of buckets must match length of targets")
        
        if abs(sum(targets) - 1.0) > 1e-10:
            raise ValueError("Targets must sum to 1")
        
        self.name = name
        self.buckets = buckets
        self.targets = targets
        
    def __repr__(self):
        return f"Category(name='{self.name}', buckets={self.buckets}, targets={self.targets})"


class Universe:
    """
    A class representing a universe for iterative raking.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to be weighted.
    categories : list of Category
        The categories to use for raking.
    """
    def __init__(self, data: pd.DataFrame, categories: List[Category]):
        self.data = data
        self.categories = categories
        
        # Validate that all category names exist in data
        for category in categories:
            if category.name not in data.columns:
                raise ValueError(f"Category {category.name} not found in data")
            
            # Validate that all buckets exist in the data
            data_values = set(data[category.name].unique())
            for bucket in category.buckets:
                if bucket not in data_values:
                    raise ValueError(f"Bucket {bucket} not found in data for category {category.name}")
        
    def __repr__(self):
        return f"Universe(data_shape={self.data.shape}, categories={len(self.categories)})"


def category(name: str, buckets: List, targets: List[float]) -> Category:
    """
    Create a category for iterative raking.
    
    Parameters
    ----------
    name : str
        The name of the category, must match a column in the data.
    buckets : list
        The possible values of the category.
    targets : list
        The target proportions for each bucket. Must sum to 1.
        
    Returns
    -------
    Category
        A Category object.
    """
    return Category(name=name, buckets=buckets, targets=targets)


def inherit_category(data: pd.DataFrame, name: str) -> Category:
    """
    Create a category from existing data, using its natural proportions as targets.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to inherit from.
    name : str
        The name of the category, must match a column in the data.
        
    Returns
    -------
    Category
        A Category object.
    """
    if name not in data.columns:
        raise ValueError(f"Column {name} not found in data")
    
    value_counts = data[name].value_counts(normalize=True)
    buckets = value_counts.index.tolist()
    targets = value_counts.values.tolist()
    
    return Category(name=name, buckets=buckets, targets=targets)


def universe(data: pd.DataFrame, *categories: Category) -> Universe:
    """
    Create a universe for iterative raking.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to be weighted.
    *categories : Category
        The categories to use for raking.
        
    Returns
    -------
    Universe
        A Universe object.
    """
    return Universe(data=data, categories=list(categories))


def compare_margins(
    universe: Universe, 
    data: Optional[pd.DataFrame] = None, 
    weight: Optional[str] = None,
    plot: bool = False
) -> pd.DataFrame:
    """
    Compare margins between the data and the target universe.
    
    Parameters
    ----------
    universe : Universe
        The target universe.
    data : pd.DataFrame, optional
        The data to compare. If None, uses the data in the universe.
    weight : str, optional
        The name of the weight column. If None, assumes unweighted data.
    plot : bool, default=False
        Whether to plot the comparison.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with the comparison.
    """
    if data is None:
        data = universe.data
        
    results = []
    
    for cat in universe.categories:
        # Calculate unweighted proportions
        uwgt_counts = data[cat.name].value_counts()
        uwgt_props = data[cat.name].value_counts(normalize=True)
        
        # Calculate weighted proportions if weights are provided
        if weight is not None:
            wgt_counts = data.groupby(cat.name)[weight].sum()
            wgt_props = wgt_counts / wgt_counts.sum()
        
        # Create a DataFrame for this category
        for i, bucket in enumerate(cat.buckets):
            row = {
                'category': cat.name,
                'bucket': bucket,
                'uwgt_n': uwgt_counts.get(bucket, 0),
                'uwgt_prop': uwgt_props.get(bucket, 0),
                'targ_prop': cat.targets[i]
            }
            
            row['uwgt_diff'] = row['uwgt_prop'] - row['targ_prop']
            
            if weight is not None:
                row['wgt_n'] = wgt_counts.get(bucket, 0)
                row['wgt_prop'] = wgt_props.get(bucket, 0)
                row['wgt_diff'] = row['wgt_prop'] - row['targ_prop']
                
            results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Plot if requested
    if plot and weight is not None:
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        
        for cat_name in result_df['category'].unique():
            cat_data = result_df[result_df['category'] == cat_name]
            
            # Reshape data for better visualization with seaborn
            plot_data = pd.DataFrame({
                'Bucket': cat_data['bucket'].tolist() * 3,
                'Proportion': (
                    cat_data['uwgt_prop'].tolist() + 
                    cat_data['wgt_prop'].tolist() + 
                    cat_data['targ_prop'].tolist()
                ),
                'Type': (
                    ['Unweighted'] * len(cat_data) +
                    ['Weighted'] * len(cat_data) +
                    ['Target'] * len(cat_data)
                )
            })
            
            # Create a more attractive plot with seaborn
            plt.figure(figsize=(10, 6))
            bar_plot = sns.barplot(
                x='Bucket', 
                y='Proportion', 
                hue='Type', 
                data=plot_data,
                palette='viridis'
            )
            
            # Customize the plot
            plt.title(f'Margin Comparison - {cat_name}', fontsize=16)
            plt.ylabel('Proportion', fontsize=14)
            plt.xlabel(cat_name, fontsize=14)
            plt.legend(title='')
            plt.tight_layout()
            
            # Add value labels on top of bars
            for p in bar_plot.patches:
                bar_plot.annotate(
                    f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, rotation=0
                )
            
            plt.show()
    
    return result_df


def iterake(
    universe: Universe, 
    max_iterations: int = 100, 
    tolerance: float = 1e-6,
    cap_weights: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Perform iterative raking to create weights.
    
    Parameters
    ----------
    universe : Universe
        The target universe.
    max_iterations : int, default=100
        The maximum number of iterations to perform.
    tolerance : float, default=1e-6
        The tolerance for convergence.
    cap_weights : tuple of (min, max), optional
        If provided, weights will be capped to these values.
        
    Returns
    -------
    pd.DataFrame
        The weighted data.
    """
    data = universe.data.copy()
    
    # Initialize weights to 1
    data['weight'] = 1.0
    
    # Track loss
    prev_loss = float('inf')
    curr_loss = float('inf')
    
    converged = False
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        for cat in universe.categories:
            # Calculate current weighted proportions
            curr_props = {}
            for bucket in cat.buckets:
                mask = data[cat.name] == bucket
                curr_props[bucket] = data.loc[mask, 'weight'].sum() / data['weight'].sum()
            
            # Adjust weights
            for i, bucket in enumerate(cat.buckets):
                target_prop = cat.targets[i]
                if curr_props[bucket] > 0:
                    adjustment = target_prop / curr_props[bucket]
                    mask = data[cat.name] == bucket
                    data.loc[mask, 'weight'] *= adjustment
            
            # If we're capping weights, do so
            if cap_weights is not None:
                min_weight, max_weight = cap_weights
                data['weight'] = data['weight'].clip(min_weight, max_weight)
                
                # Renormalize to maintain sum of weights
                data['weight'] = data['weight'] * (len(data) / data['weight'].sum())
        
        # Calculate current loss
        curr_loss = 0
        for cat in universe.categories:
            for i, bucket in enumerate(cat.buckets):
                mask = data[cat.name] == bucket
                curr_prop = data.loc[mask, 'weight'].sum() / data['weight'].sum()
                target_prop = cat.targets[i]
                curr_loss += (curr_prop - target_prop) ** 2
        
        # Check for convergence
        if abs(curr_loss - prev_loss) < tolerance:
            converged = True
            break
        
        prev_loss = curr_loss
    
    # Normalize weights to maintain sum
    data['weight'] = data['weight'] * (len(data) / data['weight'].sum())
    
    # Print summary
    print("\n-- iterake summary -------------------------------------------------------------")
    print(f" Convergence: {'Success' if converged else 'Failed'}")
    print(f"  Iterations: {iteration}")
    print()
    
    # Calculate weight statistics
    stats = weight_stats(data['weight'])
    print(f"Unweighted N: {stats['uwgt_n']:.2f}")
    print(f" Effective N: {stats['eff_n']:.2f}")
    print(f"  Weighted N: {stats['wgt_n']:.2f}")
    print(f"  Efficiency: {stats['efficiency']*100:.1f}%")
    print(f"        Loss: {curr_loss:.3f}")
    
    return data


def weight_stats(weights: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics for weights.
    
    Parameters
    ----------
    weights : pd.Series
        The weights.
        
    Returns
    -------
    dict
        A dictionary with weight statistics.
    """
    uwgt_n = len(weights)
    wgt_n = weights.sum()
    
    # Effective N calculation
    eff_n = wgt_n**2 / (weights**2).sum()
    
    # Efficiency calculation (effective N / unweighted N)
    efficiency = eff_n / uwgt_n
    
    # Loss calculation
    loss = 1 - efficiency
    
    # Min and max weights
    min_wgt = weights.min()
    max_wgt = weights.max()
    
    stats = {
        'uwgt_n': uwgt_n,
        'wgt_n': wgt_n,
        'eff_n': eff_n,
        'loss': loss,
        'efficiency': efficiency,
        'min_wgt': min_wgt,
        'max_wgt': max_wgt
    }
    
    return stats
