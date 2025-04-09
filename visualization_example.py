import pandas as pd
import numpy as np
import seaborn as sns
from pyiterake import category, universe, compare_margins, iterake, weight_stats

# Set seed for reproducibility
np.random.seed(42)

# Create a more complex sample dataset
N = 1000

# Generate age groups with specific distribution
age_groups = np.random.choice(
    ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    size=N,
    replace=True,
    p=[0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
)

# Generate region with specific distribution
region = np.random.choice(
    ['North', 'South', 'East', 'West', 'Central'],
    size=N,
    replace=True,
    p=[0.25, 0.30, 0.15, 0.20, 0.10]
)

# Generate income levels with specific distribution
income = np.random.choice(
    ['Low', 'Medium', 'High'],
    size=N,
    replace=True,
    p=[0.30, 0.50, 0.20]
)

# Create DataFrame
df = pd.DataFrame({
    'id': range(1, N+1),
    'Age': age_groups,
    'Region': region,
    'Income': income
})

print("Sample data:")
print(df.head())

# Define target proportions (different from the sample)
age_target = {
    '18-24': 0.10,
    '25-34': 0.20,
    '35-44': 0.25,
    '45-54': 0.20,
    '55-64': 0.15,
    '65+': 0.10
}

region_target = {
    'North': 0.20,
    'South': 0.25,
    'East': 0.20,
    'West': 0.25,
    'Central': 0.10
}

income_target = {
    'Low': 0.25,
    'Medium': 0.55,
    'High': 0.20
}

# Build universe
uni = universe(
    df,
    category(
        name="Age",
        buckets=list(age_target.keys()),
        targets=list(age_target.values())
    ),
    category(
        name="Region",
        buckets=list(region_target.keys()),
        targets=list(region_target.values())
    ),
    category(
        name="Income",
        buckets=list(income_target.keys()),
        targets=list(income_target.values())
    )
)

# Compare unweighted margins
print("\nUnweighted margins:")
unweighted_margins = compare_margins(uni)
print(unweighted_margins)

# Calculate weights
df_wgt = iterake(uni)

# Compare weighted margins
print("\nWeighted margins:")
weighted_margins = compare_margins(
    uni,
    data=df_wgt,
    weight="weight"
)
print(weighted_margins)

# Plot the results with our enhanced Seaborn visualizations
print("\nCreating visualizations...")

# Set figure size and style globally
sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_palette("viridis")

# Plot all categories with seaborn styling
compare_margins(
    uni,
    data=df_wgt,
    weight="weight",
    plot=True
)

# Show weight distribution
print("\nWeight distribution:")
plt_weight = sns.displot(df_wgt["weight"], kde=True, height=6, aspect=1.5)
plt_weight.fig.suptitle("Distribution of Weights", fontsize=16)
plt_weight.set_xlabels("Weight Value")
plt_weight.set_ylabels("Count")
plt_weight.fig.tight_layout()
plt_weight.fig.subplots_adjust(top=0.95)
plt_weight.ax.grid(True)

import matplotlib.pyplot as plt
plt.show() 