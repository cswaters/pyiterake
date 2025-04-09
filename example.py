import pandas as pd
import numpy as np
from pyiterake import category, universe, compare_margins, iterake, weight_stats

# Create sample data - similar to the example in the R package
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

print("Sample data:")
print(df.head())
print("\nSample data summary:")
print(df.describe(include='all'))

# Step 1: Build the universe
uni = universe(
    df,
    category(
        name="Sex",
        buckets=["Male", "Female"],
        targets=[0.4, 0.6]
    ),
    category(
        name="Under50",
        buckets=[True, False],
        targets=[0.2, 0.8]
    )
)

# Step 2: Compare margins prior to weighting
print("\nUnweighted margins:")
unweighted_margins = compare_margins(uni)
print(unweighted_margins)

# Step 3: Weight the data
df_wgt = iterake(uni)

print("\nWeighted data:")
print(df_wgt.head())

# Step 4: Compare margins after weighting
print("\nWeighted margins:")
weighted_margins = compare_margins(
    uni,
    data=df_wgt,
    weight="weight"
)
print(weighted_margins)

# Step 5: Inspect weights
print("\nWeight statistics:")
stats = weight_stats(df_wgt["weight"])
print(pd.DataFrame([stats]))

# Optional: Create a plot of the margins
print("\nCreating a plot of the margins...")
compare_margins(
    uni,
    data=df_wgt,
    weight="weight",
    plot=True
) 