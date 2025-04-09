import pandas as pd
import numpy as np
import unittest
import sys
import os

# Add parent directory to path to import pyiterake
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyiterake import category, universe, compare_margins, iterake, weight_stats


class TestPyIterake(unittest.TestCase):
    def setUp(self):
        # Create sample data
        N = 400
        np.random.seed(101)
        
        self.df = pd.DataFrame({
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
        
        # Create universe
        self.uni = universe(
            self.df,
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
    
    def test_category(self):
        cat = category(
            name="Test",
            buckets=["A", "B", "C"],
            targets=[0.3, 0.3, 0.4]
        )
        self.assertEqual(cat.name, "Test")
        self.assertEqual(cat.buckets, ["A", "B", "C"])
        self.assertEqual(cat.targets, [0.3, 0.3, 0.4])
        
        # Test validation
        with self.assertRaises(ValueError):
            category(
                name="Test",
                buckets=["A", "B"],
                targets=[0.3, 0.3, 0.4]
            )
        
        with self.assertRaises(ValueError):
            category(
                name="Test",
                buckets=["A", "B", "C"],
                targets=[0.3, 0.3, 0.3]  # Sum is 0.9, not 1.0
            )
    
    def test_universe(self):
        self.assertEqual(len(self.uni.categories), 2)
        self.assertEqual(self.uni.data.shape, self.df.shape)
    
    def test_compare_margins(self):
        margins = compare_margins(self.uni)
        self.assertEqual(len(margins), 4)
        self.assertTrue('category' in margins.columns)
        self.assertTrue('bucket' in margins.columns)
        self.assertTrue('uwgt_prop' in margins.columns)
        self.assertTrue('targ_prop' in margins.columns)
    
    def test_iterake(self):
        weighted_df = iterake(self.uni)
        self.assertEqual(len(weighted_df), len(self.df))
        self.assertTrue('weight' in weighted_df.columns)
        
        # Check that weighted proportions match targets
        margins = compare_margins(self.uni, weighted_df, "weight")
        
        # Calculate maximum absolute difference between weighted proportion and target
        max_diff = 0
        for _, row in margins.iterrows():
            diff = abs(row['wgt_prop'] - row['targ_prop'])
            max_diff = max(max_diff, diff)
        
        # Ensure the difference is small (convergence)
        self.assertLess(max_diff, 0.01)
    
    def test_weight_stats(self):
        weighted_df = iterake(self.uni)
        stats = weight_stats(weighted_df['weight'])
        
        self.assertEqual(stats['uwgt_n'], len(self.df))
        self.assertEqual(stats['wgt_n'], len(self.df))  # Sum of weights should equal original N
        self.assertTrue(0 < stats['efficiency'] <= 1)  # Efficiency should be between 0 and 1


if __name__ == '__main__':
    unittest.main() 