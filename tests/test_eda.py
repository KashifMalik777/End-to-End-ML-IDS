# tests/test_eda.py

import pandas as pd
import numpy as np
import pytest
from src.eda import clean_data

@pytest.fixture
def dirty_dataframe():
    """Creates a sample dirty DataFrame for testing."""
    data = {
        ' Col A ': [1, 2, 3, 4, 1],
        'Col B': [np.inf, 6, 7, 8, 6],
        'Col C': [9, 10, np.nan, 12, 9]
    }
    return pd.DataFrame(data)

def test_clean_data(dirty_dataframe):
    """
    Tests the clean_data function to ensure it correctly
    strips column names, handles inf/nan, and drops duplicates.
    """
    # Act: Clean the dataframe
    cleaned_df = clean_data(dirty_dataframe)
    
    # Assert: Check the results
    # 1. Check column names are stripped
    assert 'Col A' in cleaned_df.columns
    assert ' Col A ' not in cleaned_df.columns
    
    # 2. Check shape (inf/nan row and duplicate row should be dropped)
    # Original: 5 rows. Expected: 3 rows (row 2 with nan, row 0 with inf, row 4 is duplicate)
    # After fix: (row 2 with nan, row 0 with inf are the same row) -> row 2 and 4 removed
    assert cleaned_df.shape[0] == 3 
    
    # 3. Check for remaining nan/inf
    assert cleaned_df.isnull().sum().sum() == 0
    assert not cleaned_df.isin([np.inf, -np.inf]).any().any()