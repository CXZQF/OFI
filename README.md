# Order Flow Imbalances (OFI) Feature Construction

This repository contains Python code for constructing various Order Flow Imbalance (OFI) features from limit order book data, based on the methodology described in Cont et al. (2023).

## Overview

OFI measures the net pressure between buying and selling interest in financial markets. Unlike trade volume, OFI incorporates directionality and information from unexecuted orders, providing a richer signal for short-term price movements.

## Features Implemented

The code calculates four key OFI features:

1. **Best-Level OFI**: Captures imbalance at the best bid/ask prices only
2. **Multi-Level OFI**: Extends OFI calculation to multiple levels (up to 10) with depth normalization
3. **Integrated OFI**: Combines multi-level OFIs into a single feature using PCA
4. **Cross-Asset OFI**: Organizes integrated OFIs across multiple instruments for cross-impact analysis

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

## Usage

```python
import pandas as pd
from ofi import *

# Load data
file_path = 'first_25000_rows.csv'
df = pd.read_csv(file_path)

# Process all instruments
results = process_all_instruments(df, time_interval='1min', levels=10)

# Access results
best_level_ofi = results['best_level_ofi']
multi_level_ofi = results['multi_level_ofi']
integrated_ofi = results['integrated_ofi']
cross_asset_ofi = results['cross_asset_ofi']
pca_weights = results['pca_weights']
