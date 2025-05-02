import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the dataset and perform initial preprocessing
    """
    df = pd.read_csv(file_path)

    # Convert timestamp columns to datetime
    if 'ts_recv' in df.columns:
        df['ts_recv'] = pd.to_datetime(df['ts_recv'])
    if 'ts_event' in df.columns:
        df['ts_event'] = pd.to_datetime(df['ts_event'])

    # Sort by event timestamp
    df = df.sort_values(by='ts_event')

    return df


def calculate_best_level_ofi(df, time_interval='1min'):
    """
    Calculate Best-Level OFI.

    Args:
        df: DataFrame containing order book data with price and size columns
        time_interval: Sampling interval for aggregation

    Returns:
        DataFrame with timestamp and best level OFI
    """
    data = df.copy()

    # Calculate price changes
    data['bid_px_change'] = data['bid_px_00'].diff()
    data['ask_px_change'] = data['ask_px_00'].diff()

    # Calculate bid order flow
    conditions = [
        data['bid_px_change'] > 0,  # Price increased - new orders
        data['bid_px_change'] == 0,  # Price unchanged - size change
        data['bid_px_change'] < 0  # Price decreased - orders removed
    ]

    choices = [
        data['bid_sz_00'],  # New orders with price increase
        data['bid_sz_00'].diff(),  # Size change with same price
        -data['bid_sz_00'].shift(1)  # Orders removed with price decrease
    ]
    data['bid_flow'] = np.select(conditions, choices, default=0)

    # Calculate ask order flow
    conditions = [
        data['ask_px_change'] > 0,  # Price increased - orders removed
        data['ask_px_change'] == 0,  # Price unchanged - size change
        data['ask_px_change'] < 0  # Price decreased - new orders
    ]

    choices = [
        -data['ask_sz_00'].shift(1),  # Orders removed with price increase
        data['ask_sz_00'].diff(),  # Size change with same price
        data['ask_sz_00']  # New orders with price decrease
    ]
    data['ask_flow'] = np.select(conditions, choices, default=0)

    # Calculate OFI as bid_flow - ask_flow
    data['best_level_ofi'] = data['bid_flow'] - data['ask_flow']

    # Aggregate by time interval
    data.set_index('ts_event', inplace=True)
    ofi_resampled = data.resample(time_interval)['best_level_ofi'].sum().reset_index()

    return ofi_resampled


def calculate_multi_level_ofi(df, levels=10, time_interval='1min'):
    """
    Calculate Multi-Level OFI for each price level and then scale them
    according to the average depth across all levels.

    Args:
        df: DataFrame containing order book data with multiple levels
        levels: Number of order book levels to consider
        time_interval: Sampling interval for aggregation

    Returns:
        DataFrame with timestamp and scaled OFI for each level
    """
    data = df.copy()
    data.set_index('ts_event', inplace=True)
    level_ofis = {}

    # Calculate OFI for each level
    for level in range(levels):
        bid_px_col = f'bid_px_{level:02d}'
        ask_px_col = f'ask_px_{level:02d}'
        bid_sz_col = f'bid_sz_{level:02d}'
        ask_sz_col = f'ask_sz_{level:02d}'

        # Skip if required columns don't exist
        if bid_px_col not in data.columns or ask_px_col not in data.columns:
            continue

        # Create a copy for this level
        level_data = data.copy()

        # Calculate price changes
        level_data[f'bid_px_change_{level}'] = level_data[bid_px_col].diff()
        level_data[f'ask_px_change_{level}'] = level_data[ask_px_col].diff()

        # Calculate bid flow
        conditions = [
            level_data[f'bid_px_change_{level}'] > 0,
            level_data[f'bid_px_change_{level}'] == 0,
            level_data[f'bid_px_change_{level}'] < 0
        ]

        choices = [
            level_data[bid_sz_col],
            level_data[bid_sz_col].diff(),
            -level_data[bid_sz_col].shift(1)
        ]

        level_data[f'bid_flow_{level}'] = np.select(conditions, choices, default=0)

        # Calculate ask flow
        conditions = [
            level_data[f'ask_px_change_{level}'] > 0,
            level_data[f'ask_px_change_{level}'] == 0,
            level_data[f'ask_px_change_{level}'] < 0
        ]

        choices = [
            -level_data[ask_sz_col].shift(1),
            level_data[ask_sz_col].diff(),
            level_data[ask_sz_col]
        ]

        level_data[f'ask_flow_{level}'] = np.select(conditions, choices, default=0)

        # Calculate OFI for this level
        level_data[f'ofi_level_{level}'] = level_data[f'bid_flow_{level}'] - level_data[f'ask_flow_{level}']

        # Aggregate by time interval
        resampled = level_data.resample(time_interval)[f'ofi_level_{level}'].sum()
        level_ofis[level] = resampled

    # Create DataFrame with all level OFIs
    ofi_df = pd.DataFrame(level_ofis)

    # Calculate average depth across levels for normalization
    avg_depths = {}
    for level in range(levels):
        bid_sz_col = f'bid_sz_{level:02d}'
        ask_sz_col = f'ask_sz_{level:02d}'

        if bid_sz_col in data.columns and ask_sz_col in data.columns:
            # Average depth for this level in each time interval
            level_depth = data[[bid_sz_col, ask_sz_col]].mean(axis=1).resample(time_interval).mean()
            avg_depths[level] = level_depth

    # Average depth across all levels
    avg_depth_df = pd.DataFrame(avg_depths)
    avg_all_levels = avg_depth_df.mean(axis=1)

    # Scale OFIs by average depth
    scaled_ofi_df = pd.DataFrame(index=ofi_df.index)
    for level in level_ofis.keys():
        scaled_ofi_df[f'scaled_ofi_level_{level}'] = ofi_df[level] / avg_all_levels

    # Reset index to get timestamp as a column
    scaled_ofi_df = scaled_ofi_df.reset_index()

    return scaled_ofi_df


def calculate_integrated_ofi(df, levels=10, time_interval='1min'):
    """
    Calculate Integrated OFI by combining multi-level OFIs using PCA.

    Args:
        df: DataFrame containing order book data
        levels: Number of order book levels to consider
        time_interval: Sampling interval for aggregation

    Returns:
        Tuple of (DataFrame with integrated OFI, PCA weights)
    """
    # Calculate multi-level OFI
    scaled_ofi_df = calculate_multi_level_ofi(df, levels, time_interval)

    # Extract columns with scaled OFI values
    ofi_columns = [col for col in scaled_ofi_df.columns if 'scaled_ofi_level_' in col]

    if not ofi_columns:
        print("Error: No scaled OFI columns found")
        return None, None

    # Extract OFI data for PCA
    X = scaled_ofi_df[ofi_columns].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle case with only one level
    if len(ofi_columns) == 1:
        print("Only one OFI level found, no PCA needed")
        integrated_ofi = scaled_ofi_df[ofi_columns[0]].copy()
        weights = np.array([1.0])
    else:
        # Apply PCA to find first principal component
        pca = PCA(n_components=1)
        pca.fit(X_scaled)

        # Extract weights and normalize
        weights = pca.components_[0]
        normalized_weights = weights / np.abs(weights).sum()

        # Calculate integrated OFI using normalized weights
        integrated_ofi = np.zeros(len(X_scaled))
        for i, col in enumerate(ofi_columns):
            integrated_ofi += normalized_weights[i] * scaled_ofi_df[col]

    # Create result DataFrame
    result_df = pd.DataFrame({
        'ts_event': scaled_ofi_df['ts_event'],
        'integrated_ofi': integrated_ofi
    })

    return result_df, normalized_weights if len(ofi_columns) > 1 else weights


def calculate_cross_asset_ofi(integrated_ofi_dfs, time_interval='1min'):
    """
    Create a cross-asset OFI representation in a wide-format DataFrame.
    This function merges the integrated OFIs from multiple instruments
    into a single DataFrame with timestamps as rows and instruments as columns.

    Args:
        integrated_ofi_dfs: List of DataFrames containing integrated OFIs for different instruments
        time_interval: Time interval for resampling (if needed)

    Returns:
        DataFrame with timestamps as index and instruments as columns
    """
    if not integrated_ofi_dfs:
        print("No integrated OFI data provided")
        return None

    # Check that integrated_ofi_dfs is a list of DataFrames
    if not isinstance(integrated_ofi_dfs, list):
        print("Expected a list of DataFrames")
        return None

    # Create a list to hold the properly formatted DataFrames for each instrument
    formatted_dfs = []

    for df in integrated_ofi_dfs:
        if df is None or df.empty:
            continue

        # Ensure df has required columns
        required_cols = ['ts_event', 'instrument_id', 'integrated_ofi']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in DataFrame: {df.columns}")
            continue

        # Format DataFrame for this instrument
        instrument_df = df[required_cols].copy()

        # Convert ts_event to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(instrument_df['ts_event']):
            instrument_df['ts_event'] = pd.to_datetime(instrument_df['ts_event'])

        formatted_dfs.append(instrument_df)

    if not formatted_dfs:
        print("No valid DataFrames after processing")
        return None

    # Combine all DataFrames
    combined_df = pd.concat(formatted_dfs, ignore_index=True)

    # Ensure data is properly sorted
    combined_df = combined_df.sort_values('ts_event')

    # Create the wide-format DataFrame (pivot table)
    cross_asset_df = combined_df.pivot_table(
        index='ts_event',
        columns='instrument_id',
        values='integrated_ofi'
    )

    # Rename columns for clarity
    cross_asset_df.columns = [f'integrated_ofi_{col}' for col in cross_asset_df.columns]

    # Fill NaN values with 0 or forward fill
    cross_asset_df = cross_asset_df.fillna(0)

    return cross_asset_df


def process_all_instruments(df, time_interval='1min', levels=10):
    """
    Process all instruments in the dataset to calculate various OFI features.

    Args:
        df: DataFrame containing order book data for multiple instruments
        time_interval: Sampling interval for aggregation
        levels: Number of order book levels to consider

    Returns:
        Dictionary containing all calculated OFI features and PCA weights
    """
    if 'instrument_id' not in df.columns:
        print("Error: 'instrument_id' column not found in dataset")
        return {}

    # Get unique instruments
    instruments = df['instrument_id'].unique()
    print(f"Found {len(instruments)} instruments: {instruments}")

    # Initialize collections
    best_level_ofi_dfs = []
    multi_level_ofi_dfs = []  # Added this for multi-level OFI
    integrated_ofi_dfs = []
    pca_weights_dict = {}

    # Process each instrument
    for instrument in instruments:
        print(f"Processing instrument: {instrument}")

        # Filter data for this instrument
        instrument_data = df[df['instrument_id'] == instrument].copy()

        if len(instrument_data) < 10:
            print(f"  Skipping {instrument}: insufficient data")
            continue

        try:
            # Calculate best-level OFI
            best_level_ofi_df = calculate_best_level_ofi(instrument_data, time_interval)
            if best_level_ofi_df is not None and not best_level_ofi_df.empty:
                best_level_ofi_df['instrument_id'] = instrument
                best_level_ofi_dfs.append(best_level_ofi_df)
                print(f"  Successfully calculated best-level OFI for {instrument}")

            # Calculate multi-level OFI
            multi_level_ofi_df = calculate_multi_level_ofi(instrument_data, levels, time_interval)
            if multi_level_ofi_df is not None and not multi_level_ofi_df.empty:
                multi_level_ofi_df['instrument_id'] = instrument
                multi_level_ofi_dfs.append(multi_level_ofi_df)
                print(f"  Successfully calculated multi-level OFI for {instrument}")

            # Calculate integrated OFI
            integrated_ofi_df, weights = calculate_integrated_ofi(
                instrument_data, levels, time_interval)

            if integrated_ofi_df is not None and not integrated_ofi_df.empty:
                integrated_ofi_df['instrument_id'] = instrument
                integrated_ofi_dfs.append(integrated_ofi_df)
                pca_weights_dict[instrument] = weights
                print(f"  Successfully calculated integrated OFI for {instrument}")

        except Exception as e:
            print(f"  Error processing {instrument}: {e}")

    # Combine best-level OFI results
    combined_best_level_ofi = None
    if best_level_ofi_dfs:
        combined_best_level_ofi = pd.concat(best_level_ofi_dfs, ignore_index=True)
        print(f"Combined best-level OFI data: {len(combined_best_level_ofi)} rows")

    # Combine multi-level OFI results
    combined_multi_level_ofi = None
    if multi_level_ofi_dfs:
        combined_multi_level_ofi = pd.concat(multi_level_ofi_dfs, ignore_index=True)
        print(f"Combined multi-level OFI data: {len(combined_multi_level_ofi)} rows")

    # Combine integrated OFI results
    combined_integrated_ofi = None
    if integrated_ofi_dfs:
        combined_integrated_ofi = pd.concat(integrated_ofi_dfs, ignore_index=True)
        print(f"Combined integrated OFI data: {len(combined_integrated_ofi)} rows")

    # Calculate cross-asset OFI
    cross_asset_ofi_df = None
    if len(integrated_ofi_dfs) > 1:
        cross_asset_ofi_df = calculate_cross_asset_ofi(integrated_ofi_dfs, time_interval)
        print(f"Cross-asset OFI data: {cross_asset_ofi_df.shape if cross_asset_ofi_df is not None else 'None'}")

    # Return all results
    return {
        'best_level_ofi': combined_best_level_ofi,
        'multi_level_ofi': combined_multi_level_ofi,  # Added this to the results
        'integrated_ofi': combined_integrated_ofi,
        'cross_asset_ofi': cross_asset_ofi_df,
        'pca_weights': pca_weights_dict
    }


def main():
    """
    Main function to execute the OFI analysis
    """
    # Load data
    file_path = 'first_25000_rows.csv'
    df = load_data(file_path)

    # Process parameters
    time_interval = '1min'
    levels = 10

    # Process all instruments
    results = process_all_instruments(df, time_interval, levels)


if __name__ == "__main__":
    main()