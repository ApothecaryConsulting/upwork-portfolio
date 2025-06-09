"""
Utility functions for real estate transaction data analysis.

This module provides specialized calculation functions for processing real estate
transaction data, including fuzzy string matching for data standardization,
statistical calculations for various metrics, and aggregation functions for
generating analytical insights.

Key functionality includes:
- Fuzzy string matching for standardizing entity names
- Transaction ratio calculations (loan-to-purchase, purchase-to-resale)
- Relationship counting and analysis
- Property and agent statistics
"""

import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


def fuzzy_group_large_dataset(
    df: pd.DataFrame, 
    column_name: str, 
    threshold: int = 85
) -> pd.DataFrame:
    """
    Group values in a column using fuzzy matching for large datasets.
    
    This function standardizes similar strings (e.g., company names with spelling
    variations) by grouping them under a canonical form. Optimized for memory
    efficiency with large datasets.
    
    Args:
        df: Input DataFrame containing the column to standardize
        column_name: Name of column containing strings with potential spelling errors
        threshold: Similarity threshold (0-100) for fuzzy matching
        
    Returns:
        DataFrame with additional 'standardized_{column_name}' column
        
    Raises:
        KeyError: If column_name doesn't exist in DataFrame
        ValueError: If threshold is not between 0 and 100
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")
    
    # Extract unique values to reduce computation complexity
    unique_values = df[column_name].dropna().unique()
    
    if len(unique_values) == 0:
        df[f'standardized_{column_name}'] = df[column_name]
        return df
    
    # Use batching for very large datasets to manage memory
    batch_size = min(10000, len(unique_values))
    mapping = {}
    
    # Process in batches for memory efficiency
    for i in range(0, len(unique_values), batch_size):
        batch = unique_values[i:i+batch_size]
        
        # Group similar strings within batch
        groups = defaultdict(list)
        processed_values = set()
        
        for val in batch:
            if val in processed_values:
                continue
                
            # Find matches above threshold within batch
            matches = process.extract(
                val, batch, scorer=fuzz.ratio, score_cutoff=threshold
            )
            
            if matches:
                # Use longest string as canonical form
                canonical = max(matches, key=lambda x: len(x[0]))[0]
                for match_val, score, _ in matches:
                    groups[canonical].append(match_val)
                    processed_values.add(match_val)
                    mapping[match_val] = canonical
    
    # Apply mapping to create standardized column
    df[f'standardized_{column_name}'] = (
        df[column_name].map(mapping).fillna(df[column_name])
    )
    
    return df


def fuzzy_group_large_dataset_concurrent(
    df: pd.DataFrame, 
    column_name: str, 
    threshold: int = 85, 
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Concurrent version of fuzzy grouping for very large datasets.
    
    Uses thread pool to process batches in parallel for improved performance
    on multi-core systems with large datasets.
    
    Args:
        df: Input DataFrame
        column_name: Column to standardize
        threshold: Similarity threshold (0-100)
        max_workers: Maximum threads to use (default: CPU count + 4)
        
    Returns:
        DataFrame with standardized column
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    unique_values = df[column_name].dropna().unique()
    
    if len(unique_values) == 0:
        df[f'standardized_{column_name}'] = df[column_name]
        return df
    
    # Determine optimal batch size
    num_workers = max_workers or min(32, (len(unique_values) // 1000) + 4)
    batch_size = max(1000, len(unique_values) // num_workers)
    
    # Split into batches
    batches = [
        unique_values[i:i+batch_size] 
        for i in range(0, len(unique_values), batch_size)
    ]
    
    def process_batch(batch: List[str]) -> Dict[str, str]:
        """Process a single batch of values."""
        batch_mapping = {}
        groups = defaultdict(list)
        processed_values = set()
        
        for val in batch:
            if val in processed_values:
                continue
                
            matches = process.extract(
                val, batch, scorer=fuzz.ratio, score_cutoff=threshold
            )
            
            if matches:
                canonical = max(matches, key=lambda x: len(x[0]))[0]
                for match_val, score, _ in matches:
                    groups[canonical].append(match_val)
                    processed_values.add(match_val)
                    batch_mapping[match_val] = canonical
        
        return batch_mapping
    
    # Process batches concurrently
    final_mapping = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in concurrent.futures.as_completed(futures):
            batch_result = future.result()
            final_mapping.update(batch_result)
    
    # Cross-batch reconciliation for multi-batch processing
    if len(batches) > 1:
        canonical_values = list(set(final_mapping.values()))
        
        def find_canonical_matches(val: str) -> tuple:
            """Find matches between canonical values from different batches."""
            matches = process.extract(
                val, canonical_values, scorer=fuzz.ratio, score_cutoff=threshold
            )
            # Filter out self-match
            matches = [m for m in matches if m[0] != val]
            return val, matches
        
        # Process canonical reconciliation concurrently
        canonical_matches = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(find_canonical_matches, val) 
                for val in canonical_values
            ]
            
            for future in concurrent.futures.as_completed(futures):
                val, matches = future.result()
                if matches:
                    # Use longest string as final canonical form
                    best_match = max([val] + [m[0] for m in matches], key=len)
                    canonical_matches[val] = best_match
        
        # Update mappings based on cross-batch reconciliation
        for original, canonical in list(final_mapping.items()):
            if canonical in canonical_matches:
                final_mapping[original] = canonical_matches[canonical]
    
    # Apply final mapping
    df[f'standardized_{column_name}'] = (
        df[column_name].map(final_mapping).fillna(df[column_name])
    )
    
    return df


def calculate_latest_buyer(series: pd.Series) -> Union[str, float]:
    """
    Calculate the most frequent buyer agent from a series.
    
    Args:
        series: Series of buyer agent names
        
    Returns:
        Most frequent buyer agent name, or NaN if series is empty
    """
    series_clean = series.dropna()
    
    if series_clean.empty:
        return pd.NA
    
    return series_clean.value_counts().index[0]


def calculate_investorrel(group: pd.DataFrame) -> int:
    """
    Calculate the number of unique investor relationships.
    
    Args:
        group: DataFrame group containing investor column
        
    Returns:
        Count of unique investors
    """
    if 'investor' not in group.columns:
        return 0
    
    return group['investor'].nunique()


def calculate_lastpropertypurchase(group: pd.DataFrame) -> str:
    """
    Calculate the most recent property purchase details.
    
    Args:
        group: DataFrame group with property and purchase date information
        
    Returns:
        Formatted string with property address of most recent purchase
    """
    if 'purchasedate' not in group.columns or group['purchasedate'].isna().all():
        return "No purchases found"
    
    # Find row with most recent purchase date
    latest_idx = group['purchasedate'].idxmax()
    
    try:
        street_number = group.loc[latest_idx, 'property_streetnumber']
        street_name = group.loc[latest_idx, 'property_streetname']
        city = group.loc[latest_idx, 'property_city']
        
        # Handle missing values and format properly
        street_number = str(street_number) if pd.notna(street_number) else ""
        street_name = str(street_name).title() if pd.notna(street_name) else ""
        city = str(city).title() if pd.notna(city) else ""
        
        # Build address string
        address_parts = [p for p in [street_number, street_name] if p]
        address = " ".join(address_parts)
        
        if city:
            return f"{address}, {city}" if address else city
        else:
            return address if address else "Address unavailable"
            
    except (KeyError, IndexError):
        return "Property details unavailable"


def calculate_loanpurchaseprice_ratio(group: pd.DataFrame) -> float:
    """
    Calculate the average loan-to-purchase price ratio as a percentage.
    
    Args:
        group: DataFrame group with loan_amount and purchaseprice columns
        
    Returns:
        Average loan-to-purchase ratio as percentage
    """
    if 'loan_amount' not in group.columns or 'purchaseprice' not in group.columns:
        return 0.0
    
    # Filter out rows where either value is missing or zero
    valid_data = group[
        (group['loan_amount'].notna()) & 
        (group['purchaseprice'].notna()) &
        (group['loan_amount'] > 0) & 
        (group['purchaseprice'] > 0)
    ]
    
    if valid_data.empty:
        return 0.0
    
    ratio = (valid_data['loan_amount'] / valid_data['purchaseprice']).mean()
    return ratio * 100 if pd.notna(ratio) else 0.0


def calculate_averageloanperrel(group: pd.DataFrame) -> float:
    """
    Calculate the average loan amount per investor relationship.
    
    Args:
        group: DataFrame group with investor and loan_amount columns
        
    Returns:
        Average loan amount per unique investor relationship
    """
    if 'investor' not in group.columns or 'loan_amount' not in group.columns:
        return 0.0
    
    # Group by investor and calculate mean loan amount per investor
    investor_loans = group.groupby('investor')['loan_amount'].mean()
    
    # Return the mean of investor averages
    return investor_loans.mean() if not investor_loans.empty else 0.0


def calculate_marketshare(group: pd.DataFrame) -> float:
    """
    Calculate market share for a title company group.
    
    Note: This is a placeholder implementation. Actual market share calculation
    would require total market size context.
    
    Args:
        group: DataFrame group for a specific title company
        
    Returns:
        Market share percentage (currently returns 0 as placeholder)
    """
    # TODO: Implement actual market share calculation
    # This would require knowing the total market size
    return 0.0


def calculate_uniqueoffices(group: pd.DataFrame) -> int:
    """
    Calculate the number of unique offices associated with a title company.
    
    Args:
        group: DataFrame group with office-related columns
        
    Returns:
        Count of unique offices across all office columns
    """
    office_columns = [
        'listagentoffice', 'buyagentoffice', 're_listagentoffice'
    ]
    
    # Collect all office names from available columns
    all_offices = pd.Series(dtype=str)
    for col in office_columns:
        if col in group.columns:
            all_offices = pd.concat([all_offices, group[col]], ignore_index=True)
    
    return all_offices.nunique() if not all_offices.empty else 0


def calculate_uniqueagents(group: pd.DataFrame) -> int:
    """
    Calculate the number of unique agents associated with a title company.
    
    Args:
        group: DataFrame group with agent-related columns
        
    Returns:
        Count of unique agents across all agent columns
    """
    agent_columns = [
        'listagentname', 'buyagentname', 're_listagentname'
    ]
    
    # Collect all agent names from available columns
    all_agents = pd.Series(dtype=str)
    for col in agent_columns:
        if col in group.columns:
            all_agents = pd.concat([all_agents, group[col]], ignore_index=True)
    
    return all_agents.nunique() if not all_agents.empty else 0


def calculate_uniqueinvestors(group: pd.DataFrame) -> int:
    """
    Calculate the number of unique investors associated with a title company.
    
    Args:
        group: DataFrame group with investor column
        
    Returns:
        Count of unique investors
    """
    if 'investor' not in group.columns:
        return 0
    
    return group['investor'].nunique()


# Legacy function aliases for backward compatibility
def calculate_pfv_ratio(group: pd.DataFrame) -> float:
    """
    Calculate purchase-to-fair value ratio (legacy function).
    
    Args:
        group: DataFrame group with purchase and listing price columns
        
    Returns:
        Average ratio as percentage
    """
    if 'purchaseprice' not in group.columns or 'last_listing_price' not in group.columns:
        return 0.0
    
    valid_data = group[
        (group['purchaseprice'].notna()) & 
        (group['last_listing_price'].notna()) &
        (group['purchaseprice'] > 0) & 
        (group['last_listing_price'] > 0)
    ]
    
    if valid_data.empty:
        return 0.0
    
    ratio = (valid_data['purchaseprice'] / valid_data['last_listing_price']).mean()
    return ratio * 100 if pd.notna(ratio) else 0.0


def calculate_listsold_ratio(group: pd.DataFrame) -> float:
    """
    Calculate list-to-sold ratio (legacy function).
    
    Args:
        group: DataFrame group with resale and purchase price columns
        
    Returns:
        Average ratio as percentage
    """
    if 'resaleprice' not in group.columns or 'purchaseprice' not in group.columns:
        return 0.0
    
    valid_data = group[
        (group['resaleprice'].notna()) & 
        (group['purchaseprice'].notna()) &
        (group['resaleprice'] > 0) & 
        (group['purchaseprice'] > 0)
    ]
    
    if valid_data.empty:
        return 0.0
    
    ratio = (valid_data['resaleprice'] / valid_data['purchaseprice']).mean()
    return ratio * 100 if pd.notna(ratio) else 0.0


def calculate_financing(group: pd.DataFrame) -> str:
    """
    Determine financing type based on lender data (legacy function).
    
    Args:
        group: DataFrame group with lender column
        
    Returns:
        'Mixed' if both cash and financed transactions, 'Cash' otherwise
    """
    if 'lender' not in group.columns:
        return 'Cash'
    
    lenders = group['lender']
    
    # If more non-null lenders than null, it's mixed financing
    if lenders.count() >= lenders.isnull().sum():
        return 'Mixed'
    
    return 'Cash'


def calculate_doubleend(group: pd.DataFrame) -> float:
    """
    Calculate the percentage of double-end transactions (legacy function).
    
    Args:
        group: DataFrame group with doubleend column
        
    Returns:
        Percentage of transactions that are double-end
    """
    if 'doubleend' not in group.columns:
        return 0.0
    
    doubleend_series = group['doubleend']
    
    if doubleend_series.empty:
        return 0.0
    
    doubleend_count = (doubleend_series == 'Y').sum()
    total_count = len(doubleend_series)
    
    return (doubleend_count / total_count) * 100 if total_count > 0 else 0.0