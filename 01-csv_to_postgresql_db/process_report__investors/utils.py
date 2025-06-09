"""
Utility functions for data processing and analysis.

This module provides various utility functions for fuzzy string matching,
statistical calculations, and data transformations used in the investor
data processing pipeline.
"""

import concurrent.futures
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from rapidfuzz import process, fuzz


def fuzzy_group_large_dataset(
    df: pd.DataFrame, 
    column_name: str, 
    threshold: int = 85
) -> pd.DataFrame:
    """
    Group values in a column using fuzzy matching for large datasets.
    
    Uses single-threaded processing with optimized batch handling for
    memory efficiency on large datasets.
    
    Args:
        df: Input DataFrame containing the column to be processed.
        column_name: Name of column containing strings with potential spelling errors.
        threshold: Similarity threshold (0-100) for fuzzy matching.
        
    Returns:
        DataFrame with new column 'standardized_{column_name}' containing
        standardized values.
    """
    # Extract unique values to reduce computation
    unique_values = df[column_name].dropna().unique()
    
    if len(unique_values) == 0:
        df[f'standardized_{column_name}'] = df[column_name]
        return df
    
    # Process in batches for memory efficiency
    batch_size = min(10000, len(unique_values))
    mapping = {}
    
    for i in range(0, len(unique_values), batch_size):
        batch = unique_values[i:i+batch_size]
        batch_mapping = _process_fuzzy_batch(batch, threshold)
        mapping.update(batch_mapping)
    
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
    Group values using fuzzy matching with concurrent processing.
    
    Uses multiple threads for improved performance on very large datasets.
    Includes cross-batch reconciliation to ensure consistency.
    
    Args:
        df: Input DataFrame containing the column to be processed.
        column_name: Name of column containing strings with potential spelling errors.
        threshold: Similarity threshold (0-100) for fuzzy matching.
        max_workers: Maximum number of worker threads. Defaults to optimal value.
        
    Returns:
        DataFrame with new column 'standardized_{column_name}' containing
        standardized values.
    """
    unique_values = df[column_name].dropna().unique()
    
    if len(unique_values) == 0:
        df[f'standardized_{column_name}'] = df[column_name]
        return df
    
    # Determine optimal batch size and worker count
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    batch_size = min(10000, max(1000, len(unique_values) // max_workers))
    batches = [
        unique_values[i:i+batch_size] 
        for i in range(0, len(unique_values), batch_size)
    ]
    
    # Process batches concurrently
    final_mapping = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_process_fuzzy_batch, batch, threshold): i 
            for i, batch in enumerate(batches)
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_result = future.result()
            final_mapping.update(batch_result)
    
    # Cross-batch reconciliation for multiple batches
    if len(batches) > 1:
        final_mapping = _reconcile_canonical_values(final_mapping, threshold, max_workers)
    
    # Apply mapping
    df[f'standardized_{column_name}'] = (
        df[column_name].map(final_mapping).fillna(df[column_name])
    )
    
    return df


def _process_fuzzy_batch(batch: List[str], threshold: int) -> Dict[str, str]:
    """
    Process a single batch of strings for fuzzy matching.
    
    Args:
        batch: List of strings to process.
        threshold: Similarity threshold for matching.
        
    Returns:
        Dictionary mapping original strings to canonical forms.
    """
    batch_mapping = {}
    groups = defaultdict(list)
    processed_values = set()
    
    for val in batch:
        if val in processed_values:
            continue
            
        # Find matches above threshold
        matches = process.extract(
            val, batch, scorer=fuzz.ratio, score_cutoff=threshold
        )
        
        if matches:
            # Use longest string as canonical form
            canonical = max(matches, key=lambda x: len(x[0]))[0]
            for match_val, score, _ in matches:
                groups[canonical].append(match_val)
                processed_values.add(match_val)
                batch_mapping[match_val] = canonical
    
    return batch_mapping


def _reconcile_canonical_values(
    mapping: Dict[str, str], 
    threshold: int, 
    max_workers: int
) -> Dict[str, str]:
    """
    Reconcile canonical values across different batches.
    
    Args:
        mapping: Initial mapping from batch processing.
        threshold: Similarity threshold for matching.
        max_workers: Maximum number of worker threads.
        
    Returns:
        Updated mapping with reconciled canonical values.
    """
    canonical_values = list(set(mapping.values()))
    
    def find_canonical_matches(val):
        matches = process.extract(
            val, canonical_values, scorer=fuzz.ratio, score_cutoff=threshold
        )
        # Filter out self-match
        matches = [m for m in matches if m[0] != val]
        return val, matches
    
    # Find matches between canonical values
    canonical_matches = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(find_canonical_matches, val) 
            for val in canonical_values
        ]
        
        for future in concurrent.futures.as_completed(futures):
            val, matches = future.result()
            if matches:
                # Select longest string as final canonical form
                best_match = max([val] + [m[0] for m in matches], key=len)
                canonical_matches[val] = best_match
    
    # Update mappings based on canonical matches
    final_mapping = mapping.copy()
    for original, canonical in mapping.items():
        if canonical in canonical_matches:
            final_mapping[original] = canonical_matches[canonical]
    
    return final_mapping


# Statistical calculation functions for investor analysis

def calculate_pfv_ratio(group: pd.DataFrame) -> float:
    """Calculate Purchase to Fair Value ratio as percentage."""
    purchase_price = group['purchaseprice'].dropna()
    last_listing_price = group['last_listing_price'].dropna()
    
    if purchase_price.empty or last_listing_price.empty:
        return pd.NA
    
    return (purchase_price / last_listing_price).mean() * 100


def calculate_listsold_ratio(group: pd.DataFrame) -> float:
    """Calculate List to Sold ratio as percentage."""
    resale_price = group['resaleprice'].dropna()
    purchase_price = group['purchaseprice'].dropna()
    
    if resale_price.empty or purchase_price.empty:
        return pd.NA
    
    return (resale_price / purchase_price).mean() * 100


def calculate_pm_ratio(group: pd.DataFrame) -> float:
    """Calculate average days between purchase and listing."""
    last_listing_date = group['last_listing_date'].dropna()
    purchase_date = group['purchasedate'].dropna()
    
    if last_listing_date.empty or purchase_date.empty:
        return pd.NA
    
    # Ensure both series have matching indices for subtraction
    common_indices = last_listing_date.index.intersection(purchase_date.index)
    if len(common_indices) == 0:
        return pd.NA
    
    return (
        (last_listing_date.loc[common_indices] - purchase_date.loc[common_indices])
        .dt.days.mean()
    )


def calculate_purchaseresale(group: pd.DataFrame) -> float:
    """Calculate average days between purchase and resale."""
    resale_date = group['resaledate'].dropna()
    purchase_date = group['purchasedate'].dropna()
    
    if resale_date.empty or purchase_date.empty:
        return pd.NA
    
    # Ensure both series have matching indices for subtraction
    common_indices = resale_date.index.intersection(purchase_date.index)
    if len(common_indices) == 0:
        return pd.NA
    
    return (
        (resale_date.loc[common_indices] - purchase_date.loc[common_indices])
        .dt.days.mean()
    )


def calculate_financing(group: pd.DataFrame) -> str:
    """Determine financing type based on lender information."""
    lenders = group['lender'].dropna()
    
    if lenders.empty:
        return 'Unknown'
    
    # If more non-null lenders than null, it's mixed financing
    if len(lenders) >= len(group['lender']) - len(lenders):
        return 'Mixed'
    
    return 'Cash'


def calculate_agentrel(group: pd.DataFrame) -> str:
    """Calculate agent relationship statistics."""
    listing_agent = group['listagentname'].nunique()
    buying_agent = group['buyagentname'].nunique()
    relisting_agent = group['re_listagentname'].nunique()
    
    total_rel = listing_agent + buying_agent + relisting_agent
    
    return f'{total_rel}/{listing_agent}/{buying_agent}/{relisting_agent}'


def calculate_lastpropertypurchase(group: pd.DataFrame) -> str:
    """Get the most recent property purchase address."""
    purchase_dates = group['purchasedate'].dropna()
    
    if purchase_dates.empty:
        return 'No purchases found'
    
    # Get index of most recent purchase
    latest_idx = purchase_dates.idxmax()
    
    street_number = group.loc[latest_idx, 'property_streetnumber']
    street_name = str(group.loc[latest_idx, 'property_streetname']).title()
    property_city = str(group.loc[latest_idx, 'property_city']).title()
    
    return f'{street_number} {street_name}, {property_city}'


def calculate_mostfrequentvalue(series: pd.Series) -> str:
    """Calculate the most frequent value in a series."""
    series_clean = series.dropna()
    
    if series_clean.empty:
        return pd.NA
    
    return series_clean.value_counts().index[0]


# Additional utility functions (some marked as TODO in original)

def calculate_doubleend(group: pd.DataFrame) -> float:
    """Calculate percentage of double-ended transactions."""
    doubleend = group.get('doubleend', pd.Series(dtype='object'))
    
    if doubleend.empty:
        return 0.0
    
    count = (doubleend == 'Y').sum()
    return count / len(doubleend)


def calculate_investorrel(group: pd.DataFrame) -> int:
    """Calculate number of unique investors in group."""
    return group['investor'].nunique()


def calculate_loanpurchaseprice_ratio(group: pd.DataFrame) -> float:
    """Calculate loan amount to purchase price ratio as percentage."""
    loan_amount = group.get('loan_amount', pd.Series(dtype='float'))
    purchase_price = group['purchaseprice']
    
    if loan_amount.empty or purchase_price.empty:
        return pd.NA
    
    return (loan_amount / purchase_price).mean() * 100


def calculate_averageloanperrel(group: pd.DataFrame) -> float:
    """Calculate average loan amount per investor relationship."""
    loan_amount = group.get('loan_amount', pd.Series(dtype='float'))
    
    if loan_amount.empty:
        return pd.NA
    
    return group.groupby('investor')['loan_amount'].mean().mean()


def calculate_uniqueoffices(group: pd.DataFrame) -> int:
    """Calculate number of unique offices involved."""
    office_columns = ['listagentoffice', 'buyagentoffice', 're_listagentoffice']
    available_columns = [col for col in office_columns if col in group.columns]
    
    if not available_columns:
        return 0
    
    offices = pd.concat([group[col] for col in available_columns])
    return offices.nunique()


def calculate_uniqueagents(group: pd.DataFrame) -> int:
    """Calculate number of unique agents involved."""
    agent_columns = ['listagentname', 'buyagentname', 're_listagentname']
    available_columns = [col for col in agent_columns if col in group.columns]
    
    if not available_columns:
        return 0
    
    agents = pd.concat([group[col] for col in available_columns])
    return agents.nunique()


def calculate_uniqueinvestors(group: pd.DataFrame) -> int:
    """Calculate number of unique investors in group."""
    return group['investor'].nunique()


# TODO: Implement these functions when requirements are clarified
def calculate_listingssoldtoinvestors(group: pd.DataFrame) -> int:
    """Calculate listings sold to investors. Implementation needed."""
    return 0


def calculate_mostrecentbuyersagent(group: pd.DataFrame) -> str:
    """Get most recent buyer's agent. Implementation needed."""
    return "Not implemented"


def calculate_officerepresentingbuyers(group: pd.DataFrame) -> int:
    """Calculate office representing buyers. Implementation needed."""
    return 0


def calculate_buyerrepresentedtransactions(group: pd.DataFrame) -> int:
    """Calculate buyer represented transactions. Implementation needed."""
    return 0


def calculate_listingsresoldforinvestors(group: pd.DataFrame) -> int:
    """Calculate listings resold for investors. Implementation needed."""
    return 0


def calculate_marketshare(group: pd.DataFrame) -> float:
    """Calculate market share. Implementation needed."""
    return 0.0