"""
Real estate transaction data processing module.

This module processes CSV files containing real estate transaction data and generates
analytical reports for different entity types including agents, offices, lenders,
and title companies. Each report type can be enabled independently.

The module performs data cleaning, aggregation, and statistical calculations to
provide insights into transaction patterns, relationships, and performance metrics.
"""

import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import constants
import utils


def process_from_filepath(
    filepath: str, 
    agents: bool = True,
    offices: bool = False,
    lenders: bool = True,
    titlecompanies: bool = False,
) -> List[Tuple[str, str]]:
    """
    Process real estate transaction data from a CSV file and generate reports.
    
    Args:
        filepath: Path to the input CSV file
        agents: Whether to generate agent analysis report
        offices: Whether to generate office analysis report  
        lenders: Whether to generate lender analysis report
        titlecompanies: Whether to generate title company analysis report
        
    Returns:
        List of tuples containing (local_filepath, s3_filepath) for generated files
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If timestamp cannot be extracted from filename
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Extract timestamp from filename (last 8 characters before extension)
    filename_base = os.path.splitext(os.path.basename(filepath))[0]
    if len(filename_base) < 8:
        raise ValueError(f"Cannot extract timestamp from filename: {filename_base}")
    
    timestamp = filename_base[-8:]
    generated_files = []

    # Generate reports based on enabled flags
    report_generators = [
        (agents, build_agents_table, "agents"),
        (offices, build_offices_table, "offices"), 
        (lenders, build_lenders_table, "lenders"),
        (titlecompanies, build_titlecompanies_table, "title companies")
    ]
    
    for enabled, generator_func, report_type in report_generators:
        if enabled:
            try:
                file_paths = generator_func(filepath, timestamp)
                generated_files.append(file_paths)
                print(f"Successfully generated {report_type} report: {file_paths[0]}")
            except Exception as e:
                print(f"Error generating {report_type} report: {e}")
                raise

    return generated_files


def _load_and_clean_data(
    filepath: str, 
    column_mapping: dict, 
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load CSV data and perform basic cleaning operations.
    
    Args:
        filepath: Path to CSV file
        column_mapping: Dictionary mapping column names to data types
        required_columns: List of columns that cannot have null values
        
    Returns:
        Cleaned DataFrame
    """
    # Load only required columns with specified data types
    df = pd.read_csv(filepath, usecols=column_mapping.keys(), dtype=column_mapping)
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Remove rows with missing required data
    if required_columns:
        for col in required_columns:
            if col in df.columns:
                before_count = len(df)
                df = df.dropna(subset=[col])
                if len(df) < before_count:
                    print(f"Removed {before_count - len(df)} rows with missing {col}")
    
    return df


def _filter_non_doubleend_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out double-end transactions from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame excluding double-end transactions
    """
    if 'doubleend' not in df.columns:
        return df
    
    initial_count = len(df)
    filtered_df = df[df['doubleend'] != 'Y'].copy()
    removed_count = initial_count - len(filtered_df)
    
    if removed_count > 0:
        print(f"Filtered out {removed_count} double-end transactions")
    
    return filtered_df


def _build_listing_agent_stats(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Build listing agent statistics from transaction data.
    
    Args:
        df: Transaction DataFrame
        group_column: Column to group by (agent name or office)
        
    Returns:
        DataFrame with listing agent statistics
    """
    grouped = df.groupby(group_column)
    
    # Calculate aggregated statistics
    stats = pd.concat([
        grouped.agg({group_column: 'count'}),
        grouped.agg({'buyagentname': utils.calculate_latest_buyer}),
        grouped.apply(utils.calculate_investorrel),
    ], axis=1)
    
    stats.columns = ['list_count', 'most_recent_buyer', 'list_investor_count']
    return stats.reset_index()


def _build_buying_agent_stats(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Build buying agent statistics from transaction data.
    
    Args:
        df: Transaction DataFrame  
        group_column: Column to group by (agent name or office)
        
    Returns:
        DataFrame with buying agent statistics
    """
    buy_column = group_column.replace('list', 'buy') if 'list' in group_column else f'buy{group_column}'
    if buy_column not in df.columns:
        buy_column = 'buyagentname' if 'agent' in group_column else 'buyagentoffice'
    
    grouped = df.groupby(buy_column)
    
    # Calculate aggregated statistics
    stats = pd.concat([
        grouped.agg({'purchasedate': 'max'}),
        grouped.agg({buy_column: 'count'}),
        grouped.agg({'purchaseprice': 'mean'}),
        grouped.apply(utils.calculate_lastpropertypurchase),
        grouped.apply(utils.calculate_investorrel),
    ], axis=1)
    
    stats.columns = [
        'last_purchase_date', 'buy_count', 'average_purchase_price',
        'last_purchase', 'buy_investor_count'
    ]
    return stats.reset_index()


def _build_relisting_agent_stats(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Build relisting agent statistics from transaction data.
    
    Args:
        df: Transaction DataFrame
        group_column: Column to group by (agent name or office)
        
    Returns:
        DataFrame with relisting agent statistics  
    """
    relist_column = f're_{group_column}'
    if relist_column not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby(relist_column)
    
    # Calculate aggregated statistics
    stats = pd.concat([
        grouped.agg({relist_column: 'count'}),
        grouped.agg({'resaleprice': 'mean'}),
        grouped.apply(utils.calculate_investorrel),
    ], axis=1)
    
    stats.columns = ['relist_count', 'average_resale_price', 'relist_investor_count']
    return stats.reset_index()


def _build_doubleend_stats(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Build double-end transaction statistics.
    
    Args:
        df: Full DataFrame including double-end transactions
        group_column: Column to group by
        
    Returns:
        DataFrame with double-end statistics
    """
    doubleend_df = df[df['doubleend'] == 'Y']
    if doubleend_df.empty:
        return pd.DataFrame(columns=[f'doubleend_{group_column}', 'doubleend_count'])
    
    stats = doubleend_df.groupby(group_column).size().reset_index()
    stats.columns = [f'doubleend_{group_column}', 'doubleend_count']
    return stats


def build_agents_table(filepath: str, timestamp: str) -> Tuple[str, str]:
    """
    Build comprehensive agent analysis table from transaction data.
    
    Args:
        filepath: Path to input CSV file
        timestamp: Timestamp for output filename
        
    Returns:
        Tuple of (local_filepath, s3_filepath) for the generated report
    """
    # Load and clean data
    df = _load_and_clean_data(filepath, constants.agents_mapping)
    
    # Filter non-double-end transactions for most calculations
    no_doubleend_df = _filter_non_doubleend_transactions(df)
    
    # Build component statistics tables
    list_stats = _build_listing_agent_stats(no_doubleend_df, 'listagentname')
    buy_stats = _build_buying_agent_stats(no_doubleend_df, 'buyagentname')
    relist_stats = _build_relisting_agent_stats(no_doubleend_df, 're_listagentname')
    doubleend_stats = _build_doubleend_stats(df, 'listagentname')
    
    # Merge listing and buying agent data
    agent_df = pd.merge(
        list_stats, buy_stats, 
        how='outer', left_on='listagentname', right_on='buyagentname'
    )
    
    # Create unified agent name column
    agent_df['agentname'] = agent_df['listagentname'].combine_first(agent_df['buyagentname'])
    
    # Calculate total transactions
    agent_df['total_transactions'] = (
        agent_df['list_count'].fillna(0) + agent_df['buy_count'].fillna(0)
    )
    
    # Merge relisting data
    if not relist_stats.empty:
        agent_df = pd.merge(
            agent_df, relist_stats,
            how='outer', left_on='agentname', right_on='re_listagentname'
        )
        agent_df['agentname'] = agent_df['agentname'].combine_first(agent_df['re_listagentname'])
        agent_df['total_transactions'] += agent_df['relist_count'].fillna(0)
    
    # Calculate purchase/resale ratio
    agent_df['purchase_resale_ratio'] = (
        agent_df['average_purchase_price'] / agent_df['average_resale_price']
    )
    
    # Merge double-end statistics
    if not doubleend_stats.empty:
        agent_df = pd.merge(
            agent_df, doubleend_stats,
            how='left', left_on='agentname', right_on='doubleend_listagentname'
        )
    
    # Calculate double-end percentage
    agent_df['doubleend_percentage'] = (
        (agent_df['doubleend_count'] / agent_df['total_transactions']) * 100
    ).fillna(0)
    
    # Calculate total investor relationships
    investor_cols = ['list_investor_count', 'buy_investor_count', 'relist_investor_count']
    agent_df['investor_relationships'] = agent_df[
        [col for col in investor_cols if col in agent_df.columns]
    ].fillna(0).sum(axis=1)
    
    # Select and rename final columns
    final_columns = [
        'agentname', 'last_purchase_date', 'total_transactions',
        'average_purchase_price', 'average_resale_price', 'purchase_resale_ratio',
        'last_purchase', 'list_count', 'doubleend_percentage',
        'most_recent_buyer', 'relist_count', 'investor_relationships'
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in final_columns if col in agent_df.columns]
    result_df = agent_df[available_columns].copy()
    
    # Apply final column mapping
    for i, col_name in enumerate(result_df.columns):
        if i < len(constants.agents_index_mapping):
            result_df.rename(columns={col_name: constants.agents_index_mapping[i]}, inplace=True)
    
    # Generate output paths
    filename = f'agents_{timestamp}.csv'
    output_filepath = f'{os.path.dirname(filepath)}/{filename}'
    s3_filepath = f'agents/{filename}'
    
    # Save the table
    result_df.to_csv(output_filepath, index=False)
    
    return (output_filepath, s3_filepath)


def build_offices_table(filepath: str, timestamp: str) -> Tuple[str, str]:
    """
    Build comprehensive office analysis table from transaction data.
    
    This function mirrors the agent analysis but groups by office instead of agent name.
    
    Args:
        filepath: Path to input CSV file
        timestamp: Timestamp for output filename
        
    Returns:
        Tuple of (local_filepath, s3_filepath) for the generated report
    """
    # Load and clean data
    df = _load_and_clean_data(filepath, constants.offices_mapping)
    
    # Filter non-double-end transactions
    no_doubleend_df = _filter_non_doubleend_transactions(df)
    
    # Build statistics using office columns
    list_stats = _build_listing_agent_stats(no_doubleend_df, 'listagentoffice')
    buy_stats = _build_buying_agent_stats(no_doubleend_df, 'buyagentoffice')  
    relist_stats = _build_relisting_agent_stats(no_doubleend_df, 're_listagentoffice')
    doubleend_stats = _build_doubleend_stats(df, 'listagentoffice')
    
    # Follow same merging logic as agents but with office columns
    office_df = pd.merge(
        list_stats, buy_stats,
        how='outer', left_on='listagentoffice', right_on='buyagentoffice'
    )
    
    office_df['officename'] = office_df['listagentoffice'].combine_first(office_df['buyagentoffice'])
    office_df['total_transactions'] = (
        office_df['list_count'].fillna(0) + office_df['buy_count'].fillna(0)
    )
    
    # Merge additional data and apply same transformations as agents
    if not relist_stats.empty:
        office_df = pd.merge(
            office_df, relist_stats,
            how='outer', left_on='officename', right_on='re_listagentoffice'
        )
        office_df['officename'] = office_df['officename'].combine_first(office_df['re_listagentoffice'])
        office_df['total_transactions'] += office_df['relist_count'].fillna(0)
    
    office_df['purchase_resale_ratio'] = (
        office_df['average_purchase_price'] / office_df['average_resale_price']
    )
    
    if not doubleend_stats.empty:
        office_df = pd.merge(
            office_df, doubleend_stats,
            how='left', left_on='officename', right_on='doubleend_listagentoffice'
        )
    
    office_df['doubleend_percentage'] = (
        (office_df['doubleend_count'] / office_df['total_transactions']) * 100
    ).fillna(0)
    
    # Calculate investor relationships
    investor_cols = ['list_investor_count', 'buy_investor_count', 'relist_investor_count']
    office_df['investor_relationships'] = office_df[
        [col for col in investor_cols if col in office_df.columns]
    ].fillna(0).sum(axis=1)
    
    # Select final columns and apply mapping
    final_columns = [
        'officename', 'last_purchase_date', 'total_transactions',
        'average_purchase_price', 'average_resale_price', 'purchase_resale_ratio',
        'last_purchase', 'list_count', 'doubleend_percentage',
        'most_recent_buyer', 'relist_count', 'investor_relationships'
    ]
    
    available_columns = [col for col in final_columns if col in office_df.columns]
    result_df = office_df[available_columns].copy()
    
    for i, col_name in enumerate(result_df.columns):
        if i < len(constants.offices_index_mapping):
            result_df.rename(columns={col_name: constants.offices_index_mapping[i]}, inplace=True)
    
    # Generate output paths
    filename = f'offices_{timestamp}.csv'
    output_filepath = f'{os.path.dirname(filepath)}/{filename}'
    s3_filepath = f'offices/{filename}'
    
    # Save the table
    result_df.to_csv(output_filepath, index=False)
    
    return (output_filepath, s3_filepath)


def build_lenders_table(filepath: str, timestamp: str) -> Tuple[str, str]:
    """
    Build lender analysis table from transaction data.
    
    Analyzes lending patterns, loan amounts, and relationships with investors.
    
    Args:
        filepath: Path to input CSV file
        timestamp: Timestamp for output filename
        
    Returns:
        Tuple of (local_filepath, s3_filepath) for the generated report
    """
    # Load and clean data
    df = _load_and_clean_data(filepath, constants.lenders_mapping)
    
    # Format lender names
    df['lender_formatted'] = df['lender'].str.title()
    
    # Replace zero values with NaN for price calculations
    price_columns = ['loan_amount', 'purchaseprice', 'resaleprice']
    for col in price_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # Group by lender and calculate statistics
    grouped = df.groupby('lender_formatted')
    
    # Build aggregated statistics
    stats_df = pd.concat([
        grouped.agg({'loan_amount': 'mean'}),
        grouped.apply(utils.calculate_loanpurchaseprice_ratio),
        grouped.agg({'lender': 'count'}),
        grouped.apply(utils.calculate_investorrel),
        grouped.apply(utils.calculate_averageloanperrel),
    ], axis=1)
    
    # Reset index to make lender name a column
    stats_df = stats_df.reset_index()
    
    # Apply column mappings
    for i, col_name in enumerate(stats_df.columns):
        if i < len(constants.lenders_index_mapping):
            stats_df.rename(columns={col_name: constants.lenders_index_mapping[i]}, inplace=True)
    
    # Generate output paths
    filename = f'lenders_{timestamp}.csv'
    output_filepath = f'{os.path.dirname(filepath)}/{filename}'
    s3_filepath = f'lenders/{filename}'
    
    # Save the table
    stats_df.to_csv(output_filepath, index=False)
    
    return (output_filepath, s3_filepath)


def build_titlecompanies_table(filepath: str, timestamp: str) -> Tuple[str, str]:
    """
    Build title company analysis table from transaction data.
    
    Analyzes market share, relationships with offices/agents/investors for title companies.
    Uses fuzzy matching to standardize company names.
    
    Args:
        filepath: Path to input CSV file
        timestamp: Timestamp for output filename
        
    Returns:
        Tuple of (local_filepath, s3_filepath) for the generated report
    """
    # Load and clean data
    df = _load_and_clean_data(filepath, constants.titlecompanies_mapping)
    
    # Apply fuzzy grouping to standardize title company names
    df = utils.fuzzy_group_large_dataset(df, 'title_company')
    
    # Format standardized company names
    df['title_company_formatted'] = df['standardized_title_company'].str.title()
    
    # Group by standardized title company name
    grouped = df.groupby('title_company_formatted')
    
    # Build aggregated statistics
    stats_df = pd.concat([
        grouped.agg({'title_company': 'count'}),
        grouped.apply(utils.calculate_marketshare),
        grouped.apply(utils.calculate_uniqueoffices),
        grouped.apply(utils.calculate_uniqueagents),
        grouped.apply(utils.calculate_uniqueinvestors),
    ], axis=1)
    
    # Reset index to make company name a column
    stats_df = stats_df.reset_index()
    
    # Apply column mappings
    for i, col_name in enumerate(stats_df.columns):
        if i < len(constants.titlecompanies_index_mapping):
            stats_df.rename(columns={col_name: constants.titlecompanies_index_mapping[i]}, inplace=True)
    
    # Generate output paths
    filename = f'titlecompanies_{timestamp}.csv'
    output_filepath = f'{os.path.dirname(filepath)}/{filename}'
    s3_filepath = f'titlecompanies/{filename}'
    
    # Save the table
    stats_df.to_csv(output_filepath, index=False)
    
    return (output_filepath, s3_filepath)