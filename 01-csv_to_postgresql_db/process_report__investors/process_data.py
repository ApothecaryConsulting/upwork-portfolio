"""
Data processing module for investor analysis.

This module handles the main data processing pipeline, including reading
CSV files, cleaning and transforming data, and generating investor analysis
reports with various statistical calculations.
"""

import os
from typing import List, Tuple
import logging
import datetime

import pandas as pd

import constants
import utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_from_filepath(
    filepath: str, 
    investors: bool = True,
) -> List[Tuple[str, str]]:
    """
    Process data from a filepath and return paths to processed files.
    
    Main entry point for data processing pipeline. Handles file validation,
    timestamp extraction, and coordinates the processing of different report types.
    
    Args:
        filepath: Path to the input CSV file to process.
        investors: Whether to generate investors report. Default True.
        
    Returns:
        List of tuples containing (local_filepath, s3_filepath) for each
        generated output file.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If timestamp cannot be extracted from filename.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    logger.info(f"Starting processing of file: {filepath}")
    
    # Initialize results list
    filepaths = []
    
    # Extract timestamp from filename (expects format ending with 8-digit timestamp)
    try:
        timestamp = os.path.splitext(os.path.basename(filepath))[0][-8:]
        if not timestamp.isdigit() or len(timestamp) != 8:
            raise ValueError("Invalid timestamp format")
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to extract timestamp from {filepath}: {e}")
        raise ValueError(f"Invalid filename format. Expected 8-digit timestamp suffix.") from e
    
    # Process investors report if requested
    if investors:
        try:
            output_path = build_investors_table(filepath, timestamp)
            filepaths.append(output_path)
            logger.info(f"Successfully processed {filepath} -> {output_path[0]}")
        except Exception as e:
            logger.error(f"Failed to build investors table: {e}")
            raise
    
    logger.info(f"Processing completed. Generated {len(filepaths)} output files.")
    return filepaths


def build_investors_table(filepath: str, timestamp: str) -> Tuple[str, str]:
    """
    Build and save the investors table with optimized processing.
    
    Performs comprehensive data processing including:
    - Data loading with type enforcement
    - Duplicate removal and data validation
    - Fuzzy string matching for investor names
    - Date and price column standardization
    - Statistical calculations and aggregations
    - Output file generation
    
    Args:
        filepath: Path to input CSV file.
        timestamp: Timestamp string for output filename.
        
    Returns:
        Tuple of (local_output_path, s3_output_path).
        
    Raises:
        pd.errors.EmptyDataError: If input file is empty.
        KeyError: If required columns are missing.
        Exception: For other processing errors.
    """
    start_time = datetime.datetime.now()
    logger.info(f"Building investors table - Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load data with optimized settings
        df = _load_and_validate_data(filepath)
        
        # Clean and standardize data
        df = _clean_data(df)
        logger.info(f"Data cleaning completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Apply fuzzy grouping for investor names
        df = utils.fuzzy_group_large_dataset(df, 'investor')
        logger.info(f"Fuzzy grouping completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prepare data for aggregation
        df = _prepare_aggregation_data(df)
        
        # Perform aggregations
        result_df = _perform_aggregations(df)
        logger.info(f"Aggregations completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate output paths and save
        output_paths = _save_results(result_df, filepath, timestamp)
        
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"Error in build_investors_table: {e}")
        raise


def _load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data with validation and type enforcement.
    
    Args:
        filepath: Path to CSV file.
        
    Returns:
        Validated DataFrame.
        
    Raises:
        pd.errors.EmptyDataError: If file is empty.
        KeyError: If required columns are missing.
    """
    try:
        df = pd.read_csv(
            filepath,
            usecols=constants.INVESTORS_MAPPING.keys(),
            dtype=constants.INVESTORS_MAPPING,
            low_memory=False  # Ensure consistent dtypes
        )
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Input file is empty: {filepath}")
    except KeyError as e:
        raise KeyError(f"Required columns missing from input file: {e}")
    
    if df.empty:
        raise pd.errors.EmptyDataError(f"No data loaded from file: {filepath}")
    
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input data.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    # Remove duplicates and invalid records in one operation
    initial_rows = len(df)
    df = df.drop_duplicates().dropna(subset=constants.REQUIRED_PROPERTY_COLUMNS)
    
    logger.info(f"Removed {initial_rows - len(df)} duplicate/invalid rows")
    
    if df.empty:
        raise ValueError("No valid data remaining after cleaning")
    
    return df


def _prepare_aggregation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for aggregation by formatting columns.
    
    Args:
        df: Input DataFrame after fuzzy grouping.
        
    Returns:
        DataFrame ready for aggregation.
    """
    # Format entity name and address columns
    df['Entity Name'] = df['standardized_investor']
    df['Investor Mailing Address'] = df['investor_mailingaddress'].str.title()
    
    # Convert date columns with vectorized operations
    for col in constants.DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[col] = df[col].dt.tz_localize(None)
    
    # Replace zeros with NA in price columns
    df[constants.PRICE_COLUMNS] = df[constants.PRICE_COLUMNS].replace(0, pd.NA)
    
    return df


def _perform_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all statistical aggregations for the investors table.
    
    Args:
        df: Prepared DataFrame.
        
    Returns:
        Aggregated results DataFrame.
    """
    grouped = df.groupby(constants.GROUP_COLUMNS)
    
    # Standard aggregations that can be computed efficiently
    standard_aggs = {
        'purchasedate': ['max'],
        'investor': ['count'],
        'purchaseprice': ['mean'],
        'resaleprice': ['mean'],
        'investor_city': [lambda x: utils.calculate_mostfrequentvalue(x)],
        'investor_state': [lambda x: utils.calculate_mostfrequentvalue(x)],
        'investor_zip': [lambda x: utils.calculate_mostfrequentvalue(x)]
    }
    
    # Calculate standard aggregations
    result_df = grouped.agg(standard_aggs)
    
    # Flatten column names
    result_df.columns = [col[0] if col[1] == '<lambda>' else f"{col[0]}_{col[1]}" 
                        for col in result_df.columns]
    
    # Add custom calculations
    custom_calculations = {
        'pfv_ratio': utils.calculate_pfv_ratio,
        'listsold_ratio': utils.calculate_listsold_ratio,
        'pm_ratio': utils.calculate_pm_ratio,
        'purchaseresale': utils.calculate_purchaseresale,
        'financing': utils.calculate_financing,
        'agentrel': utils.calculate_agentrel,
        'lastpropertypurchase': utils.calculate_lastpropertypurchase
    }
    
    # Apply custom calculations
    for col_name, func in custom_calculations.items():
        try:
            result_df[col_name] = grouped.apply(func, include_groups=False)
        except Exception as e:
            logger.warning(f"Failed to calculate {col_name}: {e}")
            result_df[col_name] = pd.NA
    
    # Reset index and prepare final structure
    result_df = result_df.reset_index()
    
    # Reorder columns according to mapping
    try:
        result_df.columns = [
            constants.INVESTORS_INDEX_MAPPING[i] 
            for i in range(len(result_df.columns))
        ]
    except KeyError as e:
        logger.error(f"Column mapping error: {e}")
        # Fallback: use existing column names
        logger.warning("Using original column names due to mapping error")
    
    return result_df


def _save_results(
    result_df: pd.DataFrame, 
    original_filepath: str, 
    timestamp: str
) -> Tuple[str, str]:
    """
    Save the results to local file and generate S3 path.
    
    Args:
        result_df: Results DataFrame to save.
        original_filepath: Original input file path.
        timestamp: Timestamp for output filename.
        
    Returns:
        Tuple of (local_filepath, s3_filepath).
    """
    # Generate output paths
    filename = f'investors_{timestamp}.csv'
    output_dir = os.path.dirname(original_filepath)
    local_filepath = os.path.join(output_dir, filename)
    s3_filepath = f'investors/{filename}'
    
    # Save results
    try:
        result_df.to_csv(local_filepath, index=False)
        logger.info(f"Results saved to: {local_filepath}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
    
    return (local_filepath, s3_filepath)