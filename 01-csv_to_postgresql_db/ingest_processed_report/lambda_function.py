"""
AWS Lambda function for processing CSV files from S3 and loading them into Aurora PostgreSQL.

This module handles S3 PUT events, downloads CSV files, infers column types,
and efficiently loads data into PostgreSQL tables with automatic schema creation.

Author: AWS Lambda Handler
Version: 2.0
"""

import json
import boto3
import csv
import os
import re
import logging
import time
from io import StringIO
from typing import List, Tuple, Optional, Any, Dict
from database import PostgresDatabase

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global connection management
db_connection: Optional[PostgresDatabase] = None
last_conn_time: float = 0
CONN_TIMEOUT: int = 300  # 5 minutes in seconds
BATCH_SIZE: int = 1000  # Optimized batch size for Aurora
MAX_TYPE_SAMPLES: int = 5  # Number of rows to sample for type inference


def get_database_connection(
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    db_port: int
) -> PostgresDatabase:
    """
    Create or reuse a database connection with connection pooling.
    
    Implements connection reuse to reduce overhead in Lambda cold starts
    and connection establishment latency.
    
    Args:
        db_host: Database hostname
        db_name: Database name
        db_user: Database username
        db_password: Database password
        db_port: Database port number
        
    Returns:
        PostgresDatabase: Active database connection instance
        
    Raises:
        Exception: If database connection fails
    """
    global db_connection, last_conn_time
    
    current_time = time.time()
    
    # Reuse existing connection if valid and within timeout
    if (db_connection is not None and 
        current_time - last_conn_time < CONN_TIMEOUT):
        try:
            # Test connection with lightweight query
            db_connection._connection.cursor().execute('SELECT 1')
            logger.info("Reusing existing database connection")
            return db_connection
        except Exception as e:
            logger.info(f"Connection expired, creating new: {e}")
    
    # Create new connection
    try:
        logger.info("Establishing new database connection")
        new_connection = PostgresDatabase(
            db_name=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        
        db_connection = new_connection
        last_conn_time = current_time
        logger.info("Successfully established PostgreSQL connection")
        return db_connection
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def infer_column_type(value: Any, column_name: str) -> str:
    """
    Infer PostgreSQL column type from sample value and column name.
    
    Uses conservative type inference to prevent data type mismatches.
    Prioritizes TEXT type for ambiguous cases to ensure data integrity.
    
    Args:
        value: Sample value to analyze
        column_name: Name of the column (used for ID detection)
        
    Returns:
        str: PostgreSQL column type (TEXT, INTEGER, DECIMAL, BOOLEAN, DATE, TIMESTAMP)
    """
    if value is None or value == '':
        return 'TEXT'
    
    value_str = str(value).strip()
    
    # ID column detection with integer validation
    if column_name.lower().endswith('id') or column_name.lower() == 'id':
        try:
            int(value_str)
            return 'INTEGER'
        except ValueError:
            return 'TEXT'
    
    # Boolean detection (case-insensitive)
    boolean_values = {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n'}
    if value_str.lower() in boolean_values:
        return 'BOOLEAN'
    
    # Numeric type detection
    try:
        int(value_str)
        return 'INTEGER'
    except ValueError:
        try:
            float(value_str)
            return 'DECIMAL'
        except ValueError:
            pass
    
    # Date pattern matching (YYYY-MM-DD)
    if re.match(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$', value_str):
        return 'DATE'
    
    # Timestamp pattern matching (ISO format)
    if re.match(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}', 
                value_str):
        return 'TIMESTAMP'
    
    return 'TEXT'


def validate_column_types(headers: List[str], sample_rows: List[List[str]]) -> List[str]:
    """
    Determine column types using multiple sample rows for accuracy.
    
    Analyzes up to MAX_TYPE_SAMPLES rows to infer the most appropriate
    PostgreSQL data type for each column. Uses conservative approach
    to prevent type casting errors.
    
    Args:
        headers: Column names from CSV header
        sample_rows: Sample data rows for type analysis
        
    Returns:
        List[str]: PostgreSQL column types for each column
    """
    if not sample_rows:
        return ['TEXT'] * len(headers)
    
    column_types = []
    max_samples = min(MAX_TYPE_SAMPLES, len(sample_rows))
    
    for col_idx, col_name in enumerate(headers):
        # Extract sample values for current column
        sample_values = [
            row[col_idx] if col_idx < len(row) else ''
            for row in sample_rows[:max_samples]
        ]
        
        # Use TEXT if any sample suggests it (conservative approach)
        non_empty_values = [val for val in sample_values if val and str(val).strip()]
        
        if not non_empty_values:
            column_types.append('TEXT')
            continue
            
        # Check if any value requires TEXT type
        requires_text = any(
            infer_column_type(val, col_name) == 'TEXT'
            for val in non_empty_values
        )
        
        if requires_text:
            column_types.append('TEXT')
        else:
            # Use type from first non-empty value
            column_types.append(infer_column_type(non_empty_values[0], col_name))
    
    return column_types


def parse_csv_from_s3(bucket: str, key: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Download and parse CSV file from S3 with optimized memory usage.
    
    Downloads CSV content, parses headers and data, and performs
    type inference on sample rows for efficient database operations.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Tuple containing:
            - headers: List of column names
            - column_types: List of inferred PostgreSQL types
            - all_rows: All data rows from CSV
            
    Raises:
        Exception: If S3 download or CSV parsing fails
    """
    s3_client = boto3.client('s3')
    
    # Download CSV from S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        logger.info("Successfully downloaded CSV from S3")
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        raise
    
    # Parse CSV with efficient memory usage
    try:
        csv_file = StringIO(csv_content)
        csv_reader = csv.reader(csv_file)
        
        # Extract and clean headers
        headers = [h.strip() for h in next(csv_reader)]
        
        # Collect sample rows for type inference
        sample_rows = []
        for _ in range(MAX_TYPE_SAMPLES):
            try:
                sample_rows.append(next(csv_reader))
            except StopIteration:
                break
        
        # Reset and read all rows
        csv_file.seek(0)
        next(csv_reader)  # Skip header
        all_rows = list(csv_reader)
        
        # Infer column types
        column_types = validate_column_types(headers, sample_rows)
        
        logger.info(f"Parsed CSV: {len(all_rows)} rows, {len(headers)} columns")
        
        # Log column type inference for debugging
        for header, col_type in zip(headers, column_types):
            logger.info(f"Column '{header}': {col_type}")
            
        return headers, column_types, all_rows
        
    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        raise


def create_or_clear_table(
    cursor: Any,
    table_name: str,
    headers: List[str],
    column_types: List[str]
) -> None:
    """
    Create new table or clear existing table data.
    
    Checks for table existence and either creates a new table with
    inferred schema or truncates existing table for fresh data load.
    
    Args:
        cursor: Database cursor for executing SQL
        table_name: Target table name
        headers: Column names
        column_types: PostgreSQL column types
    """
    # Check table existence
    check_table_sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
    """
    
    cursor.execute(check_table_sql, (table_name,))
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        # Create table with inferred schema
        columns = [f'"{header}" {col_type}' 
                  for header, col_type in zip(headers, column_types)]
        create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns)});"
        cursor.execute(create_table_sql)
        logger.info(f"Created table: {table_name}")
    else:
        # Clear existing data efficiently
        cursor.execute(f"TRUNCATE TABLE {table_name};")
        logger.info(f"Cleared table: {table_name}")


def insert_data_in_batches(
    cursor: Any,
    table_name: str,
    headers: List[str],
    all_rows: List[List[str]]
) -> int:
    """
    Insert CSV data into PostgreSQL table using optimized batching.
    
    Processes data in batches to balance memory usage and performance.
    Handles NULL value conversion and provides progress logging.
    
    Args:
        cursor: Database cursor for executing SQL
        table_name: Target table name
        headers: Column names for INSERT statement
        all_rows: All data rows to insert
        
    Returns:
        int: Number of rows successfully inserted
    """
    if not all_rows:
        logger.info("No data rows to insert")
        return 0
    
    # Prepare INSERT statement
    quoted_columns = [f'"{header}"' for header in headers]
    placeholders = ["%s"] * len(headers)
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(quoted_columns)})
        VALUES ({', '.join(placeholders)})
    """
    
    row_count = len(all_rows)
    
    # Process in optimized batches
    for i in range(0, row_count, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, row_count)
        batch_rows = all_rows[i:batch_end]
        
        # Convert empty strings to NULL for proper database handling
        processed_batch = [
            [None if cell == '' else cell for cell in row]
            for row in batch_rows
        ]
        
        cursor.executemany(insert_sql, processed_batch)
        logger.info(f"Inserted batch {i//BATCH_SIZE + 1}: rows {i+1}-{batch_end}")
    
    return row_count


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing S3 CSV uploads to PostgreSQL.
    
    Triggered by S3 PUT events, this function downloads CSV files,
    infers schema, creates/updates PostgreSQL tables, and loads data
    efficiently using batch processing.
    
    Args:
        event: AWS Lambda event containing S3 trigger information
        context: AWS Lambda context object
        
    Returns:
        Dict containing HTTP status code and response body
        
    Environment Variables Required:
        - DB_HOST: PostgreSQL hostname
        - DB_NAME: Database name
        - DB_USER: Database username
        - DB_PASSWORD: Database password
        - DB_PORT: Database port (optional, defaults to 5432)
    """
    logger.info(f"Processing Lambda event: {json.dumps(event)}")
    
    # Extract S3 event information
    try:
        s3_record = event['Records'][0]['s3']
        bucket = s3_record['bucket']['name']
        key = s3_record['object']['key']
        table_name = os.path.split(key)[0] or 'default_table'
        
        logger.info(f"Processing: {key} from bucket: {bucket} -> table: {table_name}")
        
    except (KeyError, IndexError) as e:
        error_msg = f"Invalid S3 event structure: {e}"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps(error_msg)
        }
    
    # Get database configuration from environment
    try:
        db_config = {
            'db_host': os.environ['DB_HOST'],
            'db_name': os.environ['DB_NAME'],
            'db_user': os.environ['DB_USER'],
            'db_password': os.environ['DB_PASSWORD'],
            'db_port': int(os.environ.get('DB_PORT', 5432))
        }
        logger.info(f"Database config: {db_config['db_host']}:{db_config['db_port']}")
        
    except KeyError as e:
        error_msg = f"Missing environment variable: {e}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps(error_msg)
        }
    
    try:
        # Parse CSV from S3
        headers, column_types, all_rows = parse_csv_from_s3(bucket, key)
        
        # Get database connection
        db = get_database_connection(**db_config)
        
        # Execute database operations in transaction
        with db._connection as conn:
            cursor = conn.cursor()
            
            # Create/clear table
            create_or_clear_table(cursor, table_name, headers, column_types)
            
            # Insert data in batches
            rows_inserted = insert_data_in_batches(cursor, table_name, headers, all_rows)
            
            # Commit transaction
            conn.commit()
            
        success_msg = f"Successfully processed {key}: {rows_inserted} rows -> {table_name}"
        logger.info(success_msg)
        
        return {
            'statusCode': 200,
            'body': json.dumps(success_msg)
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps(error_msg)
        }