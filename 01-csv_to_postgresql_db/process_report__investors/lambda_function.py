"""
AWS Lambda function for processing investor data files.

This Lambda function is triggered by S3 events and processes uploaded CSV files
to generate investor analysis reports. It handles file download, processing,
and uploading results back to S3.
"""

import json
import os
import urllib.parse
import logging
from typing import Dict, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import process_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing S3 file upload events.
    
    This function is triggered when files are uploaded to the monitored S3 bucket.
    It downloads the file, processes it through the data pipeline, and uploads
    the results back to S3.
    
    Args:
        event: AWS Lambda event object containing S3 event details.
        context: AWS Lambda context object with runtime information.
        
    Returns:
        Dictionary containing status code and message.
        
    Raises:
        Exception: Re-raises any processing exceptions after logging.
    """
    logger.info(f"Received event: {json.dumps(event, indent=2)}")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise
    
    try:
        # Extract S3 event details
        bucket, key = _extract_s3_event_details(event)
        logger.info(f"Processing file: s3://{bucket}/{key}")
        
        # Download file from S3
        local_filepath = _download_file_from_s3(s3_client, bucket, key)
        
        # Process the data
        output_filepaths = process_data.process_from_filepath(local_filepath)
        
        # Upload results back to S3
        _upload_results_to_s3(s3_client, bucket, output_filepaths)
        
        # Clean up temporary files
        _cleanup_temp_files([local_filepath] + [fp[0] for fp in output_filepaths])
        
        logger.info("Processing completed successfully")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Report processed successfully",
                "input_file": key,
                "output_files": [fp[1] for fp in output_filepaths]
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing file {key} from bucket {bucket}: {e}")
        raise


def _extract_s3_event_details(event: Dict[str, Any]) -> tuple[str, str]:
    """
    Extract bucket name and object key from S3 event.
    
    Args:
        event: AWS Lambda event object.
        
    Returns:
        Tuple of (bucket_name, object_key).
        
    Raises:
        KeyError: If event structure is invalid.
        ValueError: If event contains no records.
    """
    try:
        if 'Records' not in event or not event['Records']:
            raise ValueError("Event contains no S3 records")
        
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(
            record['s3']['object']['key'], 
            encoding='utf-8'
        )
        
        return bucket, key
        
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid event structure: {e}")
        raise KeyError("Invalid S3 event structure") from e


def _download_file_from_s3(s3_client, bucket: str, key: str) -> str:
    """
    Download file from S3 to local temporary storage.
    
    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        key: S3 object key.
        
    Returns:
        Local file path where file was downloaded.
        
    Raises:
        ClientError: If S3 download fails.
        OSError: If local file operations fail.
    """
    local_filepath = f'/tmp/{os.path.basename(key)}'
    
    try:
        s3_client.download_file(bucket, key, local_filepath)
        logger.info(f"Successfully downloaded s3://{bucket}/{key} to {local_filepath}")
        
        # Verify file was downloaded and has content
        if not os.path.exists(local_filepath):
            raise OSError(f"Downloaded file not found: {local_filepath}")
        
        file_size = os.path.getsize(local_filepath)
        if file_size == 0:
            raise OSError(f"Downloaded file is empty: {local_filepath}")
        
        logger.info(f"Downloaded file size: {file_size} bytes")
        return local_filepath
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"S3 download failed [{error_code}]: {error_message}")
        raise
    except OSError as e:
        logger.error(f"File system error during download: {e}")
        raise


def _upload_results_to_s3(
    s3_client, 
    bucket: str, 
    filepaths: list[tuple[str, str]]
) -> None:
    """
    Upload processed results back to S3.
    
    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        filepaths: List of (local_path, s3_path) tuples.
        
    Raises:
        ClientError: If S3 upload fails.
        OSError: If local file cannot be read.
    """
    for local_filepath, s3_filepath in filepaths:
        try:
            # Verify local file exists before upload
            if not os.path.exists(local_filepath):
                logger.error(f"Local file not found for upload: {local_filepath}")
                continue
            
            file_size = os.path.getsize(local_filepath)
            logger.info(f"Uploading {local_filepath} ({file_size} bytes) to s3://{bucket}/{s3_filepath}")
            
            s3_client.upload_file(local_filepath, bucket, s3_filepath)
            logger.info(f"Successfully uploaded {local_filepath} to s3://{bucket}/{s3_filepath}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"S3 upload failed [{error_code}]: {error_message}")
            raise
        except OSError as e:
            logger.error(f"File system error during upload: {e}")
            raise


def _cleanup_temp_files(filepaths: list[str]) -> None:
    """
    Clean up temporary files to free Lambda storage.
    
    Args:
        filepaths: List of file paths to delete.
    """
    for filepath in filepaths:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
        except OSError as e:
            logger.warning(f"Failed to clean up {filepath}: {e}")


# Additional utility functions for enhanced functionality

def _validate_file_type(key: str) -> bool:
    """
    Validate that the uploaded file is a supported type.
    
    Args:
        key: S3 object key (filename).
        
    Returns:
        True if file type is supported.
    """
    supported_extensions = ['.csv', '.CSV']
    return any(key.endswith(ext) for ext in supported_extensions)


def _get_file_metadata(s3_client, bucket: str, key: str) -> Dict[str, Any]:
    """
    Retrieve metadata about the S3 object.
    
    Args:
        s3_client: Boto3 S3 client instance.
        bucket: S3 bucket name.
        key: S3 object key.
        
    Returns:
        Dictionary containing file metadata.
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return {
            'size': response.get('ContentLength', 0),
            'last_modified': response.get('LastModified'),
            'content_type': response.get('ContentType', 'unknown'),
            'etag': response.get('ETag', '').strip('"')
        }
    except ClientError as e:
        logger.warning(f"Could not retrieve metadata for {key}: {e}")
        return {}


def lambda_handler_with_validation(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Enhanced Lambda handler with additional validation and error handling.
    
    This version includes file type validation and metadata retrieval
    for more robust processing.
    
    Args:
        event: AWS Lambda event object containing S3 event details.
        context: AWS Lambda context object with runtime information.
        
    Returns:
        Dictionary containing detailed status and processing information.
    """
    logger.info(f"Enhanced handler received event: {json.dumps(event, indent=2)}")
    
    try:
        s3_client = boto3.client('s3')
        bucket, key = _extract_s3_event_details(event)
        
        # Validate file type
        if not _validate_file_type(key):
            logger.warning(f"Unsupported file type: {key}")
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "message": "Unsupported file type. Only CSV files are supported.",
                    "file": key
                })
            }
        
        # Get file metadata
        metadata = _get_file_metadata(s3_client, bucket, key)
        logger.info(f"File metadata: {metadata}")
        
        # Proceed with standard processing
        return lambda_handler(event, context)
        
    except Exception as e:
        logger.error(f"Enhanced handler error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": f"Processing failed: {str(e)}",
                "file": key if 'key' in locals() else "unknown"
            })
        }