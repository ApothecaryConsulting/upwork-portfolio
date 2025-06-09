"""
AWS Lambda function for processing real estate transaction data.

This module handles S3-triggered events to process uploaded CSV files containing
real estate transaction data. It generates various analytical reports based on
configurable flags for different entity types (agents, offices, lenders, etc.).

Environment Variables:
    INVESTORS_FLAG: Enable investor report generation (default: False)
    AGENTS_FLAG: Enable agent report generation (default: False)
    OFFICES_FLAG: Enable office report generation (default: False)
    LENDERS_FLAG: Enable lender report generation (default: False)
    TITLECOMPANIES_FLAG: Enable title company report generation (default: False)
"""

import json
import os
import urllib.parse
from typing import Dict, Any

import boto3

import process_data

# Configuration flags from environment variables
INVESTORS_FLAG = os.getenv("INVESTORS_FLAG", 'False').lower() in ('true', '1', 't')
AGENTS_FLAG = os.getenv("AGENTS_FLAG", 'False').lower() in ('true', '1', 't')
OFFICES_FLAG = os.getenv("OFFICES_FLAG", 'False').lower() in ('true', '1', 't')
LENDERS_FLAG = os.getenv("LENDERS_FLAG", 'False').lower() in ('true', '1', 't')
TITLECOMPANIES_FLAG = os.getenv("TITLECOMPANIES_FLAG", 'False').lower() in ('true', '1', 't')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing real estate transaction data files.
    
    Triggered by S3 upload events, downloads the file, processes it according to
    enabled flags, and uploads the generated reports back to S3.
    
    Args:
        event: AWS Lambda event containing S3 trigger information
        context: AWS Lambda context object
        
    Returns:
        Dict containing status code and message
        
    Raises:
        Exception: If file processing or S3 operations fail
    """
    print(f"Received event: {json.dumps(event, indent=2)}")

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Extract S3 bucket and object information from event
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(
            event['Records'][0]['s3']['object']['key'], 
            encoding='utf-8'
        )
    except (KeyError, IndexError) as e:
        error_msg = f"Invalid S3 event structure: {e}"
        print(error_msg)
        raise ValueError(error_msg)

    try:
        # Create local file path for temporary storage
        local_filepath = f'/tmp/{os.path.basename(key)}'
            
        # Download file from S3
        s3_client.download_file(bucket, key, local_filepath)
        print(f"Successfully downloaded {key} from {bucket} to {local_filepath}")

        # Process the data based on enabled flags
        generated_files = process_data.process_from_filepath(
            local_filepath,
            agents=AGENTS_FLAG,
            offices=OFFICES_FLAG,
            lenders=LENDERS_FLAG,
            titlecompanies=TITLECOMPANIES_FLAG,
        )

        # Upload generated files back to S3
        for local_path, s3_path in generated_files:
            s3_client.upload_file(local_path, bucket, s3_path)
            print(f"Successfully uploaded {local_path} to {bucket}/{s3_path}")
            
            # Clean up local file
            if os.path.exists(local_path):
                os.remove(local_path)

        # Clean up downloaded file
        if os.path.exists(local_filepath):
            os.remove(local_filepath)

        return {
            "statusCode": 200,
            "message": f"Successfully processed {len(generated_files)} reports"
        }

    except Exception as e:
        error_msg = (
            f'Error processing object {key} from bucket {bucket}. '
            f'Ensure the file exists and the bucket is in the same region. Error: {e}'
        )
        print(error_msg)
        raise e