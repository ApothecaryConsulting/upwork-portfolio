# CSV to PostgreSQL Database ETL Pipeline

This project implements an automated ETL (Extract, Transform, Load) pipeline that processes CSV reports and loads the data into a PostgreSQL database using AWS Lambda functions and S3 triggers.

## Architecture Overview

The pipeline consists of the following workflow:

1. **Data Ingestion**: Upload `report.csv` to S3 bucket triggers the initial processing
2. **Data Processing**: Multiple Lambda functions process the raw data into separate entity files
3. **Data Loading**: Processed CSV files are automatically ingested into PostgreSQL database tables

## Workflow Steps

1. Add `report.csv` to S3 bucket. This PUT event triggers the `process_report__investors` Lambda function
2. `process_report__investors` Lambda executes and generates `investors.csv` in the S3 bucket
3. `process_report` Lambda executes and creates multiple output files:
   - `agents.csv`
   - `offices.csv` 
   - `lenders.csv`
   - `titlecompanies.csv`
4. Any PUT event from the processing Lambda functions triggers the `ingest_processed_report` Lambda function
5. `ingest_processed_report` Lambda loads all processed data into the PostgreSQL database tables

## Technologies Used

- AWS Lambda (Serverless compute)
- Amazon S3 (File storage and event triggers)
- PostgreSQL (Database)
- Python (Lambda runtime)