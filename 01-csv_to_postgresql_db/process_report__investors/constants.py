"""
Configuration constants for investor data processing.

This module contains data type mappings and column mappings used throughout
the investor data processing pipeline.
"""

# Data type mapping for investor CSV columns
INVESTORS_MAPPING = {
    "property_streetnumber": str,
    "property_streetname": str,
    "property_city": str,
    "property_state": str,
    "county": str,
    "last_listing_date": str,
    "last_listing_price": float,
    "investor": str,
    "lender": str,
    "investor_mailingaddress": str,
    "investor_city": str,
    "investor_state": str,
    "investor_zip": str,
    "purchasedate": str,
    "purchaseprice": float,
    "listagentname": str,
    "buyagentname": str,
    "re_listagentname": str,
    "resaledate": str,
    "resaleprice": float,
}

# Column index to display name mapping for final output
INVESTORS_INDEX_MAPPING = {
    0: "Entity Name",
    1: "Investor Mailing Address",
    2: "Last Purchase",
    3: "Total Transactions",
    4: "Average Purchase",
    5: "Average Resale",
    6: "City",
    7: "State",
    8: "ZIP Code",
    9: "P/FV Ratio",
    10: "List/Sold Ratio",
    11: "P/M Ratio",
    12: "Purchase Resale",
    13: "Financing",
    14: "Agent Relationships",
    15: "Last Property Purchase",
}

# Date columns that need datetime conversion
DATE_COLUMNS = ['last_listing_date', 'purchasedate', 'resaledate']

# Price columns that should have zeros replaced with NA
PRICE_COLUMNS = ['last_listing_price', 'purchaseprice', 'resaleprice']

# Required columns for property identification
REQUIRED_PROPERTY_COLUMNS = ['property_streetnumber', 'property_streetname', 'property_city']

# Grouping columns for investor aggregation
GROUP_COLUMNS = ['Entity Name', 'Investor Mailing Address']