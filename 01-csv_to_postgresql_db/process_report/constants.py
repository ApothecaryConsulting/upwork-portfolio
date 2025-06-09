"""
Constants and mappings for real estate transaction data processing.

This module contains data type mappings for CSV columns and index mappings
for output report columns. These constants ensure consistent data processing
and standardized report formats across different entity types.

The mappings define:
- Data types for proper CSV parsing and memory optimization
- Column name standardization for output reports
- Consistent ordering of report columns
"""

from typing import Dict, Union

# Data type mappings for CSV parsing
# Using appropriate types for memory efficiency and data integrity

agents_mapping: Dict[str, Union[type, str]] = {
    "property_streetnumber": str,
    "property_streetname": str,
    "property_city": str,
    "property_state": str,
    "last_listing_date": str,
    "last_listing_price": float,
    "investor": str,
    "lender": str,
    "investor_mailingaddress": str,
    "purchasedate": str,
    "purchaseprice": float,
    "listagentname": str,
    "buyagentname": str,
    "doubleend": str,
    "re_listagentname": str,
    "resaledate": str,
    "resaleprice": float,
}

offices_mapping: Dict[str, Union[type, str]] = {
    "property_streetnumber": str,
    "property_streetname": str,
    "property_city": str,
    "property_state": str,
    "last_listing_date": str,
    "last_listing_price": float,
    "investor": str,
    "lender": str,
    "investor_mailingaddress": str,
    "purchasedate": str,
    "purchaseprice": float,
    "listagentname": str,
    "listagentoffice": str,
    "buyagentname": str,
    "buyagentoffice": str,
    "doubleend": str,
    "re_listagentname": str,
    "re_listagentoffice": str,
    "resaledate": str,
    "resaleprice": float,
}

lenders_mapping: Dict[str, Union[type, str]] = {
    "investor": str,
    "lender": str,
    "loan_amount": float,
    "purchasedate": str,
    "purchaseprice": float,
    "resaledate": str,
    "resaleprice": float,
}

titlecompanies_mapping: Dict[str, Union[type, str]] = {
    "investor": str,
    "title_company": str,
    "lender": str,
    "listagentname": str,
    "listagentoffice": str,
    "buyagentname": str,
    "buyagentoffice": str,
    "re_listagentname": str,
    "re_listagentoffice": str,
}

# Output column mappings for standardized report formats
# These ensure consistent column names and ordering across reports

agents_index_mapping: Dict[int, str] = {
    0: "Agent Name",
    1: "Last Purchase",
    2: "Total Transactions",
    3: "Average Purchase",
    4: "Average Resale",
    5: "Purchase/Resale",
    6: "Last Property Purchase",
    7: "Listings Sold to Investors",
    8: "% Double-end Trans.",
    9: "Most Recent Buyer's Agent",
    10: "Listings Re-Sold for Investors",
    11: "Unique Investor Relationships",
}

offices_index_mapping: Dict[int, str] = {
    0: "Office Name",
    1: "Last Purchase",
    2: "Total Transactions",
    3: "Average Purchase",
    4: "Average Resale",
    5: "Purchase/Resale",
    6: "Last Property Purchase",
    7: "Listings Sold to Investors",
    8: "% Double-end Trans.",
    9: "Most Recent Buyer's Office",
    10: "Listings Re-Sold for Investors",
    11: "Unique Investor Relationships",
}

lenders_index_mapping: Dict[int, str] = {
    0: "Lender Name",
    1: "Average Loan Amount",
    2: "Average Loan/Purchase Ratio",
    3: "Total Transactions to Investors",
    4: "Unique Investor Relationships",
    5: "Average Loans Per Relationship",
}

titlecompanies_index_mapping: Dict[int, str] = {
    0: "Title Company",
    1: "Total Units",
    2: "Market Share",
    3: "Unique Offices",
    4: "Unique Agents",
    5: "Unique Investors",
}