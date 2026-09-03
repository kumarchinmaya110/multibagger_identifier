import sqlite3
import pandas as pd
import logging
from datetime import datetime
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def audit_database(db_path='indian_stocks_data.db', output_excel='Database_Audit_Report.xlsx'):
    if not os.path.exists(db_path):
        logging.error(f"Database not found at {db_path}. Please check the path.")
        return

    logging.info(f"Connecting to database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logging.error(f"Failed to connect to DB: {e}")
        return

    # Load data gracefully
    logging.info("Loading tables...")
    def load_table(table_name, expected_columns):
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            return df
        except Exception as e:
            logging.warning(f"Table '{table_name}' missing or failed to load. Initializing empty dataframe. Error: {e}")
            return pd.DataFrame(columns=expected_columns)

    fundamentals_df = load_table("fundamentals", ['Symbol', 'Event_Date', 'MarketCap', 'PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Sector', 'Industry'])
    prices_df = load_table("historical_prices", ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Symbol', 'Event_Date'])
    sentiments_df = load_table("document_sentiments", ['Symbol', 'Year', 'Event_Date', 'Source', 'Doc_Type', 'Vader_Compound', 'Finbert_Label', 'Finbert_Score'])

    conn.close()

    # --- 1. Data Coverage by Stock and Year ---
    logging.info("Calculating Detailed Data Coverage...")

    # Process Prices
    prices_df['Event_Date'] = pd.to_datetime(prices_df['Event_Date'], errors='coerce')
    prices_df['Year'] = prices_df['Event_Date'].dt.year
    if not prices_df.empty:
        price_coverage = prices_df.groupby(['Symbol', 'Year']).size().reset_index(name='Trading_Days_Count')
    else:
        price_coverage = pd.DataFrame(columns=['Symbol', 'Year', 'Trading_Days_Count'])

    # Process Documents
    sentiments_df['Event_Date'] = pd.to_datetime(sentiments_df['Event_Date'], errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
    if not sentiments_df.empty:
        doc_coverage = sentiments_df.groupby(['Symbol', 'Year', 'Doc_Type']).size().unstack(fill_value=0).reset_index()
    else:
        doc_coverage = pd.DataFrame(columns=['Symbol', 'Year'])

    # Ensure expected columns exist
    for col in ['news', 'annual report', 'media presentation']:
        if col not in doc_coverage.columns:
            doc_coverage[col] = 0

    doc_coverage.rename(columns={
        'news': 'News_Count',
        'annual report': 'Annual_Reports_Count',
        'media presentation': 'Media_Presentations_Count'
    }, inplace=True)

    # Process Fundamentals
    fundamentals_df['Event_Date_Parsed'] = pd.to_datetime(fundamentals_df['Event_Date'], errors='coerce')
    fundamentals_df['Year'] = fundamentals_df['Event_Date_Parsed'].dt.year
    if not fundamentals_df.empty:
        fund_coverage = fundamentals_df[fundamentals_df['Year'].notnull()].groupby(['Symbol', 'Year']).size().reset_index(name='Fundamentals_Count')
    else:
        fund_coverage = pd.DataFrame(columns=['Symbol', 'Year', 'Fundamentals_Count'])

    # Merge Coverages
    if price_coverage.empty and doc_coverage.empty and fund_coverage.empty:
         coverage_report = pd.DataFrame(columns=['Symbol', 'Year', 'Trading_Days_Count', 'News_Count', 'Annual_Reports_Count', 'Media_Presentations_Count', 'Fundamentals_Count'])
    else:
        coverage_report = pd.merge(price_coverage, doc_coverage, on=['Symbol', 'Year'], how='outer')
        coverage_report = pd.merge(coverage_report, fund_coverage, on=['Symbol', 'Year'], how='outer')
        coverage_report.fillna(0, inplace=True)

        # Convert counts to integers
        for col in ['Trading_Days_Count', 'News_Count', 'Annual_Reports_Count', 'Media_Presentations_Count', 'Fundamentals_Count']:
            if col in coverage_report.columns:
                coverage_report[col] = coverage_report[col].astype(int)

        coverage_report.sort_values(by=['Symbol', 'Year'], inplace=True)


    # --- 2. Point-in-Time & Date Consistency Checks ---
    logging.info("Running Point-in-Time & Date Consistency Checks...")
    anomalies = []
    current_date = datetime.now()

    # A. Sentiments Consistency
    for idx, row in sentiments_df.iterrows():
        event_date = row['Event_Date']
        labeled_year = row['Year']
        symbol = row['Symbol']

        if pd.isnull(event_date):
            anomalies.append({'Symbol': symbol, 'Table': 'document_sentiments', 'Issue': 'Missing Event_Date', 'Details': f"Labeled Year: {labeled_year}"})
            continue

        actual_year = event_date.year

        # 1. Year Mismatch
        if actual_year != labeled_year:
            anomalies.append({
                'Symbol': symbol,
                'Table': 'document_sentiments',
                'Issue': 'Year Mismatch (Look-ahead/Look-behind bias)',
                'Details': f"Data for year {labeled_year} has Event_Date {event_date.strftime('%Y-%m-%d')}"
            })

        # 2. Future Leakage
        if event_date > current_date:
            anomalies.append({
                'Symbol': symbol,
                'Table': 'document_sentiments',
                'Issue': 'Future Data Leakage',
                'Details': f"Event_Date {event_date.strftime('%Y-%m-%d')} is in the future."
            })

    # B. Prices Consistency
    for idx, row in prices_df.iterrows():
        event_date = row['Event_Date']
        symbol = row['Symbol']
        if pd.isnull(event_date):
            anomalies.append({'Symbol': symbol, 'Table': 'historical_prices', 'Issue': 'Missing Event_Date', 'Details': 'Price record missing date.'})
            continue

        if event_date > current_date:
            anomalies.append({
                'Symbol': symbol,
                'Table': 'historical_prices',
                'Issue': 'Future Data Leakage',
                'Details': f"Event_Date {event_date.strftime('%Y-%m-%d')} is in the future."
            })

    anomalies_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame(columns=['Symbol', 'Table', 'Issue', 'Details'])


    # --- 3. Save to Excel ---
    logging.info(f"Saving reports to {output_excel}...")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        coverage_report.to_excel(writer, sheet_name='Detailed_Coverage', index=False)
        anomalies_df.to_excel(writer, sheet_name='Consistency_Anomalies', index=False)

    logging.info("Audit Complete!")
    logging.info(f"Found {len(anomalies_df)} anomalies.")
    logging.info(f"Processed {len(coverage_report)} Symbol/Year coverage blocks.")

if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    import os

    parser = argparse.ArgumentParser(description="Audit Indian Stocks Database")
    parser.add_argument('--db', type=str, default='user_db.db', help='Path to the SQLite database')
    parser.add_argument('--output', type=str, default='Database_Audit_Report.xlsx', help='Output Excel path')
    parser.add_argument('--download_gdrive_id', type=str, default='1aCi1veb7a3lKWZBmcRmBZ2cd-6HodJEZ', help='Google Drive File ID to download the DB from before auditing')
    args, unknown = parser.parse_known_args()

    if args.download_gdrive_id:
        logging.info(f"Downloading database from Google Drive ID: {args.download_gdrive_id}...")
        try:
            import gdown
        except ImportError:
            logging.info("Installing gdown...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f"https://drive.google.com/uc?id={args.download_gdrive_id}"
        gdown.download(url, args.db, quiet=False)

    audit_database(db_path=args.db, output_excel=args.output)
