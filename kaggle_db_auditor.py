import sqlite3
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def audit_database(db_path='indian_stocks_data.db', output_excel='Database_Audit_Report.xlsx'):
    logging.info(f"Connecting to database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logging.error(f"Failed to connect to DB: {e}")
        return

    # Load data
    logging.info("Loading tables...")
    try:
        fundamentals_df = pd.read_sql("SELECT * FROM fundamentals", conn)
        prices_df = pd.read_sql("SELECT * FROM historical_prices", conn)
        sentiments_df = pd.read_sql("SELECT * FROM document_sentiments", conn)
    except Exception as e:
        logging.error(f"Error loading tables (they might not exist yet): {e}")
        conn.close()
        return

    conn.close()

    # --- 1. Data Coverage by Stock and Year ---
    logging.info("Calculating Detailed Data Coverage...")

    # Process Prices
    prices_df['Event_Date'] = pd.to_datetime(prices_df['Event_Date'], errors='coerce')
    prices_df['Year'] = prices_df['Event_Date'].dt.year
    price_coverage = prices_df.groupby(['Symbol', 'Year']).size().reset_index(name='Trading_Days_Count')

    # Process Documents
    sentiments_df['Event_Date'] = pd.to_datetime(sentiments_df['Event_Date'], errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
    doc_coverage = sentiments_df.groupby(['Symbol', 'Year', 'Doc_Type']).size().unstack(fill_value=0).reset_index()
    # Ensure expected columns exist
    for col in ['news', 'annual report', 'media presentation']:
        if col not in doc_coverage.columns:
            doc_coverage[col] = 0

    doc_coverage.rename(columns={
        'news': 'News_Count',
        'annual report': 'Annual_Reports_Count',
        'media presentation': 'Media_Presentations_Count'
    }, inplace=True)

    # Process Fundamentals (Usually one per stock, but checking event_date year if available)
    fundamentals_df['Event_Date_Parsed'] = pd.to_datetime(fundamentals_df['Event_Date'], errors='coerce')
    fundamentals_df['Year'] = fundamentals_df['Event_Date_Parsed'].dt.year
    # For fundamentals where Year couldn't be parsed, assign a default or drop for yearly grouping
    fund_coverage = fundamentals_df[fundamentals_df['Year'].notnull()].groupby(['Symbol', 'Year']).size().reset_index(name='Fundamentals_Count')

    # Merge Coverages
    coverage_report = pd.merge(price_coverage, doc_coverage, on=['Symbol', 'Year'], how='outer')
    coverage_report = pd.merge(coverage_report, fund_coverage, on=['Symbol', 'Year'], how='outer')
    coverage_report.fillna(0, inplace=True)

    # Convert counts to integers
    for col in ['Trading_Days_Count', 'News_Count', 'Annual_Reports_Count', 'Media_Presentations_Count', 'Fundamentals_Count']:
        coverage_report[col] = coverage_report[col].astype(int)

    coverage_report.sort_values(by=['Symbol', 'Year'], inplace=True)


    # --- 2. Point-in-Time & Date Consistency Checks ---
    logging.info("Running Point-in-Time & Date Consistency Checks...")
    anomalies = []
    current_date = datetime.now()

    # A. Sentiments Consistency
    # Check if the extracted 'Year' matches the actual year of the 'Event_Date'
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
    audit_database()
