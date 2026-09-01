import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import requests
from bs4 import BeautifulSoup
import feedparser
import urllib.parse
from datetime import datetime, timedelta
import time
import logging
import warnings
import os
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndianStockPipeline:
    def __init__(self, db_name='indian_stocks_data.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)

        # Initialize NLP Models
        logging.info("Initializing NLP models...")
        # VADER for general sentiment
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        self.vader = SentimentIntensityAnalyzer()

        # FinBERT for financial sentiment
        # In a real Kaggle notebook, GPU (device=0) is recommended if available
        try:
            self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        except Exception as e:
            logging.warning(f"Failed to load FinBERT, falling back to VADER only: {e}")
            self.finbert = None

        self.coverage_data = []

    def get_stock_list(self):
        """
        Identify all stocks listed in the Indian stock market (NSE).
        Returns a DataFrame with symbols and listing dates.
        """
        logging.info("Fetching listed stocks from NSE...")
        # NSE Bhavcopy or Equity List URL (using a static link for active equities)
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }

        try:
            # For demonstration, we'll try to fetch the NSE list. If it fails due to bot-protection,
            # we will fallback to a predefined list.
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                with open('EQUITY_L.csv', 'wb') as f:
                    f.write(response.content)
                df = pd.read_csv('EQUITY_L.csv')
                symbols = df['SYMBOL'].tolist()
                dates = df[' DATE OF LISTING'].tolist()
                stocks = pd.DataFrame({'Symbol': symbols, 'ListingDate': dates})
            else:
                raise Exception("Non-200 status code")
        except Exception as e:
            logging.warning(f"Could not fetch live NSE list due to restrictions ({e}). Using sample Nifty 50 stocks.")
            sample_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN']
            stocks = pd.DataFrame({'Symbol': sample_symbols, 'ListingDate': ['01-Jan-1995'] * len(sample_symbols)})

        stocks['ListingDate'] = pd.to_datetime(stocks['ListingDate'], errors='coerce')
        # Add '.NS' for Yahoo Finance
        stocks['Yahoo_Ticker'] = stocks['Symbol'] + '.NS'
        return stocks

    def fetch_market_data(self, ticker_symbol):
        """
        Collect historical price data and fundamentals data using yfinance.
        """
        ticker = yf.Ticker(ticker_symbol)

        # Historical Data
        hist = ticker.history(period="max")

        # Fundamentals
        info = ticker.info
        fundamentals = {
            'Symbol': ticker_symbol,
            'MarketCap': info.get('marketCap'),
            'PE_Ratio': info.get('trailingPE'),
            'PB_Ratio': info.get('priceToBook'),
            'Dividend_Yield': info.get('dividendYield'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry')
        }
        fund_df = pd.DataFrame([fundamentals])

        return hist, fund_df

    def fetch_historical_documents(self, company_name, year, doc_type="news"):
        """
        Collect historical news, annual reports, or media presentations using Google News RSS.
        Aims to get at least 10 items per year.
        doc_type can be "news", "annual report", or "media presentation".
        """
        if doc_type == "news":
            query = f"{company_name} stock OR finance OR market"
        elif doc_type == "annual report":
            query = f"{company_name} annual report"
        elif doc_type == "media presentation":
            query = f"{company_name} media presentation OR investor presentation"
        else:
            query = f"{company_name}"

        query_encoded = urllib.parse.quote(query)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Google News RSS with date filters
        rss_url = f"https://news.google.com/rss/search?q={query_encoded}+after:{start_date}+before:{end_date}&hl=en-IN&gl=IN&ceid=IN:en"

        feed = feedparser.parse(rss_url)
        articles = []

        for entry in feed.entries[:15]:  # Try to get up to 15 to ensure at least 10 valid ones
            articles.append({
                'Title': entry.title,
                'Published': entry.published if hasattr(entry, 'published') else f"{year}-06-01",
                'Source': entry.source.title if hasattr(entry, 'source') else 'Unknown',
                'Doc_Type': doc_type
            })

        return articles

    def analyze_sentiment(self, text):
        """
        Use NLP to rank news articles based on their sentiments using multiple criteria.
        Criteria 1: VADER Compound Score
        Criteria 2: FinBERT Label & Score
        """
        if not text or not isinstance(text, str):
            return 0.0, 'Neutral', 0.0

        # VADER Analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # FinBERT Analysis
        finbert_label = 'Neutral'
        finbert_score = 0.0
        if self.finbert:
            try:
                # Truncate text to 512 tokens for BERT
                result = self.finbert(text[:1500])[0]
                finbert_label = result['label']
                finbert_score = result['score']
            except Exception as e:
                pass

        return vader_compound, finbert_label, finbert_score

    def run_pipeline(self, max_stocks=None):
        """
        Main pipeline function.
        max_stocks: Optional parameter to limit execution time in testing.
        """
        stocks_df = self.get_stock_list()

        count = 0
        for idx, row in stocks_df.iterrows():
            if max_stocks and count >= max_stocks:
                break

            symbol = row['Symbol']
            yahoo_ticker = row['Yahoo_Ticker']
            listing_date = row['ListingDate']

            logging.info(f"Processing {symbol} ({yahoo_ticker})...")

            # 1. Fetch Market Data & Fundamentals
            try:
                hist, fund_df = self.fetch_market_data(yahoo_ticker)
                if hist.empty:
                    logging.warning(f"No price data found for {symbol}. Skipping.")
                    continue

                # Determine actual listing year from data if listed_date is missing
                actual_start_year = hist.index.min().year
                if pd.isna(listing_date):
                    listing_date = hist.index.min()

                # Save Price Data
                hist['Symbol'] = symbol
                hist.reset_index(inplace=True)
                # Convert datetime with timezone to string to save to sqlite safely
                hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
                hist.to_sql('historical_prices', self.conn, if_exists='append', index=False)

                # Save Fundamentals
                fund_df.to_sql('fundamentals', self.conn, if_exists='append', index=False)

            except Exception as e:
                logging.error(f"Error fetching market data for {symbol}: {e}")
                continue

            # 2. Fetch News and Calculate Sentiments
            current_year = datetime.now().year
            start_year = actual_start_year

            document_records = []
            total_documents_found = 0

            for year in range(start_year, current_year + 1):
                # Fetch different document types
                documents = []
                documents.extend(self.fetch_historical_documents(symbol, year, doc_type="news"))
                documents.extend(self.fetch_historical_documents(symbol, year, doc_type="annual report"))
                documents.extend(self.fetch_historical_documents(symbol, year, doc_type="media presentation"))

                total_documents_found += len(documents)

                for doc in documents:
                    title = doc['Title']
                    # Calculate Sentiments
                    v_comp, fb_label, fb_score = self.analyze_sentiment(title)

                    document_records.append({
                        'Symbol': symbol,
                        'Year': year,
                        'Date': doc['Published'],
                        'Source': doc['Source'],
                        'Doc_Type': doc['Doc_Type'],
                        'Vader_Compound': v_comp,
                        'Finbert_Label': fb_label,
                        'Finbert_Score': fb_score
                        # Note: We purposely DO NOT save the detailed text/title to the DB
                        # to keep the database size small, as requested.
                    })

            if document_records:
                doc_df = pd.DataFrame(document_records)
                doc_df.to_sql('document_sentiments', self.conn, if_exists='append', index=False)

            # 3. Update Coverage Report
            self.coverage_data.append({
                'Symbol': symbol,
                'Listing_Date': listing_date.strftime('%Y-%m-%d') if pd.notnull(listing_date) else 'Unknown',
                'Price_Data_Start': hist['Date'].min(),
                'Price_Data_End': hist['Date'].max(),
                'Total_Trading_Days': len(hist),
                'Has_Fundamentals': not fund_df.empty,
                'Documents_Processed': total_documents_found
            })

            count += 1
            # Sleep to prevent API rate limiting
            time.sleep(2)

        # 4. Generate Coverage Report
        logging.info("Generating Data Coverage Report...")
        coverage_df = pd.DataFrame(self.coverage_data)
        coverage_df.to_excel('Data_Coverage_Report.xlsx', index=False)
        logging.info("Pipeline Complete. Files generated: indian_stocks_data.db, Data_Coverage_Report.xlsx")

    def close(self):
        """Explicitly close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    # To run on Kaggle, simply execute this script.
    # To process all stocks, do not pass max_stocks. (Note: Processing all stocks will take a long time).
    pipeline_runner = IndianStockPipeline()
    try:
        # Example for Kaggle execution. It will process all by default unless max_stocks is provided.
        pipeline_runner.run_pipeline()
    finally:
        pipeline_runner.close()
