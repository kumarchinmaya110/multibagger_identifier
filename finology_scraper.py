import io
import time
import requests # standard requests for NSE CSV
import pandas as pd
import re
from bs4 import BeautifulSoup
import logging
import argparse
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dependencies():
    packages = ["curl_cffi", "openpyxl"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            logging.info(f"{pkg} not found. Installing it...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            logging.info(f"Successfully installed {pkg}.")

ensure_dependencies()
from curl_cffi import requests as cffi_requests

def get_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            if 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].dropna().tolist()
                logging.info(f"Successfully fetched {len(symbols)} NSE symbols.")
                return symbols
    except Exception as e:
        logging.error(f"Failed to fetch NSE list: {e}")
    return []

def clean_text(text):
    if not text: return "NA"
    return re.sub(r'\s+', ' ', text).strip()

def get_metric_robust(soup, label):
    # Find span or string with the label
    elem = soup.find(string=re.compile(f"^{label}$", re.IGNORECASE))
    if not elem:
        elem = soup.find(string=re.compile(label, re.IGNORECASE))

    if elem:
        # The number could be in the parent's sibling or parent's parent's sibling
        container = elem.find_parent('div', class_=re.compile('compess'))
        if container:
            p = container.find('p')
            if p:
                return clean_text(p.get_text())
    return "NA"

def get_growth_values(soup, label):
    res = {"1yr": "NA", "3yr": "NA", "5yr": "NA"}
    cards = soup.find_all(string=re.compile(label, re.IGNORECASE))
    for c in cards:
        if c.parent and c.parent.parent:
            text = c.parent.parent.get_text(separator=' ', strip=True)
            if '1 Year' in text or '3 Year' in text:
                m1 = re.search(r'1 Year\s*([\d\.\-]+%?)', text, re.IGNORECASE)
                m3 = re.search(r'3 Year\s*([\d\.\-]+%?)', text, re.IGNORECASE)
                m5 = re.search(r'5 Year\s*([\d\.\-]+%?)', text, re.IGNORECASE)
                if m1: res['1yr'] = m1.group(1).strip()
                if m3: res['3yr'] = m3.group(1).strip()
                if m5: res['5yr'] = m5.group(1).strip()
                break
    return res

def parse_html(html, symbol):
    soup = BeautifulSoup(html, 'html.parser')
    data = {'Symbol': symbol}

    # Finstar Rating
    finstar = soup.find(id=re.compile("ManagementRating", re.IGNORECASE))
    data['Finstar Rating'] = "NA"
    if finstar:
        m = re.search(r'([\d\.]+)\s+out of', finstar.get('aria-label', ''))
        if m:
            data['Finstar Rating'] = m.group(1)

    # Sector
    data['Sector'] = "NA"
    sector_elem = soup.find('a', href=re.compile('sector=', re.IGNORECASE))
    if sector_elem:
        data['Sector'] = clean_text(sector_elem.get_text())

    # Metrics
    data['Market Cap'] = get_metric_robust(soup, 'Market Cap')
    data['Enterprise Value'] = get_metric_robust(soup, 'Enterprise Value')
    data['P/E'] = get_metric_robust(soup, 'P/E')
    data['P/B'] = get_metric_robust(soup, 'P/B')
    data['Debt'] = get_metric_robust(soup, 'DEBT')
    data['Promoter Holding'] = get_metric_robust(soup, 'Promoter Holding')
    data['EPS'] = get_metric_robust(soup, 'EPS')

    # Growth metrics
    sales_growth = get_growth_values(soup, 'Sales Growth')
    profit_growth = get_growth_values(soup, 'Profit Growth')
    roe = get_growth_values(soup, 'ROE')
    roce = get_growth_values(soup, 'ROCE')

    data['Sales Growth (1yr)'] = sales_growth['1yr']
    data['Sales Growth (3yr)'] = sales_growth['3yr']
    data['Sales Growth (5yr)'] = sales_growth['5yr']

    data['Profit Growth (1yr)'] = profit_growth['1yr']
    data['Profit Growth (3yr)'] = profit_growth['3yr']
    data['Profit Growth (5yr)'] = profit_growth['5yr']

    data['ROE (1yr)'] = roe['1yr']
    data['ROE (3yr)'] = roe['3yr']
    data['ROE (5yr)'] = roe['5yr']

    data['ROCE (1yr)'] = roce['1yr']
    data['ROCE (3yr)'] = roce['3yr']
    data['ROCE (5yr)'] = roce['5yr']

    # Strengths and Limitations
    strengths = []
    limitations = []
    for h2 in soup.find_all('h2'):
        text = h2.get_text().strip().lower()
        if "strengths" in text:
            ul = h2.find_next_sibling('ul')
            if ul: strengths = [li.get_text().strip() for li in ul.find_all('li')]
        elif "limitations" in text:
            ul = h2.find_next_sibling('ul')
            if ul: limitations = [li.get_text().strip() for li in ul.find_all('li')]

    data['Strengths'] = " | ".join(strengths) if strengths else "NA"
    data['Limitations'] = " | ".join(limitations) if limitations else "NA"

    # PAT and Total Assets (last 5 years)
    pat_values = ["NA"]*5
    assets_values = ["NA"]*5

    for table in soup.find_all('table'):
        headers = [th.get_text().strip() for th in table.find_all('th')]

        for tr in table.find_all('tr'):
            tds = tr.find_all(['th', 'td'])
            row_text = [td.get_text().strip() for td in tds]
            if row_text:
                title = row_text[0].lower()
                if 'profit after tax' in title or 'net profit' in title:
                    if len(row_text) >= 6:
                        pat_values = row_text[-5:]
                    elif len(row_text) > 1:
                        pat_values = row_text[1:] + ["NA"]*(6-len(row_text))
                elif 'total assets' in title:
                    if len(row_text) >= 6:
                        assets_values = row_text[-5:]
                    elif len(row_text) > 1:
                        assets_values = row_text[1:] + ["NA"]*(6-len(row_text))

    for i in range(5):
        data[f'PAT (Year {i+1})'] = pat_values[i] if i < len(pat_values) else "NA"
        data[f'Total Assets (Year {i+1})'] = assets_values[i] if i < len(assets_values) else "NA"

    return data

def scrape_symbol(symbol, use_wayback=False):
    url = f"https://ticker.finology.in/company/{symbol}"

    empty_data = {'Symbol': symbol, 'Finstar Rating': 'NA', 'Sector': 'NA', 'Market Cap': 'NA', 'Enterprise Value': 'NA', 'P/E': 'NA', 'P/B': 'NA', 'Debt': 'NA', 'Promoter Holding': 'NA', 'EPS': 'NA', 'Sales Growth (1yr)': 'NA', 'Sales Growth (3yr)': 'NA', 'Sales Growth (5yr)': 'NA', 'ROE (1yr)': 'NA', 'ROE (3yr)': 'NA', 'ROE (5yr)': 'NA', 'ROCE (1yr)': 'NA', 'ROCE (3yr)': 'NA', 'ROCE (5yr)': 'NA', 'Profit Growth (1yr)': 'NA', 'Profit Growth (3yr)': 'NA', 'Profit Growth (5yr)': 'NA', 'Strengths': 'NA', 'Limitations': 'NA'}
    for i in range(5):
        empty_data[f'PAT (Year {i+1})'] = 'NA'
        empty_data[f'Total Assets (Year {i+1})'] = 'NA'

    try:
        if not use_wayback:
            # Using curl_cffi to impersonate chrome and bypass Cloudflare WAF
            response = cffi_requests.get(url, impersonate="chrome", timeout=15)
            if response.status_code == 200:
                return parse_html(response.text, symbol)
            elif response.status_code in [403, 404]:
                logging.warning(f"HTTP {response.status_code} for {symbol}. Stock may not exist or blocked.")
                # We can optionally fallback to wayback if it's 403, but 404 means it's not found on the site.
                if response.status_code == 403:
                    use_wayback = True
            else:
                logging.warning(f"Failed to fetch {symbol}: HTTP {response.status_code}")
                return empty_data

        if use_wayback:
            wayback_url = f"https://web.archive.org/web/20230601000000/{url}"
            # Wayback machine doesn't need cloudflare bypass, standard requests is fine but cffi is okay too
            response_wb = requests.get(wayback_url, timeout=15)
            if response_wb.status_code == 200:
                 return parse_html(response_wb.text, symbol)
            else:
                 logging.warning(f"Wayback Machine also failed for {symbol}: HTTP {response_wb.status_code}")

    except Exception as e:
        logging.error(f"Error processing data for {symbol}: {e}")

    return empty_data

def main():
    parser = argparse.ArgumentParser(description="Scrape stock data from Finology Ticker.")
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of stocks to scrape (for testing).')
    parser.add_argument('--wayback', action='store_true', help='Force using Wayback Machine API (useful if live site is permanently blocked).')

    args, _ = parser.parse_known_args()

    symbols = get_nse_symbols()
    if not symbols:
        logging.error("No symbols found. Exiting.")
        return

    if args.limit:
        symbols = symbols[:args.limit]
        logging.info(f"Limiting scrape to {args.limit} symbols.")

    all_data = []
    total = len(symbols)

    output_file = 'finology_stocks_data.xlsx'
    for idx, sym in enumerate(symbols, 1):
        logging.info(f"Scraping data for {sym} ({idx}/{total})...")
        data = scrape_symbol(sym, use_wayback=args.wayback)
        all_data.append(data)

        # Save incrementally every 50 stocks to avoid data loss if it crashes
        if idx % 50 == 0:
            pd.DataFrame(all_data).to_excel(output_file, index=False)
            logging.info(f"Incremental save completed at stock {idx}.")

        time.sleep(1) # Polite delay between requests

    df = pd.DataFrame(all_data)
    df.to_excel(output_file, index=False)
    logging.info(f"Scraping completed. Final data saved to {output_file}")

if __name__ == "__main__":
    main()
