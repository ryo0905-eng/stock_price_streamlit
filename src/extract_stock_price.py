import yfinance as yf

def extract_stock_price(ticker, start_date, end_date):
    # S&P500のデータを取得
    data = yf.download(tickers=ticker, start=start_date, end=end_date, multi_level_index=False)
    return data