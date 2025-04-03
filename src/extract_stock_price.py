import yfinance as yf

def extract_stock_price(ticker, start_date, end_date):
    # S&P500のデータを取得
    try:
        data = yf.download(tickers=ticker, start=start_date, end=end_date, multi_level_index=False)
        if data.empty:
            raise ValueError("取得したデータが空です。ティッカーを確認してください。")
        return data
    except Exception as e:
        print(f"データ取得中にエラーが発生しました： {e}")
        return None
    

if __name__ == "__main__":
    extract_stock_price("^GSPC", "2025-1-1", "2025-4-2")
    
