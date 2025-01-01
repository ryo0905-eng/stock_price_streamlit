import pandas as pd

def transform_stock_price(data):
    # datetime形式に変換
    data.index = pd.to_datetime(data.index)

    # 欠損値は直前の値で補完
    data = data.asfreq('D', method='bfill')
    return data
