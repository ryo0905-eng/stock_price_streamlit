import pandas as pd

def transform_stock_price(data):
    # datetime形式に変換
    data.index = pd.to_datetime(data.index)
    # int型に変換
    data = data.astype(int)
    # カラム名を変更
    data = data.rename(columns={'Close': 'Actual'})
    # 必要なカラムのみに絞る
    data = data[['Actual']]
    # 欠損値は直前の値で補完
    data = data.asfreq('D', method='bfill')
    return data
