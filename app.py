
# キャッシュをクリア
st.cache_resource.clear()

import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from pycaret.time_series import *

# タイトル
st.title('S&P500 Stock price predictions')

# S&P500のデータを取得
ticker = "^GSPC"
data = yf.download(tickers=ticker, start="2020-01-01", end="2024-12-31", multi_level_index=False)

# datetime形式に変換
data.index = pd.to_datetime(data.index)

# 欠損値は直前の値で補完
data = data.asfreq('D', method='bfill')

# データの確認
st.write('データの先頭5行を表示')
st.write(data.head())

# 予測に使わない特徴量を指定
ignore_features = ['High', 'Low', 'Open', 'Volume']

# モデルのセットアップ
s = setup(data, fh = 30, fold = 5, session_id = 123, target='Close', ignore_features=ignore_features)

#　モデルを作成
arima = create_model('arima')

# 交差検証結果を表示
arima_results = pull()
#st.write('交差検証結果')
#st.write(arima_results)

# 予測
pred = predict_model(arima)

# Datetime形式に変換
pred.index = pred.index.to_timestamp()
pred.head()

# 実際の値をプロット
fig = px.line(data, x=data.index, y='Close', title='S&P500 Stock price predictions')

# 予測値をプロット
predicted_trace = go.Scatter(x=pred.index, y=pred['y_pred'], mode='lines', name='Predicted')
fig.add_trace(predicted_trace)

# 色を変更
fig.update_traces(line_color='red', selector=dict(name='Predicted'))

# グラフを表示
st.plotly_chart(fig)
