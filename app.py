import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from pycaret.time_series import *

import datetime

from src.extract_stock_price import extract_stock_price
from src.transform_stock_price import transform_stock_price


# タイトル
st.title('S&P500 Stock price predictions')

# 入力フォームを作成
st.sidebar.write('## 1. Input form')
ticker = st.sidebar.text_input('Ticker', '^GSPC')
start_date = st.sidebar.date_input('Start', value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End', value=datetime.date.today())

# session stateを初期化
if 'data' not in st.session_state:
    st.session_state.data = None

# ボタンを押してデータを取得
st.sidebar.write('## 2. Get Data')
if st.sidebar.button('Get Data'):
    st.sidebar.write('Running....')
    st.session_state.data = extract_stock_price(ticker, start_date, end_date)
    # データを変換
    st.session_state.data = transform_stock_price(st.session_state.data)

# データを表示
if st.session_state.data is not None:
    st.dataframe(st.session_state.data)

# 予測
st.sidebar.write('## 3. Predict')
if st.sidebar.button('Predict'):
    st.sidebar.write('Running....')
    # 予測に使わない特徴量を指定
    ignore_features = ['High', 'Low', 'Open', 'Volume']
    # モデルのセットアップ
    s = setup(st.session_state.data, fh = 30, fold = 5, session_id = 123, target='Close', ignore_features=ignore_features)

    #　モデルを作成
    arima = create_model('arima')

    # 交差検証結果を表示
    #arima_results = pull()
    #st.write('交差検証結果')
    #st.write(arima_results)

    # 予測
    pred = predict_model(arima)

    # Datetime形式に変換
    pred.index = pred.index.to_timestamp()
    pred.head()

    # 実際の値をプロット
    fig = px.line(st.session_state.data, x=st.session_state.data.index, y='Close', title='S&P500 Stock price predictions')

    # 予測値をプロット
    predicted_trace = go.Scatter(x=pred.index, y=pred['y_pred'], mode='lines', name='Predicted')
    fig.add_trace(predicted_trace)

    # 色を変更
    fig.update_traces(line_color='red', selector=dict(name='Predicted'))

    # グラフのズーム位置を指定
    fig.update_layout(
        xaxis_range=[pd.Timestamp.now() - pd.DateOffset(days=90), pd.Timestamp.now()],
        yaxis_range=[5500, 6400])


    # グラフを表示
    st.plotly_chart(fig)
