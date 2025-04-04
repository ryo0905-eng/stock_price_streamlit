import pandas as pd
import plotly.express as px
import streamlit as st
from pycaret.time_series import *

import datetime

from src.extract_stock_price import extract_stock_price
from src.transform_stock_price import transform_stock_price
from src.predict_stock_price import predict_stock_price

# タイトル
st.title('S&P500 Stock price predictions')

# モデルの辞書を作成
model_dict = {'ARIMA': 'arima', 'Naive': 'naive', 'ETS': 'ets', 'Prophet': 'prophet'}

# 入力フォームを作成
st.sidebar.write('## 1. Input form')
ticker = st.sidebar.text_input('Ticker', '^GSPC')
start_date = st.sidebar.date_input('Start', value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End', value=datetime.date.today())
selected_model = st.sidebar.multiselect('Model', model_dict.keys(), default=['ETS'])

# session stateを初期化
if 'data' not in st.session_state:
    st.session_state.data = None

# ボタンを押してデータを取得
st.sidebar.write('## 2. Get Data')
if st.sidebar.button('Get Data'):
    with st.spinner('Loading data...'):
        st.session_state.data = extract_stock_price(ticker, start_date, end_date)
        # データを変換
        st.session_state.data = transform_stock_price(st.session_state.data)
        # データを表示
        fig = px.line(st.session_state.data, x=st.session_state.data.index, y='Actual', title='S&P500 Stock price')
        st.plotly_chart(fig)
        st.sidebar.success('Done!')

# 予測
st.sidebar.write('## 3. Predict')
if st.sidebar.button('Predict'):
    # Spinnerを表示
    with st.spinner('Predicting...'):
        # 結果格納用データフレームリストを作成
        dfs = [st.session_state.data.copy()]
        # モデルのセットアップ
        s = setup(st.session_state.data, fh = 10, fold = 1, session_id = 123, target='Actual')
        # 予測
        for model in selected_model:
            pred = predict_stock_price(model_dict, model)
            dfs.append(pred)
        # 結果を連結
        df = pd.concat(dfs, axis=1)
        # グラフをプロット
        fig = px.line(df, x=df.index, y=['Actual']+selected_model, title='S&P500 Stock price predictions')
        # グラフのズーム位置を指定
        fig.update_layout(
            xaxis_range=[df.index.max() - pd.DateOffset(days=90), df.index.max()],
            yaxis_range=[5500, df.max().max()]
            )
        # グラフを表示
        st.plotly_chart(fig)
        # データを降順で並び替えて表示
        st.dataframe(df.sort_index(ascending=False))
