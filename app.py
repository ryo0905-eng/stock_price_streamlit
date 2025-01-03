import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import datetime

from src.extract_stock_price import extract_stock_price
from src.transform_stock_price import transform_stock_price
from src.predict_stock_price import predict_stock_price

# タイトル
st.title('S&P500 Stock price predictions')

# 入力フォームを作成
st.sidebar.write('## 1. Input form')
ticker = st.sidebar.text_input('Ticker', '^GSPC')
start_date = st.sidebar.date_input('Start', value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End', value=datetime.date.today())
model_name = st.sidebar.selectbox('Model', ['ARIMA', 'Auto ARIMA', 'ETS'])

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
        st.dataframe(st.session_state.data)
        st.sidebar.success('Done!')

# 予測
st.sidebar.write('## 3. Predict')
if st.sidebar.button('Predict'):
    # Spinnerを表示
    with st.spinner('Predicting...'):
        # 結果格納用データフレームを作成
        df = st.session_state.data.copy()

        # 予測
        pred = predict_stock_price(st.session_state.data, model_name)

        # 予測値をデータフレームに追加
        df = pd.concat([df, pred], axis=1)

        # 実際の値をプロット
        fig = px.line(df, x=df.index, y=['Actual', 'ARIMA'], title='S&P500 Stock price predictions')

        # 色を変更
#        fig.update_traces(line_color='red', selector=dict(name='Predicted'))

        # グラフのズーム位置を指定
        fig.update_layout(
            xaxis_range=[pd.Timestamp.now() - pd.DateOffset(days=90), pd.Timestamp.now()],
            yaxis_range=[5500, 6400])

        # グラフとデータを表示
        st.plotly_chart(fig)
        st.dataframe(df)
