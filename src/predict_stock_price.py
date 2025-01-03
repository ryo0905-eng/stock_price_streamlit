from pycaret.time_series import *

def predict_stock_price(df, model_name):
    # 予測に使わない特徴量を指定
    ignore_features = ['High', 'Low', 'Open', 'Volume']

    # モデルのセットアップ
    s = setup(df, fh = 30, fold = 5, session_id = 123, target='Close', ignore_features=ignore_features)

    # モデルの辞書を作成
    model_dict = {'ARIMA': 'arima', 'Auto ARIMA': 'auto_arima', 'ETS': 'ets'}

    #　モデルを作成
    model = create_model(model_dict[model_name])

    # 交差検証結果を表示
    #arima_results = pull()
    #st.write('交差検証結果')
    #st.write(arima_results)

    # 予測
    pred = predict_model(model)

    # Datetime形式に変換
    pred.index = pred.index.to_timestamp()
    return pred
