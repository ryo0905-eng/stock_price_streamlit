from pycaret.time_series import *

def predict_stock_price(model_name):
    # モデルの辞書を作成
    model_dict = {'ARIMA': 'arima', 'Auto ARIMA': 'auto_arima', 'ETS': 'ets'}

    #　モデルを作成
    model = create_model(model_dict[model_name])

    # 予測
    pred = predict_model(model, round=0)

    # カラム名を変更
    pred = pred.rename(columns={'y_pred': model_name})
    # Datetime形式に変換
    pred.index = pred.index.to_timestamp()
    return pred
