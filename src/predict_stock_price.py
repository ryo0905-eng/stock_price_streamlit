from pycaret.time_series import *

def predict_stock_price(model_dict, model_name):
    #　モデルを作成
    model = create_model(model_dict[model_name])

    # 予測
    pred = predict_model(model, fh=40, round=0)

    # カラム名を変更
    pred = pred.rename(columns={'y_pred': model_name})

    # Datetime形式に変換
    pred.index = pred.index.to_timestamp()

    # モデルをsave
    save_model(model, f'saved_model/{model_name}')

    return pred
