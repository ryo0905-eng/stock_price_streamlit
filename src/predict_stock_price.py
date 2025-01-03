from pycaret.time_series import *

import logging

# ログの設定
logging.basicConfig(
                    level=logging.INFO,
                    filename='log/saved_model.log',
                    format='%(asctime)s:%(levelname)s:%(message)s'
                    )

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

    # ログを出力
    logging.info(f'{model_name} model is saved')

    return pred
