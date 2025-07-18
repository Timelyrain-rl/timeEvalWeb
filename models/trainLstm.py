from lstm_ad.model  import LSTMAD
import pandas as pd
import os

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def trainLSTM(path,start,end):
    file_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", "month_3_processed.csv")
    df = pd.read_csv(file_path)
    df = df[["记录仪时间","8井油压","9井压力","9井流量"]]
    df.columns = ["date","p8","p9","f9"]
    data = df.fillna(method='pad',axis=0)
    data = data.reset_index()
    data = data[["p8","p9","f9"]]
    data = data.iloc[start:end,:].values
    logger.info("Finish Data load")
    
    model = LSTMAD(input_size=data.shape[1],
                   lstm_layers=2,
                   split=0.9,
                   window_size=30,
                   prediction_window_size=1,
                   output_dims=[],
                   batch_size=32,
                   validation_batch_size=128,
                   test_batch_size=128,
                   epochs=50,
                   early_stopping_delta=0.05,
                   early_stopping_patience=10,
                   optimizer='adam',
                   learning_rate=0.001,
                   random_state=42
                  )
    model.fit(data, path)
    model.save(path)
    
    logger.info("Finish train, model path is {}".format(path))


if __name__ == "__main__":
    trainLSTM("./lstm_res/0716.pt",0,800000)
    