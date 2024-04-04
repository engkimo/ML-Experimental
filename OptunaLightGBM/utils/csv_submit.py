import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from models import UnsupervisedTBNet, supervisedTBNet
from utils import save2load_pretrain_model, feature_importances, calculate_scores

def normlize(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df =pd.DataFrame(df_scaled,columns=df.columns)
    return df

def df_preprocess(df):
    df["co_ran"]=df["co_max"]-df["co_min"]
    df["o3_ran"]=df["o3_max"]-df["o3_min"]
    df["so2_ran"]=df["so2_max"]-df["so2_min"]
    df["no2_ran"]=df["no2_max"]-df["no2_min"]
    df["temperature_ran"]=df["temperature_max"]-df["temperature_min"]
    df["pressure_ran"]=df["pressure_max"]-df["pressure_min"]
    df["humidity_ran"]=df["humidity_max"]-df["humidity_min"]
    df["fst_dir"]=df["co_var"]+df["so2_var"]+df["no2_var"]
    df = df.drop(drop_list2, axis=1)
    return df

drop_list=["id", "year", "Country", "City", "month", "day", "temperature_cnt",
           "humidity_cnt", "ws_cnt", "dew_cnt", "pressure_cnt"]

drop_list2=["humidity_max", "humidity_min", "temperature_min", "temperature_max",
           "humidity_min", "humidity_max", "humidity_mid", "pressure_min", "pressure_max", "pressure_mid",
            "ws_min", "ws_max", "ws_mid", "dew_min", "dew_max", "dew_mid"]

target_name = 'pm25_mid'
df = pd.read_csv('/Users/hagi/downloads/place/test.csv')
df = df.drop(drop_list, axis=1)
df = df_preprocess(df)
print(df.shape)


def load_model(model_path="test_trained.zip"):
    loaded_clf = TabNetRegressor()
    loaded_clf.load_model(model_path)
    return loaded_clf

loaded_clf = load_model(model_path="test_trained.zip")
y_test = loaded_clf.predict(df.values)

# save to csv
dd = pd.DataFrame(y_test)
dd.set_index(dd.index+195942).to_csv("submit.csv", header=False)
