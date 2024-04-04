import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import openpyxl
# http://maruo51.com/2019/04/22/corr-matrix/

drop_list=["id", 'year', "month", "day"]
           #"temperature_cnt", "humidity_cnt", "ws_cnt", "dew_cnt", "pressure_cnt",
           #'co_cnt', 'o3_cnt', 'so2_cnt', 'no2_cnt']
           #'co_min', 'co_max', 'o3_min', 'o3_max', 'so2_min', 'so2_max',
           #'no2_min', 'no2_max', 'ws_min', 'ws_max', 'dew_min', 'dew_max',
           #"humidity_min", 'humidity_max', 'pressure_min', 'pressure_max',
          #'temperature_min', 'temperature_max']

def pd_city_concat(df):
    count_ = df.groupby('Country').transform('count')
    city_ = df.groupby('City').transform('count')
    df["City"] = count_['City'] #+item for column_name, item in df_['City'].iteritems()]
    df["Country"] = city_['Country']
    #df = df.drop(['Country'], axis=1)
    return df

def pd_country(df):
    # year
    #df.loc[df['year'] ==2019, 'year'] = 0
    #df.loc[df['year'] ==2019, 'year'] = 1
    #df.loc[df['year'] ==2019, 'year'] = 2
    # country
    #countries = df[df["pm25_mid"]>200]["Country"].value_counts()
    #count_list = list(countries.keys())
    #df["Country"] = np.where((df["Country"].isin(count_list)), 1, 0)
    return df
    
    
def df_preprocess(df):
    df = pd_city_concat(df)
    df = pd_country(df)
    df["co_mid"] = np.log10(df["co_mid"])+1
    df = df.drop(drop_list, axis=1)
    return df



def df_load(target_name = 'pm25_mid'):
    df = pd.read_csv('/Users/hagi/downloads/place/train.csv')
    df = df_preprocess(df)
    #df.sort_values(by = 'data', ascending = True, inplace = True) 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return df

def csv_corr(df, target_name='pm25_mid'):
    corr_matrix = df.corr()
    corr_y = pd.DataFrame({"features":df.columns,"corr_y":corr_matrix[target_name]},index=None)
    corr_y = corr_y.reset_index(drop=True)
    corr_y.style.background_gradient().to_excel('corr_matrix.xlsx', engine='openpyxl')
    
def df_plot_scatter(df, target_name='pm25_mid'):
    df.plot.scatter(x='Country', y=target_name)
    plt.savefig("df.plot.scatter.png")
    
    
if __name__=='__main__':
    TARGET_NAME = 'pm25_mid'
    df = df_load(target_name = TARGET_NAME)
    csv_corr(df, target_name=TARGET_NAME)
    df_plot_scatter(df, target_name=TARGET_NAME)
