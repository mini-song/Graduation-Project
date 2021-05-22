
import pandas as pd
import numpy as np

def wind_direction_dummies(dataframe, wind_direction, drop_energy=False):
    
    dataframe[wind_direction] = dataframe[wind_direction].apply(lambda x: '7.0' if x==360 else str(x//45))
    
    dataframe = pd.get_dummies(dataframe, columns=[wind_direction], drop_first=True)
    
    if drop_energy:
        c = list(dataframe.columns)
        c.remove('dangjin_floating')
        c = c + ['dangjin_floating']

        dataframe = dataframe[c]

    return dataframe






def dataframe_interpolation_by_spline(dataframe, columns):
    import time
    start = time.time()
    for column in columns:
        dataframe[column].interpolate(method='spline', order=3, inplace=True)
    print(f'processing time : {time.time()-start} s')


def dataframe_interpolation_by_linear(dataframe, columns):
    import time
    start = time.time()
    for column in columns:
        dataframe[column].interpolate(method='linear', inplace=True)
    print(f'processing time : {time.time()-start} s')






from sklearn.preprocessing import MinMaxScaler

def dataframe_normalize(dataframe, exclude_column_list):
    
    scaler = MinMaxScaler()

    normalize_columns = list(dataframe.columns)
    for column in exclude_column_list:
        normalize_columns.remove(column)

    dataframe[normalize_columns] = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=exclude_column_list)))
    

    return dataframe





import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'

def hist_features(dataframe, exclude_column_list):

    hist_columns = list(dataframe.columns)
    for column in exclude_column_list:
        hist_columns.remove(column)

    n = len(hist_columns)

    if n < 10:
        n1 = 2
        n2 = (n // 2) + 1 if n%2 else n // 2
    else:
        n1 = 3
        n2 = (n // 3) + 1 if n%3 else n // 3
    
    
    fig, axs = plt.subplots(n1, n2)
    plt.subplots_adjust(left=0.125, bottom=0.1,  right=2.0, top=1.5, wspace=0.2, hspace=0.35)    


    for i in range(n1):
        for j in range(n2):
            
            if (i*n2 + j) == n : break

            axs[i, j].hist(dataframe[hist_columns[i*n2 + j]])
            axs[i, j].set_title(hist_columns[i*n2 + j])

    

def fcst_augment(dataframe):

    start = dataframe["forecast"].iloc[0]
    end = dataframe["forecast"].iloc[-1]

    time_index = pd.Series(pd.date_range(start, end, freq='H'), name="time_index")

    dataframe["forecast"] = pd.to_datetime(dataframe["forecast"])
    dataframe = pd.merge(time_index, dataframe, how='outer', left_on="time_index", right_on="forecast").drop(columns=["time", "forecast"])

    return dataframe



def nona_mugja(dataframe):

    dataframe["6시간강수량"] = dataframe["6시간강수량"] / 6
    dataframe["6시간적설"] = dataframe["6시간적설"] / 6
    
    dataframe["6시간강수량"].fillna(method='ffill', inplace=True)
    dataframe["6시간적설"].fillna(method='ffill', inplace=True)    

    return dataframe