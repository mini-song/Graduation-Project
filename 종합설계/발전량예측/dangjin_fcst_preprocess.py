# %%
import pandas as pd
import numpy as np
import glob
import re
import time

path = './데이터/동네예보/*.csv'

file_list = glob.glob(path)

dangjin_fcst_list = [file_name for file_name in file_list if re.search(r'dangjin_\w+',file_name) is not None]
dangjin_fcst_list.sort()
print(dangjin_fcst_list)

files = [pd.read_csv(file) for file in dangjin_fcst_list]

#%%
def column_sort(dataframe):

    c = ["month", " format: day", "hour", "forecast", "3시간기온", "습도", "풍향", "풍속", "하늘상태", "강수형태", "6시간강수량", "6시간적설", "일최고기온", "일최저기온", "강수확률"]

    dataframe = dataframe[["month", " format: day", "hour", "forecast", "3시간기온", "습도", "풍향", "풍속", "하늘상태", "강수형태", "6시간강수량", "6시간적설", "일최고기온", "일최저기온", "강수확률"]]

    return dataframe

files = [column_sort(file) for file in files]
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']

#%%

def concat_all_fcst_files(files, years):

    assert len(files) == len(years)

    for i in range(len(files)):
        files[i]["year"] = years[i]
        c = files[i].columns.tolist()
        c = c[-1:] + c[:-1]
        files[i] = files[i][c]
    
    dataframe = pd.concat(files, ignore_index=True)

    return dataframe

df = concat_all_fcst_files(files, years)
df.sort_values(by=["year", "month", " format: day", "hour"], inplace=True)

#%%
import pandas as pd
import numpy as np
import glob
import re
import time

dangjin_fcst_data = pd.read_csv('../dangjin_fcst_from_2018.csv')
dangjin_fcst_data

#%%
def time_modify(dataframe):

    '''
    year, month, day, hour 로 나뉘어 있는 값을 time 컬럼에 합치는 함수
    '''

    dataframe["year"] = dataframe["year"].astype(str)
    dataframe['month'] = dataframe['month'].astype(str)
    dataframe[' format: day'] = dataframe[' format: day'].astype(str)
    dataframe['hour'] = dataframe['hour'].astype(str).apply(lambda x: x[:-2]+":00")
    dataframe['forecast'] = dataframe['forecast'].astype(float)
    dataframe["time"] = dataframe["year"] + "-" + dataframe["month"] + "-" + dataframe[" format: day"] + " " + dataframe["hour"]
    dataframe.drop(columns=["year", "month", " format: day", "hour"], inplace=True)
    dataframe = dataframe[['time']+list(dataframe.columns[:-1])]
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    
    return dataframe


dangjin_fcst_data = time_modify(dangjin_fcst_data)
dangjin_fcst_data

# %%
import os
from multiprocessing import Pool

def forecast_change(dataframe):

    print('PID :', os.getpid())

    df = dataframe[["time", "forecast"]]

    for i in dataframe.index.tolist():

        dataframe["forecast"][i] = df["time"][i] + pd.DateOffset(hours=df["forecast"][i])

    return dataframe

def parallel_processing(dataframe, function, num_cores):

    '''
    forecast 컬럼을 예보데이터가 나타내는 시간으로 변경하는 함수가 forecast_change.
    멀티 프로세싱으로 처리하기 위한 함수.
    '''
    
    import time

    start = time.time()

    df_split = np.array_split(dataframe, num_cores)
    pool = Pool(num_cores)
    dataframe = pd.concat(pool.map(function, df_split))
    pool.close()
    pool.join()

    print('\n\n\n',f'processing time : {time.time()-start} s')
    
    return dataframe 

dangjin_fcst_data = parallel_processing(dangjin_fcst_data, forecast_change, 12)


# %%
'''
forecast column 기준으로 sort
관측데이터와 예보데이터 dataframe 모두 받고 연/월/일/시간 넣으면
plot 으로 찍어서 보여줄 수 있게 디자인하기. 
'''

dangjin_obs_data = pd.read_csv('../ASOS_dangjin_from_2018.csv', encoding='cp949')
dangjin_obs_data.drop(columns=dangjin_obs_data.columns[[0, 14, 16, 17, 18, 20, 21]], inplace=True)
dangjin_obs_data

#%%
c = dangjin_obs_data.columns.tolist()
c
#%%
'''
관측 데이터와 예보 데이터에서 비교할 필요 없는 컬럼들 제거
'''
dangjin_obs_data.drop(columns=dangjin_obs_data.columns[[6,7,8,9,10,11,14,15,16,17,18,19]], inplace=True)
dangjin_obs_data
#%%
dangjin_fcst_data.drop(columns=["강수형태", "일최고기온", "일최저기온", "강수확률"], inplace=True)
dangjin_fcst_data


#%%
dangjin_fcst_data["forecast"] = dangjin_fcst_data["forecast"].astype(str)
dangjin_fcst_data["forecast"] = dangjin_fcst_data["forecast"].apply(lambda x: x[:-3])
#%%

dd = pd.merge(dangjin_fcst_data, dangjin_obs_data, left_on="forecast", right_on="일시", how="inner")
dd.sort_values(by=["forecast", "time"], inplace=True)
#%%
cc = ['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '전운량(10분위)']
c = ["3시간기온", "6시간강수량", "풍속", "풍향", "습도", "6시간적설", "하늘상태"]


# %%
def plot_features(dataframe, target, fcst_column, obs_column):

    import matplotlib.pyplot as plt

    dataframe = dataframe[dataframe["forecast"] == target]
    x = [*range(len(dataframe))]

    obs_temperature = dataframe[obs_column[1]]
    obs_rainfall = dataframe[obs_column[2]]
    obs_windspeed = dataframe[obs_column[3]]
    obs_winddirection = dataframe[obs_column[4]]
    obs_humidity = dataframe[obs_column[5]]
    obs_snowfall = dataframe[obs_column[6]]
    obs_cloudy = dataframe[obs_column[7]]

    fcst_temperature = dataframe[fcst_column[0]]
    fcst_rainfall = dataframe[fcst_column[1]]
    fcst_windspeed = dataframe[fcst_column[2]]
    fcst_winddirection = dataframe[fcst_column[3]]
    fcst_humidity = dataframe[fcst_column[4]]
    fcst_snowfall = dataframe[fcst_column[5]]
    fcst_cloudy = dataframe[fcst_column[6]]


    fig, axs = plt.subplots(2, 4)
    plt.subplots_adjust(left=0.125, bottom=0.1,  right=2.0, top=0.9, wspace=0.2, hspace=0.35)
    
    axs[0,0].plot(x, obs_temperature); axs[0,0].plot(x, fcst_temperature); axs[0,0].set_title("temperature")
    axs[0,0].legend(['obs','fcst'])

    axs[0,1].plot(x, obs_humidity); axs[0,1].plot(x, fcst_humidity); axs[0,1].set_title("humidity")
    axs[0,1].legend(['obs','fcst'])

    axs[0,2].plot(x, obs_windspeed); axs[0,2].plot(x, fcst_windspeed); axs[0,2].set_title("windspeed")
    axs[0,2].legend(['obs','fcst'])

    axs[0,3].plot(x, obs_winddirection); axs[0,3].plot(x, fcst_winddirection); axs[0,3].set_title("winddirection")
    axs[0,3].legend(['obs','fcst'])

    axs[1,0].plot(x, obs_rainfall); axs[1,0].plot(x, fcst_rainfall); axs[1,0].set_title("rainfall")
    axs[1,0].legend(['obs','fcst'])

    axs[1,1].plot(x, obs_snowfall); axs[1,1].plot(x, fcst_snowfall); axs[1,1].set_title("snowfall")
    axs[1,1].legend(['obs','fcst'])

    axs[1,2].plot(x, obs_cloudy); axs[1,2].plot(x, fcst_cloudy); axs[1,2].set_title("cloudy")
    axs[1,2].legend(['obs','fcst'])

#%%
plot_features(dd, '2018-01-15 15:00', c, cc)

