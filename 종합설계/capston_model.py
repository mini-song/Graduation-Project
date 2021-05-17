#%%
from numpy.core.numeric import NaN
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import glob
import missingno as msno
import time

energy = pd.read_csv("dangjin_floating_energy.csv")
dangjin_obs = pd.read_csv("ASOS_dangjin_from_2018.csv", encoding='cp949')
dangjin_fcst = pd.read_csv("dangjin_fcst_from_2018.csv")

dangjin_obs.drop(columns=dangjin_obs.columns[[0, 14, 16, 17, 18, 20, 21]], inplace = True)
dangjin_fcst.drop(columns=dangjin_fcst.columns[[13, 14]], inplace = True)
#%%
full_time = pd.date_range(start='2018-01-01 00:00:00', end='2021-02-01 00:00:00', freq='H')

energy[pd.to_datetime(energy['time']) == full_time] ## 27049개 모두 제대로 찍혀있음 확인.
#%%
energy['time'] = pd.to_datetime(energy['time'])
dangjin_obs['일시'] = pd.to_datetime(dangjin_obs['일시'])

#%%
data = pd.merge(energy, dangjin_obs, how='outer', left_on='time', right_on='일시')
data.drop(columns=['일시'], inplace=True)

c = list(data.columns)
c = c + [c.pop(1)]

data = data[c]
# list(data[data['일시'].isna()]['time'])[:-744]

# %%
value = {'강수량(mm)':0, '일조(hr)':0 , '일사(MJ/m2)':0, '적설(cm)':0, '시정(10m)':data['시정(10m)'].mean(),
'dangjin_floating':0}
data.fillna(value, inplace=True)
#%%
# 21년도 자료가 없어서 제거해주는거임
data.drop(index=data.index[26280:], inplace=True)
#%%
# energy[energy['dangjin_floating'].isna()]
energy.iloc[21768-72:21792-72]

# data[data['풍속(m/s)'].isna()]


#%%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'NanumGothic'
#msno.matrix(data)
msno.matrix(data)
# %%
'''
발전량 앞뒤로 보간 필요, 전운량ㅠ, 시정ㅠ , 일조일사ㅠ, 풍향ㅠ, 2021년 자료ㅠ
'''
start = time.time()
def dataframe_interpolation(dataframe, columns):
    for column in columns:
        dataframe[column].interpolate(method='spline', order=3, inplace=True)

c_need_to_be_interpolated = ['기온(°C)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', 
                             '현지기압(hPa)','해면기압(hPa)', '지면온도(°C)', '5cm 지중온도(°C)', 
                             '10cm 지중온도(°C)','20cm 지중온도(°C)', '30cm 지중온도(°C)']

dataframe_interpolation(data, c_need_to_be_interpolated)
print(time.time()-start)
#%%
msno.bar(data)

#%%
plt.figure(figsize=(17, 17))
correlations = data.corr(method='pearson')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0, annot=True)
#%%
data.info()

#%%
'''
흐음 Normalize는 경우에 따라 좋은 선택이 아닐수도..
'''
from sklearn.preprocessing import MinMaxScaler

def dataframe_normalize(dataframe):
    
    scaler = MinMaxScaler()

    dataframe[list(dataframe.columns)[1:-1]] = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=['time', 'dangjin_floating'])))
    

    return dataframe

data = dataframe_normalize(data)
#%%
data.describe()
#%%






#%%
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score

Y=data['dangjin_floating']
X =data.drop(columns=['time', 'dangjin_floating'])

X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.1)

xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

xgb_model.fit(X, Y)


xgboost.plot_importance(xgb_model)
#%%
predictions = xgb_model.predict(X_test)

print('score :', xgb_model.score(X_train, y_train))
print('explained_score :',explained_variance_score(predictions,y_test))
print('r2_score :',r2_score(predictions,y_test))
#%%
p = xgb_model.predict(data.iloc[21768-48:21792-48,1:-1]).tolist()
p
#%%
del xgb_model
del X, Y
del X_train, X_test, y_train, y_test






#%%
import torch
import torch.nn as nn

inputs = data.drop(columns=['time', 'dangjin_floating'])
outputs = data['dangjin_floating']

inputs.info()
#%%
inputs.fillna(0, inplace=True)

inputs = torch.tensor(inputs.to_numpy())
outputs = torch.tensor(outputs.to_numpy()).unsqueeze(1)
#%%

