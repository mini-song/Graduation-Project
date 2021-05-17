# -*- coding: utf-8 -*-
# %%


from numpy.core.numeric import full
import pandas as pd
import numpy as np
import glob
import missingno as msno
import time
#from parallel import parallel_processing
# +
energy = pd.read_csv("dangjin_floating_energy.csv")
dangjin_obs = pd.read_csv("ASOS_dangjin_from_2018.csv", encoding='cp949')

dangjin_obs.drop(columns=dangjin_obs.columns[[0, 14, 16, 17, 18, 20, 21]], inplace = True)
# %%
full_time = pd.date_range(start='2018-01-01 00:00:00', end='2020-12-30 23:00:00', freq='H')
# energy[pd.to_datetime(energy['time']) == full_time] ## 27049개 모두 제대로 찍혀있음 확인.
# %%
energy['time'] = pd.to_datetime(energy['time'])
dangjin_obs['일시'] = pd.to_datetime(dangjin_obs['일시'])

# +
# %%
data_obs = pd.merge(energy, dangjin_obs, how='outer', left_on='time', right_on='일시')
data_obs.drop(columns=['일시'], inplace=True)

c = list(data_obs.columns)
c = c + [c.pop(1)]

data_obs = data_obs[c]
# list(data_obs[data_obs['일시'].isna()]['time'])[:-744]

# %%
# 21년도 자료가 없어서 제거해주는거임 + 60개
data_obs.drop(index=data_obs.index[26280:], inplace=True)
# data_obs.drop(index=list(data_obs[data_obs['풍향(16방위)'].isna()].index), inplace=True)

# %%
value = {'강수량(mm)':0, '일조(hr)':0 , '일사(MJ/m2)':0, '적설(cm)':0, '시정(10m)':data_obs['시정(10m)'].mean(),
'풍향(16방위)':0}
data_obs.fillna(value, inplace=True)

# +
# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
#msno.matrix(data_obs)
msno.matrix(data_obs)
# %%
'''
전운량ㅠ, 시정ㅠ , 2021년 자료ㅠ
'''

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


c_need_to_be_interpolated = ['기온(°C)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', 
                             '현지기압(hPa)','해면기압(hPa)', '지면온도(°C)', '5cm 지중온도(°C)', 
                             '10cm 지중온도(°C)','20cm 지중온도(°C)', '30cm 지중온도(°C)']

dataframe_interpolation_by_spline(data_obs, c_need_to_be_interpolated)

# +
# %%
'''
발전량 null value 채우기.

for문 대신 numpy.array 의 broadcasting 을 이용함.

Broadcasting 이란  [1,2,3,4,5] + 4  , [1,2,3,4,5] * 4  등 원래 불가능한 연산을 아래와 같이 수행해주는 것을 말함.
=> [5,6,7,8,9]  ,  [4,8,12,16,20] 
실제로 list나 pandas.dataframe 은 이렇게 하면 연산 안 됨. numpy.array 만 지원.

그리고 tensorflow 나 torch 에서 사용하는 tensor 도 base 가 numpy.array 이기 때문에 tensor 에서도 broadcasting 을 지원.
은근 편할 때가 많이 있지만 잘못하면 의도한 것과 다르게 연산이 이루어지면서 오류 없이 돌아가는 상황이 발생할 수 있으니 사용할 때 약간 주의 필요.
'''


empty_list = data_obs[data_obs['dangjin_floating'].isna()].index.to_numpy()
# -

empty_list

data_obs['dangjin_floating'][empty_list] = (data_obs['dangjin_floating'][(empty_list-24)].to_numpy() + data_obs['dangjin_floating'][(empty_list+24)].to_numpy()) / 2
data_obs['dangjin_floating'][empty_list]

# +
# %%
'''
풍향 8방위로 change

lambda x: (뭐시기 뭐시기 코드) 는 간단한 함수를 일일히 선언하지 않고 한 줄로 간단히 만들어 쓸 때 사용.
x가 인자. lambda x: ______   요기에 만들고 싶은 함수를 쓰면 됨. x**2, x[:-1] 등등.
apply 나 map 함수와 함께 편하게 사용할 수 있음. 
apply , map 함수는 list, numpy.array, dataframe 등 iterable 한 객체에 line by line 으로 갑을 변경하거나 어떤 함수를 적용할 때 사용.
for문 보다 훨씬 빠르고 코드도 간결. 경험상 경우에 따라 10배 이상 혹은 그것보다 많이 차이날 때도 있음.
'''


data_obs['풍향(16방위)'] = data_obs['풍향(16방위)'].apply(lambda x: 7.0 if x==360 else (x//45))
data_obs['풍향(16방위)']

# +
# %%
# pd.get_dummies(data_obs)


# %%

# +
msno.bar(data_obs)

# %%
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(17, 17))
correlations = data_obs.corr(method='pearson')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0, annot=True)
# %%
data_obs.info()

# +
# %%

# +
'''
흐음 Normalize는 경우에 따라 좋은 선택이 아닐수도.. 이후에 검토가 필요하다.
'''
from sklearn.preprocessing import MinMaxScaler

def dataframe_normalize(dataframe):
    
    scaler = MinMaxScaler()

    dataframe[list(dataframe.columns)[1:-1]] = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=['time', 'dangjin_floating'])))
    

    return dataframe

data_obs = dataframe_normalize(data_obs)
# %%
data_obs.describe()
# %%

# +
# %%

# +
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score

Y=data_obs['dangjin_floating']
X =data_obs.drop(columns=['time', 'dangjin_floating'])

X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.1)

xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

xgb_model.fit(X, Y)


xgboost.plot_importance(xgb_model)

# +
# %%

# +
predictions = xgb_model.predict(X_test)

print('score :', xgb_model.score(X_train, y_train))
print('explained_score :',explained_variance_score(predictions,y_test))
print('r2_score :',r2_score(predictions,y_test))

# +
# %%

# +
'''
예측갑과 실제갑 직접 비교해보기.
'''
start = 21768 + 336 
end = 21792 + 336

pred = pd.Series(xgb_model.predict(data_obs.iloc[start:end,1:-1]), name='prediction')
real = data_obs[['time', 'dangjin_floating']].iloc[start:end].reset_index(drop=True)

pd.concat([real, pred], axis = 1)

# %%
del xgb_model
del X, Y
del X_train, X_test, y_train, y_test
del pred, real
# %%

# +
# %%

# +
dangjin_fcst = pd.read_csv("dangjin_fcst_from_2018.csv")
dangjin_fcst
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


dangjin_fcst = time_modify(dangjin_fcst)
dangjin_fcst

# +
# %%
# +
import os
from multiprocessing import Pool

def forecast_change(dataframe):

    print('PID :', os.getpid())

    df = dataframe[["time", "forecast"]]

    for i in dataframe.index.tolist():

        dataframe["forecast"][i] = df["time"][i] + pd.DateOffset(hours=df["forecast"][i])

    return dataframe

# %%
np.array([1,2])

# %%

import numpy as np
dangjin_fcst = parallel_processing(dangjin_fcst, forecast_change, 12)
dangjin_fcst

# +
# %%
# -

dangjin_fcst.sort_values(by=["forecast", "time"], inplace=True)
dangjin_fcst.drop_duplicates(subset="forecast", keep="last", inplace=True)
dangjin_fcst.drop(columns="time", inplace=True)
dangjin_fcst.reset_index(drop=True, inplace=True)
dangjin_fcst

# +
# %%
# -

dangjin_fcst = dangjin_fcst[:8759]
dangjin_fcst.info()

# +
# %%
# -

dangjin_fcst["6시간강수량"].interpolate(method="linear", inplace=True)
dangjin_fcst["6시간적설"].interpolate(method="linear", inplace=True)
dangjin_fcst["6시간강수량"].iloc[0] = 0.0
dangjin_fcst["6시간적설"].iloc[0] = 0.0
# %%
dangjin_fcst["forecast"] = pd.to_datetime(dangjin_fcst["forecast"])
full_time = pd.Series(full_time, name="time")

data_fcst = pd.merge(full_time, dangjin_fcst, how='outer', left_on="time", right_on="forecast").drop(columns="forecast")
# %%

# +
columns_spline_interpolated = ["3시간기온", "습도", "풍향", "풍속", "일최고기온", "일최저기온"]
columns_linear_interpolated = ["하늘상태", "강수형태", "6시간강수량", "6시간적설", "강수확률"]

dataframe_interpolation_by_spline(data_fcst, columns_spline_interpolated)
dataframe_interpolation_by_linear(data_fcst, columns_linear_interpolated)

# +
# %%
'''
일단 평균으로 다 채워놓음. 이전 일시 데이터 받아서 채워야함.
시정채우기, 전운량 채우기, 
'''
data_fcst = data_fcst[:-1]
data_fcst.fillna(data_fcst.mean(), inplace=True)
data_fcst.info()

# %%

# +
# %%
import torch
import torch.nn as nn

obs_inputs = data_obs.drop(columns=['time', 'dangjin_floating'])
outputs = data_obs['dangjin_floating']

fcst_inputs = data_fcst.drop(columns=["time"])


obs_inputs.info()
# %%
obs_inputs.fillna(0, inplace=True) # 전운량 일단 0으로 채우기

obs_inputs = torch.tensor(obs_inputs.to_numpy(), dtype=torch.float32).contiguous().view(-1, 24, 19)
outputs = torch.tensor(outputs.to_numpy(), dtype=torch.float32).contiguous().view(-1, 24)

fcst_inputs = torch.tensor(fcst_inputs.to_numpy(), dtype=torch.float32).contiguous().view(-1, 24, 11)

obs_inputs.shape, outputs.shape, fcst_inputs.shape
# %%
# -

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.obs_input_size = config["obs_input_size"]
        self.fcst_input_size = config["fcst_input_size"]
        self.obs_hidden_size = config["obs_hidden_size"]
        self.fcst_hidden_size = config["fcst_hidden_size"]
        self.dropout_ratio = config["dropout_ratio"]

        self.lstm_cell1 = nn.LSTM(self.obs_input_size, self.obs_hidden_size, batch_first=True)
        self.lstm_cell2 = nn.LSTM(self.fcst_input_size, self.fcst_hidden_size, batch_first=True)


        self.dropout = nn.Dropout(self.dropout_ratio)
        
        self.linear1 = nn.Linear(self.obs_hidden_size + self.fcst_hidden_size, 50)
        self.linear2 = nn.Linear(50, 1)



        self.loss_func = nn.MSELoss()
        self.eps = 1e-6
        # https://discuss.pytorch.org/t/rmse-loss-function/16540/4

    def forward(self, obs_inputs, fcst_inputs, outputs=None):
        
        out_1, _ = self.lstm_cell1(obs_inputs)
        out_2, _ = self.lstm_cell2(fcst_inputs)

        out_ = torch.cat((out_1, out_2), dim=-1)

        out_ = self.dropout(out_)
        out_ = self.linear1(out_)
        
        pred = self.linear2(out_)
        pred = pred.squeeze(-1)
        
        if outputs is not None:
            loss = torch.sqrt(self.loss_func(pred, outputs) + self.eps)

            return loss

        else:
            return pred

# +
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {'obs_input_size': 19,
          'fcst_input_size': 11,
          'obs_hidden_size': 50,
          'fcst_hidden_size': 50,
          'dropout_ratio':0.3}


# %%
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(obs_inputs,fcst_inputs, outputs)
train_dataset, valid_dataset = random_split(dataset, [1065, 30])
train_dataloader = DataLoader(train_dataset, batch_size=128)
# %%
model = Model(config).to(device)

# +
# %%
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 200


for i in range(epochs):

    losses = []

    model.train()
    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device) for t in batch)

        obs_inputs, fcst_inputs, outputs = batch[0], batch[1], batch[2]

        optimizer.zero_grad()

        loss = model(obs_inputs, fcst_inputs, outputs)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    if i%10 == 0 : print(f"{i}-epochs Average loss : {np.mean(losses)}")

# %%

n = 9

model.eval()
a = pd.Series(model(valid_dataset[n][0].unsqueeze(0).to(device), valid_dataset[n][1].unsqueeze(0).to(device)).detach().cpu().numpy().squeeze())
b = pd.Series(valid_dataset[n][2].detach().numpy())
pd.concat([a, b], axis=1)
# %%
# -









# %%

# %%
