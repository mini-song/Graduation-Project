#%%
import pandas as pd
import numpy as np
import glob
import os

from torch.nn.modules import loss

path = "./데이터"

dangjin_obs = pd.read_csv(os.path.join(path, "ASOS_hour/ASOS_dangjin (copy).csv"), encoding='cp949')
# ulsan_obs = pd.read_csv(os.path.join(path, "ASOS_hour/ASOS_ulsan.csv"), encoding='cp949')
energy = pd.read_csv(os.path.join(path, "데이콘 데이터/energy.csv"))

dangjin_obs.info()
#%%

dangjin_obs.drop(columns=dangjin_obs.columns[[0,3,11,12,13,14,17,18,20,21,23,24,25,26]], inplace=True)

mean = dangjin_obs.mean(axis=0)

dangjin_obs.fillna(mean, inplace=True)

dangjin_obs.info()
#%%
print(mean)
dangjin_obs
#%%
'''
time과 일시 컬럼 오류값 검사
'''
print(dangjin_obs["일시"].apply(len).unique())
print(energy["time"].apply(len).unique())


#%%
def abnormal_value_modify(time):
    if len(time) == 15:
        YMD, HM = time.split()
        HM = "0" + HM
        return ' '.join([YMD, HM])
    else:
        return time

dangjin_obs["일시"] = dangjin_obs["일시"].apply(abnormal_value_modify)
# %%
energy_dangjin = energy[["time", "dangjin"]].fillna(energy["dangjin"].mean(axis=0))
energy_dangjin["time"] = energy_dangjin["time"].apply(lambda x: x[:-3])

# %%
dataset = pd.merge(dangjin_obs, energy_dangjin, how="inner", left_on="일시", right_on="time")
dataset.drop(columns='time', inplace=True)


#%%
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#%%

inputs_ = torch.tensor(dataset.iloc[:,1:-1].to_numpy(), dtype=torch.float32)
outputs_ = torch.tensor(dataset.iloc[:,-1].to_numpy(), dtype=torch.float32)

inputs_.shape, outputs_.shape

#%%

class LSTM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.dropout_ratio = config["dropout_ratio"]

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(self.dropout_ratio)


        self.loss_func = nn.MSELoss()

    def forward(self, input, result):

        if result is not None:


            hidden = self.lstm(input)
            hidden = self.dropout(hidden[0])
            model_out = self.linear(hidden)

            model_out = model_out.contiguous().view(-1, 1)
            result = result.contiguous().view(-1)

            loss = self.loss_func(model_out, result)


            return loss
        else:

            hidden = self.lstm(input)
            hidden = self.dropout(hidden)
            model_out = self.linear(hidden)

            return model_out 

#%%
config = {"input_size": 12,
          "hidden_size": 50,
          "dropout_ratio": 0.3}

#%%
inputs_ = inputs_[:52416].view(-1, 24, 12)
outputs_ = outputs_[:52416].view(-1, 24)



#%%
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(inputs_, outputs_)
train_dataset, valid_dataset = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)+1])
train_dataloader = DataLoader(train_dataset, batch_size=64)

model = LSTM(config).to(device)


#%%
learning_rate = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
epoch = 8


for i in range(epoch):
    
    losses = []

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)

        inputs, outputs = batch[0], batch[1]

        optimizer.zero_grad()

        loss = model(inputs, outputs)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    print("Average loss : {}\n".format(np.mean(losses)))


#%%


# %%