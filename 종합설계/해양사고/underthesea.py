#%%
import os
path = './Desktop/major/종합설계/해양사고'
os.listdir(path)

#%%
import pandas as pd
table = pd.read_csv(os.path.join(path, './2019년 해상조난사고 통계 상세.csv'), 'r', encoding='cp949',delimiter=',')


#%%
mini = table[["발생원인", "발생유형", "발생인원", "구조", "부상", "사망", "실종"]]

a = mini[mini["발생인원"] == 0]

mini["발생유형"].value_counts()
# %%

path = './Desktop/major/종합설계/kt_data/'
os.listdir(path)
#%%
t = pd.read_csv(os.path.join(path, 'card_20200717.csv'), 'r', encoding='utf8', delimiter=',')
#%%
