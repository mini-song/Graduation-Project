#%%
import missingno as msno
import pandas as pd
import numpy as np
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# df1 = pd.read_csv('ASOS_dangjin.csv',encoding='cp949')
df2 = pd.read_csv('ASOS_dangjin_from_2018.csv',encoding='cp949')
# df3 = pd.read_csv('ASOS_ulsan.csv',encoding='cp949')
df4 = pd.read_csv('dangjin_floating_energy.csv',encoding='cp949')

df2
df22 = df2


#%%
def drop_columns(df):
    df.drop(columns=df.columns[[0, 14, 16, 17, 18, 20, 21]], inplace = True)

drop_columns(df2)
#%%

발전량 = pd.read_csv('en.csv')

발전량[발전량['time'] == '2018-01-01 01:00:00']

발전량[발전량['time'] == '2020-12-31 23:00:00']

발전량spp=발전량.iloc[26304:52606]
발전량spp = 발전량spp.reset_index()


def 발전량추가(df):
    # df['ulsan'] = 발전량spp['ulsan']
    df['dangjin_floating'] = 발전량spp['dangjin_floating']
    # df['dangjin_warehouse'] = 발전량spp['dangjin_warehouse']
    # df['dangjin'] = 발전량spp['dangjin']


발전량추가(df2)

#%%
def Spline(df,parameter):
    

    x_inter = pd.Series(df[parameter].interpolate(method='spline',order=3))
    df[parameter]= x_inter                    
    
    #return df[df2[parameter].isnull()]


def Snow_rain_0(df):
    df['적설(cm)'] = df['적설(cm)'].fillna(0)
    df['강수량(mm)'] = df['강수량(mm)'].fillna(0)


def 일조일사(df):
    df['일조(hr)'] = df['일조(hr)'].fillna(0)
    df['일사(MJ/m2)'] = df['일사(MJ/m2)'].fillna(0)

#%%
plt.rcParams['font.family'] = 'NanumGothic'
plt.figure(figsize=(17, 17))
correlations = df2.corr(method='pearson')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0, annot=True)

plt.rcParams['font.family'] = 'NanumGothic'
msno.matrix(df2)
plt.show()

msno.bar(df2)
plt.show()

Snow_rain_0(df2)

일조일사(df2)

msno.bar(df2)
plt.show()
#%%
Spline(df2,'해면기압(hPa)')


Spline(df2,'풍속(m/s)')


#Spline(df2,'풍향(16방위)') #방위 스플라인 보간 에바인것 같은데
df2['풍향(16방위)']

Spline(df2,'습도(%)')

Spline(df2,'기온(°C)')

Spline(df2,'증기압(hPa)')

Spline(df2,'이슬점온도(°C)')

Spline(df2,'현지기압(hPa)')

Spline(df2,'시정(10m)')

Spline(df2,'지면온도(°C)')

Spline(df2,'10cm 지중온도(°C)')

# Spline(df2,'ulsan')

Spline(df2,'dangjin_floating')

# Spline(df2,'dangjin_warehouse')

# Spline(df2,'dangjin')

#%%
# plt.rcParams['font.family'] = 'NanumGothic'
# msno.matrix(df2)
# plt.show()

# msno.bar(df2)
# plt.show()

df2.columns

df_nom = df2

df_nom.drop(columns='일시',inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_nom[ : ] = scaler.fit_transform(df_nom[ : ])

df_nom['일시'] = df2['일시']

# df_nom.to_csv('./정규화.csv',index=False)

df_nom.info()






