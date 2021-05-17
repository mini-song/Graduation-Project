#%%

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn import tree
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
import missingno as msno
import pandas as pd
import numpy as np
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('정규화.csv')

df

df.drop(columns=['풍향(16방위)','전운량(10분위)','시정(10m)','5cm 지중온도(°C)','20cm 지중온도(°C)','30cm 지중온도(°C)','일시'],inplace=True)

df

#%%


Y=df['dangjin_floating']
X =df.drop(columns=['ulsan','dangjin','dangjin_warehouse','dangjin_floating'])

from sklearn.datasets import load_boston
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.1)
xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

print(len(X_train), len(X_test))
xgb_model.fit(X_train,y_train)
# -
plt.rcParams['font.family'] = 'NanumGothic'
xgboost.plot_importance(xgb_model)


predictions = xgb_model.predict(X_test)

print('score :', xgb_model.score(X_train, y_train))
print('explained_score :',explained_variance_score(predictions,y_test))
print('r2_score :',r2_score(predictions,y_test))





#%%
import matplotlib
import matplotlib.font_manager as fm


[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
