# -*- coding: utf-8 -*-
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
test_df = pd.read_csv('정규화하기전.csv')
test_df['일시'] = pd.to_datetime(test_df['일시'], format='%Y-%m-%d %H:%M:%S', errors='raise')
test_df.drop(columns=['ulsan','dangjin_floating','dangjin_warehouse','dangjin'],inplace=True)

df

df.drop(columns=['풍향(16방위)','전운량(10분위)','시정(10m)','5cm 지중온도(°C)','20cm 지중온도(°C)','30cm 지중온도(°C)'],inplace=True)

df2=df.drop(columns=['ulsan','dangjin','dangjin_warehouse'])

# +


Y=df['dangjin_floating']
X =df.drop(columns=['ulsan','dangjin','dangjin_warehouse','dangjin_floating','일시'])

from sklearn.datasets import load_boston
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.1)
xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

print(len(X_train), len(X_test))
xgb_model.fit(X_train,y_train)
# -
plt.rcParams['font.family'] = 'Malgun Gothic'
xgboost.plot_importance(xgb_model)


predictions = xgb_model.predict(X_test)
predictions

r_sq = xgb_model.score(X_train, y_train)
print(r_sq)
print(explained_variance_score(predictions,y_test))

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import seaborn as sns

rf = RandomForestRegressor()
neg_mse_scores = cross_val_score(rf, X, Y, scoring="neg_mean_squared_error", cv =5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)


def get_model_cv_prediction(model, X, y):
    neg_mse_scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv =5)
    rmse_scores = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print("### {} ###".format(model.__class__.__name__))
    print("negative mse scores : ", np.round(neg_mse_scores, 2))
    print("rmse scores : ",np.round(rmse_scores, 2))
    print("avg socres : ", np.round(avg_rmse))
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
lgb = LGBMRegressor()
models = [dt, rf, gb, lgb]
for model in models:
    get_model_cv_prediction(model, X, Y)

rf.fit(X, Y)
feature_series = pd.Series(data=rf.feature_importances_,index=X.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x = feature_series, y=feature_series.index)

df.columns

df.corr()

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df_sample = df[["일사(MJ/m2)","dangjin_floating"]]
df_sample = df_sample.sample(n=10000)
plt.scatter(df_sample['일사(MJ/m2)'],df_sample.dangjin_floating)


# +
lr = LinearRegression()
rf_dep2 = DecisionTreeRegressor(max_depth=2)
rf_dep6 = DecisionTreeRegressor(max_depth=6)
rf_dep20 = DecisionTreeRegressor(max_depth=2)
X_train = df["일사(MJ/m2)"].values.reshape(-1, 1)
y_train = df["dangjin_floating"].values.reshape(-1, 1)
lr.fit(X_train, y_train)
rf_dep2.fit(X_train, y_train)
rf_dep6.fit(X_train, y_train)
rf_dep20.fit(X_train, y_train)
X_test = np.linspace(5, 8, 10000).reshape(-1, 1)

pred_lr = lr.predict(X_test)
pred_rf_dep2 = rf_dep2.predict(X_test)
pred_rf_dep6 = rf_dep6.predict(X_test)
pred_rf_dep20 = rf_dep20.predict(X_test)

fig, (ax1, ax2, ax3,ax4) = plt.subplots(figsize=(14, 4), ncols=4)
ax1.set_title("linear regression")
ax1.scatter(df_sample['일사(MJ/m2)'],df_sample.dangjin_floating)
ax1.plot(X_test, pred_lr)

ax2.set_title("RF depth 2")
ax2.scatter(df_sample['일사(MJ/m2)'],df_sample.dangjin_floating)
ax2.plot(X_test, pred_rf_dep2, color="red")

ax3.set_title("RF depth 6")
ax3.scatter(df_sample['일사(MJ/m2)'],df_sample.dangjin_floating)
ax3.plot(X_test, pred_rf_dep6, color="green")

ax4.set_title("RF depth 20")
ax4.scatter(df_sample['일사(MJ/m2)'],df_sample.dangjin_floating)
ax4.plot(X_test, pred_rf_dep20, color="black")
# -

df2.dtypes

df2['일시'] = pd.to_datetime(df['일시'], format='%Y-%m-%d %H:%M:%S', errors='raise')

df2.dtypes

from pycaret.regression import *
from pycaret.classification import *

# +
    
#clf = setup(data = df2, target = 'dangjin_floating')
#clf

# +
#best_3 = compare_models(sort='MSE', n_select = 3)

# +
#blended = blend_models(estimator_list = best_3, fold = 5)

# +

#pred_holdout = predict_model(blended)
#final_model = finalize_model(blended)
#predictions = predict_model(final_model, data = test_df)
#predictions


# +
#model_catboost = create_model('catboost', fold = 5)
#model_catboost = tune_model(model_catboost, fold=5, optimize = 'RMSE', choose_better = True)
# -

clf = setup(data = df2, target = '')
clf

df

df = pd.read_csv('ASOS_dangjin_from_2018.csv',encoding='cp949')

df

df.info()

df['풍향(16방위)'] = df['풍향(16방위)'].apply(lambda x: 7.0 if x==360 else (x//45))
df['풍향(16방위)']

df['풍향(16방위)'].value_counts()

clf = setup(data = df, 
            target = '풍향(16방위)',
            ignore_low_variance=True,
#            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            session_id = 20210302,
            combine_rare_levels = True, rare_level_threshold = 0.1)
clf

best_3 = compare_models(n_select = 3)

blended = blend_models(estimator_list = best_3, fold = 5)


pred_holdout = predict_model(blended)
final_model = finalize_model(blended)
predictions = predict_model(final_model, data = test_df)
predictions
