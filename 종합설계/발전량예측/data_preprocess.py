#%%
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import time
import os
import re


file_list = glob('데이터/데이콘 데이터/*.csv')
file_list
# %%
energy = pd.read_csv(file_list[4])
dangjin_obs = pd.read_csv(file_list[3])
ulsan_fcst = pd.read_csv(file_list[5])
dangjin_fcst = pd.read_csv(file_list[6])
ulsan_obs = pd.read_csv(file_list[7])

# %%
# def convert_time(time):
#     Ymd, HMS = time.split(' ')
#     H, M, S = HMS.split(':')
#     H = str(int(H)-1)
#     HMS = ':'.join([H, M])
#     return ' '.join([Ymd, HMS])

# energy['time'] = energy['time'].apply(convert_time)
# energy['time'] = pd.to_datetime(energy['time']) + pd.DateOffset(hours=1)

#%%
def get_filelist(path):

    files = glob(path)

    return files

def read_csv(files):
    n = len(files)

    file_list = []

    for i in range(n):
        file_list.append(pd.read_csv(files[i]))

    return file_list

path = './데이터/동네예보/당진/2016/*'
get_filelist(path)

#%%

def modify_duplicated_file(files, file_list):
    '''
    중복 존재 파일 수정
    '''

    for i, file in enumerate(files):

        marking_ = file[' format: day'][file[' format: day'].apply(len)>3]

        if marking_.__len__() > 11:
            mid = np.median(marking_.index.to_numpy()).astype(np.int)

            files[i] = file.iloc[:mid,:]
            files[i].to_csv(f'{file_list[i]}', index=False)


#%%
def add_month_column_to_csv(files, file_list):
    '''
    월 칼럼 추가. 이거로 Dataframe 한번에 Merge 할거임
    '''

    for idx, file in enumerate(files):
        marks = file[' format: day'][file[' format: day'].apply(len)>3]
        marks = list(marks.index)

        file["month"] = 0
        for i in range(len(marks)+1):
            if i==0 : file["month"].iloc[:marks[i]] = 1
            elif i == len(marks) : file["month"].iloc[marks[i-1]+1:] = i+1
            else : file["month"].iloc[marks[i-1]+1:marks[i]] = i+1

        file.to_csv(file_list[idx],index=False)

#%%

def merge_all_files(files, file_names):
    
    for file in files:
        marking_ = list(file[' format: day'][file[' format: day'].apply(len)>3].index)
        file.drop(index=marking_, inplace=True)
    
    df = file
    
    for i in range(1, len(files)):
        df = pd.merge(df, files[i], how='outer', on=["month", " format: day", "hour", "forecast"])
    
    c = [" format: day","hour","forecast",file_names[0],"month"]
    c.extend([file_names[i] for i in range(1,len(file_names))])

    df.columns = c
    cc = ["month"," format: day","hour","forecast"]
    cc.extend(file_names)
    df = df[cc]

    df.sort_values(by=["month", " format: day", "hour", "forecast"], inplace=True)

    # 당진, 울산 경로 변경 필요함.
    df.to_csv(f"./데이터/동네예보/울산/{year}/ulsan_fcst_{year}.csv", index=False)





#%%
years = [2015,2016,2017,2018,2019,2020,'2021-03']

for year in years:
    path = f'데이터/동네예보/울산/{year}/*'

    file_list = get_filelist(path)
    files = read_csv(file_list)

    num_rows = list(map(len, files))
    file_names = [re.search(r'_\w+?_', filename).group().strip('_') for filename in file_list]
        
    print(num_rows)


#%%

year = 2016
path = f'데이터/동네예보/울산/{year}/*'

file_list = get_filelist(path)
files = read_csv(file_list)

num_rows = list(map(len, files))
file_names = [re.search(r'_\w+?_', filename).group().strip('_') for filename in file_list]






#%%

for i, file in enumerate(files):
    marking_ = list(file[' format: day'][file[' format: day'].apply(len)>3].index)
    print(file_list[i], file[" format: day"].iloc[marking_])



#%%
for file in files:
    marking_ = list(file[' format: day'][file[' format: day'].apply(len)>3].index)
    file.drop(index=marking_, inplace=True)

#%%
print(files[0][" format: day"].iloc[0])
print(files[1][" format: day"].iloc[0])
files[0][" format: day"].iloc[0] == files[1][" format: day"].iloc[0]

#%%
merge_all_files(files, file_names)


#%%
files[10][" format: day"] = " " + files[10][" format: day"]
files[10][" format: day"].to_numpy()
#%%










#%%






#%%
for i in range(len(files)):
    print(file_list[i], files[i][' format: day'].nunique())
#%%
for i in range(len(files)):
    print(file_list[i], files[i]['hour'].nunique())

for i in range(len(files)):
    print(file_list[i], files[i]['forecast'].nunique())
#%%

number = 8
files[number][' format: day'].unique()
files[number][files[number][' format: day'] == '7']
#%%
files[number][' format: day'].iloc[41566] = ' 7'
#files[number][' format: day'].iloc[11123] = ' 16'
#%%
files[number].to_csv(file_list[number], index=False)





