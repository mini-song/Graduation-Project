# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np


# +

def parallel_processing(dataframe, function, num_cores):
    '''
    forecast 컬럼을 예보데이터가 나타내는 시간으로 변경하는 함수가 forecast_change.
    멀티 프로세싱으로 처리하기 위한 함수.
    '''
    import numpy as np
    import time
    import pandas as pd
    start = time.time()
    df_split = np.array_split(dataframe, num_cores)
    pool = Pool(num_cores)
    dataframe = pd.concat(pool.map(function, df_split))
    pool.close()
    pool.join()

    print('\n\n\n',f'processing time : {time.time()-start} s')
    
    return dataframe 


# -

np.array([1,2])


