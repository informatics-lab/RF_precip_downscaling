import pandas as pd
import numpy as np

def add_fcst_ids(df):
    sel = np.zeros(len(df))
    for i in range(len(df) // 20112):
        sel[20112*i:20112*(i+1)] = i
    df['fcst_id'] = sel
    return df

t = ['stratiform_rainfall_amount', 'stratiform_rainfall_amount_up',
     'stratiform_rainfall_amount_down', 'stratiform_rainfall_amount_left', 'stratiform_rainfall_amount_right'] 

for n in [2, 4, 8]:
    print('processing {:d}'.format(n))
    test = add_fcst_ids(pd.read_csv('../data/newtest{:d}.csv'.format(n)))
    print('writing...')
    test.to_csv('../data/id_test{:d}.csv'.format(n))
    print('done')
