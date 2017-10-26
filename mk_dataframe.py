import pandas as pd
import os
import numpy as np
from GridcellDataset import GridcellDataset
from pathos.multiprocessing import Pool
from random import sample

path = '../data/'
paths = [f for f in os.listdir(path) if f[:5] == 'prods']
train_paths = [p for p in paths if int(p[26:28]) < 10]
test_paths = [p for p in paths if int(p[26:28]) > 10]

'''
for i in [2, 4, 8]:
    dataset = GridcellDataset(train_paths, scale_factor=i)
    if i == 2:
        filt = np.random.choice([0, 1], size=len(dataset), p=[1-0.15, 0.15])
    dataset.set_filter(filt)
    
    with Pool(15) as p:
        lsmp = p.map(lambda i: dataset[i], range(len(dataset)))
    
    df = pd.DataFrame(lsmp)
    df.to_csv('/dev/data/train{:d}.csv'.format(i))
'''
    
for i in [2, 4, 8]:
    dataset = GridcellDataset(test_paths, scale_factor=i)
    
    with Pool(15) as p:
        lsmp = p.map(lambda i: dataset[i], range(len(dataset)))
    
    df = pd.DataFrame(lsmp)
    df.to_csv('/dev/data/newtest{:d}.csv'.format(i))