import pandas as pd
import numpy as np
import warnings
import gc
import tensorflow as tf
import logging
import os
from config import Config as cfg

def main(cat: str = 'train_data', mode: str = 'standard'):
    os.environ['CUDA_VISIBLE_DEVICES']=""
    if mode == 'standard':
        ...

    elif mode == 'technical':
        ...

    elif mode == 'diff':
        M = cfg.M
        if f'DF_diff_{M}.csv' not in os.listdir('data/processed/'):   
            data = pd.read_csv(f'data/{cfg.source_data}').sort_index().rename(columns={'Close':'bid','Date':'time'})
            
            if 'ask' not in data.columns:
                data['ask'] = data.bid * 1.01

            for m in range(1, M+1):
                # data[f'Close_{m}'] = data.bid.pct_change(m)
                data[f'Close_{m}'] = data.bid.diff(m)

            data.dropna(inplace=True)
            DF = data 
            DF.to_csv(f'data/processed/DF_diff_{M}.csv', index=False)
        else:
            DF = pd.read_csv(f'data/processed/DF_diff_{M}.csv')

        titer = range(0, len(DF), step:=1)

        X_vals = DF[[x for x in DF.columns if x not in ['bid','ask','time']]].values
        bid_ask_vals = DF[['bid','ask']].values

        X_vals = X_vals[titer.start:titer.stop:step,:]
        bid_ask_vals =  bid_ask_vals[titer.start:titer.stop:step,:]

    elif mode == 'historical':
        M = cfg.M
        if f'DF_historical_{M}.csv' not in os.listdir('data/processed/'):   
            # data = pd.read_csv('data/processed/input_table_15s.csv').sort_index()
            data = pd.read_csv(f'data/{cfg.source_data}').sort_index().rename(columns={'Close':'bid','Date':'time'})
            
            if 'ask' not in data.columns:
                data['ask'] = data.bid * 1.01

            for m in range(0, M+1):
                data[f'Close_{m}'] = data.bid.shift(m)

            data.dropna(inplace=True)
            DF = data 
            DF.to_csv(f'data/processed/DF_historical_{M}.csv', index=False)
        else:
            DF = pd.read_csv(f'data/processed/DF_historical_{M}.csv')

        titer = range(0, len(DF), step:=1)

        X_vals = DF[[x for x in DF.columns if x not in ['bid','ask','time']]].values
        bid_ask_vals = DF[['bid','ask']].values

        X_vals = X_vals[titer.start:titer.stop:step,:]
        bid_ask_vals =  bid_ask_vals[titer.start:titer.stop:step,:]

    elif mode == 'standard_trans':
        ...

    elif mode == 'trans':
        ...

    start_train = int(os.environ['START_TRAIN'])
    len_train = cfg.exp_train_len
    len_test = cfg.exp_test_len

    for cat in ['data/train_data','data/test_data']:

        if 'test' in cat:
            left=start_train+len_train
            length=len_test
        else:
            left=start_train
            length=len_train

        print(cat, left, left+length, len(X_vals))
        
        logging.info('Get vars and save')
        print(X_vals)
        XV = tf.constant( X_vals[left:left+length], tf.float32 )
        BA = tf.constant( bid_ask_vals[left:left+length], tf.float32 )                    

        np.save(f'{cat}/XV.npy', XV.numpy())
        np.save(f'{cat}/BA.npy', BA.numpy())

    logging.info('Finito')

        
def get_done_data(cat: str = 'data/train_data'):
    XV = np.load(f'{cat}/XV.npy')
    BA = np.load(f'{cat}/BA.npy')
    
    XV = tf.constant( XV, tf.float32 )
    BA = tf.constant( BA, tf.float32 )                    

    if cfg.reverse_pair:
        XV = 1/XV
        BA = 1/BA[:,::-1]
    
    return None,None, XV, BA

    
if __name__ == '__main__':
    main('', cfg.mode)