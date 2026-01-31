import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from math import log, factorial
import gc
from tqdm import tqdm
from waves import WaveGrasper
import tensorflow as tf
import logging
import os; 
from config import Config as cfg

# from enum import Enum


# class Mode(Enum):
#     standard

# os.environ['CUDA_VISIBLE_DEVICES']=""

CHANGE = cfg.wave_CHANGE
LENGTH = cfg.wave_LENGTH

MINUTE_DATA_PATH = "../PS/MINUTE_DATA/EURUSD/"


def main(cat: str = 'train_data', mode: str = 'standard'):
    os.environ['CUDA_VISIBLE_DEVICES']=""
    
    
    TRAIN_FRAC: float = 0.8
    TEST_FRAC: float = 0.19

        
    if mode == 'standard':

        logging.info('Starting system')
        data = pd.read_csv('data/processed/eurusd_big.csv')#.iloc[80_000:500_000,:].reset_index(drop=True)
        data['index'] = data.index
        data.describe()['close'].T

        subdf = data[['close']].copy()
        subdf['bid'], subdf['ask'] = subdf.close - 0.00008, subdf.close + 0.00008

        DF = subdf

        ### Pipeline

        DF.columns = [str(x) for x in DF.columns]
        pred_df = DF.copy()#[['assum','now','score']].copy()

        X = pred_df[['close']]

        wg = WaveGrasper(data.close, CHANGE, gather_point_data=True, length=LENGTH)
        wg.df.loc[0,'price'] = data.close.values[0]
        wg.df.sort_index(inplace=True)
        wg.df['index'] = wg.df.index

        len(wg.df)

        DF = pd.DataFrame([x[2]+[x[3+1],x[4+1]] for x in wg.point_data], columns=[f"tp{i}" for i in range(wg._length)][::-1]+['assum','now'], index=[x[0] for x in wg.point_data])
        time_assum = pd.DataFrame([x[0]-x[1] for x in wg.point_data], columns=['time_since'], index=[x[0] for x in wg.point_data])

        DF = pd.concat([DF, subdf, time_assum],1).dropna()

        df = wg.df
        concated_other = pd.concat( [pd.Series([x[0] for x in wg.point_data][:-1], index=[int(x[0]) for x in wg.point_data][:-1])],1)
        del concated_other[0]
        concated_other = concated_other.ffill().dropna()
        concated_other = pd.concat( [df.price.rename('y'), concated_other],1).sort_index()
        concated_other['y'] = concated_other.y.bfill()#.replace(np.nan, None)
        concated_other.dropna(inplace=True)

        DF = pd.concat([DF, concated_other['y']],1).dropna()
        # DF[['now','y']].plot()

        # dd = DF[['y']].drop_duplicates()
        # DF['prev_y'] = DF['y'].map( {k:v for k,v in zip(dd.values.reshape(-1,).tolist(), dd.shift(1).values.reshape(-1,).tolist())} )

        # DF.dropna(inplace=True)
        # DF['will_y'] = DF[f'assum'] == DF["prev_y"]
        logging.info('Main processing done')


        X = DF[[x for x in DF.columns if 'tp' in x or x in ['assum','now']]]
        # X.iloc[[0]].T.plot()

        # X = DF[['close']]
        y = (DF.y > DF.now).astype(int)

        gc.collect()

        TIMEPERIODS = 30
        RETURNPERIOD = 1000 # 100 run

        step = 1
        titer = range(max(TIMEPERIODS,RETURNPERIOD), 1_110_000, step)

        X_vals = X.values

        bid_ask_vals = DF[['bid','ask']].values

        X_vals = X_vals[titer.start:titer.stop:step,:]
        bid_ask_vals =  bid_ask_vals[titer.start:titer.stop:step,:]


        TRAIN_FRAC: float = 0.5
        TEST_FRAC: float = 0.15
        

        if 'test' in cat:
            border = int((1-TEST_FRAC)*len(X_vals))
            left = border
            length= len(X_vals)-left 
        else:
            border = int(TRAIN_FRAC*len(X_vals))
            left = 0 
            length= border
        print(cat, left, left+length, len(X_vals))
        

#         if 'test' in cat:
#             left = len(X_vals) - 100_000
#             length= len(X_vals)-left 
#         else:
#             left = 0 # len(X_vals) - 100_000
#             length= 100_000


        logging.info('Get vars and save')

        XV = tf.constant( X_vals[left:left+length], tf.float32 )
        BA = tf.constant( bid_ask_vals[left:left+length], tf.float32 )                    

#         zeros = tf.zeros((len(XV),1))
#         ones = tf.ones((len(XV),1))

#         all_dec = tf.concat( [ tf.concat([ones, zeros, zeros],1), tf.concat([zeros, ones, zeros],1), tf.concat([zeros, zeros, ones],1) ],0)

        # XVs = tf.concat([XV,XV,XV],0)

        np.save(f'{cat}/XV.npy', XV.numpy())
        np.save(f'{cat}/BA.npy', BA.numpy())

        logging.info('Finito')
        
    elif mode == 'technical':
        if f'DF_technical.csv' not in os.listdir('data/processed/'):   
            data = pd.read_csv('data/processed/input_table_15s.csv').sort_index()#.reset_index(drop=True)

            data['Close'] = data.bid/10

            # Step 2: Moving Averages
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['SMA_200'] = data['Close'].rolling(200).mean()
            data['SMA_500'] = data['Close'].rolling(500).mean()
            data['SMA_1000'] = data['Close'].rolling(1000).mean()


            data['SMA_200_50'] = data['Close'].rolling(200).mean() - data['SMA_50']
            data['SMA_500_50'] = data['Close'].rolling(500).mean() - data['SMA_50']
            data['SMA_1000_50'] = data['Close'].rolling(1000).mean() - data['SMA_50']


            data['SMA_50_500'] = data['Close'].rolling(50).mean() - data['SMA_500']
            data['SMA_200_500'] = data['Close'].rolling(200).mean() - data['SMA_500']
            data['SMA_500_500'] = data['Close'].rolling(500).mean() - data['SMA_500']
            data['SMA_1000_500'] = data['Close'].rolling(1000).mean() - data['SMA_500']

            data['SMA_50_x'] = data['Close'].rolling(50).mean() - data.Close
            data['SMA_200_x'] = data['Close'].rolling(200).mean() - data.Close
            data['SMA_500_x'] = data['Close'].rolling(500).mean() - data.Close
            data['SMA_1000_x'] = data['Close'].rolling(1000).mean() - data.Close

            del data['Close']
            del data['SMA_50']
            del data['SMA_200']
            del data['SMA_500']
            del data['SMA_1000']


            data.dropna(inplace=True)
            DF = data 
            DF.to_csv(f'data/processed/DF_technical.csv', index=False)
        else:
            DF = pd.read_csv(f'data/processed/DF_technical.csv')

        titer = range(0, len(DF), step:=1)

        X_vals = DF[[x for x in DF.columns if x not in ['bid','ask','time']]].values
        bid_ask_vals = DF[['bid','ask']].values

        X_vals = X_vals[titer.start:titer.stop:step,:]
        bid_ask_vals =  bid_ask_vals[titer.start:titer.stop:step,:]

    elif mode == 'diff':
        M = cfg.M
        if f'DF_diff_{M}.csv' not in os.listdir('data/processed/'):   
            # data = pd.read_csv('data/processed/input_table_15s.csv').sort_index()
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
        
        logging.info('Starting system')
        # data = pd.read_csv('eurusd_big.csv')#.iloc[80_000:500_000,:].reset_index(drop=True)


        if f'DF_{CHANGE}_{LENGTH}.csv' not in os.listdir('data/processed/'):    
            print('Calculate DF')
            if False:
                data = pd.concat([
                    pd.read_csv(MINUTE_DATA_PATH+x) for x in os.listdir(MINUTE_DATA_PATH)
                    if '.csv' in x], 0
                ).sort_values('time').iloc[:2_000_000].reset_index(drop=True).drop_duplicates(subset=['time'])
            else:
                data = pd.read_csv('input_table_15s.csv').sort_index()#.reset_index(drop=True)
            
            print(data)
            # return
            
            data['index'] = data.index
            DF = subdf = data

            ### Pipeline

            DF.columns = [str(x) for x in DF.columns]
            pred_df = DF.copy()#[['assum','now','score']].copy()

            X = pred_df[['bid']]

            wg = WaveGrasper(data.bid, CHANGE, gather_point_data=True, length=LENGTH)
            wg.df.loc[0,'price'] = data.bid.values[0]
            wg.df.sort_index(inplace=True)
            wg.df['index'] = wg.df.index

            print(len(wg.df))

            DF = pd.DataFrame([x[2]+[x[3+1],x[4+1]] for x in wg.point_data], columns=[f"tp{i}" for i in range(wg._length)][::-1]+['assum','now'], index=[x[0] for x in wg.point_data])
            time_assum = pd.DataFrame([x[0]-x[1] for x in wg.point_data], columns=['time_since'], index=[x[0] for x in wg.point_data])
            print(time_assum)
            DF = pd.concat([DF, subdf],1).dropna()
            DF.to_csv(f'data/processed/DF_{CHANGE}_{LENGTH}.csv', index=False)
            # DF = pd.concat([DF, subdf, time_assum],1).dropna() # important
            
            # df = wg.df
            # concated_other = pd.concat( [pd.Series([x[0] for x in wg.point_data][:-1], index=[int(x[0]) for x in wg.point_data][:-1])],1)
            # del concated_other[0]
            # concated_other = concated_other.ffill().dropna()
            # concated_other = pd.concat( [df.price.rename('y'), concated_other],1).sort_index()
            # concated_other['y'] = concated_other.y.bfill()#.replace(np.nan, None)
            # concated_other.dropna(inplace=True)

    #         DF = pd.concat([DF, concated_other['y']],1).dropna()
        else:
            DF = pd.read_csv(f'data/processed/DF_{CHANGE}_{LENGTH}.csv')

        logging.info('Main processing done')

        X = DF[[x for x in DF.columns if 'tp' in x or x in ['assum','now']]]

        gc.collect()

        TIMEPERIODS = 30
        RETURNPERIOD = 1000 # 100 run

        step = 1
        titer = range(max(TIMEPERIODS,RETURNPERIOD), len(X), step)

        X_vals = X.values

        bid_ask_vals = DF[['bid','ask']].values

        X_vals = X_vals[titer.start:titer.stop:step,:]
        bid_ask_vals =  bid_ask_vals[titer.start:titer.stop:step,:]


    elif mode == 'trans':
                
        DF = pd.read_csv('data/processed/input_table_trans_data_aud_jpy.csv', index_col=0)
        
        X_vals = DF.drop(columns=['time','bid','ask']).values
        
        X_vals = (np.sign(X_vals)*np.abs(X_vals)**0.5).astype(np.float32)
        
        print(DF.drop(columns=['time','bid','ask']))
        
        bid_ask_vals = DF[['bid','ask']].values

        if 'test' in cat:
            border = int((1-TEST_FRAC)*len(X_vals))
            left = border
            length= len(X_vals)-left 
        else:
            border = int(TRAIN_FRAC*len(X_vals))
            left = 0 
            length= border
        print(cat, left, left+length, len(X_vals))

        logging.info('Get vars and save')

        XV =  X_vals[left:left+length]
        BA = bid_ask_vals[left:left+length]                 

        np.save(f'{cat}/XV.npy', XV)
        np.save(f'{cat}/BA.npy', BA)

    
    print(os.environ)

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
#     print('INT!')
    BA = tf.constant( BA, tf.float32 )                    

#     zeros = tf.zeros((len(XV),1))
#     ones = tf.ones((len(XV),1))
#     all_dec = tf.concat( [ tf.concat([ones, zeros, zeros],1), 
#                           tf.concat([zeros, ones, zeros],1), 
#                           tf.concat([zeros, zeros, ones],1) ],0)

#     XVs = tf.concat([XV,XV,XV],0)
    if cfg.reverse_pair:
        XV = 1/XV
        BA = 1/BA[:,::-1]
    
    return None,None, XV, BA

    
def reverse_pair(XV, BA):
    XV = 1/XV
    BA = 1/BA[:,::-1]
    return XV, BA
    
    
if __name__ == '__main__':
    MODE = 'standard_trans'
    main('', cfg.mode)