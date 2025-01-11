import tensorflow as tf
import numpy as np
from get_data import get_done_data, reverse_pair
from tqdm import tqdm
import logging
from model import Trader
import time
import os
import random
import pandas as pd
import sys
from config import Config as cfg
from utils import LightStrategy, fast_rate_of_return
import argparse


if 'gpu' not in os.environ.get('CONDA_DEFAULT_ENV'):
    tf.config.set_visible_devices([], 'GPU')
    print("GPUs have been disabled because 'gpu' is not in the Conda environment name.")
else:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print(gpu_devices)
    try:
        for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print('Could not set memory growth GPU') 



def append_to_csv(data_dict, csv_file='train_log.csv'):
    try:
        # Try to read the existing CSV file
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=data_dict.keys())
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=data_dict.keys())

    # Append the dictionary to the DataFrame
    df = pd.concat([df, pd.DataFrame(data_dict)], axis=0)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

def main():
    """
    
    Train Trader
    
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('train_log_name', type=str, help='')
    args = parser.parse_args()
    XVs, all_dec, XV, BA = get_done_data()
    
    del XVs
    del all_dec
    
#     import time
#     print('sleep')
#     time.sleep(50)
    # print(XV)
    # print(BA)
    # sys.exit(1)
    trader=Trader(input_len = XV.shape[1:])    
    
    if cfg.load_weights:
        try:
            trader.decider.set_weights( np.load('weights/decider.npy', allow_pickle=True) )
        except Exception as e:
            print("attempted to load weights and failed")
            print(e)
            # if input('Do you want to init new decider and overwrite weights? (y/n)') != 'y':
            #     raise e
            
        try:
            trader.deviser.set_weights( np.load('weights/deviser.npy', allow_pickle=True) )
        except Exception as e:
            print("attempted to load weights and failed")
            print(e)
            # if input('Do you want to init new deviser and overwrite weights? (y/n)') != 'y':
            #     raise e
            
    max_reward = 0
    test_rewards = []
    rewards = []

    # LR = 0.00001
    if cfg.reward_type == 'differential_sharpe':
        lr = 0.01
        # lr = 0.001
    else:
        lr = 0.0001
    # lr = 0.001
    

    # lr = 0.001

    bs = len(XV)
    
    n_batches = int(len(XV)/bs)
    print(n_batches)

    EPOCHS =  cfg.train_max_epoch
    batch_size = cfg.train_batch_size # 3_000 

    return_rate: float = -100.
    avg_grads: float = 0.1
    

    if cfg.test_during_training:
        _, _, XV_test, BA_test = get_done_data(f'data/test_data')

    np.save('weights/deviser.npy',np.array(trader.deviser.get_weights(), dtype=object ))
    np.save('weights/decider.npy',np.array(trader.decider.get_weights(), dtype=object ))
    
    for I in range(EPOCHS):
        start = time.time()
        
        # lr = 0.00002 if return_rate > 5 else 0.0005
        # lr = lr if avg_grads > 0.01 else round(lr/max(avg_grads,lr*10),5)
        
        trader.set_lr(lr if avg_grads>0.0001 else lr*10)
        rewards = []
        grads = []
        if cfg.exp_train_len==cfg.train_batch_size:
            grads, decisions, rewards, loss_value = trader.train_iteration(XV, BA,
                                                                           offline=cfg.offline_training, 
                                                                           online_learning=cfg.reward_type=='differential_sharpe')
            grads = [sum([x.numpy().reshape(-1,).tolist() for x in grads],[])]
        else:
            tr_it = list(range(int(batch_size*np.random.uniform(1,1.5)),len(XV), batch_size))
            random.shuffle(tr_it)
            for _i in tr_it:  
                xv,ba = XV[_i-batch_size:_i], BA[_i-batch_size:_i]
                grad, decisions, reward, loss_value = trader.train_iteration(xv, ba)
                rewards.append(reward)
                grads.append(sum([x.numpy().reshape(-1,).tolist() for x in grad],[]))
        
        train_time=time.time()-start
        return_rate = round((np.prod(rewards)-1)*100,3) if 'sharpe' not in cfg.reward_type else np.round(np.mean(rewards),5)
        avg_grads = np.mean(np.abs(sum(grads,[])))
        
        if 'sharpe' in cfg.reward_type:
            print(f"{I}/{EPOCHS}| SR: {np.round(np.mean(rewards),5)} | exec_time: {round(time.time()-start,1)} | lr: {lr} | grads: {np.round(avg_grads,7)} ") #| {round(trader.decider.a.numpy(),2)} {round(trader.decider.b.numpy(),2)} | ")
        else:
            print(f"{I}/{EPOCHS}: {return_rate} | exec_time: {round(time.time()-start,1)} | lr: {lr} | grads: {np.round(avg_grads,7)}")
        

        np.save('weights/deviser.npy',np.array(trader.deviser.get_weights(), dtype=object ))
        np.save('weights/decider.npy',np.array(trader.decider.get_weights(), dtype=object ))
    
        if cfg.test_during_training:

            _grads, decisions, reward_test, _ = trader.test_iteration(XV_test, BA_test)
            decisions = decisions.numpy()
            reward_test = np.round(reward_test,5)
            
            bas = pd.DataFrame(BA_test.numpy(), columns=['bid','ask'])
            bas['d'] = decisions
            
            return_rate_test = fast_rate_of_return(bas[['bid','ask','d']].values.tolist(), lewar=1)
                
        else:
            return_rate_test=None
            reward_test = None

        if 'train_log_name' in args:
            train_log_name = args.train_log_name
        elif 'train_log_name' in os.environ.keys():
            train_log_name = os.environ['train_log_name']
        else:
            train_log_name='train_log.csv'
            print(os.environ)
            raise Exception('no train log name')

#         XV, BA = reverse_pair(XV,BA)
        if 'START_TRAIN' in os.environ.keys():
            append_to_csv(pd.DataFrame({
                'epoch':[I],
                'time':[time.time()],
                'train_time':[train_time],
                'gain':[return_rate],
                'return_rate_test':[return_rate_test],
                'sharpe_ratio':[np.round(np.mean(rewards),5)],
                'sharpe_ratio_test':[reward_test],
                'grads':[avg_grads],
                'start_data':[int(os.environ['START_TRAIN'])],
                'reward_type': [cfg.reward_type],
                'source_data': [cfg.source_data]
            }), csv_file=train_log_name)

        del rewards
        del grads
    sys.exit(1)
    
if __name__ == '__main__':
    main()