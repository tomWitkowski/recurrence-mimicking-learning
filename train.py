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
from config import Config as cfg

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# print(os.environ['CUDA_VISIBLE_DEVICES'])
print(gpu_devices)

# sys.exit()

try:
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
except: 
    ...


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
    df = df.append(data_dict, ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

def main():
    """
    
    Train Trader
    
    """
    XVs, all_dec, XV, BA = get_done_data()
    
    del XVs
    del all_dec
    
#     import time
#     print('sleep')
#     time.sleep(50)
    trader=Trader(input_len = XV.shape[1:])    
    
    try:
        trader.decider.set_weights( np.load('weights/decider.npy', allow_pickle=True) )
    except Exception as e:
        print(e)
        if input('Do you want to init new decider and overwrite weights? (y/n)') != 'y':
            raise e
        
    try:
        trader.deviser.set_weights( np.load('weights/deviser.npy', allow_pickle=True) )
    except Exception as e:
        print(e)
        if input('Do you want to init new deviser and overwrite weights? (y/n)') != 'y':
            raise e
            
    max_reward = 0

    test_rewards = []

#     LR = 0.00001
    lr = 0.0001
    
    rewards = []
    
    bs = len(XV)
    
    n_batches = int(len(XV)/bs)
    print(n_batches)

    EPOCHS =  cfg.train_max_epoch
    batch_size = cfg.train_batch_size # 3_000 

    return_rate: float = -100.
    avg_grads: float = 0.1
    
    np.save('weights/deviser.npy',trader.deviser.get_weights())
    np.save('weights/decider.npy',trader.decider.get_weights())
    
    for I in range(EPOCHS):
        start = time.time()
        
        # lr = 0.00002 if return_rate > 5 else 0.0005
        # lr = lr if avg_grads > 0.01 else round(lr/max(avg_grads,lr*10),5)
        
        trader.set_lr(lr if avg_grads>0.0001 else lr*10)
        
        rewards = []
        grads = []
        
        tr_it = list(range(int(batch_size*np.random.uniform(1,1.5)),len(XV), batch_size))
        random.shuffle(tr_it)
        for _i in tr_it:  # int(batch_size/2)):
#             print(_i,end=' ')
            xv,ba = XV[_i-batch_size:_i], BA[_i-batch_size:_i]
        
#             if (_i+I)%2:
#                 xv,ba = reverse_pair(xv,ba)
            
            # if len(xv)<batch_size:
            #     continue
            grad, decisions, reward, loss_value = trader.train_iteration(xv, ba)
            # print(f"{round(reward.numpy(),2)} {round(grad[0].numpy()[0],2)}")
            
            rewards.append(reward)
            grads.append(sum([x.numpy().reshape(-1,).tolist() for x in grad],[]))
        
        return_rate = round((np.prod(rewards)-1)*100,3)

        avg_grads = np.mean(np.abs(sum(grads,[])))
        
        print(f"{I}/{EPOCHS}: {return_rate} | exec_time: {round(time.time()-start,1)} | lr: {lr} | grads: {np.round(avg_grads,7)}")
        
        del rewards
        del grads

        np.save('weights/deviser.npy',trader.deviser.get_weights())
        np.save('weights/decider.npy',trader.decider.get_weights())
    
#         XV, BA = reverse_pair(XV,BA)
        if 'START_TRAIN' in os.environ.keys():
            append_to_csv(pd.DataFrame({
                'time':time.time(),
                'exec_time':[round(time.time()-start,3)],
                'gain':[return_rate],
                'grads':[avg_grads],
                'start_data':[int(os.environ['START_TRAIN'])]
            }))
    
if __name__ == '__main__':
    main()