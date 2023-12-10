import tensorflow as tf
import numpy as np
from get_data import get_done_data
from tqdm import tqdm
import logging
from model import Trader
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from utils import LightStrategy, fast_rate_of_return
import gc
from config import Config as cfg
os.environ['CUDA_VISIBLE_DEVICES']=""

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# try:
#     for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
# except: 
#     ...

def append_to_csv(data_dict, csv_file='test_log.csv'):
    try:
        # Try to read the existing CSV file
        try:
            df = pd.read_csv(csv_file)
        except:
            time.sleep(0.1)
            df = pd.read_csv(csv_file)

    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=data_dict.keys())

    # Append the dictionary to the DataFrame
    df = df.append(data_dict, ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)


def main(plot_i, name = 'test'):
    """
    
    Test Trader
    
    """
    unlever = True
    
    XVs, all_dec, XV, BA = get_done_data(f'{name}_data')
    del XVs
    del all_dec
    trader=Trader(input_len = XV.shape[1:])    
    
    trader.deviser.set_weights( np.load('weights/deviser.npy', allow_pickle=True) )
    trader.decider.set_weights( np.load('weights/decider.npy', allow_pickle=True) )

    test_rewards = []

    rewards = []
    
    trader.set_lr(0)
    # Grads = []
    
    xv,ba = XV, BA
    
    grads, decisions, reward, loss_value = trader.test_iteration(xv, ba)
    
    decisions = decisions.numpy()
    rewards.append(reward)
    
    decs = {k:decisions.tolist().count(k) for k in set(decisions)} | {'n_per_week':round( np.abs(np.diff(decisions)).sum()/2/(BA.shape[0]/(60*24*5)),3),'avg_time_open_h':round(np.abs(decisions).sum()/(np.abs(np.diff(decisions)).sum()/2)/60,1)}    
    
    for potential_lack in [-1,0,1]:
        if potential_lack not in decs.keys():
            decs[potential_lack] = 0
    
    
    bas = pd.DataFrame(BA.numpy(), columns=['bid','ask'])
    bas['d'] = decisions
    ls = LightStrategy()
    ev = ls.evaluate(bas, collect_result=True)
    
    if unlever:
        fast_ev = fast_rate_of_return(bas[['bid','ask','d']].values.tolist(), lewar=1)
    else:
        fast_ev = fast_rate_of_return(bas[['bid','ask','d']].values.tolist())
        
    lit_rev_metrics_ev = bas.d.shift() * bas.bid.diff()/bas.bid.shift() - (bas.ask - bas.bid) * (bas.d.diff())
    lit_rev_metrics_ev = lit_rev_metrics_ev.sum()

    fast_ev_500 = round((fast_rate_of_return(bas[['bid','ask','d']].values.tolist()[:500])-1)*100,3)
    fast_ev_1000 = round((fast_rate_of_return(bas[['bid','ask','d']].values.tolist()[:1000])-1)*100,3)
    fast_ev_2000 = round((fast_rate_of_return(bas[['bid','ask','d']].values.tolist()[:2000])-1)*100,3)
    fast_ev_5000 = round((fast_rate_of_return(bas[['bid','ask','d']].values.tolist()[:5000])-1)*100,3)
    
    print(f"""
        Result: 
            Neural: {round((np.prod(rewards)-1)*100,3)}%
            Real:   {round((ev-1)*100,3)}%
            Real:   {round((fast_ev-1)*100,3)}%
            Lit rev real: {round((lit_rev_metrics_ev)*100,3)}
        Decisions: 
            sell: {decs[-1]}
            stop: {decs[0]}
            buy:  {decs[1]}
        Transactions:
            n per week:        {decs['n_per_week']}
            avg oepn time [h]: {decs['avg_time_open_h']}
        """)

    
    if unlever:
        rewards = np.array(rewards)
        rewards = rewards-1
        rewards = rewards/cfg.lewar
        rewards = rewards+1

        x = ls.result
        ls.result = np.cumprod([1]+(1+np.diff(x)/x[:-1]/cfg.lewar).tolist()).tolist()

#         fast_ev = fast_ev-1
#         fast_ev = fast_ev/cfg.lewar
#         fast_ev = fast_ev+1

    try:
        train_log = pd.read_csv('train_log.csv').iloc[[-1]]

        if train_log.gain.values[0] == -1:
            return

        start_data = train_log.start_data.values[0]+cfg.exp_train_len


        append_to_csv(pd.DataFrame({
            'time':time.time(),
            'gain_500':[fast_ev_500],
            'gain_1000':[fast_ev_1000],
            'gain_2000':[fast_ev_2000],
            'gain_5000':[fast_ev_5000],
            'gain_all':[round((fast_ev-1)*100,3)],
            'gain_reward':[round((np.prod(rewards)-1)*100,3)],
            'gain_lit_rev':[round((lit_rev_metrics_ev)*100,3)],
            'dec_sell':[decs[-1]],
            'dec_stop':[decs[0]],
            'dec_buy':[decs[1]],
            'n_per_week':[decs['n_per_week']],
            'start_data':start_data
        }), 'test_log.csv')
    
    except FileNotFoundError:
        start_data=-1

    fig, ax = plt.subplots(figsize=(8,4))
    
    plt.plot(bas.index.values[::10], bas.bid.values[::10], color='black', lw=0.2, alpha=0.4)
    
    plt.scatter(bas[bas.d == -1].index, bas[bas.d == -1].bid, color='red', s=2)
    plt.scatter(bas[bas.d == 1].index, bas[bas.d == 1].bid, color='green', s=2)

    ax2 = ax.twinx()
    ax2.plot(np.array(ls.result)*100-100) 
    ax2.set_ylim(-8, max(np.array(ls.result)*100-100)+5)

    ax2.set_ylabel('Rate of return [%]',labelpad=15)
    
    plt.title(f"Rate of return approximation: {round((np.prod(rewards)-1)*100,3)}%\nReal rate of return: {round((fast_ev-1)*100,3)}%")
    
    plt.savefig(f'test_results/{name}_plot_{start_data}.png') # {plot_i}.png')
    
    
if __name__ == '__main__':
    i = 0
    while 'stop :)':
        try:
            main(i, 'test')
        except Exception as e:
            print(f'some error: {e}')
            time.sleep(2)
            
        # main(i, 'train')

        gc.collect()
        tf.keras.backend.clear_session()
        plt.clf()
        plt.close()

        time.sleep(5)
        i+=1
        # except: 
        #     pass
    