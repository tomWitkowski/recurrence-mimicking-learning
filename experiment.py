import os
import subprocess
import time
import csv
from config import Config as cfg
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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


def get_last_gain(col='gain', train_log_name:str='train_log.csv'):
    try:
        df = pd.read_csv(train_log_name)
        if not df.empty:
            return float(df[col].iloc[-1])
        else:
            return 0.0
    except FileNotFoundError:
        return 0.0

def main(train_log_name='train_log.csv'):
    if os.path.exists('weights/deviser.npy'):
        os.remove('weights/deviser.npy')
    if os.path.exists('weights/decider.npy'):
        os.remove('weights/decider.npy')

    print(f"Started for: {train_log_name}")

    START = cfg.exp_train_len

    if get_last_gain('start_data') != 0:
        START = int( get_last_gain('start_data')+cfg.exp_train_len+cfg.exp_test_len )
    print(START)
    for end_train in range(START, cfg.exp_data_len+cfg.exp_test_len, cfg.exp_test_len):
        os.environ['START_TRAIN'] = str( end_train - cfg.exp_train_len )

        append_to_csv(pd.DataFrame({
            'time':time.time(),
            'exec_time':-1,
            'gain':-1,
            'grads':-1,
            'start_data':[int(os.environ['START_TRAIN'])]
        }), csv_file=train_log_name)
        # Run get_data.py
        os.environ['train_log_name'] = train_log_name
        subprocess.run(['python', 'get_data.py'], env=os.environ)
        print('-'*20)
        
        append_to_csv(pd.DataFrame({
            'time':time.time(),
            'train_time':0,
            'gain':0,
            'grads':0,
            'return_rate_test':None,
            'sharpe_ratio':None,
            'sharpe_ratio_test':None,
            'start_data':[int(os.environ['START_TRAIN'])],
            'reward_type': [cfg.reward_type],
            'source_data': [cfg.source_data]
        }),csv_file=train_log_name)

        # Start train.py in the background
        train_process = subprocess.Popen(['python', 'train.py', train_log_name])
        time.sleep(1)

        while True:
            try:
                # Check last row from train_log.csv
                last_gain = get_last_gain(train_log_name=train_log_name)
                
                # Check if gain is greater than 100
                if last_gain > cfg.exp_return_stop:
                    print(f"Experiment completed. Gain: {last_gain}")
                    train_process.terminate()
                    time.sleep(5)
                    # Terminate the background process
                    break

                if train_process.poll() is not None:
                    break
                
                # Wait for a while before the next iteration
                time.sleep(0.2)

            except Exception as e:
                print(e)
    
    if os.path.exists('weights/deviser.npy'):
        os.remove('weights/deviser.npy')
    if os.path.exists('weights/decider.npy'):
        os.remove('weights/decider.npy')
    

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in [5,12,13,14,15,17,18,19,20,22,23,24,25]:
            # main(train_log_name=f'{cfg.exp_name}_{i}.csv')
            executor.submit(main, f'{cfg.exp_name}_{i}.csv')
            time.sleep(30)


        # if os.path.exists('weights/deviser.npy'):
        #     os.remove('weights/deviser.npy')
        # if os.path.exists('weights/decider.npy'):
        #     os.remove('weights/decider.npy')

        # main(train_log_name=f'{cfg.exp_name}.csv')

        # if os.path.exists('weights/deviser.npy'):
        #     os.remove('weights/deviser.npy')
        # if os.path.exists('weights/decider.npy'):
        #     os.remove('weights/decider.npy')