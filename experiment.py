import os
import subprocess
import time
import csv
from config import Config as cfg
from train import append_to_csv
import pandas as pd


def get_last_gain():
    try:
        df = pd.read_csv('train_log.csv')
        if not df.empty:
            return float(df['gain'].iloc[-1])
    except FileNotFoundError:
        return 0.0

def main():
    for end_train in range(cfg.exp_train_len, cfg.exp_data_len+cfg.exp_test_len, cfg.exp_test_len):
        os.environ['START_TRAIN'] = str( end_train - cfg.exp_train_len )

        append_to_csv(pd.DataFrame({
            'time':time.time(),
            'exec_time':-1,
            'gain':-1,
            'grads':-1,
            'start_data':[int(os.environ['START_TRAIN'])]
        }))
        # Run get_data.py
        subprocess.run(['python', 'get_data.py'])
        print('-'*20)
        
        append_to_csv(pd.DataFrame({
            'time':time.time(),
            'exec_time':0,
            'gain':0,
            'grads':0,
            'start_data':[int(os.environ['START_TRAIN'])]
        }))

        # Start train.py in the background
        train_process = subprocess.Popen(['python', 'train.py'])
        print('-'*20)
        time.sleep(10)
        print('*'*20)

        while True:
            try:
                # Check last row from train_log.csv
                last_gain = get_last_gain()
                
                # Check if gain is greater than 100
                if last_gain > cfg.exp_return_stop:
                    print(f"Experiment completed. Gain: {last_gain}")
                    # Terminate the background process
                    train_process.terminate()
                    break
                
                # Wait for a while before the next iteration
                time.sleep(0.5)

            except Exception as e:
                print(e)
                train_process.terminate()
                break

if __name__ == "__main__":
    main()