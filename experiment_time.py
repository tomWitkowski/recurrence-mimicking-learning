import os
os.environ['train_log_name']='0'
import tensorflow as tf

# for the time comparison we use standard numerical precision
os.environ['PRECISION'] = precision = '32'
tf.keras.mixed_precision.set_global_policy(f"float{precision}")  # turn off mixed/auto FP16
tf.keras.backend.set_floatx(f'float{precision}') 

import importlib
import time
import pandas as pd
from config import Config as cfg
import src.model
from src.get_data import get_done_data

if not cfg.gpu:
    tf.config.set_visible_devices([], 'GPU')
    print("GPUs have been disabled")
else:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print(gpu_devices)
    try:
        for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print('Could not set memory growth GPU') 

def initiate_agent():
    """
    Re-initialize the agent to avoid any cashing issues.
    """
    importlib.reload(src.model) 
    _, _, XV_init, BA_init = get_done_data(limit=10)
    agent=src.model.Agent(input_len = XV_init.shape[1:])    
    agent.set_lr(0.0)
    # run first time on any data to initialize fully
    _, _, _, _ = agent.train_iteration(XV_init, BA_init, offline=False, online_learning=False)
    return agent


def main(train_log_name='results/train_log.csv'):
    """
    Main routine that runs in parallel for each experiment instance.
    """
    if not os.path.exists('weights/'):
        os.makedirs('weights')
    if os.path.exists('weights/deviser.npy'):
        os.remove('weights/deviser.npy')
    if os.path.exists('weights/decider.npy'):
        os.remove('weights/decider.npy')

    _, _, XV, BA = get_done_data(limit=5_000)
    agent = initiate_agent()

    start = time.time()
    _, decisions, _, _ = agent.train_iteration(XV, BA,
                offline=False, 
                online_learning=False,
                forward_only=True)

    end = time.time()
    print(f"RML Time: {end-start}, {agent.time_counter[-1]}")
    print('Decision path: ', decisions[-50:])

    for I in range(10):
        agent = initiate_agent()

        start = time.time()
        _, decisions, _, _ = agent.train_iteration(XV, BA,
                    offline=True, 
                    online_learning=False, count_time=True,
                    forward_only=True)

        end = time.time()
        print(f"Offline Time: {end-start}")
        print('Decision path: ', decisions[-50:])
        times = pd.Series(agent.time_counter)
        times.index = range(1,len(times)+1)
        os.makedirs('results', exist_ok=True)
        times.to_csv(f'results/time_counter_offline_{I}.csv')

        agent = initiate_agent()

        ## RML loop to check times, just some of them to restrict number of runs without loss of generality
        times = []
        for i in list(range(3,100))+list(range(100,1000,10))+list(range(1000,5_001,100)):
            _, _, XV, BA = get_done_data(limit=i)
            _, decisions, _, _ = agent.train_iteration(XV, BA,
                        offline=False, 
                        online_learning=False,
                        forward_only=True)

            times.append([i,agent.time_counter[-1]])

        print(f"RML Time: {agent.time_counter[-1]}")
        pd.DataFrame(times, columns=['i','time']).set_index('i').to_csv(f'results/time_counter_rml_{I}.csv')





if __name__ == "__main__":
    main()
