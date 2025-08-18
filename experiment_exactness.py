import os
os.environ['train_log_name']='3'
import tensorflow as tf

# for this experiment we increase precision to avoid numerical issues
os.environ['PRECISION'] = precision = '64'
tf.keras.mixed_precision.set_global_policy(f"float{precision}")  # turn off mixed/auto FP16
tf.keras.backend.set_floatx(f'float{precision}') 

import time, importlib
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

    _, _, XV, BA = get_done_data(limit=200)
    
    rewards_offline = []
    rewards_rml = []

    agent=src.model.Agent(input_len = XV.shape[1:])    
    agent.set_lr(0.001)

    for e in range(100):
        grads, decisions, reward, _ = agent.train_iteration(XV, BA,
                    offline=False, 
                    online_learning=False,
                    forward_only=False)
        print(e,reward.numpy())
        rewards_offline.append(reward.numpy())

    print('--'*20)

    importlib.reload(src.model) 
    agent2=src.model.Agent(input_len = XV.shape[1:])    
    agent2.set_lr(0.001)

    for e in range(100):
        grads, decisions, reward, _ = agent2.train_iteration(XV, BA,
                    offline=True, 
                    online_learning=False, count_time=True,
                    forward_only=False)
        print(e,reward.numpy())
        rewards_rml.append(reward.numpy())
        
    df = pd.DataFrame({
        'epoch': range(1, len(rewards_offline) + 1),
        'reward_offline': rewards_offline,
        'reward_rml': rewards_rml
    })
    print(df)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/exactness.csv', index=False)


if __name__ == "__main__":
    main()
